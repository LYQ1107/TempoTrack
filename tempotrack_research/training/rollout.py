"""On-policy S5 rollout collection and PPO updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import json

import numpy as np
import torch
from torch import Tensor, nn

from ..association.edit_env import EditAction, GraphEditEnv, TrainingRewardOracle
from ..losses.policy import ppo_loss
from ..schemas import ActionTable, GraphInputs
from .checkpoint import AtomicCheckpoint


@dataclass
class RolloutBuffer:
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def transitions(self) -> int:
        return len(self.records)

    def as_tensors(self, device: torch.device | str) -> dict[str, Tensor]:
        if not self.records:
            raise ValueError("cannot tensorize an empty rollout")
        device = torch.device(device)
        values: dict[str, Tensor] = {}
        for name in ("old_logprob", "old_value", "reward", "advantage", "returns"):
            values[name] = torch.as_tensor([float(record[name]) for record in self.records], dtype=torch.float32, device=device)
        values["actions"] = torch.as_tensor([int(record["action_index"]) for record in self.records], dtype=torch.long, device=device)
        values["terminated"] = torch.as_tensor([bool(record["terminated"]) for record in self.records], dtype=torch.bool, device=device)
        values["collector_truncated"] = torch.as_tensor([bool(record["collector_truncated"]) for record in self.records], dtype=torch.bool, device=device)
        return values


def compute_gae(records: list[dict[str, Any]], gamma: float = 1.0, gae_lambda: float = 0.95) -> None:
    if not records:
        return
    # Rollout records are interleaved across environments.  GAE must follow
    # each environment's trajectory, not use the next record from another
    # video as an implicit successor.
    streams: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        streams.setdefault(int(record["env_index"]), []).append(record)
    for stream in streams.values():
        advantage = 0.0
        for record in reversed(stream):
            if record["terminated"] or record["truncated"] or record.get("collector_truncated", False):
                continuation = 0.0
            else:
                continuation = 1.0
            bootstrap = 0.0 if record["terminated"] or record["truncated"] else float(record["next_value"])
            delta = float(record["reward"]) + gamma * bootstrap - float(record["old_value"])
            advantage = delta + gamma * gae_lambda * continuation * advantage
            record["advantage"] = advantage
            record["returns"] = advantage + float(record["old_value"])


class RolloutCollector:
    """Sample concrete actions from the current policy for every rollout."""

    def _graph_inputs(self, env: GraphEditEnv, observation: Mapping[str, Any], device: torch.device) -> GraphInputs:
        source = getattr(env, "policy_inputs", None)
        if callable(source):
            source = source(observation)
        if source is None:
            source = observation
        required = ("node_features", "edge_features", "edge_index")
        if not all(name in source for name in required):
            raise ValueError("S5 rollout environment must expose deployment graph features")
        node = torch.as_tensor(source["node_features"], dtype=torch.float32, device=device)
        edge = torch.as_tensor(source["edge_features"], dtype=torch.float32, device=device)
        index = torch.as_tensor(source["edge_index"], dtype=torch.long, device=device)
        if node.ndim == 2:
            node = node.unsqueeze(0)
        if edge.ndim == 2:
            edge = edge.unsqueeze(0)
        if index.ndim == 2:
            index = index.unsqueeze(0)
        node_valid = torch.as_tensor(source.get("node_valid", np.ones(node.shape[1], dtype=bool)), dtype=torch.bool, device=device)
        edge_valid = torch.as_tensor(source.get("edge_valid", observation["edge_valid"]), dtype=torch.bool, device=device)
        initial = torch.as_tensor(source.get("initial_graph", np.zeros(edge.shape[1])), dtype=torch.float32, device=device)
        node_times = source.get("node_times")
        node_times_tensor = None if node_times is None else torch.as_tensor(node_times, dtype=torch.float32, device=device)
        if node_valid.ndim == 1:
            node_valid = node_valid.unsqueeze(0)
        if edge_valid.ndim == 1:
            edge_valid = edge_valid.unsqueeze(0)
        if initial.ndim == 1:
            initial = initial.unsqueeze(0)
        return GraphInputs(node, edge, index, node_valid, edge_valid, initial, node_times_tensor)

    def _pack_policy_batch(
        self,
        entries: Sequence[Mapping[str, Any]],
        device: torch.device,
    ) -> tuple[GraphInputs, Tensor, ActionTable, Tensor, list[GraphInputs]]:
        """Pad deployment graphs/action tables for one batched policy call."""
        if not entries:
            raise ValueError("cannot pack an empty rollout batch")
        graphs = [entry["graph"] for entry in entries]
        tables = [entry["table"] for entry in entries]
        batch_size = len(entries)
        max_nodes = max(int(graph.node_features.shape[1]) for graph in graphs)
        max_edges = max(1, max(int(graph.edge_features.shape[1]) for graph in graphs))
        max_actions = max(1, max(len(table["kind"]) for table in tables))
        node_dim = int(graphs[0].node_features.shape[-1])
        edge_dim = int(graphs[0].edge_features.shape[-1])
        node_features = torch.zeros((batch_size, max_nodes, node_dim), dtype=torch.float32, device=device)
        edge_features = torch.zeros((batch_size, max_edges, edge_dim), dtype=torch.float32, device=device)
        edge_index = torch.zeros((batch_size, 2, max_edges), dtype=torch.long, device=device)
        node_valid = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
        edge_valid = torch.zeros((batch_size, max_edges), dtype=torch.bool, device=device)
        initial_graph = torch.zeros((batch_size, max_edges), dtype=torch.float32, device=device)
        selected_edges = torch.zeros((batch_size, max_edges), dtype=torch.float32, device=device)
        action_kind = torch.full((batch_size, max_actions), 3, dtype=torch.long, device=device)
        action_edge = torch.full((batch_size, max_actions), -1, dtype=torch.long, device=device)
        action_replacement = torch.full((batch_size, max_actions), -1, dtype=torch.long, device=device)
        action_valid = torch.zeros((batch_size, max_actions), dtype=torch.bool, device=device)
        remaining = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        for row, (graph, table, entry) in enumerate(zip(graphs, tables, entries)):
            nodes = int(graph.node_features.shape[1])
            edges = int(graph.edge_features.shape[1])
            node_features[row, :nodes] = graph.node_features[0]
            node_valid[row, :nodes] = graph.node_valid[0]
            if edges:
                edge_features[row, :edges] = graph.edge_features[0]
                edge_index[row, :, :edges] = graph.edge_index[0]
                edge_valid[row, :edges] = graph.edge_valid[0]
                initial_graph[row, :edges] = graph.initial_graph[0]
                selected_edges[row, :edges] = torch.as_tensor(entry["observation"]["selected_edges"], dtype=torch.float32, device=device)
            actions = len(table["kind"])
            action_kind[row, :actions] = torch.as_tensor(table["kind"], dtype=torch.long, device=device)
            action_edge[row, :actions] = torch.as_tensor(table["edge_index"], dtype=torch.long, device=device)
            action_replacement[row, :actions] = torch.as_tensor(table["replacement_edge_index"], dtype=torch.long, device=device)
            action_valid[row, :actions] = torch.as_tensor(table["valid"], dtype=torch.bool, device=device)
            remaining[row] = float(entry["observation"]["remaining_budget"])
        return (
            GraphInputs(node_features, edge_features, edge_index, node_valid, edge_valid, initial_graph),
            selected_edges,
            ActionTable(action_kind, action_edge, action_replacement, action_valid),
            remaining,
            graphs,
        )

    def _policy_batch(
        self,
        policy: nn.Module,
        entries: Sequence[Mapping[str, Any]],
        device: torch.device,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        graph, selected, action_table, remaining, _ = self._pack_policy_batch(entries, device)
        with torch.no_grad():
            output = policy(graph, selected, action_table, remaining)
            action = policy.act(output, deterministic=False)
        return output, action["action_index"], action["logprob"], action["value"]

    def collect(self, policy: nn.Module, envs: Sequence[GraphEditEnv], reward_oracles: Sequence[TrainingRewardOracle], *, transitions: int, policy_version: int, device: torch.device | str | None = None) -> RolloutBuffer:
        if transitions < 1 or not envs:
            raise ValueError("rollout needs a positive transition budget and at least one environment")
        device = torch.device(device or next(policy.parameters()).device)
        if len(envs) != len(reward_oracles):
            raise ValueError("one reward oracle is required per environment")
        observations = [env.reset() for env in envs]
        done = [False] * len(envs)
        buffer = RolloutBuffer()
        policy.eval()
        while buffer.transitions < transitions:
            entries: list[dict[str, Any]] = []
            for slot, (env, oracle) in enumerate(zip(envs, reward_oracles)):
                if buffer.transitions + len(entries) >= transitions:
                    break
                if done[slot]:
                    observations[slot] = env.reset()
                    done[slot] = False
                observation = observations[slot]
                table = env.action_table()
                graph = self._graph_inputs(env, observation, device)
                entries.append({"slot": slot, "env": env, "oracle": oracle, "observation": observation, "before": env.selected.copy(), "table": table, "graph": graph})

            if not entries:
                break
            _output, action_indices, logprobs, values = self._policy_batch(policy, entries, device)
            next_entries: list[dict[str, Any]] = []
            round_records: list[dict[str, Any]] = []
            for row, entry in enumerate(entries):
                env = entry["env"]
                oracle = entry["oracle"]
                table = entry["table"]
                action_index = int(action_indices[row].item())
                concrete = table["actions"][action_index]
                observation = entry["observation"]
                next_observation, _environment_reward, episode_done, info = env.step(concrete)
                reward = oracle.reward(env.edge_index, entry["before"], env.selected, action_kind=concrete.kind_name)
                # At an explicit collection boundary an unfinished task is
                # collector-truncated; it is bootstrapped from its true next
                # state, never from a reset state.
                next_value = 0.0
                if not episode_done:
                    next_graph = self._graph_inputs(env, next_observation, device)
                    next_table = env.action_table()
                    next_entries.append({"slot": entry["slot"], "env": env, "oracle": oracle, "observation": next_observation, "before": env.selected.copy(), "table": next_table, "graph": next_graph})
                round_records.append({
                    "env_index": int(entry["slot"]),
                    "policy_version": int(policy_version),
                    "action_index": action_index,
                    "action": concrete,
                    "action_table": {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in table.items() if key != "actions"},
                    "observation": observation,
                    "next_observation": next_observation,
                    "old_logprob": float(logprobs[row].item()),
                    "old_value": float(values[row].item()),
                    "reward": float(reward),
                    "next_value": 0.0,
                    "terminated": bool(info.get("terminated", False)),
                    "truncated": bool(info.get("truncated", False)),
                    "collector_truncated": False,
                    "invalid_action": False,
                    "graph": {
                        "node_features": entry["graph"].node_features.detach().cpu(),
                        "edge_features": entry["graph"].edge_features.detach().cpu(),
                        "edge_index": entry["graph"].edge_index.detach().cpu(),
                        "node_valid": entry["graph"].node_valid.detach().cpu(),
                        "edge_valid": entry["graph"].edge_valid.detach().cpu(),
                        "initial_graph": entry["graph"].initial_graph.detach().cpu(),
                        "node_times": None if entry["graph"].node_times is None else entry["graph"].node_times.detach().cpu(),
                    },
                    "selected_edges": torch.as_tensor(observation["selected_edges"], dtype=torch.float32).unsqueeze(0),
                    "remaining_budget": torch.as_tensor([float(observation["remaining_budget"])], dtype=torch.float32),
                })
                observations[entry["slot"]] = next_observation
                done[entry["slot"]] = bool(episode_done)

            if next_entries:
                _next_output, _next_actions, _next_logprobs, next_values = self._policy_batch(policy, next_entries, device)
                next_by_slot = {int(entry["slot"]): float(next_values[row].item()) for row, entry in enumerate(next_entries)}
            else:
                next_by_slot = {}
            for record in round_records:
                record["next_value"] = float(next_by_slot.get(int(record["env_index"]), 0.0))
                buffer.records.append(record)
        last_by_env: dict[int, dict[str, Any]] = {}
        for record in buffer.records:
            last_by_env[int(record["env_index"])] = record
        for record in last_by_env.values():
            if not record["terminated"] and not record["truncated"]:
                record["collector_truncated"] = True
        compute_gae(buffer.records)
        return buffer


class PPOTrainer:
    """PPO updater which discards each buffer before collecting a new one."""

    def __init__(self, policy: nn.Module, collector: RolloutCollector | None = None):
        self.policy = policy
        self.collector = collector or RolloutCollector()

    def update(self, buffer: RolloutBuffer, optimizer: torch.optim.Optimizer, *, epochs: int = 4, clip_ratio: float = 0.2, entropy_weight: float = 0.01, value_weight: float = 0.5) -> dict[str, float]:
        device = next(self.policy.parameters()).device
        tensors = buffer.as_tensors(device)
        metrics: dict[str, float] = {}
        self.policy.train()
        entries: list[dict[str, Any]] = []
        for record in buffer.records:
            graph_value = record["graph"]
            graph = GraphInputs(
                graph_value["node_features"], graph_value["edge_features"], graph_value["edge_index"],
                graph_value["node_valid"], graph_value["edge_valid"], graph_value["initial_graph"],
                graph_value["node_times"],
            )
            remaining = record["remaining_budget"]
            entries.append({
                "graph": graph,
                "table": record["action_table"],
                "observation": {
                    "selected_edges": record["selected_edges"][0].numpy(),
                    "remaining_budget": float(remaining.reshape(-1)[0]),
                },
            })
        batch_size = max(1, int(getattr(self, "ppo_microbatch_size", 64)))
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            for start in range(0, len(entries), batch_size):
                stop = min(start + batch_size, len(entries))
                graph, selected, table, remaining, _ = self.collector._pack_policy_batch(entries[start:stop], device)
                output = self.policy(graph, selected, table, remaining)
                new_logprob = self.policy.distribution(output).log_prob(tensors["actions"][start:stop])
                entropy = self.policy.distribution(output).entropy()
                values = ppo_loss(
                    new_logprob,
                    tensors["old_logprob"][start:stop],
                    tensors["advantage"][start:stop],
                    output["value"],
                    tensors["returns"][start:stop],
                    entropy,
                    clip_ratio,
                    entropy_weight,
                    value_weight,
                )
                # Each mini-batch contributes its exact fraction of the
                # rollout loss; gradients are accumulated before one PPO
                # optimizer step.  The size is configuration-controlled and
                # does not alter the concrete action trajectories.
                loss = values["total"] * ((stop - start) / max(len(entries), 1))
                loss.backward()
                for name, value in values.items():
                    if torch.is_tensor(value) and value.ndim == 0:
                        sums[name] = sums.get(name, 0.0) + float(value.detach()) * (stop - start)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0))
            optimizer.step()
            for name, value in sums.items():
                metrics[name] = value / max(len(buffer.records), 1)
            metrics["grad_norm"] = grad_norm
        metrics["clip_fraction"] = float(metrics.get("ratio_mean", 1.0) > 1.2 or metrics.get("ratio_mean", 1.0) < 0.8)
        metrics["transitions"] = float(buffer.transitions)
        return metrics

    def train(
        self,
        run_spec: Any,
        *,
        bc_checkpoint: str | None,
        envs: Sequence[GraphEditEnv] | None = None,
        reward_oracles: Sequence[TrainingRewardOracle] | None = None,
        transitions: int | None = None,
        checkpoint_path: str | Path | None = None,
        progress_path: str | Path | None = None,
        resume: str = "never",
        expected: Mapping[str, Any] | None = None,
        on_update: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if bc_checkpoint is None:
            raise FileNotFoundError("PPO requires a verified BC checkpoint")
        if envs is None or reward_oracles is None:
            raise RuntimeError("PPO requires real deployment environments and external training reward oracles")
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=float(getattr(run_spec, "optimizer", {}).get("lr", 2e-4)))
        self.optimizer = optimizer
        train_config = getattr(run_spec, "train", {})
        self.ppo_microbatch_size = max(1, int(train_config.get("ppo_microbatch_size", train_config.get("microbatch_size", 64))))
        budget = int(transitions or getattr(run_spec, "train", {}).get("ppo_transitions", 50000))
        version = 0
        total = 0
        updates = 0
        history: list[dict[str, Any]] = []

        # PPO has its own resumable lineage.  A partially completed rollout
        # is intentionally discarded on restart, while the last completed
        # update restores model, optimizer, RNG, transition count and policy
        # version.  This keeps resume exact at update boundaries and never
        # silently reloads the preceding BC weights over a PPO checkpoint.
        ppo_checkpoint = None if checkpoint_path is None else AtomicCheckpoint(checkpoint_path)
        if resume == "auto" and ppo_checkpoint is not None and ppo_checkpoint.path.exists():
            payload = ppo_checkpoint.load(
                self.policy,
                optimizer,
                expected=expected,
                map_location=next(self.policy.parameters()).device,
            )
            metadata = dict(payload.get("metadata", {}))
            total = int(metadata.get("transitions", 0))
            updates = int(payload.get("optimizer_step", metadata.get("ppo_updates", 0)))
            version = int(metadata.get("policy_version", updates))
            saved_history = metadata.get("ppo_history", [])
            if isinstance(saved_history, list):
                history = [dict(item) for item in saved_history if isinstance(item, Mapping)]
            if progress_path is not None and Path(progress_path).exists():
                progress = json.loads(Path(progress_path).read_text(encoding="utf-8"))
                if int(progress.get("transitions", total)) != total:
                    raise ValueError("PPO progress/checkpoint transition counts disagree")
            if total > budget:
                raise ValueError(f"PPO checkpoint transitions {total} exceed requested budget {budget}")
        else:
            payload = torch.load(bc_checkpoint, map_location=next(self.policy.parameters()).device, weights_only=False)
            state = payload.get("model", payload.get("model_state"))
            if state is None:
                raise ValueError("BC checkpoint has no model state")
            self.policy.load_state_dict(state, strict=True)

        def persist(state: Mapping[str, Any]) -> None:
            if on_update is not None:
                on_update(state)

        while total < budget:
            version += 1
            buffer = self.collector.collect(self.policy, envs, reward_oracles, transitions=min(2048, budget - total), policy_version=version)
            metrics = self.update(
                buffer,
                optimizer,
                epochs=int(getattr(run_spec, "train", {}).get("ppo_epochs", 4)),
                clip_ratio=float(getattr(run_spec, "train", {}).get("clip", 0.2)),
                entropy_weight=float(getattr(run_spec, "train", {}).get("entropy", 0.01)),
                value_weight=float(getattr(run_spec, "train", {}).get("value", 0.5)),
            )
            total += buffer.transitions
            updates += 1
            action_counts: dict[str, int] = {}
            for record in buffer.records:
                action = record.get("action")
                kind = action.kind_name if isinstance(action, EditAction) else str(record.get("action_kind", "UNKNOWN"))
                action_counts[kind] = action_counts.get(kind, 0) + 1
            history.append({"policy_version": version, **metrics, "action_counts": action_counts})
            persist({"status": "RUNNING", "transitions": total, "updates": updates, "policy_version": version, "history": history})
        cumulative_actions: dict[str, int] = {}
        for item in history:
            for kind, count in dict(item.get("action_counts", {})).items():
                cumulative_actions[str(kind)] = cumulative_actions.get(str(kind), 0) + int(count)
        return {"status": "COMPLETED", "transitions": total, "updates": updates, "history": history, "action_counts": cumulative_actions}
