"""Executable training runtime for the repaired TempoTrack suite.

Training consumes reference-based JSONL episodes backed by immutable
Detic/MASA ledgers.  There is intentionally no NPZ batch fallback: every
optimizer step obtains a (possibly repeated after a new epoch) record from a
real episode loader and the run artifact records its source hashes.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..config import file_hash, object_hash
from ..data.datasets import (
    ContinuationEpisodeDataset,
    EditDemonstrationDataset,
    GraphEpisodeDataset,
    MemoryEpisodeDataset,
    PairEpisodeDataset,
    collate_training_batches,
)
from ..errors import DataUnavailable
from ..losses.policy import masked_behavior_cloning_loss
from ..memory.predictive_dual import PredictiveDualMemory
from ..models.continuation_flow import ContinuationFlowModel, SuccessorStateTransform
from ..models.edit_policy import EditPolicy
from ..models.graph_diffusion import GraphDiffusionMatcher
from ..models.graph_flow import GraphFlowMatcher
from ..models.identity_predictor import JEPAIdentityLinker, PairMetricLinker
from ..schemas import ActionTable, GraphInputs, RunSpec
from ..association.edit_env import GraphEditEnv, TrainingRewardOracle
from .checkpoint import AtomicCheckpoint
from .engine import TrainConfig, TrainingEngine, seed_everything
from .memory_trainer import MemoryInputs, MemoryTargets, MemoryTrainingTask
from .rollout import PPOTrainer


_BUDGETS = {
    "ordinary_metric": (3000, 100000),
    "s1_jepa": (3000, 100000),
    "s2_state_fm": (3000, 100000),
    "s3_graph_fm": (3000, 120000),
    "s4_graph_diffusion": (3000, 120000),
    "predictive_dual": (3000, 60000),
    "s5_rl_edit": (3000, 30000),
}


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _first(value: Any) -> Any:
    """Unwrap the one-element list introduced for non-tensor table fields."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _collect_episode_uids(value: Any, output: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"episode_uid", "uid"}:
                if isinstance(item, str) and item:
                    output.add(item)
                elif isinstance(item, (list, tuple)):
                    output.update(str(candidate) for candidate in item if isinstance(candidate, str) and candidate)
            else:
                _collect_episode_uids(item, output)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_episode_uids(item, output)


def _episode_kind(method: str) -> str:
    return {
        "ordinary_metric": "pair",
        "s1_jepa": "pair",
        "predictive_dual": "memory",
        "s2_state_fm": "continuation",
        "s3_graph_fm": "graph",
        "s4_graph_diffusion": "graph",
        "s5_rl_edit": "edit",
    }.get(method, method)


def _dataset_class(kind: str) -> type:
    return {
        "pair": PairEpisodeDataset,
        "metric": PairEpisodeDataset,
        "memory": MemoryEpisodeDataset,
        "continuation": ContinuationEpisodeDataset,
        "graph": GraphEpisodeDataset,
        "edit": EditDemonstrationDataset,
    }[kind]


def _resolve_episode_manifest(run_spec: RunSpec | None, repo: Path, kind: str, episode_manifest: str | Path | None, split: str) -> Path:
    candidates: list[Path] = []
    if episode_manifest:
        candidates.append(Path(episode_manifest))
    if run_spec is not None:
        for key in ("episode_manifest", "episodes_manifest"):
            value = run_spec.data.get(key)
            if value:
                candidates.append(Path(value))
    candidates.extend([
        repo / "outputs" / "research_v2" / "episodes" / split / "episodes_manifest.json",
        repo / "outputs" / "research" / "episodes" / split / "episodes_manifest.json",
        repo / "outputs" / "research_v2" / "episodes" / "episodes_manifest.json",
    ])
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = repo / candidate
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if "files" in payload:
            return candidate
        item = payload.get("kinds", {}).get(kind)
        if item and item.get("files"):
            return candidate
    joined = ", ".join(str(item) for item in candidates)
    raise DataUnavailable(f"missing real {kind} episode manifest; checked {joined}")


def _materialize_kind_manifest(overall: Path, kind: str, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(overall.read_text(encoding="utf-8"))
    if "files" in payload:
        return overall, payload
    item = dict(payload.get("kinds", {}).get(kind, {}))
    if not item.get("files") or not int(item.get("count", 0)):
        raise DataUnavailable(f"episode kind {kind} is not ready in {overall}")
    # Resolve relative files against the overall manifest, then write a
    # run-local manifest whose references are explicit and immutable.
    files = []
    for value in item["files"]:
        path = Path(value)
        files.append(str(path if path.is_absolute() else overall.parent / path))
    item["files"] = files
    item["kind"] = kind
    path = run_dir / f"{kind}_manifest.json"
    _atomic_json(item, path)
    return path, item


def _model_config(run_spec: RunSpec | None, method: str) -> dict[str, Any]:
    values = dict(run_spec.model if run_spec else {})
    defaults = {
        "hidden_dim": 64,
        "layers": 1,
        "heads": 4,
        "ff_dim": 256,
        "dynamic_dim": 32,
        "graph_hidden_dim": 64,
        "graph_layers": 2,
        "latent_dim": 64,
        "diffusion_steps": 1000,
        "memory_hidden_dim": 64,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    if method in {"s3_graph_fm", "s4_graph_diffusion", "s5_rl_edit"}:
        values.setdefault("hidden_dim", values["graph_hidden_dim"])
    return values


def _make_model(method: str, sample: Mapping[str, Any], config: Mapping[str, Any]) -> nn.Module:
    if method == "ordinary_metric":
        return PairMetricLinker(
            int(sample["left_appearance"].shape[-1]), int(config["hidden_dim"]),
            int(config["layers"]), int(config["heads"]), int(config["ff_dim"]),
            dynamic_dim=int(config["dynamic_dim"]),
        )
    if method == "s1_jepa":
        return JEPAIdentityLinker(
            int(sample["context_appearance"].shape[-1]), int(config["hidden_dim"]),
            int(config["layers"]), int(config["heads"]), int(config["ff_dim"]),
            int(config["dynamic_dim"]),
        )
    if method == "predictive_dual":
        dim = int(sample["observation"].shape[-1])
        return MemoryTrainingTask(
            PredictiveDualMemory(
                history_dim=2 * dim,
                observation_dim=dim,
                evidence_dim=8,
                hidden_dim=int(config["memory_hidden_dim"]),
            ),
            unroll=int(config.get("unroll", 16)),
            loss_weights=config.get("loss_weights"),
        )
    if method == "s2_state_fm":
        return ContinuationFlowModel(
            latent_dim=int(config.get("latent_dim", 64)),
            hidden_dim=int(config["hidden_dim"]),
            layers=int(config["layers"]),
            appearance_dim=int(sample["source_appearance"].shape[-1]),
        )
    if method == "s3_graph_fm":
        return GraphFlowMatcher(
            int(sample["node_features"].shape[-1]), int(sample["edge_features"].shape[-1]),
            int(config["graph_hidden_dim"]), int(config["graph_layers"]),
        )
    if method == "s4_graph_diffusion":
        return GraphDiffusionMatcher(
            int(sample["node_features"].shape[-1]), int(sample["edge_features"].shape[-1]),
            int(config["graph_hidden_dim"]), int(config["graph_layers"]), int(config["diffusion_steps"]),
        )
    if method == "s5_rl_edit":
        return EditPolicy(
            int(sample["node_features"].shape[-1]), int(sample["edge_features"].shape[-1]),
            int(config["graph_hidden_dim"]),
        )
    raise ValueError(f"no real model factory for {method}")


def _graph_inputs(batch: Mapping[str, Any], device: torch.device) -> GraphInputs:
    return GraphInputs(
        batch["node_features"].to(device), batch["edge_features"].to(device),
        batch["edge_index"].long().to(device), batch["node_valid"].bool().to(device),
        batch["edge_valid"].bool().to(device), batch["initial_graph"].float().to(device),
        None,
    )


def _action_table(batch: Mapping[str, Any], device: torch.device) -> ActionTable:
    raw = batch["action_table"]
    if not isinstance(raw, Mapping):
        raise ValueError("edit episode did not provide a complete action table")
    fields = {name: torch.as_tensor(raw[name], device=device) for name in ("kind", "edge_index", "replacement_edge_index", "valid")}
    if fields["kind"].ndim == 1:
        fields = {name: value.unsqueeze(0) for name, value in fields.items()}
    if any(value.ndim != 2 for value in fields.values()):
        raise ValueError("collated ActionTable fields must have shape [B,A]")
    shape = fields["kind"].shape
    if any(value.shape != shape for value in fields.values()):
        raise ValueError("collated ActionTable fields have inconsistent shapes")
    return ActionTable(
        fields["kind"].long(), fields["edge_index"].long(),
        fields["replacement_edge_index"].long(), fields["valid"].bool(),
    )


def _s2_transform(dataset: ContinuationEpisodeDataset, run_dir: Path, config: Mapping[str, Any]) -> SuccessorStateTransform:
    path = run_dir / "state_transform.json"
    if path.exists():
        return SuccessorStateTransform(int(dataset[0]["source_appearance"].shape[-1]), snapshot=json.loads(path.read_text(encoding="utf-8")))
    residuals: list[Tensor] = []
    for index in range(len(dataset)):
        item = dataset[index]
        source = item["source_appearance"].float().mean(0)
        target = item["target_appearance"].float().mean(0)
        residuals.append(target - source)
    if len(residuals) < 2:
        raise DataUnavailable("S2 state transform requires at least two real continuation episodes")
    transform = SuccessorStateTransform(int(residuals[0].shape[-1]), mode=str(config.get("state_transform_mode", "pca")))
    snapshot = transform.fit(torch.stack(residuals)).to_dict()
    _atomic_json(snapshot, path)
    return transform


def _loss_for(method: str, model: nn.Module, batch: Mapping[str, Any], *, device: torch.device, state_transform: SuccessorStateTransform | None = None, generator: torch.Generator | None = None) -> Mapping[str, Tensor]:
    if method == "ordinary_metric":
        return model.compute_loss(batch, {"same_identity": batch["same_identity"], "candidate_known": batch["candidate_known"]})
    if method == "s1_jepa":
        return model.compute_loss(batch, {"positive": batch["positive"]})
    if method == "predictive_dual":
        inputs = MemoryInputs(
            prototype=batch["prototype"], observations=batch["appearance"],
            history_states=batch["history_state"], causal_evidence=batch["causal_evidence"],
            frames=batch["relative_time"], bboxes=batch["geometry"],
        )
        targets = MemoryTargets(
            future_embedding=batch["future_embedding"],
            positive_mask=batch["positive_mask"],
            candidate_known=batch["candidate_known"],
            reliability=batch["reliability"],
            reliability_known=batch["reliability_known"],
            valid=batch.get("valid_steps"),
        )
        return model(inputs, targets)
    if method == "s2_state_fm":
        if state_transform is None:
            raise ValueError("S2 loss requires a fitted state transform")
        source = {"appearance": batch["source_appearance"], "geometry": batch["source_geometry"], "relative_time": batch["source_time"], "valid": batch["source_valid"]}
        target = {"appearance": batch["target_appearance"], "geometry": batch["target_geometry"], "relative_time": batch["target_time"], "valid": batch["target_valid"]}
        source_valid = batch["source_valid"].bool()
        target_valid = batch["target_valid"].bool()
        source_last = source["source_time" if False else "relative_time"]
        source_last_index = source_valid.long().sum(-1) - 1
        target_first_index = torch.zeros(target_valid.shape[0], dtype=torch.long, device=device)
        source_last_time = source["relative_time"][torch.arange(source_valid.shape[0], device=device), source_last_index]
        target_first_time = target["relative_time"][torch.arange(target_valid.shape[0], device=device), target_first_index]
        gap = target_first_time - source_last_time
        target_state = state_transform.encode_target(source, target)
        condition = model.encode_condition(source, gap)
        return model.compute_loss(
            torch.zeros_like(target_state), target_state, condition,
            batch.get("exists"), existence_known=batch.get("existence_known"),
            target_state_valid=batch.get("target_state_valid"), generator=generator,
        )
    if method in {"s3_graph_fm", "s4_graph_diffusion"}:
        graph = _graph_inputs(batch, device)
        if method == "s3_graph_fm":
            return model.compute_loss(graph.target_graph if hasattr(graph, "target_graph") else batch["target_graph"].float(), graph.node_features, graph.edge_features, graph.edge_index, graph.edge_valid, graph.node_valid, graph.initial_graph, generator=generator, loss_edge_mask=batch.get("target_graph_known"))
        return model.compute_loss(batch["target_graph"].float().to(device), graph.node_features, graph.edge_features, graph.edge_index, graph.edge_valid, graph.initial_graph, graph.node_valid, generator=generator, loss_edge_mask=batch.get("target_graph_known"))
    if method == "s5_rl_edit":
        graph = _graph_inputs(batch, device)
        table = _action_table(batch, device)
        selected = batch["selected_edges"].float().to(device)
        remaining = batch["remaining_budget"].float().reshape(-1).to(device)
        output = model(graph, selected, table, remaining)
        target = batch["actions"].long().reshape(-1).to(device)
        mask = table.valid
        loss = masked_behavior_cloning_loss(output["logits"], target, mask)
        return {"total": loss, "bc": loss.detach(), "action_count": mask.sum().detach()}
    raise ValueError(f"no loss registered for {method}")


def _build_ppo_envs(dataset: EditDemonstrationDataset, *, max_edits: int | None = None) -> tuple[list[GraphEditEnv], list[TrainingRewardOracle]]:
    envs: list[GraphEditEnv] = []
    oracles: list[TrainingRewardOracle] = []
    for record in dataset.records:
        metadata = dict(record.get("metadata", {}))
        identities = np.asarray(metadata.get("node_identities_loss_only", []), dtype=np.int64)
        edges = np.asarray(record.get("edge_index", []), dtype=np.int64).reshape(2, -1)
        valid = np.asarray(record.get("edge_valid", [True] * edges.shape[1]), dtype=bool)
        initial = np.asarray(record.get("initial_graph", [False] * edges.shape[1]), dtype=bool)
        env = GraphEditEnv(len(identities), edges, valid, max_edits=max_edits if max_edits is not None else max(1, 2 * len(identities)))
        try:
            env.reset(selected=initial)
        except ValueError:
            # The graph builder normally emits a legal initial graph.  An
            # invalid serialized initial state is a data-contract failure,
            # not permission to introduce an oracle graph.
            raise DataUnavailable(f"edit episode {record.get('episode_uid')} has an illegal initial graph")
        sample = dataset[dataset.records.index(record)]
        env.policy_inputs = {
            "node_features": sample["node_features"].numpy(),
            "edge_features": sample["edge_features"].numpy(),
            "edge_index": sample["edge_index"].numpy(),
            "node_valid": sample["node_valid"].numpy(),
            "edge_valid": sample["edge_valid"].numpy(),
            "initial_graph": sample["initial_graph"].numpy(),
        }
        envs.append(env)
        oracles.append(TrainingRewardOracle(identities, identities >= 0))
    if not envs:
        raise DataUnavailable("PPO requires at least one real edit episode")
    return envs, oracles


def run_available_training(
    repo: str | Path,
    method: str,
    frontend: str,
    profile: str,
    seed: int,
    resume: str = "auto",
    device: str | None = None,
    *,
    run_spec: RunSpec | None = None,
    phase: str | None = None,
    episode_manifest: str | Path | None = None,
    run_root: str | Path | None = None,
    max_steps: int | None = None,
    ppo_transitions: int | None = None,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    seed_everything(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    method = str(method)
    phase = phase or ("bc" if method == "s5_rl_edit" else "train")
    kind = _episode_kind(method)
    data = dict(run_spec.data if run_spec else {})
    split = str(data.get("episode_split", data.get("split", "train_base")))
    overall_manifest = _resolve_episode_manifest(run_spec, repo, kind, episode_manifest or data.get("episode_manifest"), split)
    base_run_root = Path(run_root or (run_spec.run_root if run_spec else repo / "outputs" / "research_v2" / "runs"))
    # Public CLI ``--run-root`` names the experiment root (which contains
    # prepared/, episodes/, runs/, ...).  Preserve direct ``.../runs`` paths
    # for older explicit invocations while keeping suite and standalone runs
    # on the same lineage.
    if base_run_root.name != "runs":
        base_run_root = base_run_root / "runs"
    # The profile is a stopping policy, not a new run lineage.  A trial can
    # therefore resume into the same full-schedule checkpoint instead of
    # restarting with a short cosine schedule under a different directory.
    run_dir = base_run_root / f"{frontend}_{method}_{phase}_seed{int(seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    kind_manifest, manifest_payload = _materialize_kind_manifest(overall_manifest, kind, run_dir)
    dataset = _dataset_class(kind)(kind_manifest, transform_snapshot=data.get("transform_snapshot"), cache_videos=int(data.get("cache_videos", 2)), epoch=0)
    if len(dataset) < 2:
        raise DataUnavailable(f"{kind} training loader has only {len(dataset)} episode; at least two distinct records are required")
    loader_generator = torch.Generator().manual_seed(int(seed))
    train_config = dict(run_spec.train if run_spec else {})
    microbatch_size = int(train_config.get("microbatch_size", train_config.get("batch_size", 1)))
    accumulation_steps = int(train_config.get("accumulation_steps", 1))
    if microbatch_size < 1 or accumulation_steps < 1:
        raise ValueError("microbatch_size and accumulation_steps must be positive")
    effective_batch = int(train_config.get("effective_batch", microbatch_size * accumulation_steps))
    if effective_batch != microbatch_size * accumulation_steps:
        raise ValueError("effective_batch must equal microbatch_size * accumulation_steps")
    loader = DataLoader(
        dataset, batch_size=microbatch_size, shuffle=True,
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=bool(train_config.get("pin_memory", False)),
        collate_fn=collate_training_batches, generator=loader_generator,
    )
    sample = dataset[0]
    config = _model_config(run_spec, method)
    data_hash = object_hash({"manifest": file_hash(overall_manifest), "kind": kind, "count": len(dataset)})
    metadata = {
        "schema_version": 2,
        "method": method,
        "frontend": frontend,
        "phase": phase,
        "profile": profile,
        "seed": int(seed),
        "episode_manifest": str(overall_manifest),
        "episode_kind": kind,
        "episode_count": len(dataset),
        "data_hash": data_hash,
        "model_config": config,
        "input_contract": "frozen_predicted_boxes_masa_features_gt_identity_supervision_only",
        "loader_config": {"microbatch_size": microbatch_size, "accumulation_steps": accumulation_steps, "effective_batch": effective_batch, "num_workers": int(train_config.get("num_workers", 0))},
    }
    _atomic_json(metadata, run_dir / "resolved_run.json")

    # S5 PPO uses the same real graph episodes and the preceding BC
    # checkpoint.  It is an on-policy loop, not a repeated offline batch.
    if method == "s5_rl_edit" and phase == "ppo":
        model = _make_model(method, sample, config).to(selected_device)
        bc_checkpoint = data.get("bc_checkpoint") or (run_spec.provenance.get("bc_checkpoint") if run_spec else None)
        if not bc_checkpoint:
            raise DataUnavailable("S5 PPO requires an explicit verified BC checkpoint")
        configured_max_edits = train_config.get("max_edits")
        max_edits = None if configured_max_edits is None else int(configured_max_edits)
        envs, oracles = _build_ppo_envs(dataset, max_edits=max_edits)
        trainer = PPOTrainer(model)
        declared_transitions = train_config.get("ppo_transitions", data.get("ppo_transitions", 50000))
        ppo_checkpoint = AtomicCheckpoint(run_dir / "last.pt")
        ppo_progress = run_dir / "ppo_progress.json"

        def persist_ppo_update(state: Mapping[str, Any]) -> None:
            checkpoint_metadata = {
                **metadata,
                "transitions": int(state["transitions"]),
                "ppo_updates": int(state["updates"]),
                "policy_version": int(state["policy_version"]),
                "ppo_history": list(state["history"]),
                "checkpoint_boundary": "completed_ppo_update",
            }
            ppo_checkpoint.save(
                model,
                trainer.optimizer,
                metadata=checkpoint_metadata,
                optimizer_step=int(state["updates"]),
                attempted_steps=int(state["updates"]),
                components={"ppo": {"policy_versions": int(state["policy_version"]), "transitions": int(state["transitions"])}},
            )
            _atomic_json(dict(state), ppo_progress)

        result = trainer.train(
            run_spec or type("Run", (), {"optimizer": {}, "train": {"ppo_transitions": ppo_transitions or 50000}})(),
            bc_checkpoint=str(bc_checkpoint), envs=envs, reward_oracles=oracles,
            transitions=int(ppo_transitions if ppo_transitions is not None else declared_transitions),
            checkpoint_path=ppo_checkpoint.path,
            progress_path=ppo_progress,
            resume=resume,
            expected={"method": method, "frontend": frontend, "data_hash": data_hash},
            on_update=persist_ppo_update,
        )
        optimizer = getattr(trainer, "optimizer", None)
        ppo_checkpoint.save(model, optimizer, metadata={**metadata, "transitions": result.get("transitions", 0), "ppo_updates": result.get("updates", 0), "policy_version": result.get("updates", 0), "ppo_history": result.get("history", []), "checkpoint_boundary": "completed_ppo_run"}, optimizer_step=int(result.get("updates", 0)), attempted_steps=int(result.get("updates", 0)), components={"ppo": {"policy_versions": result.get("updates", 0), "transitions": result.get("transitions", 0)}})
        _atomic_json({**result, "checkpoint": str(ppo_checkpoint.path)}, ppo_progress)
        _atomic_json(result, run_dir / "train_result.json")
        return {"status": "COMPLETED", "run_dir": str(run_dir), "checkpoint": str(ppo_checkpoint.path), **result}

    state_transform = _s2_transform(dataset, run_dir, config) if method == "s2_state_fm" else None
    model = _make_model(method, sample, config).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float((run_spec.optimizer if run_spec else {}).get("lr", 2e-4)), weight_decay=float((run_spec.optimizer if run_spec else {}).get("weight_decay", 0.05)))
    # Always build the scheduler for the declared full budget.  Integration
    # and trial runs only stop earlier; they must not exhaust a 3k-step cosine
    # schedule and then masquerade as a resumable 100k-step run.
    declared_steps = train_config.get("full_steps" if profile == "full" else "trial_steps")
    if declared_steps is None:
        declared_steps = _BUDGETS.get(method, (3000, 3000))[1 if profile == "full" else 0]
    requested_steps = int(max_steps if max_steps is not None else declared_steps)
    if requested_steps < 1:
        raise ValueError("requested training steps must be positive")
    schedule_steps = int(train_config.get("full_steps", _BUDGETS.get(method, (requested_steps, requested_steps))[1]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, schedule_steps))
    engine = TrainingEngine(
        model, optimizer,
        TrainConfig(
            max_steps=requested_steps,
            validate_every=int((run_spec.train if run_spec else {}).get("validate_every", 0)),
            save_every=int((run_spec.train if run_spec else {}).get("save_every", 500)),
            grad_clip=float((run_spec.train if run_spec else {}).get("grad_clip", 1.0)),
            accumulation_steps=accumulation_steps,
            amp=str((run_spec.train if run_spec else {}).get("amp", "bf16_if_supported")),
        ),
        selected_device, scheduler=scheduler,
    )
    checkpoint = AtomicCheckpoint(run_dir / "last.pt")
    if resume == "auto" and checkpoint.path.exists():
        payload = checkpoint.load(model, optimizer, scheduler=scheduler, scaler=engine.scaler, expected={"method": method, "frontend": frontend, "data_hash": data_hash}, map_location=selected_device)
        engine.global_step = int(payload.get("optimizer_step", payload.get("metadata", {}).get("global_step", 0)))
        engine.optimizer_steps = int(payload.get("optimizer_step", engine.global_step))
        engine.attempted_steps = int(payload.get("attempted_steps", engine.optimizer_steps))
        engine.epoch = int(payload.get("epoch", 0))
        engine.consumed_batch_cursor = int(payload.get("consumed_batch_cursor", 0))

    metrics_path = run_dir / "metrics.jsonl"
    distinct_uids: set[str] = set()

    def loss_fn(batch: Mapping[str, Any]) -> Mapping[str, Tensor]:
        moved = _move(batch, selected_device)
        _collect_episode_uids(moved.get("metadata", {}), distinct_uids)
        return _loss_for(method, model, moved, device=selected_device, state_transform=state_transform)

    def on_step(step: int, values: Mapping[str, float]) -> None:
        if method == "s1_jepa" and values.get("optimizer_step_success", 0.0) > 0:
            model.update_target(True, optimizer_step=step, schedule_steps=requested_steps)
        record = {"step": int(step), "method": method, "frontend": frontend, "phase": phase, "seed": int(seed), **{str(k): float(v) for k, v in values.items()}}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        if step == 1 or step % max(1, engine.config.save_every) == 0 or step == requested_steps:
            checkpoint.save(
                model, optimizer, scheduler=scheduler, scaler=engine.scaler,
                metadata={**metadata, "global_step": step, "optimizer_steps": engine.optimizer_steps},
                optimizer_step=engine.optimizer_steps, attempted_steps=engine.attempted_steps,
                epoch=engine.epoch, consumed_batch_cursor=engine.consumed_batch_cursor,
                sampler_state={"seed": int(seed), "distinct_episode_uids": sorted(distinct_uids)},
                ema_schedule={"momentum": float(model.current_momentum(requested_steps))} if method == "s1_jepa" else {},
                components={"state_transform": state_transform.snapshot().to_dict()} if state_transform is not None else {},
            )

    result = engine.run(loader, loss_fn, on_step=on_step)
    checkpoint.save(
        model, optimizer, scheduler=scheduler, scaler=engine.scaler,
        metadata={**metadata, "global_step": engine.global_step, "optimizer_steps": engine.optimizer_steps},
        optimizer_step=engine.optimizer_steps, attempted_steps=engine.attempted_steps,
        epoch=engine.epoch, consumed_batch_cursor=engine.consumed_batch_cursor,
        sampler_state={"seed": int(seed), "distinct_episode_uids": sorted(distinct_uids)},
        ema_schedule={"momentum": float(model.current_momentum(requested_steps))} if method == "s1_jepa" else {},
        components={"state_transform": state_transform.snapshot().to_dict()} if state_transform is not None else {},
    )
    result_payload = {
        "status": "COMPLETED", "method": method, "frontend": frontend, "phase": phase,
        "profile": profile, "seed": int(seed), "run_dir": str(run_dir),
        "checkpoint": str(checkpoint.path), "episode_manifest": str(overall_manifest),
        "episode_count": len(dataset), "distinct_episode_uids": len(distinct_uids),
        "optimizer_steps": int(engine.optimizer_steps), "requested_steps": requested_steps,
        "data_hash": data_hash, **result,
    }
    _atomic_json(result_payload, run_dir / "train_result.json")
    return result_payload


__all__ = ["DataUnavailable", "run_available_training"]
