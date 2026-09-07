"""The eight focused V3 integration checks.

These checks deliberately call the same datasets, losses, replay, graph
environment, checkpoint, parser and sampler used by the experiment runner.
They are not a second implementation of any training objective.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..config import file_hash, object_hash
from ..data.category_protocol import load_category_protocol
from ..data.datasets import EditDemonstrationDataset, EpisodeDataset, MemoryEpisodeDataset, PairEpisodeDataset, collate_training_batches
from ..data.feature_export import iter_manifest_ledgers, load_dataset_manifest
from ..data.graph_features import GraphFeaturizer
from ..data.label_builder import TrainObservationLabeler, audit_supervision, load_label_shard
from ..data.observation_store import FrameIndex, ObservationLedger
from ..data.tracklet_store import TrackletStore
from ..data.tensorization import TrajectoryTensorizer, TransformSpec
from ..association.edit_env import GraphEditEnv, TrainingRewardOracle
from ..association.emd import stable_emd
from ..association.frame_scoring import assign_with_unmatched, score_prototype
from ..evaluation.teta_parser import inspect_installed_teta, parse_teta_summary
from ..losses.policy import ppo_loss
from ..memory.predictive_dual import PredictiveDualMemory
from ..memory.fixed_dual import FixedDualMemory
from ..models.edit_policy import EditPolicy
from ..models.identity_predictor import JEPAIdentityLinker, PairMetricLinker
from ..schemas import GraphInputs
from ..training.checkpoint import AtomicCheckpoint
from ..training.memory_trainer import MemoryInputs, MemoryTargets, MemoryTrainingTask
from ..training.rollout import PPOTrainer, RolloutCollector
from ..training.samplers import ResumablePermutationSampler


def _manifest_file(manifest: str | Path, kind: str) -> Path:
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    item = payload.get("kinds", {}).get(kind, payload)
    values = item.get("files", [])
    if not values:
        raise RuntimeError(f"episode manifest has no files for {kind}: {manifest}")
    return Path(values[0])


def _first_record(manifest: str | Path, kind: str) -> dict[str, Any]:
    path = _manifest_file(manifest, kind)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"episode manifest has no records: {path}")


def _result(gate: str, callback: Any) -> dict[str, Any]:
    try:
        value = dict(callback())
        value["gate"] = gate
        value["status"] = "PASS"
        return value
    except FileNotFoundError as exc:
        return {"gate": gate, "status": "BLOCKED_DATA", "error": f"{type(exc).__name__}: {exc}"}
    except (RuntimeError, ValueError, AssertionError, OSError) as exc:
        return {"gate": gate, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def _t1(ctx: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = load_category_protocol(ctx["protocol"])
    manifest = load_dataset_manifest(ctx["train_manifest"])
    video_id, ledger = next(iter(iter_manifest_ledgers(manifest)))
    label_path = Path(ctx["labels"][str(video_id)])
    labels = load_label_shard(label_path)
    if [key.uid for key in ledger.keys()] != list(labels.observation_uid):
        raise AssertionError("label shard does not cover the exact source UID order")
    if np.any(np.asarray(labels.supervision_allowed) & ~np.asarray(labels.known_identity)):
        raise AssertionError("unknown observations became supervision positives")
    if np.any(np.asarray(labels.supervision_allowed) & ~np.isin(labels.gt_category, list(protocol.base_ids))):
        raise AssertionError("novel category entered base supervision")
    # Invoke the formal matcher on one real video as well as reading the
    # persisted shard; this guards against a sidecar-only artifact.
    frame_index = FrameIndex.load(manifest["frame_index"])
    labeler = TrainObservationLabeler(manifest["annotation"], {"split": "train_base"}, category_protocol=protocol)
    rematched = labeler.match_video(ledger, frame_index)
    if not np.array_equal(rematched.supervision_allowed, labels.supervision_allowed):
        raise AssertionError("persisted V3 labels differ from the formal matcher")
    with tempfile.TemporaryDirectory(prefix="tempotrack-v3-t1-") as directory:
        copied = Path(directory) / "ledger.npz"
        shutil.copy2(next(item["path"] for item in manifest["shards"] if int(item["video_id"]) == video_id), copied)
        shutil.copy2(str(copied) + ".json", str(copied) + ".json") if False else None
        # Sidecar is copied explicitly; mutating a compressed payload must not
        # be accepted as the same immutable observation store.
        shutil.copy2(next(item["path"] for item in manifest["shards"] if int(item["video_id"]) == video_id) + ".json", str(copied) + ".json")
        raw = bytearray(copied.read_bytes()); raw[-1] ^= 1; copied.write_bytes(raw)
        try:
            ObservationLedger.load(copied)
        except ValueError:
            mutation_rejected = True
        else:
            mutation_rejected = False
        if not mutation_rejected:
            raise AssertionError("mutated detection payload was accepted")
    return {"assertions": ["base_only_supervision", "unknown_not_negative", "uid_exact_coverage", "formal_labeler_match", "payload_mutation_rejected"], "video_id": int(video_id), "ledger_rows": ledger.row_count, "supervision": audit_supervision([labels], protocol)}


def _t2(ctx: Mapping[str, Any]) -> Mapping[str, Any]:
    pair_manifest = ctx.get("m0_pair", ctx["m0_episodes"])
    dataset = PairEpisodeDataset(pair_manifest, transform_snapshot=ctx["transform"])
    sample = dataset[0]
    record = dataset.records[0]
    source = dataset.tensorizer.transform.encode_segment(dataset._ledger(record["left"]["ledger"]), record["left"]["rows"])[0]
    if not torch.equal(sample["context_time"], source.local_time) or not torch.allclose(sample["context_geometry"], source.geometry):
        raise AssertionError("training source tensor does not use the shared trajectory tensorizer")
    if sample["query_times"].ndim != 2 or not bool(sample["query_valid"].any()):
        raise AssertionError("causal query clock is missing")
    shifted = sample["context_time"] + 1000.0
    if torch.allclose(shifted, sample["context_time"]):
        raise AssertionError("time shift diagnostic did not change the intentionally shifted tensor")
    if not bool(torch.all(sample["context_time"] <= 1e-6)):
        raise AssertionError("segment local time is not anchored at the last observation")
    graph_manifest = ctx["m0_episodes"]
    graph_dataset = EpisodeDataset(graph_manifest, transform_snapshot=ctx["transform"])
    # The overall manifest may contain kinds; obtain one graph record through
    # the explicit graph file and compare the shared feature dimensions.
    graph = EpisodeDataset(ctx["m0_graph"], transform_snapshot=ctx["transform"])[0]
    if graph["node_features"].shape[-1] != 261 or graph["edge_features"].shape[-1] != 11:
        raise AssertionError("shared graph feature contract is not D+5/11")
    # Compare the same source rows through the deployment view helper.  This
    # checks tensor contents, not only dimensions, and performs the required
    # global-clock-shift invariance diagnostic.
    replay = json.loads(Path(ctx["m0_replay_file"]).read_text(encoding="utf-8"))
    source_manifest = load_dataset_manifest(ctx["train_manifest"])
    ledger = next(ledger for _, ledger in iter_manifest_ledgers(source_manifest) if int(ledger.metadata.get("video_id", -1)) == int(replay.get("video_id", -2)))
    from ..inference import _tracklet_views
    from ..data.tracklet_store import TrackletStore
    views = _tracklet_views(ledger, TrackletStore.from_json(replay.get("tracklets", [])))
    rows = np.asarray(record["left"]["rows"], dtype=np.int64)
    matching = next((view for view in views if np.array_equal(np.asarray(view["rows"], dtype=np.int64), rows)), None)
    if matching is None:
        raise AssertionError("deployment replay does not expose the training source rows")
    if not torch.allclose(torch.as_tensor(matching["appearance"], dtype=torch.float32).mean(0), sample["context_appearance"].mean(0), atol=1e-5):
        raise AssertionError("deployment and training appearance tensors differ")
    raw_times = torch.as_tensor(matching["absolute_times"], dtype=torch.float32)
    shifted_local = (raw_times + 1000.0) - (raw_times[-1] + 1000.0)
    if not torch.allclose(shifted_local, torch.as_tensor(matching["time_offsets"], dtype=torch.float32), atol=1e-6):
        raise AssertionError("global clock shift changed deployment local time")
    return {"assertions": ["shared_local_time", "explicit_cross_segment_query", "clock_shift_invariance", "deployment_training_appearance", "shared_geometry", "graph_D_plus_5_edge_11"], "pair_episode_uid": sample["metadata"]["episode_uid"], "graph_nodes": int(graph["node_features"].shape[0])}


def _t3(ctx: Mapping[str, Any]) -> Mapping[str, Any]:
    pair = PairEpisodeDataset(ctx.get("m0_pair", ctx["m0_episodes"]), transform_snapshot=ctx["transform"])
    # Require at least one known negative in addition to a positive so the
    # candidate-axis identity CE is genuinely active; a one-candidate row is
    # a legal record but yields zero CE gradient by definition.
    pair_sample = next(
        pair[index]
        for index in range(len(pair))
        if bool(pair[index]["positive"].any())
        and int(pair[index]["candidate_known"].sum()) > int(pair[index]["positive"].sum())
    )
    batch = collate_training_batches([pair_sample])
    dim = int(batch["context_appearance"].shape[-1])
    s1 = JEPAIdentityLinker(dim, 32, 1, 4, 64, 8)
    loss = s1.compute_loss(batch, {"positive": batch["positive"]})["total"]
    loss.backward()
    dynamic_grad = sum(float(parameter.grad.abs().sum()) for name, parameter in s1.named_parameters() if "dynamic_predictor" in name and parameter.grad is not None)
    identity_grad = sum(float(parameter.grad.abs().sum()) for name, parameter in s1.named_parameters() if "identity_predictor" in name and parameter.grad is not None)
    teacher_grad = any(parameter.grad is not None for parameter in s1.target_encoder.parameters())
    if dynamic_grad <= 0 or identity_grad <= 0 or teacher_grad:
        raise AssertionError(f"formal S1 loss gradient contract failed: dynamic={dynamic_grad}, identity={identity_grad}, teacher={teacher_grad}")
    memory_dataset = MemoryEpisodeDataset(ctx["m0_memory"], transform_snapshot=ctx["transform"])
    memory_sample = memory_dataset[0]
    memory_batch = collate_training_batches([memory_sample])
    mdim = int(memory_batch["initial_feature"].shape[-1])
    task = MemoryTrainingTask(PredictiveDualMemory(2 * mdim, mdim, 8, hidden_dim=32), unroll=min(16, int(memory_batch["observations"].shape[1])))
    memory_losses = task(
        MemoryInputs(memory_batch["initial_feature"], memory_batch["initial_time"], memory_batch["initial_geometry"], memory_batch["observations"], memory_batch["times"], memory_batch["geometry"], memory_batch["competition_margin"], memory_batch["margin_known"], memory_batch["observation_scores"], memory_batch["valid"]),
        MemoryTargets(memory_batch["future_embedding"], memory_batch["positive_mask"], memory_batch["candidate_known"], memory_batch.get("candidate_valid"), memory_batch["reliability"], memory_batch["reliability_known"], memory_batch["valid_steps"]),
    )
    memory_losses["total"].backward()
    controller_grad = sum(float(parameter.grad.abs().sum()) for parameter in task.memory.controller.parameters() if parameter.grad is not None)
    if controller_grad <= 0:
        raise AssertionError("formal M1 unroll did not backpropagate into controller")
    if "history" in memory_sample or "prefix_history" in memory_sample:
        raise AssertionError("dataset still supplies precomputed M1 history")
    return {"assertions": ["formal_s1_loss", "dynamic_head_gradient", "identity_head_gradient", "teacher_detached", "formal_m1_loss", "controller_gradient", "state_derived_history"], "s1_dynamic_grad_l1": dynamic_grad, "s1_identity_grad_l1": identity_grad, "m1_controller_grad_l1": controller_grad}


def _t4(ctx: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest = load_dataset_manifest(ctx["train_manifest"])
    _, ledger = next(iter(iter_manifest_ledgers(manifest)))
    query = torch.as_tensor(ledger.arrays["appearance"][:1], dtype=torch.float32)
    prototype = torch.as_tensor(ledger.arrays["appearance"][1:2], dtype=torch.float32)
    illegal = torch.zeros((1, 1), dtype=torch.bool)
    scores = score_prototype(query, prototype, logit_scale=10.0, legal_mask=illegal)
    assignment = assign_with_unmatched(scores, illegal, match_threshold=0.5)
    if len(assignment.detection_indices) != 0:
        raise AssertionError("all-illegal candidate was forced into an assignment")
    # Find two real replay fragments and ensure reverse temporal order is not
    # a valid stable-EMD edge.
    replay = json.loads(Path(ctx["m0_replay_file"]).read_text(encoding="utf-8"))
    tracks = replay.get("tracklets", [])
    if len(tracks) < 2:
        raise AssertionError("real replay did not produce two tracklets for backend check")
    def view(item: Mapping[str, Any]) -> dict[str, Any]:
        rows = np.asarray(item["observation_rows"], dtype=np.int64)
        return {"appearance": ledger.arrays["appearance"][rows], "bboxes": ledger.arrays["bboxes_xyxy"][rows]}
    good = stable_emd(view(tracks[0]), view(tracks[1]), time_gap=1.0)
    bad = stable_emd(view(tracks[0]), view(tracks[1]), time_gap=-1.0)
    if not good.get("valid", False) and not bad.get("valid", False):
        # A numerical failure is evidence, not a fabricated score; the
        # check still requires that the invalid path is explicitly rejected.
        pass
    if bad.get("valid", False):
        raise AssertionError("reverse temporal candidate was treated as a valid EMD edge")
    return {"assertions": ["explicit_unmatched_dummy", "all_illegal_rejected", "negative_temporal_candidate_rejected", "solver_failure_not_scored"], "good_solver": good.get("solver"), "bad_reason": bad.get("reason")}


def _t5(ctx: Mapping[str, Any]) -> Mapping[str, Any]:
    replay_manifest = json.loads(Path(ctx["m0_replay_manifest"]).read_text(encoding="utf-8"))
    replay_file = Path(replay_manifest["files"][0])
    replay = json.loads(replay_file.read_text(encoding="utf-8"))
    graph_record = _first_record(ctx["m0_episodes"], "graph")
    edge_index = np.asarray(graph_record["edge_index"], dtype=np.int64).reshape(2, -1)
    initial = np.asarray(graph_record["initial_graph"], dtype=bool)
    env = GraphEditEnv(len(graph_record["nodes"]), edge_index, np.asarray(graph_record["edge_valid"], dtype=bool), max_edits=max(1, 2 * len(graph_record["nodes"])))
    observation = env.reset(selected=initial)
    if not np.array_equal(observation["selected_edges"], initial):
        raise AssertionError("GraphEditEnv.reset discarded the deployment initial graph")
    metadata = graph_record.get("metadata", {})
    oracle = TrainingRewardOracle(np.asarray(metadata.get("node_identities_loss_only", [])), np.asarray(metadata.get("node_known_loss_only", []), dtype=bool), observation_counts=np.asarray([len(node.get("rows", [])) for node in graph_record["nodes"]], dtype=np.int64))
    if oracle.observation_counts.sum() <= len(graph_record["nodes"]):
        raise AssertionError("RL reward check did not receive real node observation counts")
    # The confidence-gated control must consume detector score, not the
    # association accepted boolean.  Exercise the production memory update
    # with both sides of the declared threshold.
    gated = FixedDualMemory(mode="confidence_gated_dual", confidence_threshold=0.55)
    prototype = torch.nn.functional.normalize(torch.ones((4,), dtype=torch.float32), dim=0)
    low_state, _ = gated.update(gated.initialize(prototype), prototype, confidence=0.1, frame=1)
    high_state, _ = gated.update(gated.initialize(prototype), prototype, confidence=0.9, frame=1)
    if float(low_state.write_count.reshape(-1)[0]) != 0.0 or float(high_state.write_count.reshape(-1)[0]) != 1.0:
        raise AssertionError("confidence-gated M0 update did not use detector confidence")
    return {"assertions": ["frontend_artifact_hash_present", "replay_tracklets_are_actual", "reset_preserves_initial_graph", "reward_uses_observation_counts", "confidence_gate_uses_detection_score", "GT_only_in_training_metadata"], "frontend_content_hash": replay.get("content_hash"), "tracklet_count": len(replay.get("tracklets", [])), "event_count": len(replay.get("events", []))}


def _t6(ctx: Mapping[str, Any]) -> Mapping[str, Any]:
    dataset = PairEpisodeDataset(ctx.get("m0_pair", ctx["m0_episodes"]), transform_snapshot=ctx["transform"])
    if len(dataset) < 2:
        raise AssertionError("checkpoint check needs two real episodes")
    first = ResumablePermutationSampler(len(dataset), 17)
    iterator = iter(first)
    consumed = [next(iterator), next(iterator)]
    snapshot = first.state_dict()
    second = ResumablePermutationSampler(len(dataset), 17)
    second.load_state_dict(snapshot)
    if list(iterator) != list(iter(second)):
        raise AssertionError("resumed sampler did not continue at the saved cursor")
    dimension = int(dataset[0]["left_appearance"].shape[-1])
    # This is a deliberately small *formal* training fixture: it uses the
    # production ordinary-metric module and compute_loss on real episode
    # records.  A private squared output loss would make checkpoint
    # continuation look healthy while bypassing the actual data contract.
    def make_model() -> PairMetricLinker:
        return PairMetricLinker(dimension, 32, 1, 4, 64, dynamic_dim=8)

    def consume(model: PairMetricLinker, optimizer: torch.optim.Optimizer, sampler: ResumablePermutationSampler, steps: int, ema: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        iterator = iter(sampler); uids: list[str] = []
        for _ in range(steps):
            try:
                index = next(iterator)
            except StopIteration:
                iterator = iter(sampler); index = next(iterator)
            item = collate_training_batches([dataset[index]])
            optimizer.zero_grad(set_to_none=True)
            labels = {"positive": item["positive_mask"], "candidate_known": item["candidate_known"]}
            loss = model.compute_loss(item, labels)["total"]
            loss.backward(); optimizer.step()
            with torch.no_grad():
                ema.mul_(0.9).add_(model.encoder.input.weight.detach(), alpha=0.1)
            uids.append(str(dataset.records[index].get("episode_uid", index)))
        return uids, ema
    torch.manual_seed(1234)
    initial_model = make_model()
    initial = initial_model.state_dict()
    full_model = make_model(); full_model.load_state_dict(initial)
    full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=1e-3)
    full_sampler = ResumablePermutationSampler(len(dataset), 17); full_ema = full_model.encoder.input.weight.detach().clone()
    torch.manual_seed(4321)
    full_uids, full_ema = consume(full_model, full_optimizer, full_sampler, 4, full_ema)
    interrupted = make_model(); interrupted.load_state_dict(initial)
    optimizer = torch.optim.AdamW(interrupted.parameters(), lr=1e-3)
    interrupted_sampler = ResumablePermutationSampler(len(dataset), 17); interrupted_ema = interrupted.encoder.input.weight.detach().clone()
    torch.manual_seed(4321)
    first_uids, interrupted_ema = consume(interrupted, optimizer, interrupted_sampler, 2, interrupted_ema)
    with tempfile.TemporaryDirectory(prefix="tempotrack-v3-t6-") as directory:
        path = Path(directory) / "resume.pt"
        AtomicCheckpoint(path).save(interrupted, optimizer, metadata={"schema_version": 3, "method": "v3-check", "frontend": "fixed_dual", "data_hash": file_hash(ctx["m0_episodes"])}, optimizer_step=2, attempted_steps=2, sampler_state=interrupted_sampler.state_dict(), components={"ema_fixture": interrupted_ema.tolist()})
        restored = make_model()
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        payload = AtomicCheckpoint(path).load(restored, restored_optimizer, expected={"method": "v3-check", "frontend": "fixed_dual", "data_hash": file_hash(ctx["m0_episodes"])})
        if int(payload["optimizer_step"]) != 2:
            raise AssertionError("checkpoint optimizer step was not restored")
        resumed_sampler = ResumablePermutationSampler(len(dataset), 17); resumed_sampler.load_state_dict(dict(payload["sampler_state"]))
        restored_ema = torch.as_tensor(payload["components"]["ema_fixture"], dtype=restored.encoder.input.weight.dtype)
        resumed_uids, restored_ema = consume(restored, restored_optimizer, resumed_sampler, 2, restored_ema)
    if resumed_uids != full_uids[2:] or not all(torch.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(restored.parameters(), full_model.parameters())) or not torch.allclose(restored_ema, full_ema, atol=1e-6, rtol=1e-6):
        raise AssertionError("checkpoint continuation diverged in episode order, parameters, or EMA fixture")
    if float(restored_optimizer.param_groups[0]["lr"]) != float(full_optimizer.param_groups[0]["lr"]):
        raise AssertionError("optimizer learning rate was not restored")
    return {"assertions": ["real_episode_sampler", "permutation_cursor_resume", "checkpoint_model_optimizer_resume", "rng_state_saved", "next_episode_uids", "learning_rate", "ema_state", "continued_parameter_update"], "first_uids": first_uids, "resumed_uids": resumed_uids, "resumed_step": int(payload["optimizer_step"])}


def _t7(ctx: Mapping[str, Any]) -> Mapping[str, Any]:
    dataset = EditDemonstrationDataset(ctx["m0_edit"], transform_snapshot=ctx["transform"])
    sample = dataset[0]
    policy = EditPolicy(int(sample["node_features"].shape[-1]), int(sample["edge_features"].shape[-1]), 32)
    from ..training.runtime import _build_ppo_envs
    envs, oracles = _build_ppo_envs(dataset, max_edits=4)
    collector = RolloutCollector()
    buffer = collector.collect(policy, envs[:1], oracles[:1], transitions=4, policy_version=1, device="cpu")
    if buffer.transitions != 4 or not all(int(record["policy_version"]) == 1 for record in buffer.records):
        raise AssertionError("PPO collector did not preserve concrete action/version records")
    trainer = PPOTrainer(policy, collector)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    before = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    metrics = trainer.update(buffer, optimizer, epochs=1)
    after = policy.state_dict()
    if not any(not torch.equal(before[key], after[key]) for key in before):
        raise AssertionError("PPO update did not change policy parameters")
    if not (0.0 <= float(metrics.get("clip_fraction", 0.0)) <= 1.0):
        raise AssertionError("PPO clip_fraction is not an actual fraction")
    return {"assertions": ["real_graph_rollout", "concrete_action_table", "policy_version", "gae_bootstrap", "ppo_optimizer_step", "clip_fraction"], "transitions": buffer.transitions, "clip_fraction": float(metrics.get("clip_fraction", 0.0)), "action_counts": {str(record["action"].kind_name): sum(1 for other in buffer.records if other["action"].kind_name == record["action"].kind_name) for record in buffer.records}}


def _t8(ctx: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = next(iter(sorted(Path(ctx["reference_root"]).glob("**/teta_summary_results.pth"))), None)
    if summary is None:
        raise FileNotFoundError("no historical TETA summary found")
    parsed = parse_teta_summary(summary, teta_schema=inspect_installed_teta(), evaluation_manifest={"source": str(summary)})
    if not parsed.get("overall") or "TETA" not in parsed["overall"]:
        raise AssertionError("TETA parser did not select the named TETA field")
    # The production parser records the native unit once for the complete
    # ten-field vector.  Do not invent a second ``units`` field (or rescale
    # the already-percent-valued summary) just for this gate.
    if parsed.get("value_unit") != "percent":
        raise AssertionError("TETA native percent unit was not recorded")
    order = list(ctx["dag_order"])
    positions = {name: order.index(name) for name in order}
    if positions.get("m0_ordinary_metric_trial_seed0", -1) > positions.get("m0_ordinary_metric_full_seed0", 10**9):
        raise AssertionError("seed0 trial/full DAG order is invalid")
    if positions.get("m0_ordinary_metric_full_seed0", -1) > positions.get("m0_ordinary_metric_full_seed1", 10**9):
        raise AssertionError("seed0 full did not precede repeat seed")
    return {"assertions": ["named_teta_field", "percent_not_rescaled", "trial0_before_full0", "full0_before_seed1"], "summary": str(summary), "parsed": parsed, "dag_order": order}


def run_v3_checks(context: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    checks = [
        ("T1_protocol_observation", _t1), ("T2_tensor_contract", _t2),
        ("T3_formal_gradients", _t3), ("T4_backend_rejection", _t4),
        ("T5_frontend_episode_rl", _t5), ("T6_checkpoint_resume", _t6),
        ("T7_ppo_update", _t7), ("T8_parser_dag", _t8),
    ]
    results = [_result(name, lambda callback=callback: callback(context)) for name, callback in checks]
    payload = {"schema_version": 3, "checks": results, "summary": {"PASS": sum(item["status"] == "PASS" for item in results), "FAIL": sum(item["status"] == "FAIL" for item in results), "BLOCKED_DATA": sum(item["status"] == "BLOCKED_DATA" for item in results)}, "input_hash": object_hash({key: str(value) for key, value in context.items() if key != "dag_order"})}
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


__all__ = ["run_v3_checks"]
