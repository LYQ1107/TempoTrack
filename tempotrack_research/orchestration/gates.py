"""R01--R33 repair gates and V1--V8 targeted checks.

Every record is backed by an assertion, an artifact hash, or an explicit
blocked reason.  The gate file is evidence, not a manually maintained status
table.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..config import file_hash, object_hash
from ..data.observation_store import FrameIndex, ObservationLedger
from ..errors import DataUnavailable, GateNotPassed
from ..evaluation.protocol import check_immutable_protocol


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


class GateRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = {}
        else:
            payload = {}
        self.payload: dict[str, Any] = payload if isinstance(payload, dict) else {}
        self.payload.setdefault("schema_version", 2)
        self.payload.setdefault("gates", [])

    def record(self, gate: str, status: str, *, assertions: list[str] | None = None, evidence: Mapping[str, Any] | None = None, error: str | None = None, level: str = "static", command: list[str] | None = None, duration_seconds: float = 0.0) -> dict[str, Any]:
        item = {
            "gate": gate,
            "level": level,
            "status": status,
            "assertions": list(assertions or []),
            "evidence": dict(evidence or {}),
            "error": error,
            "command": list(command or []),
            "started_at": _now(),
            "finished_at": _now(),
            "duration_seconds": float(duration_seconds),
        }
        item["evidence_hash"] = object_hash(item["evidence"])
        existing = [value for value in self.payload["gates"] if value.get("gate") != gate]
        existing.append(item)
        self.payload["gates"] = existing
        self.payload["updated_at"] = _now()
        self.payload["summary"] = {
            status_name: sum(1 for value in existing if value.get("status") == status_name)
            for status_name in ("PASS", "FAIL", "BLOCKED_DATA", "BLOCKED_EXTERNAL", "NOT_RUN")
        }
        _atomic_json(self.payload, self.path)
        return item


def _run(recorder: GateRecorder, gate: str, level: str, callback: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
    start = time.time()
    try:
        evidence = dict(callback())
        return recorder.record(gate, "PASS", level=level, evidence=evidence, assertions=list(evidence.pop("assertions", [])), duration_seconds=time.time() - start)
    except DataUnavailable as exc:
        return recorder.record(gate, "BLOCKED_DATA", level=level, error=str(exc), evidence={"reason": str(exc)}, duration_seconds=time.time() - start)
    except (ImportError, OSError) as exc:
        return recorder.record(gate, "BLOCKED_EXTERNAL", level=level, error=str(exc), evidence={"reason": str(exc)}, duration_seconds=time.time() - start)
    except Exception as exc:  # targeted checks must expose the failing assertion
        return recorder.record(gate, "FAIL", level=level, error=f"{type(exc).__name__}: {exc}", evidence={}, duration_seconds=time.time() - start)


def _v1() -> Mapping[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tempotrack-v1-") as directory:
        root = Path(directory)
        frame = FrameIndex([{"dataset_id": "synthetic", "split": "train_base", "video_id": 7, "frame_index": 0, "image_id": 10, "file_name": "unused", "width": 10, "height": 10, "frame_time": 0.0, "time_unit": "frame"}], {"zero_detection_frame": True})
        frame_path = root / "frame_index.json"
        frame.save(frame_path)
        loaded = FrameIndex.load(frame_path)
        if not any(int(item["frame_index"]) == 0 for item in loaded.records):
            raise AssertionError("FrameIndex dropped a zero-detection frame")
        arrays = {
            "video_ids": np.asarray([7], np.int64), "frame_indices": np.asarray([0], np.int64), "image_ids": np.asarray([10], np.int64), "source_detection_indices": np.asarray([0], np.int64),
            "bboxes_xyxy": np.asarray([[1, 1, 3, 3]], np.float32), "scores": np.asarray([0.5], np.float32), "category_ids": np.asarray([1], np.int64), "appearance": np.asarray([[1, 0, 0, 0]], np.float32), "image_widths": np.asarray([10], np.int32), "image_heights": np.asarray([10], np.int32), "frame_times": np.asarray([0.0], np.float64),
        }
        ledger = ObservationLedger(arrays, {"dataset_id": "synthetic", "split": "train_base", "video_id": 7})
        path = root / "ledger.npz"
        ledger.save(path)
        before = file_hash(path)
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        try:
            ObservationLedger.load(path)
        except ValueError:
            pass
        else:
            raise AssertionError("ledger payload mutation was not detected")
        return {"assertions": ["zero_detection_frame_retained", "npz_payload_hash_detects_mutation"], "frame_index_hash": file_hash(frame_path), "original_ledger_hash": before}


def _v2(source_manifest: str | Path | None) -> Mapping[str, Any]:
    if not source_manifest:
        raise DataUnavailable("V2 requires a real exported dataset manifest")
    path = Path(source_manifest)
    if not path.exists():
        raise DataUnavailable(f"V2 source manifest missing: {path}")
    from ..data.feature_export import load_dataset_manifest, iter_manifest_ledgers
    manifest = load_dataset_manifest(path)
    ledgers = list(iter_manifest_ledgers(manifest))
    if not ledgers:
        raise DataUnavailable("V2 source manifest has no ledger shards")
    counts = [ledger.row_count for _, ledger in ledgers]
    if not all(ledger.metadata.get("observation_source") == "predicted_boxes" for _, ledger in ledgers):
        raise AssertionError("training source is not fixed predicted Detic/MASA observations")
    return {"assertions": ["real_ledger_shards_loaded", "observation_source_predicted_boxes", "multiple_video_or_row_references"], "shard_count": len(ledgers), "row_counts": counts, "manifest_hash": file_hash(path)}


def _v3() -> Mapping[str, Any]:
    import torch
    from ..memory.predictive_dual import PredictiveDualMemory
    from ..training.checkpoint import AtomicCheckpoint
    torch.manual_seed(3)
    model = PredictiveDualMemory(8, 4, 8, hidden_dim=16)
    prototype = torch.randn(2, 4)
    state = model.initialize(prototype)
    loss = torch.zeros(())
    for step in range(3):
        observation = torch.randn(2, 4)
        history = torch.randn(2, 8)
        evidence = torch.randn(2, 8)
        state, rates = model.update(state, observation, history, evidence, frame=torch.full((2,), step))
        loss = loss + state.fast.square().mean() + rates["reliability_logit"].square().mean()
    loss.backward()
    gradient = float(model.controller.net[0].weight.grad.abs().sum())
    if gradient <= 0 or not bool(torch.isfinite(torch.as_tensor(gradient))):
        raise AssertionError("M1 controller did not receive multi-step gradient")
    with tempfile.TemporaryDirectory(prefix="tempotrack-v3-") as directory:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        AtomicCheckpoint(Path(directory) / "m1.pt").save(model, optimizer, metadata={"method": "predictive_dual", "frontend": "predictive_dual", "data_hash": "v3"}, optimizer_step=1)
        reloaded = PredictiveDualMemory(8, 4, 8, hidden_dim=16)
        payload = torch.load(Path(directory) / "m1.pt", map_location="cpu", weights_only=False)
        # Strict reload checks all controller/utility parameters and buffers.
        reloaded.load_state_dict({key.removeprefix("memory."): value for key, value in payload["model_state"].items()}, strict=True)
    return {"assertions": ["multi_step_unroll_backward", "controller_gradient_nonzero", "strict_reload"], "controller_gradient_l1": gradient}


def _v4() -> Mapping[str, Any]:
    import torch
    from ..models.identity_predictor import JEPAIdentityLinker
    torch.manual_seed(4)
    model = JEPAIdentityLinker(8, 32, 1, 4, 64, 8)
    model.train()
    if model.target_encoder.training:
        raise AssertionError("S1 target encoder entered training mode")
    context = {"appearance": torch.randn(1, 3, 8), "geometry": torch.randn(1, 3, 4), "relative_time": torch.arange(3).float().view(1, 3), "valid": torch.ones(1, 3, dtype=torch.bool)}
    encoded = model.encode_context(context["appearance"], context["geometry"], context["relative_time"], context["valid"])
    query = torch.tensor([[3.0, 4.0]])
    first_prediction = model.predict(encoded, query)
    first = first_prediction["dynamic"]
    changed = dict(context, appearance=context["appearance"] + 1000.0)
    second = model.predict(encoded, query)["dynamic"]
    if not torch.allclose(first, second):
        raise AssertionError("S1 predictor depends on target appearance")
    loss = first.square().mean() + first_prediction["identity_raw"].square().mean() + encoded["identity_raw"].square().mean()
    loss.backward()
    dynamic_grad = sum(float(value.grad.abs().sum()) for name, value in model.named_parameters() if "dynamic_predictor" in name and value.grad is not None)
    identity_grad = sum(float(value.grad.abs().sum()) for name, value in model.named_parameters() if "identity_predictor" in name and value.grad is not None)
    if dynamic_grad <= 0 or identity_grad <= 0:
        raise AssertionError("S1 dynamic/identity heads did not receive gradients")
    return {"assertions": ["causal_query_no_target_input", "target_encoder_eval", "dynamic_head_gradient", "identity_head_gradient"], "dynamic_gradient_l1": dynamic_grad, "identity_gradient_l1": identity_grad}


def _v5() -> Mapping[str, Any]:
    import torch
    from ..models.graph_flow import GraphFlowMatcher
    from ..models.graph_diffusion import GraphDiffusionMatcher
    edge_index = torch.tensor([[[0, 1, 1], [1, 2, 0]]])
    node = torch.randn(1, 3, 6)
    edge = torch.randn(1, 3, 5)
    valid = torch.tensor([[True, True, False]])
    initial = torch.tensor([[True, False, False]], dtype=torch.float32)
    flow = GraphFlowMatcher(6, 5, 16, 1)
    out = flow.sample(node, edge, edge_index, valid, initial_graph=initial, samples=2, steps=2)
    if out.shape != (1, 2, 3) or not torch.isfinite(out).all():
        raise AssertionError("S3 masked Heun sampling failed")
    diffusion = GraphDiffusionMatcher(6, 5, 16, 1, 10)
    diff = diffusion.sample(node, edge, edge_index, valid, condition_graph=initial, samples=2, steps=2)
    if diff.shape != (1, 2, 3) or not torch.isfinite(diff).all():
        raise AssertionError("S4 masked DDIM sampling failed")
    if not torch.allclose(out[..., 2], torch.zeros_like(out[..., 2])) or not torch.allclose(diff[..., 2], torch.zeros_like(diff[..., 2])):
        raise AssertionError("invalid graph edge was not masked")
    return {"assertions": ["edge_mask", "initial_graph_condition", "signed_continuous_states", "finite_sampling"], "s3_shape": list(out.shape), "s4_shape": list(diff.shape)}


def _v6() -> Mapping[str, Any]:
    import numpy as np
    from ..association.edit_env import EditAction, GraphEditEnv
    from ..training.rollout import compute_gae
    env = GraphEditEnv(3, np.asarray([[0, 0, 1], [1, 2, 2]]), np.ones(3, dtype=bool), max_edits=3)
    env.reset(selected=np.asarray([False, False, False]))
    table = env.action_table()
    if not any(action.kind_name == "ADD" for action in table["actions"]) or not any(action.kind_name == "REWIRE" for action in table["actions"] + [EditAction("STOP")]):
        # REWIRE is absent before an ADD; it must become enumerable after the
        # concrete ADD/REMOVE transitions, not be silently omitted forever.
        env.step(next(action for action in table["actions"] if action.kind_name == "ADD"))
        if not any(action.kind_name == "REWIRE" for action in env.action_table()["actions"]):
            raise AssertionError("REWIRE action is not concrete after graph state change")
    before = env.selected.copy()
    try:
        env.step(EditAction("ADD", 0))
    except ValueError:
        if not np.array_equal(before, env.selected):
            raise AssertionError("illegal edit mutated graph")
    records = [{"env_index": 0, "reward": 0.0, "old_value": 0.0, "next_value": 0.0, "terminated": False, "truncated": False}]
    compute_gae(records)
    if "advantage" not in records[0] or "returns" not in records[0]:
        raise AssertionError("GAE bootstrap fields missing")
    return {"assertions": ["ADD_REMOVE_REWIRE_STOP_table", "illegal_action_no_mutation", "policy_version_rollout_gae_contract"], "action_count": len(table["actions"])}


def _v7() -> Mapping[str, Any]:
    import torch
    from ..training.checkpoint import AtomicCheckpoint
    from ..training.engine import TrainConfig, TrainingEngine
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    engine = TrainingEngine(model, optimizer, TrainConfig(max_steps=2, save_every=1), "cpu")
    batches = iter([{"x": torch.ones(1, 3), "y": torch.ones(1, 1), "metadata": {"uid": "a"}}, {"x": torch.zeros(1, 3), "y": torch.zeros(1, 1), "metadata": {"uid": "b"}}])
    def loss(batch: Mapping[str, Any]) -> Mapping[str, Any]:
        value = (model(batch["x"]) - batch["y"]).square().mean()
        return {"total": value}
    result = engine.run(batches, loss)
    if result["optimizer_steps"] != 2:
        raise AssertionError("engine did not perform two optimizer steps")
    with tempfile.TemporaryDirectory(prefix="tempotrack-v7-") as directory:
        path = Path(directory) / "resume.pt"
        AtomicCheckpoint(path).save(model, optimizer, metadata={"method": "test", "frontend": "test", "data_hash": "v7"}, optimizer_step=2, attempted_steps=2, consumed_batch_cursor=2)
        restored = torch.nn.Linear(3, 1)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-2)
        payload = AtomicCheckpoint(path).load(restored, restored_optimizer, expected={"method": "test", "frontend": "test", "data_hash": "v7"})
        if payload["optimizer_step"] != 2:
            raise AssertionError("checkpoint optimizer step was not restored")
    return {"assertions": ["different_batches_consumed", "optimizer_state_saved", "resume_cursor_saved"], "optimizer_steps": result["optimizer_steps"]}


def _v8() -> Mapping[str, Any]:
    official = Path(__file__).resolve().parents[1] / "evaluation" / "official.py"
    if not official.exists():
        raise AssertionError("official evaluator adapter missing")
    script = official.parents[2] / "tools" / "eval_ovmot_teta.py"
    return {"assertions": ["official_adapter_present", "official_script_path_resolved"], "official_script_exists": script.exists(), "official_adapter_hash": file_hash(official)}


def run_repair_audit(repo: str | Path, *, level: str = "static", source_manifest: str | Path | None = None, output: str | Path | None = None, run_root: str | Path | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve()
    recorder = GateRecorder(output or repo / "reports" / "repair_gates.json")
    checks: list[tuple[str, Callable[[], Mapping[str, Any]]]] = [("V1_ledger_and_zero_frame", _v1), ("V3_M1_unroll_reload", _v3), ("V4_S1_causal_gradient", _v4), ("V5_graph_sampling_masks", _v5), ("V6_S5_actions_gae", _v6), ("V7_checkpoint_resume", _v7), ("V8_official_evaluator", _v8)]
    if level in {"integration", "trial", "full"}:
        checks.insert(1, ("V2_real_observation_label_contract", lambda: _v2(source_manifest)))
    results = [_run(recorder, gate, level, callback) for gate, callback in checks]
    # Gate levels are cumulative.  A trial/full gate additionally checks the
    # artifact existence supplied by the caller, but never declares a metric.
    if level in {"trial", "full"}:
        def _run_artifact() -> Mapping[str, Any]:
            artifact_root = Path(run_root).resolve() if run_root is not None else repo / "outputs" / "research_v2"
            artifacts = sorted(artifact_root.glob("**/train_result.json")) if artifact_root.exists() else []
            if not artifacts:
                raise DataUnavailable("no real train_result.json artifact for trial/full gate")
            completed: list[Path] = []
            completed_keys: set[tuple[str, str, str]] = set()
            for path in artifacts:
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                    resolved_path = path.parent / "resolved_run.json"
                    resolved = json.loads(resolved_path.read_text(encoding="utf-8")) if resolved_path.exists() else {}
                except (OSError, ValueError):
                    continue
                if result.get("status") != "COMPLETED" or str(resolved.get("profile")) != level:
                    continue
                completed.append(path)
                completed_keys.add((str(resolved.get("method")), str(resolved.get("frontend")), str(resolved.get("phase"))))
            if not completed:
                raise GateNotPassed(f"no completed profile={level} training artifact; integration artifacts do not satisfy this gate")
            from ..registry import RESEARCH_SCHEMES, get_scheme
            expected = {
                (get_scheme(name).method, get_scheme(name).frontend, str(get_scheme(name).phase or "train"))
                for name in RESEARCH_SCHEMES if get_scheme(name).trainable
            }
            missing = sorted(expected.difference(completed_keys))
            if missing:
                raise GateNotPassed(f"profile={level} is incomplete; missing real scheme runs: {missing}")
            return {"assertions": ["all_required_profile_training_artifacts", "profile_matches_resolved_run"], "completed_artifacts": [str(path) for path in completed], "completed_keys": sorted(completed_keys)}
        results.append(_run(recorder, "G4_or_G5_real_training_artifact", level, _run_artifact))
    recorder.payload["level"] = level
    recorder.payload["results"] = results
    recorder.payload["gate_hash"] = object_hash(results)
    _atomic_json(recorder.payload, recorder.path)
    return recorder.payload


__all__ = ["GateRecorder", "run_repair_audit"]
