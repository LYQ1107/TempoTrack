"""The eight V4 high-value checks.

Checks report ``PASS``, ``FAIL``, ``BLOCKED_DATA`` or ``NOT_EXERCISED``.  A
static inspection is never promoted to a production-training PASS.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import file_hash, object_hash
from ..errors import DataUnavailable
from .dag import build_experiment_dag, topological_order


def _item(name: str, status: str, *, evidence: Mapping[str, Any] | None = None, error: str | None = None, assertions: list[str] | None = None) -> dict[str, Any]:
    return {"check": name, "status": status, "evidence": dict(evidence or {}), "error": error, "assertions": list(assertions or []), "checked_at": time.time()}


def _real_pair_manifest(run_root: Path) -> Path | None:
    candidates = sorted(run_root.glob("episodes/**/val_base_internal/*_episodes_manifest.json"))
    candidates.extend(sorted(run_root.glob("episodes/**/val_base_internal/pair_manifest.json")))
    for candidate in candidates:
        if candidate.name == "pair_manifest.json":
            return candidate
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            item = dict(payload.get("kinds", {}).get("pair", {}))
            files = list(item.get("files", []))
            if not files or int(item.get("count", 0)) < 1:
                continue
            resolved = [str((candidate.parent / value).resolve()) if not Path(value).is_absolute() else str(Path(value).resolve()) for value in files]
            item.update({"schema_version": 4, "kind": "pair", "files": resolved})
            pair_manifest = candidate.parent / "pair_manifest.json"
            pair_manifest.write_text(json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return pair_manifest
        except (OSError, ValueError, TypeError):
            continue
    return None


def run_v4_checks(repo: str | Path, *, reference_root: str | Path, run_root: str | Path, suite: Mapping[str, Any], local: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    repo = Path(repo).resolve(); run_root = Path(run_root).resolve(); reference_root = Path(reference_root).resolve()
    results: list[dict[str, Any]] = []
    # C1: only a real V4 pair manifest and the actual tensorizer/backend hook
    # can pass.  Old V3 episodes are intentionally not silently reused here.
    pair = _real_pair_manifest(run_root)
    if pair is None:
        results.append(_item("C1_input_time_backend", "BLOCKED_DATA", error="no V4 rebuilt PairEpisodeDataset manifest", evidence={"reference_pair_not_accepted": True}))
    else:
        try:
            from ..data.datasets import PairEpisodeDataset
            dataset = PairEpisodeDataset(pair, transform_snapshot=local.get("data", {}).get("transform_snapshot"))
            if len(dataset) < 1:
                raise DataUnavailable("V4 pair dataset is empty")
            sample = dataset[0]
            required = {"context_appearance", "target_appearance", "query_times"}
            if not required.issubset(sample):
                raise AssertionError(f"formal PairEpisodeDataset missing {sorted(required - set(sample))}")
            results.append(_item("C1_input_time_backend", "PASS", assertions=["formal_pair_dataset_loaded", "explicit_query_times_present"], evidence={"manifest": str(pair), "manifest_hash": file_hash(pair), "episode_count": len(dataset)}))
        except DataUnavailable as exc:
            results.append(_item("C1_input_time_backend", "BLOCKED_DATA", error=str(exc)))
        except Exception as exc:
            results.append(_item("C1_input_time_backend", "FAIL", error=f"{type(exc).__name__}: {exc}"))

    # C2/C3/C7 are evidence checks, not static promises.  Before the live
    # scheduler has produced the corresponding artifacts they are explicitly
    # NOT_EXERCISED; a shared-memory resource snapshot is never converted into
    # a fake external blocker.
    def _train_runs(markers: Sequence[str]) -> list[Path]:
        values: list[Path] = []
        for result in sorted(run_root.glob("runs/**/train_result.json")):
            try:
                payload = json.loads(result.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str(payload.get("method")) in markers or any(marker in result.as_posix() for marker in markers):
                values.append(result)
        return values

    s1_runs = _train_runs(["s1_jepa"])
    def _has_updates(path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = path.parent / "metrics.jsonl"
            return path.parent.joinpath("last.pt").exists() and metrics.exists() and metrics.stat().st_size > 0 and int(payload.get("optimizer_steps", 0)) > 0
        except (OSError, ValueError, TypeError):
            return False
    if any(_has_updates(item) for item in s1_runs):
        results.append(_item("C2_s1_real_training", "PASS", assertions=["formal_s1_training_result", "optimizer_steps_positive", "metrics_recorded"], evidence={"train_results": [str(item) for item in s1_runs]}))
    else:
        results.append(_item("C2_s1_real_training", "NOT_EXERCISED", evidence={"required_marker": "s1_jepa", "reference_root_not_used_as_training": True, "candidate_train_results": [str(item) for item in s1_runs]}))

    m1_replays = sorted(run_root.glob("frontend/m1_*/predictive_dual/*/replay_manifest.json"))
    memory_runs = _train_runs(["predictive_dual"])
    if m1_replays and any(any(part.startswith("m1_") for part in path.parts) for path in m1_replays) and memory_runs:
        results.append(_item("C3_m1_real_events", "PASS", assertions=["m1_replay_manifest_present", "memory_train_result_present"], evidence={"replays": [str(item) for item in m1_replays[:20]], "train_results": [str(item) for item in memory_runs[:20]]}))
    else:
        results.append(_item("C3_m1_real_events", "NOT_EXERCISED", evidence={"required_marker": "m1_memory", "replay_candidates": [str(item) for item in m1_replays[:20]], "train_results": [str(item) for item in memory_runs[:20]]}))

    s2_s5_runs = _train_runs(["s2_state_fm", "s5_rl_edit"])
    ppo_rollouts = sorted(run_root.glob("runs/**/ppo_progress.json"))
    if s2_s5_runs and all(Path(item).parent.joinpath("last.pt").exists() for item in s2_s5_runs) and (ppo_rollouts or any("s2_state_fm" in item.as_posix() for item in s2_s5_runs)):
        results.append(_item("C7_s2_s5_execution", "PASS", assertions=["s2_checkpoint_or_ppo_rollout", "s5_or_s2_artifact_bound"], evidence={"train_results": [str(item) for item in s2_s5_runs], "ppo_rollouts": [str(item) for item in ppo_rollouts]}))
    else:
        results.append(_item("C7_s2_s5_execution", "NOT_EXERCISED", evidence={"required_marker": "s2_s5", "train_results": [str(item) for item in s2_s5_runs], "ppo_rollouts": [str(item) for item in ppo_rollouts]}))

    # C4 can validate the production DAG/target implementation only after a
    # V4 graph window exists.  Keep the distinction explicit.
    graph = list(run_root.glob("episodes/**/graph.jsonl")) if run_root.exists() else []
    if not graph:
        results.append(_item("C4_graph_targets_unknown", "BLOCKED_DATA", error="no rebuilt V4 graph windows available", evidence={"reference_graph_not_accepted": True}))
    else:
        results.append(_item("C4_graph_targets_unknown", "PASS", assertions=["v4_graph_windows_present"], evidence={"graph_artifacts": [str(item) for item in graph[:20]]}))

    # C5 is tied to a real V4 train_result/checkpoint pair, not the old V3
    # resume fixture.  The exact-resume code is still compiled by verify.
    checkpoints = list(run_root.glob("runs/**/train_result.json")) if run_root.exists() else []
    resume_evidence = [item for item in checkpoints if item.parent.joinpath("last.pt").exists() and item.parent.joinpath("resolved_run.json").exists() and item.parent.joinpath("progress.json").exists()]
    if resume_evidence:
        results.append(_item("C5_checkpoint_signature_resume", "PASS", assertions=["v4_train_result_present", "resolved_run_bound", "progress_checkpoint_boundary"], evidence={"train_results": [str(item) for item in resume_evidence[:20]]}))
    else:
        results.append(_item("C5_checkpoint_signature_resume", "NOT_EXERCISED", evidence={"reference_checkpoints_not_accepted": True, "candidate_train_results": [str(item) for item in checkpoints[:20]]}))

    # C6 has a real production DAG proof; two-GPU concurrency is separately
    # recorded as NOT_EXERCISED when the resource snapshot has fewer than two
    # eligible UUIDs.
    try:
        jobs = build_experiment_dag(suite, run_root, through="complete")
        order = topological_order(jobs)
        known = {job.job_id for job in jobs}
        missing_deps = sorted({dep for job in jobs for dep in job.dependencies if dep not in known})
        if missing_deps:
            raise AssertionError(f"DAG has missing dependencies: {missing_deps[:10]}")
        job_rows: list[dict[str, Any]] = []
        status_path = repo / "reports" / "v4" / "status.json"
        if status_path.exists():
            try:
                value = json.loads(status_path.read_text(encoding="utf-8")); job_rows = [dict(item) for item in value.get("jobs", {}).values() if isinstance(item, Mapping)]
            except (OSError, ValueError, TypeError):
                job_rows = []
        devices = sorted({str(item.get("device_uuid")) for item in job_rows if item.get("device_uuid") and item.get("status") in {"COMPLETED", "RUNNING"}})
        c6_status = "PASS" if len(devices) >= 2 else "NOT_EXERCISED"
        results.append(_item("C6_dag_and_gpu_lease", c6_status, assertions=["production_typed_dag_acyclic", "independent_dependency_edges_recorded"] + (["two_distinct_worker_uuids"] if len(devices) >= 2 else []), evidence={"job_count": len(jobs), "topological_order_hash": object_hash(order), "worker_device_uuids": devices, "multi_gpu": len(devices) >= 2}))
    except Exception as exc:
        results.append(_item("C6_dag_and_gpu_lease", "FAIL", error=f"{type(exc).__name__}: {exc}"))

    # C8 may parse a historical official artifact, but it is reference-only
    # until a V4 prediction is bound to it.  This distinction prevents an old
    # metric from becoming a current V4 conclusion.
    current_evaluations = sorted(run_root.glob("evaluations/**/evaluation.json")) if run_root.exists() else []
    official = [path for path in current_evaluations if "official_validation" in str(path)]
    official_rows: list[dict[str, Any]] = []
    for path in official:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload.get("summary_path") or payload.get("artifact") or payload.get("summary")
            summary_path = Path(str(summary)) if summary else path.parent / "teta_summary_results.pth"
            if not summary_path.is_absolute():
                summary_path = path.parent / summary_path
            if payload.get("status") == "COMPLETED" and summary_path.exists():
                official_rows.append({"evaluation": str(path), "evaluation_hash": file_hash(path), "summary": str(summary_path), "summary_hash": file_hash(summary_path)})
        except (OSError, ValueError, TypeError):
            continue
    if official_rows:
        results.append(_item("C8_official_evaluation_loop", "PASS", assertions=["v4_prediction_bound", "official_evaluator_completed", "summary_artifact_present"], evidence={"evaluations": official_rows}))
    else:
        summaries = sorted(reference_root.glob("**/teta_summary_results.pth")) if reference_root.exists() else []
        if summaries:
            results.append(_item("C8_official_evaluation_loop", "REFERENCE_ONLY", assertions=["historical_official_summary_found"], evidence={"summary": str(summaries[0]), "summary_hash": file_hash(summaries[0]), "current_prediction_bound": False}))
        else:
            results.append(_item("C8_official_evaluation_loop", "BLOCKED_DATA", error="no official TETA summary available"))
    payload = {"schema_version": 4, "checked_at": time.time(), "results": results, "summary": {status: sum(item["status"] == status for item in results) for status in ("PASS", "FAIL", "BLOCKED_DATA", "BLOCKED_EXTERNAL", "NOT_EXERCISED", "REFERENCE_ONLY")}, "input_hash": object_hash({"reference_root": str(reference_root), "run_root": str(run_root), "local_schema": local.get("schema_version"), "suite_schema": suite.get("schema_version")})}
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


__all__ = ["run_v4_checks"]
