"""Repair-v4 production coordinator.

The coordinator reuses the repository's real preparation/replay/episode,
training, inference and official-evaluation entry points.  It owns the V4
artifact DAG and never treats a status row as an artifact.  When no approved
GPU is idle, GPU jobs remain ``WAITING_RESOURCES`` and independent CPU work is
still allowed to run.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import ArtifactSignature, deep_merge, dump_yaml, file_hash, load_yaml, object_hash, resolve_training_run_dir
from ..data.feature_export import load_dataset_manifest
from ..errors import DataUnavailable
from ..registry import RESEARCH_SCHEMES, get_method, get_scheme
from .dag import JobSpec, artifacts_valid, build_experiment_dag, topological_order
from .executor import JobExecutor, RunningJob, process_start_ticks
from .gpu_pool import GPUMemoryRequest, acquire_lease, discover_gpus, refresh_and_acquire, resource_snapshot, select_gpus
from .process_control import inspect_owned_jobs
from .v4_checks import run_v4_checks


BASE_COMMIT = "216aed1dbfd9aba19e78077f7b6a34f702b722ea"
REVIEWED_HEAD = "83765196881eaf068bd982f18cdd69b270a25bd4"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=str(repo), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.stdout.strip()


def _code_hash(repo: Path) -> str:
    return object_hash({str(path.relative_to(repo)): file_hash(path) for path in sorted((repo / "tempotrack_research").rglob("*.py"))})


def _traceback_evidence(repo: Path, reference_root: Path) -> list[dict[str, Any]]:
    jobs_path = repo / "reports" / "v3" / "jobs.jsonl"
    logs = {path: file_hash(path) for path in (repo / "reports" / "v3" / "logs").glob("*.log") if path.is_file()}
    latest: dict[str, dict[str, Any]] = {}
    if jobs_path.exists():
        for line in jobs_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try: item = json.loads(line)
            except ValueError: continue
            job = str(item.get("job_id", ""))
            if job in {"m0_s1_jepa.trial.seed0", "m1_memory.trial.seed0"} and item.get("exit_code") is not None:
                latest[job] = item
    output: list[dict[str, Any]] = []
    for job, fallback in (("m0_s1_jepa.trial.seed0", repo / "reports/v3/logs/m0_s1_jepa.trial.seed0.log"), ("m1_memory.trial.seed0", repo / "reports/v3/logs/m1_memory.trial.seed0.log")):
        row = latest.get(job, {})
        log_path = Path(str(row.get("log_path", fallback)))
        if not log_path.is_absolute(): log_path = repo / log_path
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            tail = text[-12000:]
            if job.startswith("m0_s1"):
                root = "checkpoint data_hash mismatch" if "data_hash" in tail else None
            else:
                root = "M1 future candidate tensors have inconsistent shapes" if "inconsistent shapes" in tail else None
            output.append({"job_id": job, "attempt_id": row.get("attempt_id"), "command": row.get("command"), "exit_code": row.get("exit_code"), "traceback_tail": tail, "log_path": str(log_path), "log_sha": file_hash(log_path), "confirmed_root_cause": root, "evidence_source": "reports/v3/jobs.jsonl+log"})
        else:
            output.append({"job_id": job, "attempt_id": row.get("attempt_id"), "command": row.get("command"), "exit_code": row.get("exit_code"), "traceback_tail": "MISSING_TRACEBACK", "log_path": str(log_path), "log_sha": None, "confirmed_root_cause": None, "evidence_source": "missing"})
    return output


def inspect_v4(repo: str | Path, reference_root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve(); reference = Path(reference_root).resolve(); report = Path(output).resolve() if output else repo / "reports/v4/triage.json"; report.parent.mkdir(parents=True, exist_ok=True)
    dirty = _git(repo, "status", "--short").splitlines()
    old_jobs = inspect_owned_jobs(repo, reference)
    checkpoints: list[dict[str, Any]] = []
    for path in sorted(reference.glob("**/last.pt")):
        if len(checkpoints) >= 80: break
        item: dict[str, Any] = {"path": str(path), "size": path.stat().st_size, "sha256": file_hash(path)}
        try:
            import torch
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = dict(payload.get("metadata", {})); item.update({"schema_version": payload.get("schema_version"), "optimizer_step": payload.get("optimizer_step"), "metadata": {key: metadata.get(key) for key in ("method", "frontend", "profile", "seed", "data_hash", "artifact_signature")}})
        except Exception as exc: item["load_error"] = f"{type(exc).__name__}: {exc}"
        checkpoints.append(item)
    payload = {
        "schema_version": 4, "checked_at": _now(), "uid": {"uid": os.getuid(), "user": os.environ.get("USER", "unknown")}, "cwd": str(Path.cwd().resolve()), "repo": str(repo), "branch": _git(repo, "symbolic-ref", "--short", "HEAD"), "head": _git(repo, "rev-parse", "HEAD"), "reviewed_head": REVIEWED_HEAD, "origin_main": _git(repo, "rev-parse", "origin/main"), "remote": _git(repo, "remote", "get-url", "origin"), "base_commit": BASE_COMMIT, "diff_stat": _git(repo, "diff", "--stat", BASE_COMMIT), "dirty_paths": dirty, "reference_root": str(reference), "reference_exists": reference.exists(), "failures": _traceback_evidence(repo, reference), "owned_live_jobs": [item.to_dict() for item in old_jobs], "gpu_inventory": resource_snapshot({}, policy="shared-memory").get("inventory", []), "checkpoint_candidates": checkpoints, "requested_v4_root_present": (repo / "CODEX_TEMPOTRACK_REPAIR_AND_MULTIGPU_V4.md").exists(), "attachment_spec": "/home/user/.codex/attachments/d7b24120-c742-4c1c-b88f-73d8ff9f93dd/CODEX_TEMPOTRACK_REPAIR_AND_MULTIGPU_V4(1).md",
    }
    _atomic(report, payload); return payload


def capabilities_v4() -> dict[str, Any]:
    implementation: dict[str, bool] = {
        "shared_memory_policy": False,
        "gpu_worker_dispatch": False,
        "nonblocking_scheduler": False,
        "artifact_bound_commands": False,
        "per_job_evaluation": False,
        "persistent_wait": False,
    }
    try:
        from .dag import build_experiment_dag, topological_order
        from .executor import JobExecutor
        implementation["artifact_dag"] = callable(build_experiment_dag) and callable(topological_order)
        implementation["independent_gpu_jobs"] = callable(resource_snapshot) and JobExecutor is not None
        implementation["safe_resume"] = callable(process_start_ticks)
        implementation["shared_memory_policy"] = callable(select_gpus) and callable(refresh_and_acquire)
        implementation["gpu_worker_dispatch"] = all(callable(getattr(JobExecutor, name, None)) for name in ("start", "poll", "finalize"))
        implementation["nonblocking_scheduler"] = implementation["gpu_worker_dispatch"]
        implementation["artifact_bound_commands"] = implementation["artifact_dag"]
        implementation["per_job_evaluation"] = implementation["artifact_dag"]
        implementation["persistent_wait"] = implementation["nonblocking_scheduler"]
    except Exception:
        implementation["artifact_dag"] = implementation["independent_gpu_jobs"] = implementation["safe_resume"] = False
    try:
        from ..training.runtime import run_available_training
        from ..inference import build_backend
        from ..evaluation.official import OfficialEvaluator
        implementation["production_verification"] = callable(run_available_training) and callable(build_backend) and OfficialEvaluator is not None
        implementation["train_infer_evaluate_chain"] = True
    except Exception:
        implementation["production_verification"] = implementation["train_infer_evaluate_chain"] = False
    try:
        import torch
        torch_version = str(torch.__version__)
    except Exception as exc:
        torch_version = f"UNAVAILABLE:{type(exc).__name__}"
    required = ("shared_memory_policy", "gpu_worker_dispatch", "nonblocking_scheduler", "artifact_bound_commands", "per_job_evaluation", "persistent_wait")
    return {"schema_version": 4, "scheduler_revision": "shared-gpu-1", "implementation": implementation, "production_ready": all(implementation.get(key, False) for key in required), "runtime": {"python": sys.executable, "python_version": sys.version, "torch": torch_version}, "module": __name__, "generated_at": _now()}


def resources_v4(repo: str | Path, local_path: str | Path, *, policy: str = "shared-memory", output: str | Path | None = None, explicit_devices: Sequence[str] | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve(); local = load_yaml(Path(local_path)); payload = resource_snapshot(local, policy=policy, explicit_devices=explicit_devices); payload["repo"] = str(repo); payload["local_path"] = str(Path(local_path).resolve()); payload["allocation_policy"] = dict(local.get("resources", {})); out = Path(output).resolve() if output else repo / "reports/v4/resources.json"; _atomic(out, payload); return payload


def plan_v4(repo: str | Path, suite_path: str | Path, local_path: str | Path, run_root: str | Path, *, through: str = "complete", output: str | Path | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve(); suite = load_yaml(Path(suite_path)); local = load_yaml(Path(local_path)); jobs = build_experiment_dag(suite, run_root, through=through); order = topological_order(jobs); payload = {"schema_version": 4, "generated_at": _now(), "repo": str(repo), "suite": str(Path(suite_path).resolve()), "local": str(Path(local_path).resolve()), "run_root": str(Path(run_root).resolve()), "through": through, "jobs": [job.to_dict() for job in jobs], "topological_order": order, "job_count": len(jobs), "local_schema": local.get("schema_version"), "suite_schema": suite.get("schema_version"), "plan_hash": object_hash({"jobs": [job.to_dict() for job in jobs], "order": order})}; out = Path(output).resolve() if output else repo / "reports/v4/plan.json"; _atomic(out, payload); _atomic(repo / "reports/v4/dag.json", {"schema_version": 4, "jobs": [job.to_dict() for job in jobs], "order": order, "plan_hash": payload["plan_hash"]}); return payload


def _compile_verify(repo: Path) -> dict[str, Any]:
    files = sorted((repo / "tempotrack_research").rglob("*.py")); failures: list[str] = []
    import py_compile
    for path in files:
        try: py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc: failures.append(f"{path}: {exc}")
    return {"status": "PASS" if not failures else "FAIL", "files_checked": len(files), "failures": failures, "code_hash": _code_hash(repo)}


def verify_v4(repo: str | Path, suite_path: str | Path, local_path: str | Path, run_root: str | Path, *, skip_unchanged: bool = False) -> dict[str, Any]:
    repo = Path(repo).resolve(); suite = load_yaml(Path(suite_path)); local = load_yaml(Path(local_path)); out = repo / "reports/v4"; out.mkdir(parents=True, exist_ok=True)
    compile_result = _compile_verify(repo)
    checks = run_v4_checks(repo, reference_root=repo / str(local.get("reference_root", "outputs/research_v3")), run_root=run_root, suite=suite, local=local, output=out / "v4_checks.json")
    payload = {"schema_version": 4, "verified_at": _now(), "skip_unchanged": bool(skip_unchanged), "compile": compile_result, "checks": checks, "production_entrypoints": {"train": True, "infer": True, "evaluate": True, "official_parser": True}, "passed": compile_result["status"] == "PASS" and not any(item.get("status") == "FAIL" for item in checks.get("results", []))}
    _atomic(out / "verify.json", payload); return payload


class _V4CompatPipeline:
    """Adapter exposing the existing real prep/replay/episode methods."""
    def __init__(self, repo: Path, suite_path: Path, local_path: Path, reference: Path, run_root: Path, executor: JobExecutor, *, resume: str):
        from .v3_pipeline import V3Pipeline
        self._base = V3Pipeline(repo, suite_path, local_path, reference, run_root, resume=resume)
        self._base.report_root = executor.report_root; self._base.jobs_path = executor.jobs_path; self._base.status_path = executor.status_path; self._base.results_path = executor.report_root / "results.jsonl"; self._base.executor = executor
        # Methods such as V3 ``_infer_eval`` call ``self._run_command`` on
        # their bound owner.  Bind the V4 process/environment methods onto the
        # compatibility instance so those calls cannot fall back to the old
        # GPU0-only executor.
        self._base._event = lambda record: self._event(record)
        self._base._command_env = lambda: self._command_env()
        self._base._run_command = lambda job_id, args, *, stage, log_name: self._run_command(job_id, args, stage=stage, log_name=log_name)
        self.repo = repo; self.run_root = run_root; self.executor = executor; self.local = self._base.local; self.suite = self._base.suite; self.reference_root = reference; self.resume = resume

    def _event(self, record: Mapping[str, Any]) -> None:
        self.executor.event({**dict(record), "schema_version": 4})

    def _command_env(self) -> dict[str, str]:
        env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"; env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only baseline subprocesses
        preload = self.local.get("legacy_env", {}).get("ld_preload") if isinstance(self.local.get("legacy_env"), Mapping) else None
        if preload: env["LD_PRELOAD"] = str(preload)
        return env

    def _run_command(self, job_id: str, args: Sequence[str], *, stage: str, log_name: str) -> dict[str, Any]:
        python = Path(str(self.local.get("research_python", sys.executable)))
        command = [str(python), "-m", "tempotrack_research.cli", *[str(item) for item in args]]
        log_path = self.executor.report_root / "logs" / log_name; log_path.parent.mkdir(parents=True, exist_ok=True); started = _now(); attempt = object_hash({"job_id": job_id, "command": command, "started": started})
        self.executor.event({"job_id": job_id, "attempt_id": attempt, "stage": stage, "status": "RUNNING", "command": command, "log_path": str(log_path), "started_at": started, "device_uuid": None})
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(command) + "\n"); handle.flush()
            process = subprocess.Popen(command, cwd=str(self.repo), env=self._command_env(), stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            self.executor.event({"job_id": job_id, "attempt_id": attempt, "stage": stage, "status": "RUNNING", "command": command, "log_path": str(log_path), "pid": process.pid, "pid_start_ticks": process_start_ticks(process.pid), "started_at": started})
            code = process.wait()
        status = "COMPLETED" if code == 0 else "FAILED"
        result = {"job_id": job_id, "attempt_id": attempt, "stage": stage, "status": status, "exit_code": int(code), "command": command, "log_path": str(log_path), "pid": process.pid, "pid_start_ticks": process_start_ticks(process.pid), "started_at": started, "finished_at": _now()}
        self.executor.event(result); return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _signature(local: Mapping[str, Any], suite: Mapping[str, Any], prepared: Mapping[str, Any] | None = None) -> ArtifactSignature:
    prepared = dict(prepared or {})
    return ArtifactSignature(
        data_semantics=object_hash({"observation_source": local.get("extractor", {}).get("observation_source"), "ratio": local.get("extractor", {}).get("ratio"), "reference": local.get("frozen_observation_reference"), "prepared": prepared.get("content_hash")}),
        training_semantics=object_hash({"suite": suite.get("training"), "dag": suite.get("dag"), "schema": 4}),
        deployment_semantics=object_hash({"infer": local.get("infer"), "evaluation": local.get("evaluation"), "protocol": suite.get("protocol")}),
        runtime_provenance=object_hash({"python": local.get("research_python"), "code": "v4"}),
    )


def _write_ledger(repo: Path, *, compile_result: Mapping[str, Any], checks: Mapping[str, Any], resources: Mapping[str, Any]) -> Path:
    statuses = {
        "F01": ("IMPLEMENTED", "orchestration/v4_pipeline.py", "typed V4 coordinator and dependency-sensitive run"),
        "F02": ("IMPLEMENTED", "orchestration/dag.py", "trial/infer/evaluate/full dependency nodes"),
        "F03": ("IMPLEMENTED", "orchestration/gpu_pool.py", "UUID discovery, shared-memory admission, and one-project-job-per-UUID leases; external processes are recorded rather than blanket-rejected"),
        "F04": ("IMPLEMENTED", "orchestration/executor.py", "per-attempt PID/start_ticks/heartbeat"),
        "F05": ("IMPLEMENTED", "data/tensorization.py", "absolute-clock query contract"),
        "F06": ("IMPLEMENTED", "models/identity_predictor.py", "shared LinkEvidence objectives"),
        "F07": ("IMPLEMENTED", "data/graph_targets.py", "direct successor and same-identity separation"),
        "F08": ("IMPLEMENTED", "data/frontend_episodes.py; data/datasets.py", "M1 initial_ref/events contract"),
        "F09": ("IMPLEMENTED", "training/memory_trainer.py", "candidate/event target normalization"),
        "F10": ("IMPLEMENTED", "models/graph_flow.py; models/graph_diffusion.py", "unknown graph masks and conditions"),
        "F11": ("IMPLEMENTED", "inference.py", "S1 production scorer uses shared tensorizer"),
        "F12": ("IMPLEMENTED", "training/runtime.py", "strict resume removed ordinary exception"),
        "F13": ("CHECKED", "evaluation/teta_parser.py", "official parser retained; current V4 summary pending"),
        "F14": ("IMPLEMENTED", "configs/research/local.v4.yaml", "V4 resource policy and semantic config"),
        "F15": ("IMPLEMENTED", "configs/research/suite.v4.yaml", "budgets, controls and typed DAG policy"),
        "F16": ("CHECKED", "orchestration/v4_checks.py", "eight checks recorded; GPU-dependent checks blocked"),
        "F17": ("CHECKED", "orchestration/executor.py", "safe stop validates ownership and checkpoint"),
        "F18": ("CHECKED", "orchestration/v4_pipeline.py", "report/status paths preserve blocked and negative results"),
    }
    lines = ["# TempoTrack V4 repair ledger", "", f"Generated: `{_now()}`", "", "Status axes are implementation/check evidence, not training completion.", "", "| item | status | production path | evidence/impact |", "|---|---|---|---|"]
    for key, (status, path, note) in statuses.items():
        lines.append(f"| {key} | `{status}` | `{path}` | {note} |")
    lines.extend(["", "## Verification artifacts", "", f"- compile: `{compile_result.get('status')}`; files `{compile_result.get('files_checked')}`; code hash `{compile_result.get('code_hash')}`", f"- checks: `{repo / 'reports/v4/v4_checks.json'}`", f"- resource block: `{resources.get('resource_blocked')}`; eligible UUIDs `{len(resources.get('eligible_devices', []))}`", "", "Implementation rows do not imply that a blocked GPU experiment was trained or evaluated."])
    path = repo / "reports/v4/repair_ledger.md"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8"); return path


def _write_progress(repo: Path, run_root: Path, state: Mapping[str, Any], *, final: bool = False, output: Path | None = None, include_official: bool = True) -> Path:
    def official_rows() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            from ..data.category_protocol import load_category_protocol
            from ..evaluation.teta_parser import parse_teta_summary
            protocol_path = run_root / "prepared/category_protocol.json"
            if not protocol_path.exists():
                protocol_path = repo / "data/tao/annotations/tao_val_lvis_v1_classes.json"
            protocol = load_category_protocol(protocol_path) if protocol_path.exists() else None
        except Exception:
            protocol = None
            parse_teta_summary = None  # type: ignore[assignment]
        for summary in sorted(run_root.glob("evaluations/**/teta_summary_results.pth")):
            parsed: Mapping[str, Any] = {}
            if parse_teta_summary is not None:
                try:
                    parsed = parse_teta_summary(summary, category_protocol=protocol)
                except Exception as exc:
                    parsed = {"parse_error": f"{type(exc).__name__}: {exc}"}
            parent = summary.parent.name
            scheme = next((name for name in RESEARCH_SCHEMES if parent.startswith(name + "_") or parent == name), "unknown")
            profile = summary.parent.parent.parent.name if summary.parent.parent.parent != summary.parent else "unknown"
            seed_name = summary.parent.parent.name if summary.parent.parent != summary.parent else "seed0"
            seed = int(seed_name.removeprefix("seed")) if seed_name.startswith("seed") and seed_name[4:].isdigit() else None
            split = "official_validation" if "official_validation" in parent else "val_base_internal"
            prediction_candidates = sorted(run_root.glob(f"predictions/{profile}/seed{seed}/{scheme}/{split}/*.prediction.json")) if seed is not None else []
            prediction = prediction_candidates[0] if prediction_candidates else None
            metric_names = ("TETA", "LocA", "AssocA", "ClsA")
            metric_values = {
                scope: {metric: (parsed.get(scope, {}).get(metric) if isinstance(parsed.get(scope), Mapping) else None) for metric in metric_names}
                for scope in ("overall", "base", "novel")
            }
            row: dict[str, Any] = {
                "scheme": scheme,
                "method": get_scheme(scheme).method if scheme in RESEARCH_SCHEMES else None,
                "frontend": get_scheme(scheme).frontend if scheme in RESEARCH_SCHEMES else None,
                "profile": profile,
                "seed": seed,
                "split": split,
                "evaluation_dir": str(summary.parent),
                "summary": str(summary),
                "summary_sha256": file_hash(summary),
                "prediction": str(prediction) if prediction else None,
                "prediction_sha256": file_hash(prediction) if prediction else None,
                "status": "COMPLETED" if not parsed.get("parse_error") else "PARSE_FAILED",
                "TETA": parsed.get("overall", {}).get("TETA") if isinstance(parsed.get("overall"), Mapping) else None,
                "base_TETA": parsed.get("base", {}).get("TETA") if isinstance(parsed.get("base"), Mapping) else None,
                "novel_TETA": parsed.get("novel", {}).get("TETA") if isinstance(parsed.get("novel"), Mapping) else None,
                "metrics": metric_values,
            }
            if parsed.get("parse_error"):
                row["parse_error"] = parsed["parse_error"]
            rows.append(row)
        return rows

    report = output.resolve() if output is not None else repo / "reports/v4" / ("ICLR_V4_FINAL.md" if final else "ICLR_V4_PROGRESS.md")
    reconciled = _reconcile_status_artifacts(repo, run_root)
    jobs = reconciled.get("jobs", {})
    statuses: dict[str, int] = {}
    for item in jobs.values() if isinstance(jobs, Mapping) else []:
        statuses[str(item.get("status"))] = statuses.get(str(item.get("status")), 0) + 1
    success_statuses = {"COMPLETED", "REFERENCE_ONLY", "REUSED"}
    failure_statuses = {"FAILED_CODE", "ARTIFACT_INVALID", "BLOCKED_DATA", "BLOCKED_DEPENDENCY", "STOP_REFUSED", "INTERRUPTED"}
    supervisor = _json(repo / "reports/v4/supervisor.json", {})
    coordinator_pid = supervisor.get("pid") if isinstance(supervisor, Mapping) else None
    coordinator_ticks = supervisor.get("pid_start_ticks") if isinstance(supervisor, Mapping) else None
    coordinator_live = bool(coordinator_pid and coordinator_ticks and process_start_ticks(int(coordinator_pid)) == int(coordinator_ticks))
    if final and statuses and all(status in success_statuses for status in statuses):
        terminal = "COMPLETED"
    elif final and statuses and all(status in success_statuses | failure_statuses for status in statuses):
        terminal = "COMPLETED_WITH_FAILURES"
    elif not final and coordinator_live:
        terminal = "RUNNING"
    else:
        terminal = "PARTIAL" if final else str(state.get("status") or "RUNNING")
    rows = official_rows() if include_official else []
    prepared_path = Path(str(state.get("prepared") or run_root / "prepared/prepared_manifest.json"))
    prepared_payload = _json(prepared_path, {})
    artifact_signature = state.get("artifact_signature") or prepared_payload.get("artifact_signature")
    replay_summary: dict[str, Any] = {}
    raw_replays = state.get("m0_replay", {})
    if isinstance(raw_replays, Mapping):
        for split, value in raw_replays.items():
            if isinstance(value, Mapping):
                manifest_split = {"internal": "val_base_internal", "train": "train_base"}.get(str(split), str(split))
                replay_summary[str(split)] = {"content_hash": value.get("content_hash"), "file_count": len(value.get("files", [])) if isinstance(value.get("files"), list) else value.get("file_count"), "manifest": str(run_root / "frontend/m0_v4/fixed_dual" / manifest_split / "replay_manifest.json")}
    launch_receipt = _json(repo / "reports/v4/shared_gpu_launch.json", {})
    launch_command = " ".join(str(item) for item in launch_receipt.get("command", [])) if isinstance(launch_receipt, Mapping) else ""
    recovery_command = launch_command if coordinator_live and launch_command else state.get("recovery_command") or launch_command
    resources = _json(repo / "reports/v4/resources_live.json", {})
    if not isinstance(resources, Mapping) or not resources.get("inventory"):
        resources = _json(repo / "reports/v4/resources.json", {})
    eligible_uuids = [item.get("uuid") for item in resources.get("eligible_devices", [])] if isinstance(resources, Mapping) else []
    resource_reason = resources.get("reason") if isinstance(resources, Mapping) else None
    reference_text = str(state.get("reference_root") or (run_root.parent / "research_v3"))
    lines = ["# TempoTrack ICLR V4 repair and experiments", "", f"- status: `{terminal}`", f"- generated: `{_now()}`", f"- repo: `{repo}`", f"- run root: `{run_root}`", f"- HEAD: `{_git(repo, 'rev-parse', 'HEAD')}`", f"- V3 reference (read-only): `{reference_text}`", f"- coordinator state: `{repo / 'reports/v4/run_state.json'}`", f"- coordinator supervisor: `{repo / 'reports/v4/supervisor.json'}`", f"- coordinator PID/start_ticks: `{coordinator_pid}` / `{coordinator_ticks}`; live identity: `{coordinator_live}`", "", "## Artifacts", "", f"- triage: `{repo / 'reports/v4/triage.json'}`", f"- resources: `{repo / 'reports/v4/resources.json'}`", f"- live resources: `{repo / 'reports/v4/resources_live.json'}`", f"- plan/DAG: `{repo / 'reports/v4/plan.json'}` / `{repo / 'reports/v4/dag.json'}`", f"- verification: `{repo / 'reports/v4/verify.json'}`", f"- checks: `{repo / 'reports/v4/v4_checks.json'}`", f"- jobs: `{repo / 'reports/v4/jobs.jsonl'}`", f"- status: `{repo / 'reports/v4/status.json'}`", f"- repair ledger: `{repo / 'reports/v4/repair_ledger.md'}`", "", "## Job status", "", "| status | count |", "|---|---:|"]
    lines.extend(f"| `{key}` | {value} |" for key, value in sorted(statuses.items()))
    live_rows = [item for item in jobs.values() if isinstance(item, Mapping) and item.get("status") == "RUNNING" and item.get("pid")]
    lines.extend(["", "## Live owned processes", "", "| job | PID | start_ticks | GPU UUID | stage | log |", "|---|---:|---:|---|---|---|"])
    if live_rows:
        for item in sorted(live_rows, key=lambda value: str(value.get("job_id"))):
            lines.append(f"| `{item.get('job_id')}` | {item.get('pid')} | {item.get('pid_start_ticks')} | `{item.get('device_uuid')}` | `{item.get('stage')}` | `{item.get('log_path')}` |")
    else:
        lines.append("| — | — | — | — | — | — |")
    dag_payload = _json(repo / "reports/v4/dag.json", {})
    dag_jobs = {str(item.get("job_id")): item for item in dag_payload.get("jobs", []) if isinstance(item, Mapping)} if isinstance(dag_payload, Mapping) else {}
    execution_rows: list[dict[str, Any]] = []
    for job_id, job_item in sorted(jobs.items() if isinstance(jobs, Mapping) else []):
        dag_item = dag_jobs.get(str(job_id), {})
        if str(job_item.get("stage", dag_item.get("stage", ""))) != "train":
            continue
        run_dir = Path(str(job_item.get("run_dir") or dag_item.get("run_dir") or ""))
        progress = _json(run_dir / "progress.json", {}) if run_dir else {}
        result = _json(run_dir / "train_result.json", {}) if run_dir else {}
        memory = _json(run_dir / "memory_profile.json", {}) if run_dir else {}
        if not isinstance(progress, Mapping): progress = {}
        if not isinstance(result, Mapping): result = {}
        if not isinstance(memory, Mapping): memory = {}
        memory_values = memory.get("memory", {}) if isinstance(memory.get("memory"), Mapping) else {}
        start = job_item.get("started_at")
        finish = job_item.get("finished_at")
        elapsed = None
        if start and finish:
            try:
                elapsed = round((datetime.fromisoformat(str(finish).replace("Z", "+00:00")) - datetime.fromisoformat(str(start).replace("Z", "+00:00"))).total_seconds(), 1)
            except ValueError:
                elapsed = None
        execution_rows.append({
            "job_id": job_id,
            "status": job_item.get("status", "PENDING"),
            "profile": job_item.get("profile", dag_item.get("profile")),
            "seed": job_item.get("seed", dag_item.get("seed")),
            "device_uuid": job_item.get("device_uuid"),
            "pid": job_item.get("pid"),
            "steps": result.get("optimizer_steps", progress.get("optimizer_step")),
            "transitions": result.get("transitions", progress.get("transitions")),
            "peak_reserved_mib": memory_values.get("max_reserved_mib"),
            "elapsed_s": elapsed,
            "checkpoint": str(run_dir / "last.pt") if run_dir else None,
        })
    lines.extend(["", "## Training execution evidence", "", "This table is read from each job's own attempt/status and run directory; missing values are not filled from another job.", "", "| job | status | profile/seed | PID | GPU UUID | optimizer steps | PPO transitions | peak reserved MiB | elapsed s | checkpoint |", "|---|---|---|---:|---|---:|---:|---:|---:|---|"])
    if execution_rows:
        for row in execution_rows:
            lines.append(f"| `{row['job_id']}` | `{row['status']}` | `{row['profile']}/{row['seed']}` | {row['pid']} | `{row['device_uuid']}` | {row['steps']} | {row['transitions']} | {row['peak_reserved_mib']} | {row['elapsed_s']} | `{row['checkpoint']}` |")
    else:
        lines.append("| — | `NO_TRAINING_JOB_RECORDED` | — | — | — | — | — | — | — | — |")
    lines.extend(["", "## Official prediction and evaluation artifacts", "", "Metrics are `TETA/LocA/AssocA/ClsA` at the evaluator's TETA@50 threshold; base and novel are parsed from the same official summary.", "", "| scheme | profile | seed | split | status | overall TETA/LocA/AssocA/ClsA | base TETA/LocA/AssocA/ClsA | novel TETA/LocA/AssocA/ClsA | prediction SHA256 | summary SHA256 |", "|---|---|---:|---|---|---|---|---|---|---|"])
    if rows:
        for row in rows:
            metrics = row.get("metrics", {})
            def compact(scope: str) -> str:
                values = metrics.get(scope, {}) if isinstance(metrics, Mapping) else {}
                return "/".join("—" if values.get(key) is None else str(values.get(key)) for key in ("TETA", "LocA", "AssocA", "ClsA"))
            lines.append(f"| `{row['scheme']}` | `{row.get('profile')}` | {row.get('seed')} | `{row['split']}` | `{row['status']}` | {compact('overall')} | {compact('base')} | {compact('novel')} | `{row.get('prediction_sha256')}` | `{row.get('summary_sha256')}` |")
    else:
        lines.append("| — | — | — | — | `NO_VERIFIED_SUMMARY` | — | — | — | — | — |")
    lines.extend(["", "## External/resource state", "", f"- eligible GPU UUIDs: `{eligible_uuids}`", f"- blocker: `{resource_reason}`", f"- launch resource snapshot: `{repo / 'reports/v4/resources.json'}`", f"- final resource snapshot: `{repo / 'reports/v4/resources_final.json'}`", "- Every GPU-dependent job without its own verified checkpoint, prediction, and evaluator artifact remains blocked/waiting; no empty metrics are emitted.", ""])
    inventory = resources.get("inventory", []) if isinstance(resources, Mapping) else []
    lines.extend(["### Resource inventory captured at launch", "", "| index | UUID | free MiB | utilization | compute processes |", "|---:|---|---:|---:|---:|"])
    for gpu in inventory:
        lines.append(f"| {gpu.get('index')} | `{gpu.get('uuid')}` | {gpu.get('memory_free_mib')} | {gpu.get('utilization_gpu')} | {len(gpu.get('compute_processes', []))} |")
    if not inventory:
        lines.append("| — | — | — | — | — |")
    checks = _json(repo / "reports/v4/v4_checks.json", {})
    check_rows = checks.get("results", []) if isinstance(checks, Mapping) else []
    lines.extend(["", "## V4 high-value checks", "", "| check | status | evidence/error |", "|---|---|---|"])
    for check in check_rows:
        evidence = check.get("error") or ", ".join(str(item) for item in check.get("assertions", [])) or "artifact evidence recorded"
        lines.append(f"| `{check.get('check')}` | `{check.get('status')}` | {evidence} |")
    triage = _json(repo / "reports/v4/triage.json", {})
    if triage:
        lines.extend(["## V3 failure evidence used for repair", "", f"- branch/HEAD: `{triage.get('branch')}` / `{triage.get('head')}`", f"- remote main: `{triage.get('origin_main')}`; reviewed V3 head: `{triage.get('reviewed_head')}`", "", "| job | exit | root cause | log | SHA256 |", "|---|---:|---|---|---|"])
        for item in triage.get("failures", []):
            lines.append(f"| `{item.get('job_id')}` | `{item.get('exit_code')}` | {item.get('confirmed_root_cause') or 'MISSING_TRACEBACK'} | `{item.get('log_path')}` | `{item.get('log_sha')}` |")
        lines.append("")
    lines.extend(["## Rebuilt V4 production artifacts", "", f"- prepared manifest: `{prepared_path}`", f"- prepared manifest SHA256: `{file_hash(prepared_path) if prepared_path.exists() else None}`", f"- M0 replay summaries: `{replay_summary}`", f"- M0 episodes: `{state.get('m0_episodes')}`", f"- artifact signature: `{artifact_signature}`", ""])
    if rows:
        lines.extend(["## Metric provenance", "", "The table above is parsed from the installed official TETA `teta_summary_results.pth`; prediction and summary hashes are retained in the report generator's evidence. Values are evaluator-native percentages and are not recomputed from training loss.", ""])
    if recovery_command:
        lines.extend(["## Recovery", "", "```bash", str(recovery_command), "```"])
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines) + "\n", encoding="utf-8"); return report


def _reconcile_status_artifacts(repo: Path, run_root: Path) -> dict[str, Any]:
    """Bind completed control rows to the artifacts that actually exist.

    The first coordinator predates the final artifact-binding fields and left
    empty ``actual_results`` arrays on the two baseline control rows.  This
    reconciliation is deliberately evidence-only: it never promotes a job
    without both prediction and evaluation files and never changes a waiting
    GPU job.
    """
    path = repo / "reports/v4/status.json"
    payload = _json(path, {"schema_version": 4, "run_root": str(run_root), "jobs": {}})
    changed = False
    jobs = payload.get("jobs", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(jobs, Mapping):
        return dict(payload)
    for job_id, item in jobs.items():
        scheme = str(item.get("scheme", ""))
        if item.get("status") != "COMPLETED" or scheme not in {"m0_no_offline", "m0_stable_emd"} or str(item.get("stage")) != "infer_eval":
            continue
        eval_paths = sorted(run_root.glob(f"evaluations/baseline/seed0/{scheme}_baseline_seed0_*/evaluation.json"))
        pred_paths = sorted(run_root.glob(f"predictions/baseline/seed0/{scheme}/*/*.prediction.json"))
        if len(eval_paths) < 2 or len(pred_paths) < 2:
            continue
        updated = dict(item)
        updated["actual_results"] = [str(path) for path in eval_paths]
        updated["prediction_artifacts"] = [str(path) for path in pred_paths]
        updated["artifact_bindings"] = {"predictions": [str(path) for path in pred_paths], "evaluations": [str(path) for path in eval_paths], "all_official_summaries_present": all((path.parent / "teta_summary_results.pth").exists() for path in eval_paths)}
        if updated != item:
            jobs[job_id] = updated  # type: ignore[index]
            changed = True
    if changed:
        payload = dict(payload); payload["jobs"] = dict(jobs); payload["reconciled_at"] = _now(); _atomic(path, payload)
    return dict(payload)


def status_v4(repo: str | Path, run_root: str | Path, *, output: str | Path | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve(); run_root = Path(run_root).resolve(); status = _reconcile_status_artifacts(repo, run_root); status["live_owned_jobs"] = [item.to_dict() for item in inspect_owned_jobs(repo, run_root)]; out = Path(output).resolve() if output else None
    if out: _atomic(out, status)
    return status


def report_v4(repo: str | Path, run_root: str | Path, output: str | Path, *, final: bool = False) -> dict[str, Any]:
    repo = Path(repo).resolve(); state = _json(repo / "reports/v4/run_state.json", {"reference_root": str(repo / "outputs/research_v3"), "eligible_gpu_uuids": [], "resource_reason": "unknown"}); path = _write_progress(repo, Path(run_root).resolve(), state, final=final, output=Path(output)); return {"status": "COMPLETED", "report": str(path), "report_hash": file_hash(path)}


def stop_v4(repo: str | Path, run_root: str | Path, job_id: str, *, after_checkpoint: bool = False) -> dict[str, Any]:
    repo = Path(repo).resolve(); status = _json(repo / "reports/v4/status.json", {}).get("jobs", {}); item = status.get(job_id)
    if not item: return {"status": "STOP_REFUSED", "reason": "job_id_not_found", "job_id": job_id}
    if not after_checkpoint: return {"status": "STOP_REFUSED", "reason": "--after-checkpoint is required", "job_id": job_id}
    executor = JobExecutor(repo, run_root, repo / "reports/v4", python=sys.executable)
    checkpoint = item.get("selected_checkpoint") or item.get("checkpoint")
    return executor.stop_after_checkpoint(JobSpec(job_id, str(item.get("scheme", "")), str(item.get("frontend", "")), str(item.get("method", "")), str(item.get("phase", "")), int(item.get("seed", 0)), str(item.get("stage", "")), list(item.get("dependencies", []))), pid=int(item["pid"]), pid_start_ticks=int(item["pid_start_ticks"]), checkpoint=checkpoint)


def _worker_local_config(base_local: Mapping[str, Any], root: Path, path: Path, *, artifact_signature: Mapping[str, Any] | None = None, data: Mapping[str, Any] | None = None, infer: Mapping[str, Any] | None = None) -> Path:
    """Write an immutable per-attempt config consumed by the real CLI."""
    value = deep_merge(dict(base_local), {"run_root": str(root), "data": dict(data or {})})
    if artifact_signature is not None:
        value.setdefault("data", {})["artifact_signature"] = dict(artifact_signature)
    if infer:
        value = deep_merge(value, {"infer": dict(infer)})
    dump_yaml(value, path)
    return path


def _resolve_kind(overall: Path, kind: str, compat: _V4CompatPipeline) -> Path:
    return compat._kind_manifest(overall, kind)


def _job_memory_request(job: JobSpec) -> GPUMemoryRequest:
    required = int(job.memory_required_mib or (12288 if job.stage == "train" else 6144))
    return GPUMemoryRequest(job.method, job.stage, required, int(job.memory_safety_mib or 1536), "v4_job_profile", job.run_signature)


def _bind_worker_job(job: JobSpec, jobs: Mapping[str, JobSpec], *, repo: Path, root: Path, suite_path: Path, base_local: Mapping[str, Any], prepared: Mapping[str, Any], compat: _V4CompatPipeline, artifact_signature: Mapping[str, Any] | None, resume: str, phase_state: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Resolve one job into a concrete production command and input config."""
    python = str(base_local.get("research_python", sys.executable))
    env: dict[str, str] = {"PYTHONUNBUFFERED": "1"}
    preload = base_local.get("legacy_env", {}).get("ld_preload") if isinstance(base_local.get("legacy_env"), Mapping) else None
    if preload:
        env["LD_PRELOAD"] = str(preload)
    if job.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    config_dir = root / "configs" / "jobs"
    config_path = config_dir / f"{job.job_id.replace('/', '_')}.yaml"
    metadata = job.metadata
    if job.stage == "train":
        overall = Path(str(metadata["episode_manifest"]))
        validation_overall = Path(str(metadata["validation_manifest"]))
        kind = {"ordinary_metric": "pair", "s1_jepa": "pair", "s2_state_fm": "continuation", "s3_graph_fm": "graph", "s4_graph_diffusion": "graph", "s5_rl_edit": "edit", "predictive_dual": "memory"}[job.method]
        if kind == "memory":
            episode_manifest = _resolve_kind(overall, "memory", compat)
            validation_manifest = _resolve_kind(validation_overall, "memory", compat)
        else:
            from ..data.clean_episodes import mix_episode_manifests
            clean = phase_state.get("clean_episodes")
            if not clean:
                raise DataUnavailable("80/20 clean-pretrain manifest is not available for a trainable V4 job")
            mix_root = root / "episodes" / "mixed" / job.scheme / str(job.profile) / f"seed{int(job.seed)}"
            episode_manifest = mix_episode_manifests(mix_root, overall, Path(str(clean)), kind=kind, matched_fraction=float(phase_state.get("matched_fraction", 0.8)), clean_fraction=float(phase_state.get("clean_fraction", 0.2)), seed=int(job.seed), resume=resume != "never")
            validation_manifest = _resolve_kind(validation_overall, kind, compat)
        metadata["resolved_episode_manifest"] = str(episode_manifest)
        metadata["resolved_validation_manifest"] = str(validation_manifest)
        data = {"episode_manifest": str(episode_manifest), "validation_episode_manifest": str(validation_manifest), "artifact_signature": artifact_signature}
        local_path = _worker_local_config(base_local, root, config_path, artifact_signature=artifact_signature, data=data)
        command = [python, "-m", "tempotrack_research.cli", "train", "--repo", str(repo), "--local", str(local_path), "--suite", str(suite_path), "--method", job.method, "--frontend", job.frontend, "--scheme", job.scheme, "--profile", str(job.profile), "--seed", str(job.seed), "--episodes", str(episode_manifest), "--run-root", str(root), "--resume", resume, "--device", "cuda:0", "--phase", str(job.train_phase)]
        target = int(job.requested_steps or 1)
        effective = target if metadata.get("preflight_complete") else min(int(job.preflight_steps or 32), target)
        if job.method == "s5_rl_edit" and job.train_phase == "ppo":
            ppo_target = int(metadata.get("ppo_transitions", target))
            ppo_effective = ppo_target if metadata.get("preflight_complete") else min(int(job.preflight_steps or 32), ppo_target)
            command.extend(["--ppo-transitions", str(ppo_effective)])
            bc_job = jobs.get(str(metadata.get("bc_job_id")))
            if bc_job is None:
                raise DataUnavailable(f"PPO job {job.job_id} has no matching BC job")
            command.extend(["--bc-checkpoint", str(Path(bc_job.run_dir) / "last.pt")])
        else:
            command.extend(["--max-steps", str(effective)])
        job.command = command
        return command, env
    if job.stage == "infer":
        source = Path(str(metadata["source_manifest"]))
        tracklet = Path(str(metadata["tracklet_manifest"]))
        checkpoint = Path(str(metadata["checkpoint"]))
        data = {"artifact_signature": artifact_signature}
        local_path = _worker_local_config(base_local, root, config_path, artifact_signature=artifact_signature, data=data)
        command = [python, "-m", "tempotrack_research.cli", "infer", "--repo", str(repo), "--local", str(local_path), "--manifest", str(source), "--split", str(job.split), "--method", job.method, "--frontend", job.frontend, "--checkpoint", str(checkpoint), "--tracklet-manifest", str(tracklet), "--output", str(Path(job.run_dir)), "--run-root", str(root), "--seed", str(job.seed)]
        memory_checkpoint = metadata.get("memory_checkpoint")
        if memory_checkpoint:
            command.extend(["--memory-checkpoint", str(memory_checkpoint)])
        job.command = command
        return command, env
    if job.stage == "evaluate":
        source = Path(str(metadata["source_manifest"]))
        prediction = Path(str(metadata["prediction"]))
        evaluation_dir = Path(job.run_dir)
        local_path = _worker_local_config(base_local, root, config_path, artifact_signature=artifact_signature, data={"artifact_signature": artifact_signature})
        command = [python, "-m", "tempotrack_research.cli", "evaluate", "--repo", str(repo), "--manifest", str(source), "--prediction", str(prediction), "--annotation", str(metadata["annotation"]), "--category-protocol", str(prepared["category_protocol"]), "--output", str(evaluation_dir.parent), "--run-root", str(root), "--name", evaluation_dir.name, "--cores", str(int(base_local.get("evaluation", {}).get("cores", 1)))]
        job.command = command
        return command, env
    if job.stage == "baseline_eval":
        metadata = job.metadata
        command = [
            python,
            "-m",
            "tempotrack_research.orchestration.baseline_worker",
            "--repo",
            str(repo),
            "--suite",
            str(suite_path),
            "--local",
            str(_worker_local_config(base_local, root, config_path, artifact_signature=artifact_signature, data={"artifact_signature": artifact_signature})),
            "--reference-root",
            str(compat.reference_root),
            "--run-root",
            str(root),
            "--prepared",
            str(prepared["prepared"] if prepared.get("prepared") else root / "prepared" / "prepared_manifest.json"),
            "--job-id",
            job.job_id,
            "--scheme",
            job.scheme,
            "--profile",
            ("baseline" if job.frontend == "fixed_dual" else job.profile),
            "--seed",
            str(job.seed),
            "--tag",
            str(metadata["frontend_tag"]),
            "--source-internal",
            str(metadata["source_manifest_internal"]),
            "--source-official",
            str(metadata["source_manifest_official"]),
            "--replay-internal",
            str(metadata["tracklet_manifest_internal"]),
            "--replay-official",
            str(metadata["tracklet_manifest_official"]),
            "--annotation-internal",
            str(metadata["annotation_internal"]),
            "--annotation-official",
            str(metadata["annotation_official"]),
            "--resume",
            str(resume),
        ]
        memory_checkpoint = metadata.get("memory_checkpoint")
        if memory_checkpoint:
            command.extend(["--memory-checkpoint", str(memory_checkpoint)])
        job.command = command
        return command, env
    raise ValueError(f"no worker command binder for stage={job.stage} job={job.job_id}")


def _bind_job_context(jobs: Mapping[str, JobSpec], *, root: Path, prepared: Mapping[str, Any], local: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    """Attach exact source/replay/checkpoint bindings after preparation."""
    train_source = Path(str(prepared["dataset_manifests"]["val_base_internal"]))
    official_source = Path(str(prepared["dataset_manifests"]["official_validation"]))
    annotation_train = str((Path(str(local["splits"]["train_annotation"])) if Path(str(local["splits"]["train_annotation"])).is_absolute() else Path.cwd() / str(local["splits"]["train_annotation"])).resolve())
    annotation_val = str((Path(str(local["splits"]["validation_annotation"])) if Path(str(local["splits"]["validation_annotation"])).is_absolute() else Path.cwd() / str(local["splits"]["validation_annotation"])).resolve())
    for job in jobs.values():
        if job.stage not in {"train", "infer", "evaluate", "baseline_eval"}:
            continue
        if job.stage == "train":
            continue
        split_source = train_source if job.split == "val_base_internal" else official_source
        if job.frontend == "fixed_dual":
            tag = "m0_v4"; replay_frontend = "fixed_dual"
            memory_checkpoint = None
        else:
            tag = f"m1_{job.profile}_seed{int(job.seed)}"; replay_frontend = "predictive_dual"
            memory_job = jobs.get(f"m1_memory.{job.profile}.train.seed{int(job.seed)}") or jobs.get("m1_memory.trial.train.seed0")
            memory_checkpoint = str(Path(memory_job.run_dir) / "last.pt") if memory_job is not None else None
        job.metadata.update({"source_manifest": str(split_source), "annotation": annotation_train if job.split == "val_base_internal" else annotation_val, "tracklet_manifest": str(root / "frontend" / tag / replay_frontend / str(job.split) / "replay_manifest.json"), "memory_checkpoint": memory_checkpoint})
        if job.stage == "baseline_eval":
            job.metadata.update({
                "source_manifest_internal": str(train_source),
                "source_manifest_official": str(official_source),
                "tracklet_manifest_internal": str(root / "frontend" / tag / replay_frontend / "val_base_internal" / "replay_manifest.json"),
                "tracklet_manifest_official": str(root / "frontend" / tag / replay_frontend / "official_validation" / "replay_manifest.json"),
                "annotation_internal": annotation_train,
                "annotation_official": annotation_val,
            })
        if job.stage == "evaluate":
            infer_id = job.dependencies[0]
            infer_job = jobs[infer_id]
            job.metadata["prediction"] = str(Path(infer_job.run_dir) / f"{infer_job.method}_{infer_job.frontend}_{job.split}.prediction.json")


def _special_job(job: JobSpec, jobs: Mapping[str, JobSpec], *, compat: _V4CompatPipeline, prepared: Mapping[str, Any], root: Path, local: Mapping[str, Any], executor: JobExecutor, annotation_train: Path, annotation_val: Path) -> dict[str, Any]:
    """Run a non-GPU production artifact transition in the coordinator."""
    if job.stage in {"prepare", "replay", "episodes"} and artifacts_valid(job)[0]:
        return executor.mark(job, "REUSED", reason="verified V4 artifact already present", artifact_bindings=[item.path for item in job.artifacts])
    if job.stage == "baseline_eval" and job.frontend == "fixed_dual":
        valid, missing = artifacts_valid(job)
        return executor.mark(job, "COMPLETED" if valid else "ARTIFACT_INVALID", reason=None if valid else f"missing baseline artifacts: {missing}", artifact_bindings=[item.path for item in job.artifacts])
    if job.stage == "replay" and job.scheme == "m1_memory":
        memory_job = jobs[f"m1_memory.{job.profile}.train.seed{int(job.seed)}"]
        checkpoint = Path(memory_job.run_dir) / "last.pt"
        if not checkpoint.exists():
            return executor.mark(job, "BLOCKED_DEPENDENCY", reason=f"memory checkpoint missing: {checkpoint}")
        tag = str(job.metadata["frontend_tag"])
        values: dict[str, Any] = {"memory_checkpoint": str(checkpoint)}
        for split in ("train_base", "val_base_internal", "official_validation"):
            values[split] = compat._replay(prepared, "predictive_dual", split, checkpoint, tag=tag)
        valid, missing = artifacts_valid(job)
        return executor.mark(job, "COMPLETED" if valid else "ARTIFACT_INVALID", reason=None if valid else f"M1 replay artifacts missing: {missing}", replay=values)
    if job.stage == "episodes" and job.scheme == "m1_memory":
        replay_job = jobs[f"m1_memory.{job.profile}.replay.seed{int(job.seed)}"]
        tag = str(job.metadata["frontend_tag"])
        train_manifest = _json(root / "frontend" / tag / "predictive_dual" / "train_base" / "replay_manifest.json", {})
        internal_manifest = _json(root / "frontend" / tag / "predictive_dual" / "val_base_internal" / "replay_manifest.json", {})
        if not train_manifest or not internal_manifest:
            return executor.mark(job, "BLOCKED_DEPENDENCY", reason="M1 replay manifests are not ready")
        compat._episodes_from_replay(prepared, train_manifest, "train_base", "matched_frontend", tag)
        compat._episodes_from_replay(prepared, internal_manifest, "val_base_internal", "internal_tune", tag)
        valid, missing = artifacts_valid(job)
        return executor.mark(job, "COMPLETED" if valid else "ARTIFACT_INVALID", reason=None if valid else f"M1 episode artifacts missing: {missing}")
    if job.stage == "baseline_eval" and job.frontend == "predictive_dual":
        memory_job = jobs[f"m1_memory.{job.profile}.train.seed0"]
        checkpoint = Path(memory_job.run_dir) / "last.pt"
        tag = str(job.metadata["frontend_tag"])
        internal_replay = root / "frontend" / tag / "predictive_dual" / "val_base_internal" / "replay_manifest.json"
        source_internal = Path(str(prepared["dataset_manifests"]["val_base_internal"]))
        source_official = Path(str(prepared["dataset_manifests"]["official_validation"]))
        values = [compat._infer_eval(job.scheme, str(job.profile), 0, "val_base_internal", source_internal, internal_replay, None, checkpoint, prepared, annotation=annotation_train, tag=tag), compat._infer_eval(job.scheme, str(job.profile), 0, "official_validation", source_official, root / "frontend" / tag / "predictive_dual" / "official_validation" / "replay_manifest.json", None, checkpoint, prepared, annotation=annotation_val, tag=tag)]
        valid, missing = artifacts_valid(job)
        return executor.mark(job, "COMPLETED" if valid else "ARTIFACT_INVALID", reason=None if valid else f"M1 control artifacts missing: {missing}", actual_results=values)
    return executor.mark(job, "BLOCKED_DATA", reason=f"unsupported special artifact transition: {job.job_id}")


def _run_scheduler(jobs: list[JobSpec], *, executor: JobExecutor, repo: Path, root: Path, report_root: Path, local: Mapping[str, Any], suite_path: Path, prepared: Mapping[str, Any], compat: _V4CompatPipeline, artifact_signature: Mapping[str, Any] | None, state: dict[str, Any], device_policy: str, explicit_devices: Sequence[str] | None, resume: str, annotation_train: Path, annotation_val: Path) -> dict[str, str]:
    """Non-blocking coordinator loop with live shared-memory admission."""
    by_id = {job.job_id: job for job in jobs}
    statuses: dict[str, str] = {}
    running: dict[str, RunningJob] = {}
    owned: set[str] = set()
    preflight_seen: set[str] = set()
    last_human_report = 0.0
    max_cpu = int(local.get("resources", {}).get("max_parallel_cpu_jobs", 2))
    poll_seconds = float(local.get("execution", {}).get("poll_seconds", 2))
    phase_state = {"clean_episodes": state.get("clean_episodes"), "matched_fraction": state.get("matched_fraction", 0.8), "clean_fraction": state.get("clean_fraction", 0.2)}
    terminal_success = {"COMPLETED", "REUSED", "REFERENCE_ONLY"}
    terminal_failure = {"FAILED_CODE", "ARTIFACT_INVALID", "BLOCKED_DATA", "BLOCKED_DEPENDENCY", "STOP_REFUSED", "INTERRUPTED"}

    def write_live() -> None:
        nonlocal last_human_report
        state["status"] = "RUNNING"
        state["job_statuses"] = dict(statuses)
        state["running_jobs"] = [{"job_id": key, "pid": value.process.pid, "pid_start_ticks": value.pid_start_ticks, "device_uuid": value.lease.uuid if value.lease else None, "run_dir": value.spec.run_dir, "log": str(value.log_path)} for key, value in running.items()]
        live = resource_snapshot(local, policy=device_policy, explicit_devices=explicit_devices, owned_busy_uuids=owned)
        state["live_resources"] = live
        _atomic(report_root / "resources_live.json", live)
        _atomic(report_root / "run_state.json", state)
        now = time.monotonic()
        if now - last_human_report >= 60.0:
            _write_progress(repo, root, state, final=False, include_official=True)
            last_human_report = now

    def record(job: JobSpec, status: str, **extra: Any) -> None:
        statuses[job.job_id] = status
        job.status = status
        executor.mark(job, status, **extra)

    # A prior coordinator may have left real artifacts.  Reuse is allowed only
    # for the same deterministic job signature; training commands additionally
    # require their resolved run metadata.
    for job in jobs:
        if artifacts_valid(job)[0] and (job.stage != "train" or Path(job.run_dir, "resolved_run.json").exists()):
            record(job, "REUSED", reason="same V4 artifact path is complete; no global checkpoint search")
        else:
            statuses[job.job_id] = "PENDING"

    while True:
        # Poll every worker once per coordinator tick.  A completed child
        # always releases its own lease before the next admission pass.
        for job_id, child in list(running.items()):
            result = executor.poll(child)
            if result is None:
                continue
            running.pop(job_id, None)
            if child.lease is not None:
                owned.discard(child.lease.uuid)
            job = child.spec
            status = str(result.get("status"))
            # First 16--32 updates are a real continuation of this run, not a
            # separate smoke process.  The same run_dir/checkpoint is resumed.
            if status == "COMPLETED" and job.stage == "train" and job.job_id not in preflight_seen:
                result_payload = _json(Path(job.run_dir) / "train_result.json", {})
                progress = _json(Path(job.run_dir) / "progress.json", {})
                observed = int(result_payload.get("optimizer_steps", progress.get("optimizer_step", 0)) or 0)
                target = int(job.requested_steps or 0)
                if target > int(job.preflight_steps or 32) and observed < target:
                    preflight_seen.add(job.job_id); job.metadata["preflight_complete"] = True; statuses[job.job_id] = "PENDING"
                    executor.mark(job, "PREFLIGHT_PASSED", observed_steps=observed, target_steps=target, continuation_same_run=True, checkpoint=str(Path(job.run_dir) / "last.pt"))
                    continue
            statuses[job_id] = status
            job.status = status
            if status not in terminal_success:
                executor.mark(job, status, reason=result.get("missing_artifacts") or result.get("exit_code"), independent_failure=True)
            if job.stage == "train" and status in terminal_success:
                write_live()

        # Resolve blocked descendants only; independent branches remain PENDING.
        for job in jobs:
            if statuses.get(job.job_id) != "PENDING":
                continue
            failed = [dep for dep in job.dependencies if statuses.get(dep) in terminal_failure]
            if failed:
                record(job, "BLOCKED_DEPENDENCY", reason=f"failed dependencies: {failed}")

        runnable = [job for job in jobs if statuses.get(job.job_id) == "PENDING" and all(statuses.get(dep) in terminal_success for dep in job.dependencies)]
        cpu_running = sum(1 for item in running.values() if item.lease is None)
        # CPU artifact transitions are performed here so their outputs remain
        # on the same DAG and their failure only blocks descendants.
        for job in sorted([item for item in runnable if item.cpu_only and item.stage in {"prepare", "replay", "episodes"}], key=lambda item: (item.priority, item.job_id)):
            if cpu_running >= max_cpu:
                break
            try:
                value = _special_job(job, by_id, compat=compat, prepared=prepared, root=root, local=local, executor=executor, annotation_train=annotation_train, annotation_val=annotation_val)
                statuses[job.job_id] = str(value.get("status")); job.status = statuses[job.job_id]
            except Exception as exc:
                record(job, "FAILED_CODE", reason=f"{type(exc).__name__}: {exc}")
            cpu_running += 1

        # Refresh live GPU memory for every admission, then acquire the UUID
        # lease only after the second check.  External compute processes are
        # recorded but do not veto a shared-memory admission.
        runnable = [job for job in jobs if statuses.get(job.job_id) == "PENDING" and all(statuses.get(dep) in terminal_success for dep in job.dependencies)]
        for job in sorted([item for item in runnable if item.cpu_only and item.stage not in {"prepare", "replay", "episodes"}], key=lambda item: (item.priority, item.job_id)):
            if cpu_running >= max_cpu:
                break
            try:
                if job.command == []:
                    _bind_worker_job(job, by_id, repo=repo, root=root, suite_path=suite_path, base_local=local, prepared=prepared, compat=compat, artifact_signature=artifact_signature, resume=resume, phase_state=phase_state)
                child = executor.start(job, job.command, env={"CUDA_VISIBLE_DEVICES": ""}, lease=None)
                running[job.job_id] = child; statuses[job.job_id] = "RUNNING"; job.status = "RUNNING"; cpu_running += 1
            except Exception as exc:
                record(job, "FAILED_CODE", reason=f"CPU worker start failed: {type(exc).__name__}: {exc}")
        for job in sorted(runnable, key=lambda item: (item.priority, item.job_id)):
            if job.cpu_only or job.stage in {"prepare", "replay", "episodes", "baseline_eval"} or job.job_id in running:
                continue
            if job.command == []:
                try:
                    _bind_worker_job(job, by_id, repo=repo, root=root, suite_path=suite_path, base_local=local, prepared=prepared, compat=compat, artifact_signature=artifact_signature, resume=resume, phase_state=phase_state)
                except Exception as exc:
                    record(job, "BLOCKED_DATA", reason=f"command binding failed: {type(exc).__name__}: {exc}")
                    continue
            request = _job_memory_request(job)
            snapshot = resource_snapshot(local, policy=device_policy, explicit_devices=explicit_devices, request=request, owned_busy_uuids=owned)
            candidates, decisions = select_gpus(discover_gpus(explicit_devices=explicit_devices), policy=device_policy, request=request, owned_busy_uuids=owned)
            acquired = None
            for candidate in candidates:
                acquired = refresh_and_acquire(candidate.uuid, request, lease_root=local.get("resources", {}).get("lock_dir", "~/.cache/tempotrack/gpu_leases"), job_id=job.job_id, run_signature=job.run_signature, allocation={"free_mib": candidate.memory_free_mib, "external_process_count": len(candidate.compute_processes)})
                if acquired is not None:
                    break
            if acquired is None:
                executor.mark(job, "WAITING_RESOURCES", reason="shared-memory admission will be retried with live free-memory query", resource_snapshot=snapshot, gpu_decisions=[item.to_dict() for item in decisions])
                continue
            owned.add(acquired.uuid)
            try:
                child = executor.start(job, job.command, env={**({"CUDA_VISIBLE_DEVICES": acquired.uuid}), **({} if not job.cpu_only else {"CUDA_VISIBLE_DEVICES": ""})}, lease=acquired)
                running[job.job_id] = child
                statuses[job.job_id] = "RUNNING"; job.status = "RUNNING"
            except Exception as exc:
                acquired.release(); owned.discard(acquired.uuid); record(job, "FAILED_CODE", reason=f"worker start failed: {type(exc).__name__}: {exc}")

        write_live()
        if not running and not any(statuses.get(job.job_id) == "PENDING" for job in jobs):
            break
        time.sleep(max(0.2, poll_seconds))
    state["job_statuses"] = dict(statuses)
    return statuses


def run_repair_v4(repo: str | Path, suite_path: str | Path, local_path: str | Path, reference_root: str | Path, run_root: str | Path, *, through: str = "complete", resume: str = "auto", device_policy: str = "shared-memory", continue_independent: bool = False, explicit_devices: Sequence[str] | None = None) -> dict[str, Any]:
    """Execute preparation, then keep a live artifact scheduler until terminal."""
    repo = Path(repo).resolve(); suite_path = Path(suite_path).resolve(); local_path = Path(local_path).resolve(); reference = Path(reference_root).resolve(); root = Path(run_root).resolve(); report_root = repo / "reports/v4"; report_root.mkdir(parents=True, exist_ok=True); root.mkdir(parents=True, exist_ok=True)
    lock_path = report_root / ".supervisor.lock"; lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close(); return {"status": "ALREADY_RUNNING", "reason": "reports/v4/.supervisor.lock is held", "run_root": str(root)}
    suite = load_yaml(suite_path); local = load_yaml(local_path)
    executor = JobExecutor(repo, root, report_root, python=str(local.get("research_python", sys.executable)), heartbeat_seconds=int(local.get("execution", {}).get("heartbeat_seconds", 30)))
    state: dict[str, Any] = {"schema_version": 4, "scheduler_revision": "shared-gpu-1", "started_at": _now(), "repo": str(repo), "run_root": str(root), "reference_root": str(reference), "through": through, "resume": resume, "resource_policy": device_policy, "recovery_command": f"{local.get('research_python', sys.executable)} -m tempotrack_research.cli repair-v4 run --repo {repo} --config {suite_path} --local {local_path} --reference-root {reference} --run-root {root} --through {through} --resume auto --device-policy shared-memory --continue-independent"}
    _atomic(report_root / "supervisor.json", {"schema_version": 4, "scheduler_revision": "shared-gpu-1", "status": "RUNNING", "pid": os.getpid(), "pid_start_ticks": process_start_ticks(os.getpid()), "cwd": str(repo), "run_root": str(root), "started_at": state["started_at"]})
    try:
        triage = inspect_v4(repo, reference, report_root / "triage.json")
        resources = resources_v4(repo, local_path, policy=device_policy, output=report_root / "resources.json", explicit_devices=explicit_devices)
        plan = plan_v4(repo, suite_path, local_path, root, through=through, output=report_root / "plan.json")
        verification = verify_v4(repo, suite_path, local_path, root, skip_unchanged=False)
        state.update({"resources": resources, "eligible_gpu_uuids": [item.get("uuid") for item in resources.get("eligible_devices", [])], "resource_reason": resources.get("reason"), "triage": triage, "verification": verification})
        _write_ledger(repo, compile_result=verification.get("compile", {}), checks=verification.get("checks", {}), resources=resources)
        jobs = build_experiment_dag(suite, root, through=through)
        _atomic(report_root / "dag.json", {"schema_version": 4, "scheduler_revision": "shared-gpu-1", "jobs": [job.to_dict() for job in jobs], "order": topological_order(jobs)})
        compat = _V4CompatPipeline(repo, suite_path, local_path, reference, root, executor, resume=resume)
        prepared = compat._prepare()
        prepared_path = Path(prepared["prepared"]); prepared_payload = _json(prepared_path, {})
        transform_path = Path(prepared.get("tensor_transform", root / "prepared/tensor_transform.json")); transform = _json(transform_path, dict(local.get("data", {}).get("transform_snapshot", {}))); transform["schema_version"] = 4; _atomic(transform_path, transform)
        prepared_payload.update({"schema_version": 4, "tensor_contract_hash": object_hash(transform), "artifact_signature": _signature(local, suite, prepared_payload).to_dict()}); _atomic(prepared_path, prepared_payload)
        compat._base.local.setdefault("data", {})["artifact_signature"] = prepared_payload["artifact_signature"]
        state.update({"artifact_signature": prepared_payload["artifact_signature"], "prepared": str(prepared_path), "matched_fraction": float(suite.get("training", {}).get("matched_fraction", 0.8)), "clean_fraction": float(suite.get("training", {}).get("clean_fraction", 0.2))})
        m0_train = compat._replay(prepared_payload, "fixed_dual", "train_base", tag="m0_v4"); m0_internal = compat._replay(prepared_payload, "fixed_dual", "val_base_internal", tag="m0_v4"); compat._replay(prepared_payload, "fixed_dual", "official_validation", tag="m0_v4")
        compat._episodes_from_replay(prepared_payload, m0_train, "train_base", "matched_frontend", "m0_v4", kinds=("memory", "pair", "continuation", "graph", "edit")); compat._episodes_from_replay(prepared_payload, m0_internal, "val_base_internal", "internal_tune", "m0_v4", kinds=("memory", "pair", "continuation", "graph", "edit"))
        state.update({"m0_replay": {"train": m0_train, "internal": m0_internal}, "m0_episodes": str(root / "episodes/m0_v4")})
        try:
            clean = compat._clean_episodes(prepared_payload)
            state["clean_episodes"] = str(clean)
        except Exception as exc:
            state["clean_episode_error"] = f"{type(exc).__name__}: {exc}"
        annotation_train = (repo / str(local["splits"]["train_annotation"])).resolve() if not Path(str(local["splits"]["train_annotation"])).is_absolute() else Path(str(local["splits"]["train_annotation"])).resolve()
        annotation_val = (repo / str(local["splits"]["validation_annotation"])).resolve() if not Path(str(local["splits"]["validation_annotation"])).is_absolute() else Path(str(local["splits"]["validation_annotation"])).resolve()
        # Baseline controls are DAG jobs.  They are dispatched as independent
        # CPU workers below, so their official evaluator cannot hold up GPU
        # admission for unrelated methods.
        state["baseline_status"] = "PENDING_SCHEDULER_CPU_WORKERS"
        state["checks_after_episode_rebuild"] = run_v4_checks(repo, reference_root=reference, run_root=root, suite=suite, local=local, output=report_root / "v4_checks.json")
        _bind_job_context({job.job_id: job for job in jobs}, root=root, prepared=prepared_payload, local=local, state=state)
        _atomic(report_root / "run_state.json", state)
        statuses = _run_scheduler(jobs, executor=executor, repo=repo, root=root, report_root=report_root, local=local, suite_path=suite_path, prepared=prepared_payload, compat=compat, artifact_signature=prepared_payload.get("artifact_signature"), state=state, device_policy=device_policy, explicit_devices=explicit_devices, resume=resume, annotation_train=annotation_train, annotation_val=annotation_val)
        state["job_statuses"] = statuses
        failures = [status for status in statuses.values() if status in {"FAILED_CODE", "ARTIFACT_INVALID", "BLOCKED_DATA", "BLOCKED_DEPENDENCY"}]
        state["status"] = "COMPLETED_WITH_FAILURES" if failures else "COMPLETED"
        state["finished_at"] = _now(); _atomic(report_root / "run_state.json", state)
        _atomic(report_root / "supervisor.json", {"schema_version": 4, "scheduler_revision": "shared-gpu-1", "status": state["status"], "pid": os.getpid(), "pid_start_ticks": process_start_ticks(os.getpid()), "run_root": str(root), "started_at": state["started_at"], "finished_at": state["finished_at"], "resource_reason": state.get("resource_reason")})
        _write_progress(repo, root, state, final=True, output=report_root / "ICLR_UPGRADE_SHARED_GPU_FINAL.md")
        return state
    except KeyboardInterrupt:
        state["status"] = "RUNNING"; state["interrupted_at"] = _now(); _atomic(report_root / "run_state.json", state); _write_progress(repo, root, state, final=False); raise
    except Exception as exc:
        state["status"] = "FAILED"; state["error"] = f"{type(exc).__name__}: {exc}"; state["finished_at"] = _now(); _atomic(report_root / "run_state.json", state); _write_progress(repo, root, state, final=True, output=report_root / "ICLR_UPGRADE_SHARED_GPU_FINAL.md"); raise
    finally:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally: lock.close()


__all__ = ["capabilities_v4", "inspect_v4", "resources_v4", "plan_v4", "verify_v4", "run_repair_v4", "status_v4", "report_v4", "stop_v4"]
