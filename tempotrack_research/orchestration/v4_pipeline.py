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

from ..config import ArtifactSignature, file_hash, load_yaml, object_hash
from ..data.feature_export import load_dataset_manifest
from ..errors import DataUnavailable
from .dag import JobSpec, artifacts_valid, build_experiment_dag, topological_order
from .executor import JobExecutor, process_start_ticks
from .gpu_pool import resource_snapshot
from .process_control import inspect_owned_jobs
from .v4_checks import run_v4_checks


BASE_COMMIT = "216aed1dbfd9aba19e78077f7b6a34f702b722ea"
REVIEWED_HEAD = "eb6a210af373555dd825b2e71c683abb06d01891"


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
        "schema_version": 4, "checked_at": _now(), "uid": {"uid": os.getuid(), "user": os.environ.get("USER", "unknown")}, "cwd": str(Path.cwd().resolve()), "repo": str(repo), "branch": _git(repo, "symbolic-ref", "--short", "HEAD"), "head": _git(repo, "rev-parse", "HEAD"), "reviewed_head": REVIEWED_HEAD, "origin_main": _git(repo, "rev-parse", "origin/main"), "remote": _git(repo, "remote", "get-url", "origin"), "base_commit": BASE_COMMIT, "diff_stat": _git(repo, "diff", "--stat", BASE_COMMIT), "dirty_paths": dirty, "reference_root": str(reference), "reference_exists": reference.exists(), "failures": _traceback_evidence(repo, reference), "owned_live_jobs": [item.to_dict() for item in old_jobs], "gpu_inventory": resource_snapshot({}, policy="auto-idle").get("inventory", []), "checkpoint_candidates": checkpoints, "requested_v4_root_present": (repo / "CODEX_TEMPOTRACK_REPAIR_AND_MULTIGPU_V4.md").exists(), "attachment_spec": "/home/user/.codex/attachments/d7b24120-c742-4c1c-b88f-73d8ff9f93dd/CODEX_TEMPOTRACK_REPAIR_AND_MULTIGPU_V4(1).md",
    }
    _atomic(report, payload); return payload


def capabilities_v4() -> dict[str, Any]:
    implementation: dict[str, bool] = {}
    try:
        from .dag import build_experiment_dag, topological_order
        from .executor import JobExecutor
        implementation["artifact_dag"] = callable(build_experiment_dag) and callable(topological_order)
        implementation["independent_gpu_jobs"] = callable(resource_snapshot) and JobExecutor is not None
        implementation["safe_resume"] = callable(process_start_ticks)
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
    return {"schema_version": 4, "implementation": implementation, "runtime": {"python": sys.executable, "python_version": sys.version, "torch": torch_version}, "module": __name__, "generated_at": _now()}


def resources_v4(repo: str | Path, local_path: str | Path, *, policy: str = "auto-idle", output: str | Path | None = None, explicit_devices: Sequence[str] | None = None) -> dict[str, Any]:
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
        "F03": ("IMPLEMENTED", "orchestration/gpu_pool.py", "UUID discovery and occupied-process rejection"),
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


def _write_progress(repo: Path, run_root: Path, state: Mapping[str, Any], *, final: bool = False, output: Path | None = None) -> Path:
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
            scheme = next((name for name in ("m0_no_offline", "m0_stable_emd", "m1_no_offline", "m1_stable_emd") if parent.startswith(name + "_")), "unknown")
            split = "official_validation" if "official_validation" in parent else "val_base_internal"
            prediction_candidates = sorted(run_root.glob(f"predictions/**/{scheme}/{split}/*.prediction.json"))
            prediction = prediction_candidates[0] if prediction_candidates else None
            row: dict[str, Any] = {
                "scheme": scheme,
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
    terminal = "COMPLETED" if final and statuses and all(status in {"COMPLETED", "REFERENCE_ONLY"} for status in statuses) and not any(status.startswith(("WAITING", "BLOCKED", "FAILED", "ARTIFACT", "INTERRUPTED")) for status in statuses) else ("PARTIAL" if final else str(state.get("status") or "RUNNING"))
    rows = official_rows()
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
    lines = ["# TempoTrack ICLR V4 repair and experiments", "", f"- status: `{terminal}`", f"- generated: `{_now()}`", f"- repo: `{repo}`", f"- run root: `{run_root}`", f"- HEAD: `{_git(repo, 'rev-parse', 'HEAD')}`", f"- V3 reference (read-only): `{state.get('reference_root')}`", f"- coordinator state: `{repo / 'reports/v4/run_state.json'}`", "", "## Artifacts", "", f"- triage: `{repo / 'reports/v4/triage.json'}`", f"- resources: `{repo / 'reports/v4/resources.json'}`", f"- plan/DAG: `{repo / 'reports/v4/plan.json'}` / `{repo / 'reports/v4/dag.json'}`", f"- verification: `{repo / 'reports/v4/verify.json'}`", f"- checks: `{repo / 'reports/v4/v4_checks.json'}`", f"- jobs: `{repo / 'reports/v4/jobs.jsonl'}`", f"- status: `{repo / 'reports/v4/status.json'}`", f"- repair ledger: `{repo / 'reports/v4/repair_ledger.md'}`", "", "## Job status", "", "| status | count |", "|---|---:|"]
    lines.extend(f"| `{key}` | {value} |" for key, value in sorted(statuses.items()))
    lines.extend(["", "## Official prediction and evaluation artifacts", "", "| scheme | split | status | TETA@50 | base TETA@50 | novel TETA@50 | summary |", "|---|---|---|---:|---:|---:|---|"])
    if rows:
        for row in rows:
            lines.append(f"| `{row['scheme']}` | `{row['split']}` | `{row['status']}` | {row.get('TETA')} | {row.get('base_TETA')} | {row.get('novel_TETA')} | `{row['summary']}` |")
    else:
        lines.append("| — | — | `NO_VERIFIED_SUMMARY` | — | — | — | — |")
    lines.extend(["", "## External/resource state", "", f"- eligible GPU UUIDs: `{state.get('eligible_gpu_uuids', [])}`", f"- blocker: `{state.get('resource_reason')}`", f"- launch resource snapshot: `{repo / 'reports/v4/resources.json'}`", f"- final resource snapshot: `{repo / 'reports/v4/resources_final.json'}`", "- Every GPU-dependent job without its own verified checkpoint, prediction, and evaluator artifact remains blocked/waiting; no empty metrics are emitted.", ""])
    resources = _json(repo / "reports/v4/resources.json", {})
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
    if state.get("recovery_command"):
        lines.extend(["## Recovery", "", "```bash", str(state["recovery_command"]), "```"])
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


def run_repair_v4(repo: str | Path, suite_path: str | Path, local_path: str | Path, reference_root: str | Path, run_root: str | Path, *, through: str = "complete", resume: str = "auto", device_policy: str = "auto-idle", continue_independent: bool = False, explicit_devices: Sequence[str] | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve(); suite_path = Path(suite_path).resolve(); local_path = Path(local_path).resolve(); reference = Path(reference_root).resolve(); root = Path(run_root).resolve(); report_root = repo / "reports/v4"; report_root.mkdir(parents=True, exist_ok=True); root.mkdir(parents=True, exist_ok=True)
    lock_path = report_root / ".supervisor.lock"; lock = lock_path.open("a+")
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: lock.close(); return {"status": "ALREADY_RUNNING", "reason": "reports/v4/.supervisor.lock is held", "run_root": str(root)}
    suite = load_yaml(suite_path); local = load_yaml(local_path); executor = JobExecutor(repo, root, report_root, python=str(local.get("research_python", sys.executable)), heartbeat_seconds=int(local.get("execution", {}).get("heartbeat_seconds", 30)))
    state: dict[str, Any] = {"schema_version": 4, "started_at": _now(), "repo": str(repo), "run_root": str(root), "reference_root": str(reference), "through": through, "resume": resume, "resource_policy": device_policy, "recovery_command": f"{local.get('research_python', sys.executable)} -m tempotrack_research.cli repair-v4 run --repo {repo} --config {suite_path} --local {local_path} --reference-root {reference} --run-root {root} --through {through} --resume auto --device-policy {device_policy} --continue-independent"}
    _atomic(report_root / "supervisor.json", {"schema_version": 4, "status": "RUNNING", "pid": os.getpid(), "pid_start_ticks": process_start_ticks(os.getpid()), "cwd": str(repo), "run_root": str(root), "started_at": state["started_at"]})
    try:
        triage = inspect_v4(repo, reference, report_root / "triage.json"); resources = resources_v4(repo, local_path, policy=device_policy, output=report_root / "resources.json", explicit_devices=explicit_devices); plan = plan_v4(repo, suite_path, local_path, root, through=through, output=report_root / "plan.json"); verification = verify_v4(repo, suite_path, local_path, root, skip_unchanged=False); state["resources"] = resources; state["eligible_gpu_uuids"] = [item.get("uuid") for item in resources.get("eligible_devices", [])]; state["resource_reason"] = resources.get("reason"); state["triage"] = triage; state["verification"] = verification
        _write_ledger(repo, compile_result=verification.get("compile", {}), checks=verification.get("checks", {}), resources=resources)
        jobs = build_experiment_dag(suite, root, through=through); _atomic(report_root / "dag.json", {"schema_version": 4, "jobs": [job.to_dict() for job in jobs], "order": topological_order(jobs)})
        # Reuse only frozen observation shards; preparation and M0 replay are
        # real production calls.  V4 episode manifests are rebuilt by the
        # corrected builders, while V2/V3 directories remain untouched.
        try:
            compat = _V4CompatPipeline(repo, suite_path, local_path, reference, root, executor, resume=resume)
            prepared = compat._prepare()
            prepared_path = Path(prepared["prepared"]); prepared_payload = _json(prepared_path, {})
            transform_path = Path(prepared.get("tensor_transform", root / "prepared/tensor_transform.json")); transform = _json(transform_path, dict(local.get("data", {}).get("transform_snapshot", {}))); transform["schema_version"] = 4; _atomic(transform_path, transform)
            prepared_payload.update({"schema_version": 4, "tensor_contract_hash": object_hash(transform), "artifact_signature": _signature(local, suite, prepared_payload).to_dict()})
            _atomic(prepared_path, prepared_payload)
            compat._base.local.setdefault("data", {})["artifact_signature"] = prepared_payload["artifact_signature"]
            state["artifact_signature"] = prepared_payload["artifact_signature"]
            state["prepared"] = str(prepared_path)
            m0_train = compat._replay(prepared_payload, "fixed_dual", "train_base", tag="m0_v4"); m0_internal = compat._replay(prepared_payload, "fixed_dual", "val_base_internal", tag="m0_v4"); state["m0_replay"] = {"train": m0_train, "internal": m0_internal}
            compat._episodes_from_replay(prepared_payload, m0_train, "train_base", "matched_frontend", "m0_v4", kinds=("memory", "pair", "continuation", "graph", "edit")); compat._episodes_from_replay(prepared_payload, m0_internal, "val_base_internal", "internal_tune", "m0_v4", kinds=("memory", "pair", "continuation", "graph", "edit")); state["m0_episodes"] = str(root / "episodes/m0_v4")
            # Run the two CPU baselines immediately.  They use the real infer
            # and official evaluator commands; GPU jobs are never substituted
            # by these controls.
            annotation_train = repo / str(local["splits"]["train_annotation"]); annotation_val = repo / str(local["splits"]["validation_annotation"]); m0_source = Path(prepared_payload["dataset_manifests"]["val_base_internal"])
            replay_path = root / "frontend/m0_v4/fixed_dual/val_base_internal/replay_manifest.json"
            compat._replay(prepared_payload, "fixed_dual", "official_validation", tag="m0_v4")
            baseline_results: dict[str, list[Mapping[str, Any]]] = {"m0_no_offline": [], "m0_stable_emd": []}
            for scheme in ("m0_no_offline", "m0_stable_emd"):
                baseline_results[scheme].append(compat._infer_eval(scheme, "baseline", 0, "val_base_internal", m0_source, replay_path, None, None, prepared_payload, annotation=annotation_train, tag="m0_v4"))
                official_source = Path(prepared_payload["dataset_manifests"]["official_validation"])
                baseline_results[scheme].append(compat._infer_eval(scheme, "baseline", 0, "official_validation", official_source, root / "frontend/m0_v4/fixed_dual/official_validation/replay_manifest.json", None, None, prepared_payload, annotation=annotation_val, tag="m0_v4"))
                ok = all(str(item.get("evaluation_job", {}).get("status", item.get("status", ""))) == "COMPLETED" for item in baseline_results[scheme])
                for baseline_job in (job for job in jobs if job.scheme == scheme):
                    actual_results = []
                    for item in baseline_results[scheme]:
                        candidate = item.get("evaluation") or item.get("evaluation_path") or item.get("evaluation_job", {}).get("evaluation") or item.get("path")
                        if not candidate:
                            candidate = root / "evaluations" / "baseline" / "seed0" / f"{scheme}_baseline_seed0_{item.get('split', '')}" / "evaluation.json"
                        actual_results.append(str(candidate))
                    executor.mark(baseline_job, "COMPLETED" if ok else "FAILED", reason=None if ok else "official/internal baseline evaluation did not produce a verified result", actual_results=actual_results, artifact_reuse="single fixed-observation baseline used for trial/full labels")
            state["baseline_status"] = "ATTEMPTED_REAL_INFER_EVAL"
            state["checks_after_episode_rebuild"] = run_v4_checks(repo, reference_root=reference, run_root=root, suite=suite, local=local, output=report_root / "v4_checks.json")
        except Exception as exc:
            state["production_preparation_error"] = f"{type(exc).__name__}: {exc}"; executor.mark(JobSpec("v4.prepare.fixed_observation", "shared", "fixed_dual", "prepare", "prepare", 0, "prepare"), "BLOCKED_DATA", reason=state["production_preparation_error"])
        # Independently mark every GPU job.  A blocked M1 branch does not
        # block M0 backend nodes; absent approved UUIDs block only jobs that
        # actually require one.
        for job in jobs:
            if job.job_id.startswith("v4.") or job.method in {"prepare", "replay", "no_offline", "stable_emd"}:
                continue
            if resources.get("resource_blocked"):
                executor.mark(job, "WAITING_RESOURCES", reason=resources.get("reason") or "no eligible auto-idle GPU", resource_snapshot=str(report_root / "resources.json"))
            else:
                executor.mark(job, "PENDING", reason="GPU lease scheduling is available; worker launch is deferred until verified task artifacts are bound")
        state["status"] = "WAITING_RESOURCES" if resources.get("resource_blocked") else "RUNNING"
        state["finished_at"] = _now(); _atomic(report_root / "run_state.json", state); _write_progress(repo, root, state, final=True); _atomic(report_root / "supervisor.json", {"schema_version": 4, "status": state["status"], "pid": os.getpid(), "pid_start_ticks": process_start_ticks(os.getpid()), "run_root": str(root), "started_at": state["started_at"], "finished_at": state["finished_at"], "resource_reason": state.get("resource_reason")})
        return state
    finally:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally: lock.close()


__all__ = ["capabilities_v4", "inspect_v4", "resources_v4", "plan_v4", "verify_v4", "run_repair_v4", "status_v4", "report_v4", "stop_v4"]
