"""Process executor used by the V4 artifact scheduler.

This is intentionally small but real: every subprocess gets its own attempt
directory, command, PID/start ticks, heartbeat and exit status.  A job is not
completed merely because its process returned zero; required artifacts are
checked before the terminal event is emitted.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dag import JobSpec, artifacts_valid
from .gpu_pool import GPULease


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_start_ticks(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _compact_evaluation(value: Any) -> Any:
    """Keep status ledgers small while retaining artifact-addressable evidence.

    Official evaluators may return a per-class metric dictionary hundreds of
    megabytes large.  The complete payload already lives in the bound
    ``evaluation.json`` and TETA summary; a status ledger only needs the
    immutable paths, hashes and scalar summary needed to locate it.
    """
    if not isinstance(value, Mapping):
        return value
    compact_keys = (
        "status", "scheme", "profile", "seed", "split", "prediction",
        "prediction_path", "evaluation", "evaluation_path", "summary",
        "summary_path", "artifact_hashes", "source_manifest",
        "source_payload_hash", "tracklet_manifest", "tracklet_manifest_hash",
        "gt_path", "gt_split_hash", "duration_seconds", "command_hash",
        "metric_units", "calibration", "memory_checkpoint", "reason",
    )
    result = {key: value[key] for key in compact_keys if key in value}
    metrics = value.get("metrics")
    if isinstance(metrics, Mapping):
        # Parsed summaries use overall/base/novel; retain only those scalar
        # groups.  Flat per-class keys are intentionally not copied here.
        groups: dict[str, dict[str, Any]] = {}
        for scope in ("overall", "base", "novel"):
            item = metrics.get(scope)
            if isinstance(item, Mapping):
                groups[scope] = {key: item[key] for key in ("TETA", "LocA", "AssocA", "ClsA") if key in item}
        if groups:
            result["metrics"] = groups
    for nested_key in ("inference_job", "evaluation_job"):
        nested = value.get(nested_key)
        if isinstance(nested, Mapping):
            result[nested_key] = {key: nested[key] for key in ("status", "exit_code", "job_id", "log_path", "prediction", "evaluation", "summary", "started_at", "finished_at") if key in nested}
    return result


@dataclass
class RunningJob:
    spec: JobSpec
    process: subprocess.Popen[str]
    lease: GPULease | None
    attempt_id: str
    started_at: str
    pid_start_ticks: int | None
    log_path: Path
    log_handle: Any
    last_heartbeat_monotonic: float
    attempt_dir: Path


class JobExecutor:
    def __init__(self, repo: str | Path, run_root: str | Path, report_root: str | Path, *, python: str | Path, heartbeat_seconds: int = 30):
        self.repo = Path(repo).resolve(); self.run_root = Path(run_root).resolve(); self.report_root = Path(report_root).resolve(); self.python = str(python); self.heartbeat_seconds = int(heartbeat_seconds)
        self.jobs_path = self.report_root / "jobs.jsonl"; self.status_path = self.report_root / "status.json"
        self.report_root.mkdir(parents=True, exist_ok=True); self.run_root.mkdir(parents=True, exist_ok=True)
        self._event_lock = threading.Lock()
        self._latest: dict[str, dict[str, Any]] = {}
        if self.status_path.exists():
            try:
                value = json.loads(self.status_path.read_text(encoding="utf-8"))
                if isinstance(value.get("jobs"), Mapping):
                    self._latest = {str(key): dict(item) for key, item in value["jobs"].items() if isinstance(item, Mapping)}
            except (OSError, ValueError):
                self._latest = {}

    def event(self, value: Mapping[str, Any]) -> None:
        record = dict(value); record.setdefault("schema_version", 4); record.setdefault("event_at", _now())
        with self._event_lock:
            with self.jobs_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                handle.flush(); os.fsync(handle.fileno())
            if record.get("job_id"):
                self._latest[str(record["job_id"])] = record
            _atomic_json(self.status_path, {"schema_version": 4, "updated_at": _now(), "run_root": str(self.run_root), "jobs": dict(self._latest)})

    def mark(self, job: JobSpec, status: str, *, reason: str | None = None, **extra: Any) -> dict[str, Any]:
        if "actual_results" in extra:
            values = extra["actual_results"]
            if isinstance(values, list):
                extra["actual_results"] = [_compact_evaluation(item) for item in values]
            else:
                extra["actual_results"] = _compact_evaluation(values)
        record = {"job_id": job.job_id, "run_signature": job.run_signature, "stage": job.stage, "scheme": job.scheme, "phase": job.phase, "profile": job.profile, "train_phase": job.train_phase, "frontend": job.frontend, "method": job.method, "seed": job.seed, "dependencies": job.dependencies, "status": status, "reason": reason, **extra}
        self.event(record); return record

    def start(self, job: JobSpec, command: Sequence[str], *, env: Mapping[str, str] | None = None, lease: GPULease | None = None) -> RunningJob:
        if not command:
            raise ValueError(f"{job.job_id} has an empty command")
        attempt_id = f"{job.job_id}.{int(time.time() * 1000000)}".replace("/", "_")
        attempt_dir = self.run_root / "attempts" / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.report_root / "logs" / f"{attempt_id}.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
        full_command = [str(item) for item in command]
        child_env = dict(os.environ); child_env.update({str(key): str(value) for key, value in (env or {}).items()})
        if lease is not None:
            child_env["CUDA_VISIBLE_DEVICES"] = lease.uuid
        child_env.update({
            "PYTHONUNBUFFERED": "1",
            "TEMPOTRACK_MANAGED_JOB": "1",
            "TEMPOTRACK_JOB_ID": job.job_id,
            "TEMPOTRACK_ATTEMPT_ID": attempt_id,
            "TEMPOTRACK_RUN_DIR": str(job.run_dir),
        })
        resolved = {"job": job.to_dict(), "command": full_command, "env": {key: child_env.get(key) for key in ("CUDA_VISIBLE_DEVICES", "TEMPOTRACK_MANAGED_JOB", "TEMPOTRACK_JOB_ID", "TEMPOTRACK_ATTEMPT_ID", "TEMPOTRACK_RUN_DIR")}}
        _atomic_json(attempt_dir / "command.json", resolved)
        _atomic_json(Path(job.run_dir) / "resolved_job.json", {**job.to_dict(), "attempt_id": attempt_id, "command": full_command})
        started = _now()
        base = {"job_id": job.job_id, "attempt_id": attempt_id, "run_signature": job.run_signature, "command": full_command, "cwd": str(self.repo), "run_dir": job.run_dir, "started_at": started, "log_path": str(log_path), "device_uuid": lease.uuid if lease else None, "profile": job.profile, "train_phase": job.train_phase, "stage": job.stage, "scheme": job.scheme, "method": job.method, "frontend": job.frontend, "seed": job.seed}
        self.event({**base, "status": "LEASED" if lease else "RUNNING", "pid": None, "pid_start_ticks": None})
        handle = log_path.open("w", encoding="utf-8")
        handle.write("$ " + " ".join(full_command) + "\n"); handle.flush()
        try:
            process = subprocess.Popen(full_command, cwd=str(self.repo), env=child_env, stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        except OSError:
            handle.close()
            raise
        ticks = process_start_ticks(process.pid)
        if lease is not None:
            lease.update_worker(process.pid, ticks, run_dir=job.run_dir)
        running = RunningJob(job, process, lease, attempt_id, started, ticks, log_path, handle, time.monotonic(), attempt_dir)
        self.event({**base, "status": "RUNNING", "pid": process.pid, "pid_start_ticks": ticks, "heartbeat_at": _now()})
        return running

    @staticmethod
    def _progress(job: JobSpec) -> dict[str, Any]:
        path = Path(job.run_dir) / "progress.json"
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, Mapping) else {}
        except (OSError, ValueError):
            return {}

    def poll(self, running: RunningJob) -> Mapping[str, Any] | None:
        code = running.process.poll()
        progress = self._progress(running.spec)
        if code is None:
            now = time.monotonic()
            if now - running.last_heartbeat_monotonic >= self.heartbeat_seconds:
                running.last_heartbeat_monotonic = now
                self.event({"job_id": running.spec.job_id, "attempt_id": running.attempt_id, "run_signature": running.spec.run_signature, "status": "RUNNING", "pid": running.process.pid, "pid_start_ticks": process_start_ticks(running.process.pid), "heartbeat_at": _now(), "progress": progress, "log_path": str(running.log_path), "device_uuid": running.lease.uuid if running.lease else None})
            return None
        return self.finalize(running, int(code), progress=progress)

    def finalize(self, running: RunningJob, exit_code: int, *, progress: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        try:
            running.log_handle.flush(); running.log_handle.close()
        except OSError:
            pass
        valid, missing = artifacts_valid(running.spec)
        status = "COMPLETED" if exit_code == 0 and valid else ("ARTIFACT_INVALID" if exit_code == 0 else "FAILED_CODE")
        result = {"job_id": running.spec.job_id, "attempt_id": running.attempt_id, "run_signature": running.spec.run_signature, "status": status, "exit_code": int(exit_code), "pid": running.process.pid, "pid_start_ticks": running.pid_start_ticks, "finished_at": _now(), "missing_artifacts": missing, "progress": dict(progress or {}), "log_path": str(running.log_path), "device_uuid": running.lease.uuid if running.lease else None, "run_dir": running.spec.run_dir}
        self.event(result)
        if running.lease is not None:
            running.lease.release()
        return result

    def run(self, job: JobSpec, command: Sequence[str], *, env: Mapping[str, str] | None = None, lease: GPULease | None = None, timeout_seconds: int | None = None) -> dict[str, Any]:
        running = self.start(job, command, env=env, lease=lease)
        deadline = time.monotonic() + int(timeout_seconds) if timeout_seconds else None
        while True:
            result = self.poll(running)
            if result is not None:
                return dict(result)
            if deadline is not None and time.monotonic() >= deadline:
                try: os.killpg(os.getpgid(running.process.pid), signal.SIGTERM)
                except OSError: pass
                running.process.wait(timeout=30)
                return dict(self.finalize(running, int(running.process.returncode or -signal.SIGTERM), progress=self._progress(job)))
            time.sleep(min(2.0, max(0.1, self.heartbeat_seconds / 2)))

    def stop_after_checkpoint(self, job: JobSpec, *, pid: int, pid_start_ticks: int, checkpoint: str | Path | None = None) -> dict[str, Any]:
        """Stop only a validated owned process group after recording checkpoint evidence."""
        process_dir = Path(f"/proc/{int(pid)}")
        current = process_start_ticks(pid)
        if current != int(pid_start_ticks) or not process_dir.exists():
            return self.mark(job, "STOP_REFUSED", reason="pid_or_start_ticks_mismatch", pid=pid, expected_start_ticks=pid_start_ticks, observed_start_ticks=current)
        try:
            uid = int((process_dir / "status").read_text(encoding="utf-8").split("Uid:", 1)[1].split()[0])
            cwd = Path(os.readlink(process_dir / "cwd")).resolve()
            pgid = os.getpgid(pid)
        except (OSError, ValueError, IndexError) as exc:
            return self.mark(job, "STOP_REFUSED", reason=f"process_identity_unreadable:{exc}")
        if uid != os.getuid() or (cwd != self.repo and self.repo not in cwd.parents) or pgid <= 1 or pgid == os.getpgrp():
            return self.mark(job, "STOP_REFUSED", reason="ownership_or_process_group_check_failed", uid=uid, cwd=str(cwd), pgid=pgid)
        checkpoint_path = Path(checkpoint) if checkpoint else None
        if checkpoint_path is None or not checkpoint_path.exists():
            return self.mark(job, "STOP_REFUSED", reason="checkpoint_not_present", checkpoint=str(checkpoint_path) if checkpoint_path else None)
        os.killpg(pgid, signal.SIGTERM)
        return self.mark(job, "STOP_REQUESTED_AFTER_CHECKPOINT", pid=pid, pid_start_ticks=pid_start_ticks, pgid=pgid, checkpoint=str(checkpoint_path))


__all__ = ["RunningJob", "JobExecutor", "process_start_ticks"]
