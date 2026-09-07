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
import time
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


class JobExecutor:
    def __init__(self, repo: str | Path, run_root: str | Path, report_root: str | Path, *, python: str | Path, heartbeat_seconds: int = 30):
        self.repo = Path(repo).resolve(); self.run_root = Path(run_root).resolve(); self.report_root = Path(report_root).resolve(); self.python = str(python); self.heartbeat_seconds = int(heartbeat_seconds)
        self.jobs_path = self.report_root / "jobs.jsonl"; self.status_path = self.report_root / "status.json"
        self.report_root.mkdir(parents=True, exist_ok=True); self.run_root.mkdir(parents=True, exist_ok=True)

    def event(self, value: Mapping[str, Any]) -> None:
        record = dict(value); record.setdefault("schema_version", 4); record.setdefault("event_at", _now())
        with self.jobs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        latest: dict[str, dict[str, Any]] = {}
        try:
            for line in self.jobs_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if item.get("job_id"):
                    latest[str(item["job_id"])] = item
        except (OSError, ValueError):
            pass
        _atomic_json(self.status_path, {"schema_version": 4, "updated_at": _now(), "run_root": str(self.run_root), "jobs": latest})

    def mark(self, job: JobSpec, status: str, *, reason: str | None = None, **extra: Any) -> dict[str, Any]:
        record = {"job_id": job.job_id, "run_signature": job.run_signature, "stage": job.stage, "scheme": job.scheme, "phase": job.phase, "seed": job.seed, "dependencies": job.dependencies, "status": status, "reason": reason, **extra}
        self.event(record); return record

    def run(self, job: JobSpec, command: Sequence[str], *, env: Mapping[str, str] | None = None, lease: GPULease | None = None, timeout_seconds: int | None = None) -> dict[str, Any]:
        attempt_id = f"{job.job_id}.{int(time.time() * 1000000)}"
        attempt_dir = self.run_root / "attempts" / attempt_id.replace("/", "_")
        attempt_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.report_root / "logs" / f"{attempt_id.replace('/', '_')}.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
        full_command = [str(item) for item in command]
        started = _now(); process: subprocess.Popen[str] | None = None
        child_env = dict(os.environ); child_env.update({str(key): str(value) for key, value in (env or {}).items()})
        if lease is not None:
            child_env["CUDA_VISIBLE_DEVICES"] = lease.uuid
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        base = {"job_id": job.job_id, "attempt_id": attempt_id, "run_signature": job.run_signature, "command": full_command, "cwd": str(self.repo), "run_dir": job.run_dir, "started_at": started, "log_path": str(log_path), "device_uuid": lease.uuid if lease else None}
        self.event({**base, "status": "RUNNING", "pid": None, "pid_start_ticks": None})
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(full_command) + "\n"); log.flush()
            try:
                process = subprocess.Popen(full_command, cwd=str(self.repo), env=child_env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            except OSError as exc:
                result = {**base, "status": "FAILED_START", "error": f"{type(exc).__name__}: {exc}", "finished_at": _now()}
                self.event(result); return result
            pid_ticks = process_start_ticks(process.pid)
            self.event({**base, "status": "RUNNING", "pid": process.pid, "pid_start_ticks": pid_ticks, "heartbeat_at": _now()})
            heartbeat = time.monotonic(); deadline = time.monotonic() + int(timeout_seconds) if timeout_seconds else None
            while process.poll() is None:
                if deadline is not None and time.monotonic() >= deadline:
                    try: os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    except OSError: pass
                    result = {**base, "status": "INTERRUPTED_TIMEOUT", "pid": process.pid, "pid_start_ticks": pid_ticks, "finished_at": _now()}
                    self.event(result); return result
                time.sleep(min(2.0, max(0.1, self.heartbeat_seconds / 2)))
                if time.monotonic() - heartbeat >= self.heartbeat_seconds:
                    self.event({**base, "status": "RUNNING", "pid": process.pid, "pid_start_ticks": process_start_ticks(process.pid), "heartbeat_at": _now()})
                    heartbeat = time.monotonic()
            exit_code = int(process.returncode)
        valid, missing = artifacts_valid(job)
        status = "COMPLETED" if exit_code == 0 and valid else ("ARTIFACT_MISSING" if exit_code == 0 else "FAILED")
        result = {**base, "status": status, "exit_code": exit_code, "pid": process.pid, "pid_start_ticks": pid_ticks, "finished_at": _now(), "missing_artifacts": missing}
        self.event(result); return result

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


__all__ = ["JobExecutor", "process_start_ticks"]
