"""Conservative GPU discovery and per-worker UUID leases for repair-v4.

The allocator treats every visible compute application as occupied.  It never
infers idleness from free memory alone and it never signals a process owned by
another task.  A worker receives a UUID in ``CUDA_VISIBLE_DEVICES``; CUDA then
sees that UUID as its local device zero.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _start_ticks(pid: int) -> int | None:
    try:
        tail = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def _proc_record(pid: int) -> dict[str, Any]:
    value: dict[str, Any] = {"pid": int(pid), "start_ticks": _start_ticks(pid)}
    try:
        value["uid"] = int(Path(f"/proc/{pid}/status").read_text(encoding="utf-8").split("Uid:", 1)[1].split()[0])
    except (OSError, ValueError, IndexError):
        value["uid"] = None
    try:
        value["cwd"] = str(Path(os.readlink(f"/proc/{pid}/cwd")).resolve())
    except OSError:
        value["cwd"] = None
    try:
        value["argv"] = [item.decode("utf-8", errors="replace") for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if item]
    except OSError:
        value["argv"] = []
    return value


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_free_mib: int
    utilization_gpu: int | None
    compute_processes: tuple[dict[str, Any], ...]
    visible: bool = True

    @property
    def occupied(self) -> bool:
        return bool(self.compute_processes)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["compute_processes"] = [dict(item) for item in self.compute_processes]
        value["occupied"] = self.occupied
        return value


@dataclass
class GPULease:
    uuid: str
    lease_path: Path
    lock_handle: Any
    owner_pid: int
    owner_start_ticks: int | None
    acquired_at: str

    def metadata(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "owner_pid": self.owner_pid,
            "owner_start_ticks": self.owner_start_ticks,
            "acquired_at": self.acquired_at,
            "lease_path": str(self.lease_path),
        }

    def release(self) -> None:
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_handle.close()

    def __enter__(self) -> "GPULease":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


def _query_rows(query: str) -> list[list[str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    rows: list[list[str]] = []
    for line in result.stdout.splitlines():
        if line.strip():
            rows.append([item.strip() for item in line.split(",")])
    return rows


def discover_gpus(*, explicit_devices: Sequence[str] | None = None) -> list[GPUInfo]:
    gpu_rows = _query_rows("gpu=index,uuid,name,memory.total,memory.free,utilization.gpu")
    app_rows = _query_rows("compute-apps=gpu_uuid,pid,process_name,used_memory")
    apps: dict[str, list[dict[str, Any]]] = {}
    for row in app_rows:
        if len(row) < 3:
            continue
        uuid = row[0]
        try:
            pid = int(row[1])
        except ValueError:
            continue
        item = {"pid": pid, "process_name": row[2], "used_memory_mib": row[3] if len(row) > 3 else None}
        item.update(_proc_record(pid))
        apps.setdefault(uuid, []).append(item)
    requested = {str(item).strip() for item in (explicit_devices or ()) if str(item).strip()}
    values: list[GPUInfo] = []
    for row in gpu_rows:
        if len(row) < 6:
            continue
        try:
            index = int(row[0]); total = int(float(row[3])); free = int(float(row[4]))
        except ValueError:
            continue
        uuid = row[1]
        visible = not requested or uuid in requested or str(index) in requested
        try:
            util = int(float(row[5]))
        except ValueError:
            util = None
        values.append(GPUInfo(index, uuid, row[2], total, free, util, tuple(apps.get(uuid, ())), visible))
    return values


def scheduler_context() -> dict[str, Any]:
    names = ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "NVIDIA_VISIBLE_DEVICES", "CUDA_MIG_STRICT_PARTITIONING")
    return {name: os.environ.get(name) for name in names if os.environ.get(name) is not None}


def select_idle_gpus(inventory: Sequence[GPUInfo], *, policy: str = "auto-idle", min_free_mib: int = 2048) -> tuple[list[GPUInfo], list[dict[str, Any]]]:
    eligible: list[GPUInfo] = []
    reasons: list[dict[str, Any]] = []
    for gpu in inventory:
        reason: list[str] = []
        if not gpu.visible:
            reason.append("outside_explicit_or_scheduler_visibility")
        if policy not in {"auto-idle", "auto_idle", "auto_allocated_idle"}:
            reason.append(f"unsupported_policy:{policy}")
        if gpu.compute_processes:
            reason.append("compute_process_present")
        if gpu.memory_free_mib < int(min_free_mib):
            reason.append("insufficient_free_memory")
        if reason:
            reasons.append({"uuid": gpu.uuid, "index": gpu.index, "reasons": reason, "processes": list(gpu.compute_processes)})
        else:
            eligible.append(gpu)
    return eligible, reasons


def acquire_lease(gpu: GPUInfo, root: str | Path, *, job_id: str, run_signature: str, nonblocking: bool = True) -> GPULease | None:
    root = Path(root).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f"{gpu.uuid}.lock"
    handle = lock_path.open("a+")
    flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
    try:
        fcntl.flock(handle.fileno(), flags)
    except BlockingIOError:
        handle.close(); return None
    record = {"schema_version": 4, "job_id": job_id, "run_signature": run_signature, "pid": os.getpid(), "start_ticks": _start_ticks(os.getpid()), "acquired_at": _now(), "uuid": gpu.uuid}
    handle.seek(0); handle.truncate(); handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush(); os.fsync(handle.fileno())
    return GPULease(gpu.uuid, lock_path, handle, os.getpid(), _start_ticks(os.getpid()), record["acquired_at"])


def resource_snapshot(local: Mapping[str, Any] | None = None, *, policy: str = "auto-idle", explicit_devices: Sequence[str] | None = None) -> dict[str, Any]:
    local = dict(local or {})
    resources = dict(local.get("resources", {}))
    requested = explicit_devices
    configured = resources.get("allowed_devices")
    if requested is None and isinstance(configured, list):
        requested = [str(item) for item in configured]
    inventory = discover_gpus(explicit_devices=requested)
    headroom = int(float(resources.get("gpu_memory_headroom_gib", 2)) * 1024)
    eligible, reasons = select_idle_gpus(inventory, policy=policy, min_free_mib=headroom)
    return {
        "schema_version": 4,
        "checked_at": _now(),
        "policy": policy,
        "explicit_devices": list(explicit_devices or []),
        "configured_allowed_devices": configured,
        "scheduler_context": scheduler_context(),
        "inventory": [item.to_dict() for item in inventory],
        "eligible_devices": [item.to_dict() for item in eligible],
        "blocked_devices": reasons,
        "resource_blocked": not bool(eligible),
        "reason": "no_visible_gpu_without_compute_process" if not eligible else None,
        "host_pid": os.getpid(),
        "host_start_ticks": _start_ticks(os.getpid()),
    }


__all__ = ["GPUInfo", "GPULease", "discover_gpus", "select_idle_gpus", "acquire_lease", "resource_snapshot"]
