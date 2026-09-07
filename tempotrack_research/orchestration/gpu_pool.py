"""GPU discovery, shared-memory admission and per-project UUID leases.

External compute applications are evidence for the report, not a blanket
rejection rule.  The lease prevents two TempoTrack workers from using one
UUID; the live free-memory check is the admission rule for shared GPUs.
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


@dataclass(frozen=True)
class GPUMemoryRequest:
    method: str
    stage: str
    required_mib: int
    safety_mib: int
    source: str
    workload_signature: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GPUDecision:
    uuid: str
    eligible: bool
    free_mib: int
    required_mib: int
    safety_mib: int
    external_process_count: int
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def update_worker(self, pid: int, start_ticks: int | None, *, run_dir: str | None = None, request: GPUMemoryRequest | None = None) -> None:
        """Persist the child identity while keeping the coordinator lock."""
        try:
            self.lock_handle.seek(0)
            payload = json.loads(self.lock_handle.read() or "{}")
        except (OSError, ValueError):
            payload = {}
        payload.update({"worker_pid": int(pid), "worker_start_ticks": start_ticks})
        if run_dir is not None:
            payload["run_dir"] = str(run_dir)
        if request is not None:
            payload["memory_request"] = request.to_dict()
        try:
            self.lock_handle.seek(0); self.lock_handle.truncate()
            self.lock_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.lock_handle.flush(); os.fsync(self.lock_handle.fileno())
        except OSError:
            pass

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


def _scheduler_devices() -> set[str] | None:
    """Read hard scheduler/container visibility, not a stale shell default."""
    values: list[str] = []
    for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "NVIDIA_VISIBLE_DEVICES"):
        raw = os.environ.get(name)
        if raw and raw not in {"all", "void", "none"}:
            values.extend(item.strip() for item in raw.replace(" ", ",").split(",") if item.strip())
    return set(values) if values else None


def select_gpus(
    inventory: Sequence[GPUInfo], *,
    policy: str,
    request: GPUMemoryRequest,
    owned_busy_uuids: set[str],
) -> tuple[list[GPUInfo], list[GPUDecision]]:
    """Select shared GPUs by live free memory and this project's leases."""
    shared = str(policy).lower().replace("_", "-") in {"shared-memory", "shared-gpu-1"}
    ranked = sorted(inventory, key=lambda item: (-item.memory_free_mib, item.utilization_gpu if item.utilization_gpu is not None else 10_000, item.uuid))
    eligible: list[GPUInfo] = []
    decisions: list[GPUDecision] = []
    for gpu in ranked:
        reason: str | None = None
        if not gpu.visible:
            reason = "DEVICE_NOT_ACCESSIBLE"
        elif not shared:
            reason = f"UNSUPPORTED_POLICY:{policy}"
        elif gpu.uuid in owned_busy_uuids:
            reason = "OWN_JOB_LEASED"
        elif gpu.memory_free_mib < int(request.required_mib) + int(request.safety_mib):
            reason = "INSUFFICIENT_MEMORY_FOR_JOB"
        decision = GPUDecision(gpu.uuid, reason is None, gpu.memory_free_mib, int(request.required_mib), int(request.safety_mib), len(gpu.compute_processes), reason)
        decisions.append(decision)
        if reason is None:
            eligible.append(gpu)
    return eligible, decisions


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


def refresh_and_acquire(
    gpu_uuid: str,
    request: GPUMemoryRequest,
    *,
    lease_root: str | Path,
    job_id: str,
    run_signature: str,
    allocation: Mapping[str, Any] | None = None,
) -> GPULease | None:
    """Refresh NVML/nvidia-smi and acquire only after the current check."""
    inventory = discover_gpus()
    gpu = next((item for item in inventory if item.uuid == str(gpu_uuid) and item.visible), None)
    if gpu is None or gpu.memory_free_mib < request.required_mib + request.safety_mib:
        return None
    lease = acquire_lease(gpu, lease_root, job_id=job_id, run_signature=run_signature, nonblocking=True)
    if lease is not None:
        lease.update_worker(os.getpid(), _start_ticks(os.getpid()), request=request)
        try:
            path = lease.lease_path
            handle = lease.lock_handle
            handle.seek(0); payload = json.loads(handle.read() or "{}")
            payload["allocation"] = dict(allocation or {})
            payload["memory_request"] = request.to_dict()
            handle.seek(0); handle.truncate(); handle.write(json.dumps(payload, ensure_ascii=False) + "\n"); handle.flush(); os.fsync(handle.fileno())
        except (OSError, ValueError):
            pass
    return lease


def resource_snapshot(local: Mapping[str, Any] | None = None, *, policy: str = "shared-memory", explicit_devices: Sequence[str] | None = None, request: GPUMemoryRequest | None = None, owned_busy_uuids: set[str] | None = None) -> dict[str, Any]:
    local = dict(local or {})
    resources = dict(local.get("resources", {}))
    requested = explicit_devices
    configured = resources.get("allowed_devices")
    if requested is None and isinstance(configured, list):
        requested = [str(item) for item in configured]
    elif requested is None and isinstance(configured, str) and configured not in {"all_accessible", "all"}:
        requested = [item.strip() for item in configured.split(",") if item.strip()]
    scheduler_devices = _scheduler_devices() if bool(resources.get("respect_scheduler_allocation", True)) else None
    if scheduler_devices:
        # An explicit empty intersection means there is no allocated device;
        # passing [] to discover_gpus would otherwise mean "all visible".
        requested = sorted(set(requested or scheduler_devices) & scheduler_devices)
        if not requested:
            return {
                "schema_version": 4,
                "checked_at": _now(),
                "policy": policy,
                "explicit_devices": list(explicit_devices or []),
                "configured_allowed_devices": configured,
                "scheduler_context": scheduler_context(),
                "inventory": [],
                "eligible_devices": [],
                "memory_request": (request or GPUMemoryRequest("bootstrap", "train", int(resources.get("memory_bootstrap_mib", 4096)), int(resources.get("memory_safety_mib", 2048)), "bootstrap_estimate", "bootstrap")).to_dict(),
                "blocked_devices": [],
                "gpu_decisions": [],
                "resource_blocked": True,
                "reason": "NO_SCHEDULER_VISIBLE_DEVICE",
                "host_pid": os.getpid(),
                "host_start_ticks": _start_ticks(os.getpid()),
            }
    inventory = discover_gpus(explicit_devices=requested)
    headroom = resources.get("gpu_memory_headroom_gib", 2)
    try:
        safety_default = int(float(headroom) * 1024)
    except (TypeError, ValueError):
        safety_default = 2048
    required = request or GPUMemoryRequest("bootstrap", "train", int(resources.get("memory_bootstrap_mib", 4096)), int(resources.get("memory_safety_mib", safety_default)), "bootstrap_estimate", "bootstrap")
    owned = set(owned_busy_uuids or set())
    eligible, decisions = select_gpus(inventory, policy=policy, request=required, owned_busy_uuids=owned)
    blocked = [decision.to_dict() for decision in decisions if not decision.eligible]
    return {
        "schema_version": 4,
        "checked_at": _now(),
        "policy": policy,
        "explicit_devices": list(explicit_devices or []),
        "configured_allowed_devices": configured,
        "scheduler_context": scheduler_context(),
        "inventory": [item.to_dict() for item in inventory],
        "eligible_devices": [item.to_dict() for item in eligible],
        "memory_request": required.to_dict(),
        "blocked_devices": blocked,
        "gpu_decisions": [decision.to_dict() for decision in decisions],
        "resource_blocked": not bool(eligible),
        "reason": "MEMORY_QUERY_FAILED" if not inventory else ("INSUFFICIENT_MEMORY_FOR_JOB" if not eligible else None),
        "host_pid": os.getpid(),
        "host_start_ticks": _start_ticks(os.getpid()),
    }


__all__ = ["GPUInfo", "GPUMemoryRequest", "GPUDecision", "GPULease", "discover_gpus", "select_idle_gpus", "select_gpus", "acquire_lease", "refresh_and_acquire", "resource_snapshot"]
