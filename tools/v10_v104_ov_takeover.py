#!/usr/bin/env python3
"""Safely expand the V10.4 OVTrack prefetch after a coordinator handoff.

The original prefetch coordinator deliberately started only three shared-GPU
workers.  This helper is used only after that coordinator is stopped, and it
attaches to the already-running worker PIDs recorded in its state instead of
restarting them.  New shards have distinct complete-video output directories;
the helper never signals an inference worker and never writes a second
producer into an active shard directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import v10_v104_ov_prefetch as base


POLL_SECONDS = 20
MIN_FREE_MB = base.MIN_FREE_MB
MIN_MEM_AVAILABLE_MB = 20_000
ALL_GPUS = list(range(10))


def live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def mem_available_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def set_job(state: dict[str, Any], split: str, index: int, pid: int, **updates: Any) -> dict[str, Any]:
    lane = state.setdefault("lanes", {}).setdefault(
        split, {"status": "RUNNING", "completed_shards": [], "jobs": []}
    )
    for job in lane.setdefault("jobs", []):
        if int(job.get("index", -1)) == index and int(job.get("pid", -1)) == pid:
            job.update(updates)
            return job
    job = {"index": index, "pid": pid, **updates}
    lane["jobs"].append(job)
    return job


def mark_complete(state: dict[str, Any], split: str, index: int, pid: int, returncode: int) -> None:
    stream = base.ROOT / split / f"shard_{index:02d}" / "stream" / "tao_track.json"
    ok = returncode == 0 and stream.is_file()
    job = set_job(
        state,
        split,
        index,
        pid,
        returncode=int(returncode),
        ended_at=base.iso(),
        status="COMPLETED" if ok else "FAILED",
        output=str(stream),
    )
    lane = state["lanes"][split]
    if ok:
        lane.setdefault("completed_shards", []).append(index)
        lane["completed_shards"] = sorted(set(lane["completed_shards"]))
    else:
        state.update({"status": "FAILED", "reason": f"{split}_SHARD_{index:02d}_FAILED", "failed_job": job})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--max-active", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    state_path = args.state.resolve()
    state = base.read_json(state_path)
    if not isinstance(state, dict):
        raise RuntimeError(f"OV_STATE_MISSING:{state_path}")
    if root != base.ROOT.resolve():
        raise RuntimeError(f"OV_ROOT_MISMATCH:{root} != {base.ROOT.resolve()}")

    # The active list is the only authoritative handoff list.  Old failed
    # attempts may remain in lanes.jobs with stale RUNNING labels; they are
    # intentionally not attached here.
    active: dict[tuple[str, int], dict[str, Any]] = {}
    for item in state.get("active", []):
        if not isinstance(item, dict):
            continue
        try:
            key = (str(item["split"]), int(item["index"]))
            pid = int(item["pid"])
            gpu = int(item["gpu"])
        except (KeyError, TypeError, ValueError):
            continue
        active[key] = {"pid": pid, "gpu": gpu, "process": None, "handle": None}

    pending = []
    for item in state.get("pending", []):
        if isinstance(item, dict):
            try:
                key = (str(item["split"]), int(item["index"]))
            except (KeyError, TypeError, ValueError):
                continue
            if key not in active:
                pending.append(key)

    state["helper"] = "v10_v104_ov_takeover"
    state["pid"] = os.getpid()
    state["takeover_started_at"] = base.iso()
    state["takeover_max_active"] = max(1, int(args.max_active))
    state["status"] = "RUNNING"
    base.atomic_json(state_path, state)

    checkpoint = next((item for item in base.OV_CHECKPOINT_ALTERNATES if item.is_file()), None)
    if checkpoint is None:
        raise RuntimeError("OVTRACK_CHECKPOINT_MISSING")

    while pending or active:
        free = base.gpu_free_mb()
        leased = {int(item["gpu"]) for item in active.values()}
        available = [
            gpu for gpu in ALL_GPUS
            if gpu not in leased and free.get(gpu, 0) >= MIN_FREE_MB
        ]

        # First adopt already-running workers.  No child is reaped here;
        # only the output and /proc liveness are observed after handoff.
        for key, record in list(active.items()):
            if record["process"] is not None:
                continue
            if live(int(record["pid"])):
                continue
            split, index = key
            stream = root / split / f"shard_{index:02d}" / "stream" / "tao_track.json"
            if stream.is_file():
                mark_complete(state, split, index, int(record["pid"]), 0)
                active.pop(key)
            else:
                mark_complete(state, split, index, int(record["pid"]), 1)
                base.atomic_json(state_path, state)
                return 4

        # Do not admit a new model if the host is already near its safety
        # floor.  This leaves room for the currently running COV/MASA jobs.
        while pending and available and len(active) < max(1, int(args.max_active)):
            if mem_available_mb() < MIN_MEM_AVAILABLE_MB:
                break
            split, index = pending.pop(0)
            gpu = available.pop(0)
            command, env = base.make_command(split, index, gpu, root, checkpoint)
            log = root / "logs" / f"takeover_{split}_{index:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
            handle.write(
                f"\n[{base.iso()}] takeover gpu={gpu} free_mb={free.get(gpu)} "
                f"mem_available_mb={mem_available_mb()} $ {' '.join(command)}\n"
            )
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=str(base.HARD_REPO),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            key = (split, index)
            active[key] = {"pid": process.pid, "gpu": gpu, "process": process, "handle": handle}
            set_job(
                state,
                split,
                index,
                process.pid,
                gpu=gpu,
                log=str(log),
                command=command,
                started_at=base.iso(),
                status="RUNNING",
                takeover=True,
            )

        state["active"] = [
            {"split": key[0], "index": key[1], "gpu": int(record["gpu"]), "pid": int(record["pid"])}
            for key, record in sorted(active.items())
        ]
        state["pending"] = [{"split": split, "index": index} for split, index in pending]
        state["current_stage"] = "OV_TAKEOVER_WORKERS"
        state["next_action"] = "monitor adopted and expanded workers"
        state["heartbeat"] = base.iso()
        state["resource_snapshot"] = {
            "gpu_free_mb": free,
            "mem_available_mb": mem_available_mb(),
        }
        base.atomic_json(state_path, state)

        for key, record in list(active.items()):
            process = record["process"]
            if process is None:
                continue
            code = process.poll()
            if code is None:
                continue
            record["handle"].close()
            split, index = key
            mark_complete(state, split, index, int(record["pid"]), int(code))
            active.pop(key)
            if code != 0:
                base.atomic_json(state_path, state)
                return 4

        if pending or active:
            time.sleep(POLL_SECONDS)

    state.pop("active", None)
    state.pop("pending", None)
    for split in ("test", "val"):
        state["lanes"][split]["status"] = "MERGING"
        state["heartbeat"] = base.iso()
        base.atomic_json(state_path, state)
        try:
            result = base.merge_and_evaluate(split, root)
        except Exception as exc:
            state.update({"status": "FAILED", "reason": f"{split}_MERGE_OR_EVAL_FAILED", "error": repr(exc)})
            base.atomic_json(state_path, state)
            return 5
        state["lanes"][split].update(result)
        state["lanes"][split]["status"] = "PASS"
        state["heartbeat"] = base.iso()
        base.atomic_json(state_path, state)

    state.update({"status": "COMPLETED", "completed_at": base.iso(), "current_stage": "COMPLETED", "next_action": "consume OV results"})
    base.atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
