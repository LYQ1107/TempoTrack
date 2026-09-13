#!/usr/bin/env python3
"""Run the already-defined OVTrack native+memory-only lanes alongside COV/MASA.

This is an execution helper, not a new tracking method.  It reuses the exact
V10.4 OV entry point and annotation shards, but writes to an isolated root so
that the sequential downstream supervisor cannot race its output writers.
Only one prefetch worker is placed on each selected physical GPU.  Existing
project jobs are allowed to share a card when the live free-VRAM check passes;
no process outside this helper is signalled or reprioritised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any


HARD_REPO = Path("/data2/usr_for_deadline/tempotrack_v104_search_hardening")
V10_ROOT = Path("/data2/usr_for_deadline/tempotrack_v10_unified")
ROOT = V10_ROOT / "v104_downstream" / "ov_early_parallel_20260913"
SHARD_ROOT = V10_ROOT / "v104_downstream" / "annotation_shards"
IMAGE_PREFIX = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames")
OV_SOURCE = V10_ROOT / "ovtrack_full_source"
OV_RUNTIME = HARD_REPO / "configs/research/v10/ovtrack_full_runtime.py"
OV_TEMPO = HARD_REPO / "configs/research/v10/ovtrack_full_tempo_memory_only.yaml"
OV_CHECKPOINT_ALTERNATES = (
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/ovtrack/saved_models/ovtrack_detpro_prompt.pth"),
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/l3/OVTrack/saved_models/ovtrack_detpro_prompt.pth"),
)
TEST_ANNOTATION = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json"
)
VAL_ANNOTATION = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json"
)
STREAM_PY = "/home/lwr/anaconda3/envs/ovtr/bin/python"
AUDIT_PY = "/home/lwr/anaconda3/envs/masaenv/bin/python"
SHARED_GPUS = [5, 6, 7]
MIN_FREE_MB = 10_000
POLL_SECONDS = 20


def iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
    finally:
        try:
            os.unlink(raw)
        except FileNotFoundError:
            pass


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def common_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "LD_PRELOAD": env.get("LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0"),
            "V10_OVTRACK_SOURCE": str(OV_SOURCE),
        }
    )
    entries = [str(HARD_REPO), str(OV_SOURCE), "/data2/usr_for_deadline/tet_a62a9c0_clean/teta"]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def gpu_free_mb() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    values: dict[int, int] = {}
    if result.returncode != 0:
        return values
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            values[int(fields[0])] = int(float(fields[1]))
        except ValueError:
            continue
    return values


def git_head(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def annotations() -> dict[str, Path]:
    return {"test": TEST_ANNOTATION, "val": VAL_ANNOTATION}


def make_command(split: str, index: int, gpu: int, root: Path, checkpoint: Path) -> tuple[list[str], dict[str, str]]:
    shard = SHARD_ROOT / f"OV_{split}" / f"shard_{index:02d}.json"
    out = root / split / f"shard_{index:02d}"
    # The stream wrapper intentionally requires these paths to exist before
    # startup; create them in the parent so an entry-point failure cannot be
    # mistaken for a model/runtime failure.
    (out / "work").mkdir(parents=True, exist_ok=True)
    (out / "stream").mkdir(parents=True, exist_ok=True)
    env = common_env()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "V10_WORK_DIR": str(out / "work"),
            "V10_STREAM_RESULTS_DIR": str(out / "stream"),
        }
    )
    command = [
        STREAM_PY,
        str(HARD_REPO / "tools/v10_ovtrack_test_tempo_stream.py"),
        str(OV_RUNTIME),
        str(checkpoint),
        "--out", str(out / "native_results.pkl"),
        "--eval-options", f"resfile_path={out / 'internal_results.pth'}",
        "--cfg-options",
        f"data.test.ann_file={shard}",
        f"data.test.img_prefix={IMAGE_PREFIX}/",
        "data.workers_per_gpu=0",
        f"model.tracker.tempo.config_path={OV_TEMPO}",
    ]
    return command, env


def merge_and_evaluate(split: str, root: Path) -> dict[str, Any]:
    annotation = annotations()[split]
    manifest = read_json(SHARD_ROOT / f"OV_{split}" / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("shards"), list):
        raise RuntimeError(f"OV_{split}_MANIFEST_INVALID")
    merged = root / split / "tao_track.json"
    merge_manifest = root / split / "merge_manifest.json"
    command = [
        AUDIT_PY,
        str(HARD_REPO / "tools/v10_merge_complete_video_tao.py"),
        "--annotation", str(annotation),
        "--output", str(merged),
        "--manifest", str(merge_manifest),
    ]
    for shard in manifest["shards"]:
        index = int(shard["index"])
        command.extend(["--shard", str(shard["path"]), str(root / split / f"shard_{index:02d}" / "stream/tao_track.json")])
    log = root / split / "merge.log"
    with log.open("w", encoding="utf-8") as handle:
        handle.write(f"[{iso()}] $ {' '.join(command)}\n")
        result = subprocess.run(command, cwd=str(HARD_REPO), env=common_env(), stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0 or not merged.is_file():
        raise RuntimeError(f"{split}_MERGE_FAILED:{result.returncode}")
    eval_dir = root / split / "evaluation"
    eval_command = [
        AUDIT_PY,
        str(HARD_REPO / "tools/eval_ovmot_teta.py"),
        "--gt", str(annotation),
        "--pred", str(merged),
        "--out", str(eval_dir),
        "--name", f"OV_V10_EARLY_{split.upper()}",
        "--cores", "2",
    ]
    eval_log = root / split / "evaluation.log"
    with eval_log.open("w", encoding="utf-8") as handle:
        handle.write(f"[{iso()}] $ {' '.join(eval_command)}\n")
        result = subprocess.run(eval_command, cwd=str(HARD_REPO), env=common_env(), stdout=handle, stderr=subprocess.STDOUT, text=True)
    summary = eval_dir / f"OV_V10_EARLY_{split.upper()}" / "teta_summary_results.pth"
    if result.returncode != 0 or not summary.is_file():
        raise RuntimeError(f"{split}_EVALUATION_FAILED:{result.returncode}")
    return {
        "status": "PASS",
        "split": split,
        "annotation": str(annotation),
        "annotation_sha256": sha256(annotation),
        "prediction": str(merged),
        "prediction_sha256": sha256(merged),
        "summary": str(summary),
        "summary_sha256": sha256(summary),
        "merge_command": command,
        "evaluation_command": eval_command,
        "completed_at": iso(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--max-active", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    checkpoint = next((item for item in OV_CHECKPOINT_ALTERNATES if item.is_file()), None)
    if checkpoint is None:
        atomic_json(state_path, {"status": "BLOCKED", "reason": "OVTRACK_CHECKPOINT_MISSING", "updated_at": iso()})
        return 2
    if not OV_SOURCE.is_dir() or not OV_RUNTIME.is_file() or not OV_TEMPO.is_file():
        atomic_json(state_path, {"status": "BLOCKED", "reason": "OV_INPUT_MISSING", "updated_at": iso()})
        return 3

    state = read_json(state_path)
    if not isinstance(state, dict):
        state = {
            "schema_version": 1,
            "status": "RUNNING",
            "helper": "v10_v104_ov_prefetch",
            "created_at": iso(),
            "hard_repo": str(HARD_REPO),
            "hard_repo_head": git_head(HARD_REPO),
            "ov_source": str(OV_SOURCE),
            "ov_source_head": git_head(OV_SOURCE),
            "runtime_config": str(OV_RUNTIME),
            "runtime_config_sha256": sha256(OV_RUNTIME),
            "tempo_config": str(OV_TEMPO),
            "tempo_config_sha256": sha256(OV_TEMPO),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "shared_gpus": SHARED_GPUS,
            "min_free_vram_mb": MIN_FREE_MB,
            "lanes": {},
        }
    state["pid"] = os.getpid()
    state["status"] = "RUNNING"
    atomic_json(state_path, state)

    pending = [(split, index) for index in range(8) for split in ("test", "val")]
    # Put one Test and one Val job first, then continue filling the shared
    # three-card lane.  All eight complete-video shards remain mandatory.
    pending = [("test", 0), ("val", 0), ("test", 1), ("val", 1)] + [
        (split, index) for index in range(2, 8) for split in ("test", "val")
    ]
    active: dict[tuple[str, int], tuple[subprocess.Popen[str], Any, int, Path]] = {}
    done: set[tuple[str, int]] = set()
    for split in ("test", "val"):
        lane = state.setdefault("lanes", {}).setdefault(split, {"status": "PENDING", "completed_shards": []})
        for index in lane.get("completed_shards", []):
            done.add((split, int(index)))
        if lane.get("status") == "PASS":
            for index in range(8):
                done.add((split, index))
    pending = [item for item in pending if item not in done]

    while pending or active:
        free = gpu_free_mb()
        available = [gpu for gpu in SHARED_GPUS if gpu not in {item[2] for item in active.values()} and free.get(gpu, 0) >= MIN_FREE_MB]
        while pending and available and len(active) < max(1, args.max_active):
            split, index = pending.pop(0)
            gpu = available.pop(0)
            out = root / split / f"shard_{index:02d}"
            out.mkdir(parents=True, exist_ok=True)
            command, env = make_command(split, index, gpu, root, checkpoint)
            log = root / "logs" / f"{split}_{index:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
            handle.write(f"\n[{iso()}] gpu={gpu} shared_free_mb={free.get(gpu)} $ {' '.join(command)}\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=str(HARD_REPO), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            key = (split, index)
            active[key] = (process, handle, gpu, log)
            state.setdefault("lanes", {}).setdefault(split, {"status": "RUNNING", "completed_shards": []})["status"] = "RUNNING"
            state["lanes"][split].setdefault("jobs", []).append({
                "index": index, "gpu": gpu, "pid": process.pid, "log": str(log),
                "command": command, "started_at": iso(), "status": "RUNNING",
            })

        state["active"] = [{"split": k[0], "index": k[1], "gpu": v[2], "pid": v[0].pid} for k, v in active.items()]
        state["pending"] = [{"split": split, "index": index} for split, index in pending]
        state["resource_snapshot"] = {str(k): v for k, v in free.items()}
        state["heartbeat"] = iso()
        atomic_json(state_path, state)

        for key, (process, handle, gpu, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            split, index = key
            lane = state["lanes"][split]
            job = next(item for item in lane.get("jobs", []) if item.get("index") == index and item.get("pid") == process.pid)
            job.update({"returncode": int(code), "ended_at": iso(), "status": "COMPLETED" if code == 0 else "FAILED"})
            stream = root / split / f"shard_{index:02d}" / "stream" / "tao_track.json"
            if code != 0 or not stream.is_file():
                state["status"] = "FAILED"
                state["reason"] = f"{split}_SHARD_{index:02d}_FAILED"
                state["failed_job"] = job
                atomic_json(state_path, state)
                for _key, (_proc, _handle, _gpu, _log) in active.items():
                    if _key != key:
                        # Do not signal healthy external workers.  This only
                        # records that no new prefetch work is admitted.
                        pass
                return 4
            done.add(key)
            lane.setdefault("completed_shards", []).append(index)
            lane["completed_shards"] = sorted(set(lane["completed_shards"]))
            active.pop(key)

        if pending or active:
            time.sleep(POLL_SECONDS)

    for split in ("test", "val"):
        state["lanes"][split]["status"] = "MERGING"
        state["heartbeat"] = iso()
        atomic_json(state_path, state)
        try:
            result = merge_and_evaluate(split, root)
        except Exception as exc:
            state["status"] = "FAILED"
            state["reason"] = f"{split}_MERGE_OR_EVAL_FAILED"
            state["error"] = repr(exc)
            atomic_json(state_path, state)
            return 5
        state["lanes"][split].update(result)
        state["lanes"][split]["status"] = "PASS"
        state["heartbeat"] = iso()
        atomic_json(state_path, state)

    state["status"] = "COMPLETED"
    state["completed_at"] = iso()
    state.pop("active", None)
    state.pop("pending", None)
    atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
