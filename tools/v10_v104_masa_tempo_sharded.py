#!/usr/bin/env python3
"""Run the MASA-R50 Tempo lane on complete-video shards.

This is an execution accelerator for the already-defined V10.4 MASA Tempo
configuration.  It does not change the detector, association model, or
Tempo configuration.  Each annotation shard contains complete videos, so a
fresh tracker state per shard is equivalent to the original per-video reset
semantics.  Outputs are written below a new immutable root and are merged
only after every shard has produced an official-format prediction.

The existing full-split MASA jobs are intentionally not touched.  This
helper is safe to run alongside them and is useful when the full-split
process is CPU-bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import v10_v104_masa_downstream as canonical  # noqa: E402
from v10_make_video_shards import build_shards  # noqa: E402


HARD_REPO = canonical.HARD_REPO
MASA_PY = canonical.MASA_PY
MASA_TEST = canonical.MASA_TEST
MASA_CHECKPOINT = canonical.MASA_CHECKPOINT
TEMPO_CONFIG = canonical.TEMPO_CONFIG
IMAGE_PREFIX = canonical.IMAGE_PREFIX
RUN_CWD = canonical.RUN_CWD
TETA_SOURCE = Path("/data2/usr_for_deadline/tet_a62a9c0_clean/teta")

ANNOTATIONS = dict(canonical.ANNOTATIONS)
PUBLIC_DETECTIONS = dict(canonical.PUBLIC_DETECTIONS)
SPLITS = ("val", "test")
DEFAULT_MAX_ACTIVE = 6
MIN_FREE_VRAM_MB = 8_000
MIN_MEM_AVAILABLE_MB = 20_000
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


def git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def available_memory_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, IndexError, ValueError):
        pass
    return 0


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
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            values[int(fields[0])] = int(float(fields[1]))
        except ValueError:
            continue
    return values


def common_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "LD_PRELOAD": env.get(
                "LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0"
            ),
        }
    )
    # ``eval_ovmot_teta.py`` and the parser import their dependencies from
    # the hardened repository; keep this environment independent of any
    # optional constant in the canonical runner module.
    entries = [str(HARD_REPO), str(TETA_SOURCE)]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def command_for(
    split: str,
    index: int,
    annotation: Path,
    output: Path,
    gpu: int,
) -> tuple[list[str], dict[str, str]]:
    official = output / "official_format"
    prediction = output / "predictions.pkl"
    output.mkdir(parents=True, exist_ok=True)
    env = common_env()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "V10_WORK_DIR": str(output / "work")})
    command = [
        MASA_PY,
        str(MASA_TEST),
        str(TEMPO_CONFIG),
        str(MASA_CHECKPOINT),
        "--work-dir",
        str(output / "work"),
        "--out",
        str(prediction),
        "--cfg-options",
        f"model.public_det_path={PUBLIC_DETECTIONS[split]}",
        f"test_dataloader.dataset.ann_file={annotation}",
        f"test_dataloader.dataset.data_prefix.img_path={IMAGE_PREFIX}/",
        "test_dataloader.num_workers=0",
        "test_dataloader.persistent_workers=False",
        f"test_evaluator.ann_file={annotation}",
        f"test_evaluator.outfile_prefix={official}",
    ]
    return command, env


def merge_and_evaluate(split: str, root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    split_root = root / split
    merged = split_root / "tao_track.json"
    merge_manifest = split_root / "merge_manifest.json"
    command = [
        MASA_PY,
        str(HARD_REPO / "tools/v10_merge_complete_video_tao.py"),
        "--annotation",
        str(ANNOTATIONS[split]),
        "--output",
        str(merged),
        "--manifest",
        str(merge_manifest),
    ]
    for item in manifest["shards"]:
        index = int(item["index"])
        command.extend(
            [
                "--shard",
                str(item["path"]),
                str(split_root / f"shard_{index:02d}" / "official_format/tao_track.json"),
            ]
        )
    merge_log = split_root / "merge.log"
    with merge_log.open("w", encoding="utf-8") as handle:
        handle.write(f"[{iso()}] $ {' '.join(command)}\n")
        result = subprocess.run(
            command,
            cwd=str(RUN_CWD),
            env=common_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0 or not merged.is_file():
        raise RuntimeError(f"{split}_MASA_SHARD_MERGE_FAILED:{result.returncode}")

    name = f"MASA_R50_COVDET_TEMPO_SHARDED_{split.upper()}"
    eval_dir = split_root / "evaluation"
    eval_command = [
        MASA_PY,
        str(HARD_REPO / "tools/eval_ovmot_teta.py"),
        "--gt",
        str(ANNOTATIONS[split]),
        "--pred",
        str(merged),
        "--out",
        str(eval_dir),
        "--name",
        name,
        "--cores",
        "8",
    ]
    eval_log = split_root / "evaluation.log"
    with eval_log.open("w", encoding="utf-8") as handle:
        handle.write(f"[{iso()}] $ {' '.join(eval_command)}\n")
        result = subprocess.run(
            eval_command,
            cwd=str(RUN_CWD),
            env=common_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    summaries = sorted(eval_dir.rglob("teta_summary_results.pth"))
    if result.returncode != 0 or not summaries:
        raise RuntimeError(f"{split}_MASA_SHARD_EVALUATION_FAILED:{result.returncode}")
    summary = summaries[-1]
    return {
        "status": "PASS",
        "split": split,
        "method": "tempo_sharded",
        "protocol": "tempo_memory_only_complete_video_shards",
        "annotation": canonical.annotation_inventory(ANNOTATIONS[split]),
        "public_detection_root": str(PUBLIC_DETECTIONS[split]),
        "public_detection_root_status": "frozen_existing_audited_input",
        "config": str(TEMPO_CONFIG),
        "config_sha256": sha256(TEMPO_CONFIG),
        "checkpoint": str(MASA_CHECKPOINT),
        "checkpoint_sha256": sha256(MASA_CHECKPOINT),
        "prediction": str(merged),
        "prediction_sha256": sha256(merged),
        "summary": str(summary),
        "summary_sha256": sha256(summary),
        "metrics": canonical.parse_summary(summary, ANNOTATIONS[split]),
        "shard_manifest": str(root / "annotations" / split / "manifest.json"),
        "shard_manifest_sha256": sha256(root / "annotations" / split / "manifest.json"),
        "merge_command": command,
        "evaluation_command": eval_command,
        "created_at": iso(),
    }


def prepare(root: Path, count: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state = read_json(state_path)
    if not isinstance(state, dict):
        state = {
            "schema_version": 1,
            "artifact": "tempotrack_v10_masa_r50_covdet_tempo_complete_video_shards",
            "status": "RUNNING",
            "created_at": iso(),
            "hard_repo": str(HARD_REPO),
            "hard_repo_head": git_head(HARD_REPO),
            "tempo_config": str(TEMPO_CONFIG),
            "tempo_config_sha256": sha256(TEMPO_CONFIG),
            "checkpoint": str(MASA_CHECKPOINT),
            "checkpoint_sha256": sha256(MASA_CHECKPOINT),
            "shard_count": count,
            "min_free_vram_mb": MIN_FREE_VRAM_MB,
            "min_mem_available_mb": MIN_MEM_AVAILABLE_MB,
            "splits": {},
        }
    for split in SPLITS:
        ann_dir = root / "annotations" / split
        manifest_path = ann_dir / "manifest.json"
        if not manifest_path.is_file():
            manifest = build_shards(ANNOTATIONS[split].resolve(), ann_dir, count)
        else:
            manifest = read_json(manifest_path)
        if not isinstance(manifest, dict) or int(manifest.get("shard_count", -1)) != count:
            raise RuntimeError(f"{split}_MASA_SHARD_MANIFEST_INVALID")
        state["splits"].setdefault(
            split,
            {
                "status": "PENDING",
                "annotation": str(ANNOTATIONS[split]),
                "annotation_sha256": sha256(ANNOTATIONS[split]),
                "public_detection_root": str(PUBLIC_DETECTIONS[split]),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "jobs": [],
            },
        )
    state["pid"] = os.getpid()
    state["heartbeat"] = iso()
    atomic_json(state_path, state)
    return state


def run(root: Path, state: dict[str, Any], max_active: int) -> int:
    state_path = root / "state.json"
    jobs: list[dict[str, Any]] = []
    for split in SPLITS:
        manifest = read_json(root / "annotations" / split / "manifest.json")
        if not isinstance(manifest, dict):
            raise RuntimeError(f"{split}_MASA_SHARD_MANIFEST_MISSING")
        for item in manifest["shards"]:
            index = int(item["index"])
            output = root / split / f"shard_{index:02d}"
            official = output / "official_format/tao_track.json"
            old = next(
                (
                    x
                    for x in state["splits"][split].get("jobs", [])
                    if int(x.get("index", -1)) == index
                ),
                None,
            )
            if isinstance(old, dict) and old.get("status") == "COMPLETED" and official.is_file():
                continue
            jobs.append({"split": split, "index": index, "annotation": Path(item["path"]), "output": output})

    active: list[tuple[dict[str, Any], subprocess.Popen[str], Any]] = []
    failed = False
    while jobs or active:
        free = gpu_free_mb()
        leased = {int(record["gpu"]) for record, _process, _handle in active}
        available = [gpu for gpu in sorted(free) if gpu not in leased and free[gpu] >= MIN_FREE_VRAM_MB]
        while jobs and available and len(active) < max(1, max_active):
            if available_memory_mb() < MIN_MEM_AVAILABLE_MB:
                break
            job = jobs.pop(0)
            gpu = available.pop(0)
            command, env = command_for(job["split"], job["index"], job["annotation"], job["output"], gpu)
            log = root / "logs" / f"{job['split']}_{job['index']:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
            handle.write(
                f"\n[{iso()}] gpu={gpu} free_vram_mb={free[gpu]} "
                f"mem_available_mb={available_memory_mb()} $ {' '.join(command)}\n"
            )
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=str(RUN_CWD),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            record = {
                "split": job["split"],
                "index": job["index"],
                "gpu": gpu,
                "pid": process.pid,
                "status": "RUNNING",
                "started_at": iso(),
                "log": str(log),
                "output": str(job["output"]),
                "command": command,
            }
            state["splits"][job["split"]].setdefault("jobs", []).append(record)
            state["splits"][job["split"]]["status"] = "RUNNING"
            active.append((record, process, handle))

        state["status"] = "RUNNING"
        state["active"] = [
            {"split": r["split"], "index": r["index"], "gpu": r["gpu"], "pid": r["pid"]}
            for r, _p, _h in active
        ]
        state["pending"] = [{"split": j["split"], "index": j["index"]} for j in jobs]
        state["resource_snapshot"] = {
            "gpu_free_mb": free,
            "mem_available_mb": available_memory_mb(),
        }
        state["heartbeat"] = iso()
        atomic_json(state_path, state)

        remaining: list[tuple[dict[str, Any], subprocess.Popen[str], Any]] = []
        for record, process, handle in active:
            code = process.poll()
            if code is None:
                remaining.append((record, process, handle))
                continue
            handle.close()
            output = Path(record["output"])
            official = output / "official_format/tao_track.json"
            ok = int(code) == 0 and official.is_file()
            record.update({"returncode": int(code), "ended_at": iso(), "status": "COMPLETED" if ok else "FAILED"})
            if not ok:
                failed = True
        active = remaining
        if failed:
            jobs.clear()
        if jobs or active:
            time.sleep(POLL_SECONDS)

    state.pop("active", None)
    state.pop("pending", None)
    if failed:
        state.update({"status": "BLOCKED", "current_stage": "SHARD_RUN", "next_action": "inspect failed MASA shard logs"})
        atomic_json(state_path, state)
        return 4

    rows = []
    for split in SPLITS:
        state["splits"][split]["status"] = "MERGING"
        state["heartbeat"] = iso()
        atomic_json(state_path, state)
        manifest = read_json(root / "annotations" / split / "manifest.json")
        if not isinstance(manifest, dict):
            raise RuntimeError(f"{split}_MASA_SHARD_MANIFEST_MISSING")
        try:
            row = merge_and_evaluate(split, root, manifest)
        except Exception as exc:
            state.update({"status": "BLOCKED", "current_stage": f"{split.upper()}_MERGE_EVAL", "error": repr(exc)})
            atomic_json(state_path, state)
            return 5
        state["splits"][split].update({"status": "PASS", "result": row})
        rows.append(row)
        state["heartbeat"] = iso()
        atomic_json(state_path, state)

    aggregate = {"status": "PASS", "artifact": "tempotrack_v10_masa_tempo_sharded_results", "rows": rows, "created_at": iso()}
    atomic_json(root / "masa_tempo_sharded_results.json", aggregate)
    state.update({"status": "COMPLETED", "current_stage": "COMPLETED", "next_action": "adopt valid sharded rows into final report", "completed_at": iso()})
    atomic_json(state_path, state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-active", type=int, default=DEFAULT_MAX_ACTIVE)
    args = parser.parse_args()
    root = args.root.resolve()
    if not MASA_CHECKPOINT.is_file() or not TEMPO_CONFIG.is_file():
        raise SystemExit("MASA_TEMPO_INPUT_MISSING")
    state = prepare(root, max(1, int(args.count)))
    return run(root, state, max(1, int(args.max_active)))


if __name__ == "__main__":
    raise SystemExit(main())
