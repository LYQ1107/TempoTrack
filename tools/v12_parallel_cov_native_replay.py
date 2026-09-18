#!/usr/bin/env python3
"""Parallelize the detector-free native COV replay used by Test event build.

The frontend cache is already complete and immutable.  This utility only
replays each disjoint frontend shard through the pinned native tracker, then
merges the resulting prediction rows with deterministic track-id offsets.  It
does not load annotations or run detector forwards.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _prediction_sort_key(row: Mapping[str, Any], image_order: Mapping[str, int]) -> tuple[Any, ...]:
    return (
        image_order[str(row["image_id"])],
        str(row.get("video_id")),
        int(row.get("track_id", -1)),
        int(row.get("category_id", -1)),
        tuple(float(value) for value in row.get("bbox", ())),
        float(row.get("score", 0.0)),
    )


def _capture_environment(capture: Path, repo: Path, cov_source: Path) -> dict[str, str]:
    document = read_json(capture)
    if document.get("capture_source") != "observed_live_proc_environ":
        raise RuntimeError("FAIL_CLOSED_CAPTURE_PROVENANCE")
    captured = document.get("environment", {})
    if not isinstance(captured, Mapping):
        raise RuntimeError("FAIL_CLOSED_CAPTURE_ENVIRONMENT_MISSING")
    environment = dict(os.environ)
    for key in ("LD_PRELOAD",):
        if captured.get(key):
            environment[key] = str(captured[key])
    captured_pythonpath = str(captured.get("PYTHONPATH", ""))
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(repo), str(cov_source), captured_pythonpath) if value
    )
    environment["V10_COV_SOURCE"] = str(cov_source)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    return environment


def _launch_shards(args: argparse.Namespace, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    cache_root = args.frontend_root.resolve()
    shard_ids = [f"{index:02d}" for index in range(int(args.shard_count))]
    gpus = [value.strip() for value in str(args.gpus).split(",") if value.strip()]
    if len(gpus) != len(shard_ids):
        raise ValueError("one GPU is required for each replay shard")
    environment = _capture_environment(args.capture.resolve(), args.repo.resolve(), args.cov_source.resolve())
    processes: list[tuple[str, str, subprocess.Popen[Any]]] = []
    records: list[dict[str, Any]] = []
    for shard, gpu in zip(shard_ids, gpus):
        cache = cache_root / f"shard_{shard}" / "frontend_cache"
        output_dir = args.parallel_root.resolve() / f"shard_{shard}"
        output = output_dir / "cov_native_prediction.json"
        if not (cache / "manifest.json").is_file():
            raise FileNotFoundError(f"frontend shard cache missing: {cache}")
        if output.exists() or output.with_name("cov_native_prediction.manifest.json").exists():
            raise RuntimeError(f"refusing to overwrite replay shard output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        child_environment = dict(environment)
        child_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            str(args.python.resolve()),
            str(args.repo.resolve() / "tools" / "v11_covtrack_replay_cache.py"),
            "--cache", str(cache.resolve()),
            "--tempo-config", str(args.tempo_config.resolve()),
            "--cov-source", str(args.cov_source.resolve()),
            "--cov-config", str(args.cov_config.resolve()),
            "--cov-checkpoint", str(args.cov_checkpoint.resolve()),
            "--output", str(output.resolve()),
            "--device", "cuda:0",
            "--track-offset-scope", "global",
        ]
        log = output_dir / "replay.log"
        handle = log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(args.cov_source.resolve()),
            env=child_environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        handle.close()
        record = {
            "shard": f"shard_{shard}",
            "gpu": str(gpu),
            "pid": int(process.pid),
            "cache": str(cache.resolve()),
            "output": str(output.resolve()),
            "log": str(log.resolve()),
            "argv": command,
            "status": "RUNNING",
            "started_at_unix": time.time(),
        }
        records.append(record)
        processes.append((shard, str(gpu), process))
    runtime["shards"] = records
    runtime["status"] = "REPLAY_RUNNING"
    write_json(args.runtime, runtime)

    remaining = {shard: process for shard, _gpu, process in processes}
    while remaining:
        for shard, process in list(remaining.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            for record in records:
                if record["shard"] == f"shard_{shard}":
                    record.update({"returncode": int(returncode), "status": "PASS" if returncode == 0 else "FAILED", "ended_at_unix": time.time()})
                    break
            del remaining[shard]
            runtime["shards"] = records
            write_json(args.runtime, runtime)
        if remaining:
            time.sleep(5.0)
    failures = [record for record in records if record.get("status") != "PASS"]
    if failures:
        raise RuntimeError(f"parallel native replay failed: {failures}")
    return records


def _merge(args: argparse.Namespace, runtime: dict[str, Any], records: list[dict[str, Any]]) -> None:
    full_cache = args.full_cache.resolve()
    full_manifest_path = full_cache / "manifest.json"
    full_manifest = read_json(full_manifest_path)
    image_order = {str(value): index for index, value in enumerate(full_manifest["ordered_image_ids"])}
    rows: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    offset = 0
    native_calls = 0
    frames = 0
    videos = 0
    for record in records:
        shard = record["shard"]
        shard_dir = args.parallel_root.resolve() / shard
        manifest_path = shard_dir / "cov_native_prediction.manifest.json"
        prediction_path = shard_dir / "cov_native_prediction.json"
        manifest = read_json(manifest_path)
        if manifest.get("status") != "PASS" or manifest.get("artifact") != "v11_covtrack_frontend_replay":
            raise RuntimeError(f"invalid native replay manifest: {manifest_path}")
        if int(manifest.get("detector_forward_calls", -1)) != 0 or manifest.get("gt_loaded_during_replay") is not False:
            raise RuntimeError(f"native replay causal guard failed: {manifest_path}")
        if not prediction_path.is_file() or sha256_file(prediction_path) != manifest.get("prediction_sha256"):
            raise RuntimeError(f"native replay prediction hash mismatch: {prediction_path}")
        local_rows = read_json(prediction_path)
        if not isinstance(local_rows, list):
            raise RuntimeError(f"native replay prediction is not a list: {prediction_path}")
        local_ids = [int(row.get("track_id", -1)) for row in local_rows if int(row.get("track_id", -1)) >= 0]
        track_count = max(local_ids, default=-1) + 1
        for row in local_rows:
            copied = dict(row)
            copied["track_id"] = int(copied["track_id"]) + offset
            rows.append(copied)
        shard_records.append(
            {
                "shard": shard,
                "gpu": record["gpu"],
                "pid": record["pid"],
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "prediction": str(prediction_path.resolve()),
                "prediction_sha256": manifest.get("prediction_sha256"),
                "frames": int(manifest.get("frames", 0)),
                "videos": int(manifest.get("videos", 0)),
                "rows": len(local_rows),
                "track_offset": offset,
                "track_count": track_count,
                "native_tracker_match_calls": int(manifest.get("native_tracker_match_calls", 0)),
            }
        )
        offset += track_count
        native_calls += int(manifest.get("native_tracker_match_calls", 0))
        frames += int(manifest.get("frames", 0))
        videos += int(manifest.get("videos", 0))
    rows.sort(key=lambda row: _prediction_sort_key(row, image_order))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prediction = output_root / "cov_native_prediction.json"
    manifest_path = output_root / "cov_native_prediction.manifest.json"
    if prediction.exists() or manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite native replay output: {output_root}")
    prediction.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    merged = {
        "status": "PASS",
        "artifact": "v11_covtrack_frontend_replay_merged_parallel",
        "cache": str(full_cache),
        "cache_manifest_sha256": sha256_file(full_manifest_path),
        "tempo_config": str(args.tempo_config.resolve()),
        "tempo_config_sha256": sha256_file(args.tempo_config.resolve()),
        "cov_source": str(args.cov_source.resolve()),
        "cov_config": str(args.cov_config.resolve()),
        "cov_checkpoint": str(args.cov_checkpoint.resolve()),
        "device": "parallel_one_gpu_per_frontend_shard",
        "frames": frames,
        "videos": videos,
        "rows": len(rows),
        "prediction": str(prediction),
        "prediction_sha256": sha256_file(prediction),
        "detector_forward_calls": 0,
        "native_tracker_match_calls": native_calls,
        "gt_loaded_during_replay": False,
        "track_offset_scope": "parallel_shard_global_offsets",
        "shard_count": len(shard_records),
        "shards": shard_records,
        "created_at_unix": time.time(),
    }
    write_json(manifest_path, merged)
    runtime.update(
        {
            "status": "PASS",
            "completed_at_unix": time.time(),
            "merged_prediction": str(prediction),
            "merged_manifest": str(manifest_path),
            "merged_prediction_sha256": merged["prediction_sha256"],
            "shard_records": shard_records,
        }
    )
    write_json(args.runtime, runtime)
    print(json.dumps(runtime, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-root", type=Path, required=True)
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--parallel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tempo-config", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cov-source", type=Path, required=True)
    parser.add_argument("--cov-config", type=Path, required=True)
    parser.add_argument("--cov-checkpoint", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    runtime = {
        "status": "STARTING",
        "artifact": "v12_parallel_cov_native_replay",
        "frontend_root": str(args.frontend_root.resolve()),
        "full_cache": str(args.full_cache.resolve()),
        "parallel_root": str(args.parallel_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "capture": str(args.capture.resolve()),
        "capture_source_required": "observed_live_proc_environ",
        "started_at_unix": time.time(),
    }
    write_json(args.runtime, runtime)
    try:
        records = _launch_shards(args, runtime)
        _merge(args, runtime, records)
        return 0
    except Exception as error:
        runtime.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}", "ended_at_unix": time.time()})
        write_json(args.runtime, runtime)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
