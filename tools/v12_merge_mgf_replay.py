#!/usr/bin/env python3
"""Merge V12 cached replay shards without loading annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--trial-id", default="s00_m00")
    args = parser.parse_args()
    shard_root = args.shard_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty merge root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    full_manifest_path = args.full_cache.resolve() / "manifest.json"
    full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    image_order = {str(value): index for index, value in enumerate(full_manifest["ordered_image_ids"])}
    rows: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    offset = 0
    reference: dict[str, Any] | None = None
    for index in range(int(args.shard_count)):
        shard = f"shard_{index:02d}"
        manifest_path = shard_root / shard / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"missing V12 replay shard manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or manifest.get("artifact") != "v12_qdic_mgf_causal_replay":
            raise RuntimeError(f"invalid V12 replay manifest: {manifest_path}")
        if int(manifest.get("detector_forward_calls", -1)) != 0 or manifest.get("gt_loaded_during_replay") is not False:
            raise RuntimeError(f"V12 replay causal guard failed: {manifest_path}")
        if reference is None:
            reference = manifest
        elif manifest.get("thresholds") != reference.get("thresholds"):
            raise RuntimeError("V12 replay threshold mismatch across shards")
        prediction = Path(str(manifest["prediction"])).resolve()
        if not prediction.is_file() or sha256_file(prediction) != manifest.get("prediction_sha256"):
            raise RuntimeError(f"V12 prediction hash mismatch: {prediction}")
        local_rows = json.loads(prediction.read_text(encoding="utf-8"))
        if not isinstance(local_rows, list):
            raise RuntimeError(f"V12 prediction is not a list: {prediction}")
        local_ids = [int(row["track_id"]) for row in local_rows if int(row.get("track_id", -1)) >= 0]
        local_count = max(local_ids, default=-1) + 1
        for row in local_rows:
            copied = dict(row)
            copied["track_id"] = int(copied["track_id"]) + offset
            rows.append(copied)
        shard_records.append({
            "shard": shard,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "prediction": str(prediction),
            "prediction_sha256": manifest.get("prediction_sha256"),
            "rows": len(local_rows),
            "frames": int(manifest.get("frames", 0)),
            "videos": int(manifest.get("videos", 0)),
            "track_offset": offset,
            "track_count": local_count,
            "cache": manifest.get("cache"),
            "cache_manifest_sha256": manifest.get("cache_manifest_sha256"),
        })
        offset += local_count
    rows.sort(key=lambda row: (
        image_order.get(str(row.get("image_id")), 10**12),
        str(row.get("video_id")),
        int(row.get("track_id", -1)),
        int(row.get("category_id", -1)),
    ))
    prediction = output_root / "tao_track.json"
    prediction.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "PASS",
        "artifact": "v12_qdic_mgf_causal_replay_merged",
        "trial_id": str(args.trial_id),
        "full_cache": str(args.full_cache.resolve()),
        "full_cache_manifest_sha256": sha256_file(full_manifest_path),
        "thresholds": reference.get("thresholds") if reference else None,
        "frames": sum(int(item.get("frames", 0)) for item in shard_records),
        "videos": sum(int(item.get("videos", 0)) for item in shard_records),
        "rows": len(rows),
        "prediction": str(prediction),
        "prediction_sha256": sha256_file(prediction),
        "detector_forward_calls": 0,
        "gt_loaded_during_replay": False,
        "shard_count": int(args.shard_count),
        "shards": shard_records,
        "mgf_provenance": reference.get("mgf_provenance") if reference else None,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
