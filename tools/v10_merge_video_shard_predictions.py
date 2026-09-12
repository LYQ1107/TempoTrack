#!/usr/bin/env python3
"""Validate and merge complete-video predictions from one logical smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge(manifest_path: Path, trials_root: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen_videos: set[int] = set()
    rows: list[dict[str, Any]] = []
    shard_receipts: list[dict[str, Any]] = []
    frame_total = 0
    for item in manifest.get("shards", []):
        index = int(item["index"])
        trial = trials_root / f"shard_{index:02d}"
        receipt_path = trial / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"missing shard receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "COMPLETED":
            raise RuntimeError(f"shard is not completed: {receipt_path}: {receipt.get('status')}")
        stream_manifest_path = trial / "stream" / "stream_manifest.json"
        prediction_path = trial / "stream" / "tao_track.json"
        stream_manifest = json.loads(stream_manifest_path.read_text(encoding="utf-8"))
        if stream_manifest.get("status") != "PASS":
            raise RuntimeError(f"stream manifest failed: {stream_manifest_path}")
        expected_frames = int(item["frame_count"])
        if int(stream_manifest.get("frames", -1)) != expected_frames:
            raise RuntimeError(
                f"frame count mismatch for shard {index}: "
                f"{stream_manifest.get('frames')} != {expected_frames}"
            )
        video_ids = {int(value) for value in item["video_ids"]}
        overlap = seen_videos.intersection(video_ids)
        if overlap:
            raise RuntimeError(f"video overlap across shards: {sorted(overlap)[:8]}")
        seen_videos.update(video_ids)
        shard_rows = json.loads(prediction_path.read_text(encoding="utf-8"))
        if not isinstance(shard_rows, list):
            raise RuntimeError(f"prediction is not a list: {prediction_path}")
        bad = [row for row in shard_rows if int(row.get("video_id", -1)) not in video_ids]
        if bad:
            raise RuntimeError(f"prediction row escaped shard {index}: {bad[0]}")
        rows.extend(shard_rows)
        frame_total += expected_frames
        shard_receipts.append(
            {
                "index": index,
                "trial_id": receipt.get("trial_id"),
                "receipt": str(receipt_path),
                "receipt_sha256": _sha256(receipt_path),
                "prediction": str(prediction_path),
                "prediction_sha256": _sha256(prediction_path),
                "frame_count": expected_frames,
                "video_count": len(video_ids),
                "prediction_rows": len(shard_rows),
            }
        )
    expected_video_count = int(manifest.get("source_video_count", -1))
    expected_frame_count = int(manifest.get("source_frame_count", -1))
    if len(seen_videos) != expected_video_count or frame_total != expected_frame_count:
        raise RuntimeError(
            f"combined coverage mismatch: videos={len(seen_videos)}/{expected_video_count}, "
            f"frames={frame_total}/{expected_frame_count}"
        )
    rows.sort(key=lambda row: (int(row["video_id"]), int(row["image_id"]), int(row["track_id"])))
    _atomic_json(output, rows)
    receipt = {
        "schema_version": 1,
        "artifact": "tempotrack_v10_q1_full_smoke_combined_prediction",
        "status": "PASS",
        "annotation_shard_manifest": str(manifest_path),
        "annotation_shard_manifest_sha256": _sha256(manifest_path),
        "complete_video_disjoint": True,
        "source_video_count": expected_video_count,
        "source_frame_count": expected_frame_count,
        "covered_video_count": len(seen_videos),
        "covered_frame_count": frame_total,
        "prediction": str(output),
        "prediction_sha256": _sha256(output),
        "prediction_rows": len(rows),
        "shards": shard_receipts,
    }
    receipt_path = output.with_name(output.stem + ".merge_receipt.json")
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--trials-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge(args.manifest.resolve(), args.trials_root.resolve(), args.output.resolve()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
