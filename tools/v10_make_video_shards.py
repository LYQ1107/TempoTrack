#!/usr/bin/env python3
"""Make deterministic, complete-video annotation shards for one logical run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_id(item: dict[str, Any]) -> int:
    return int(item["id"])


def build_shards(annotation: Path, output: Path, count: int) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be positive")
    source = json.loads(annotation.read_text(encoding="utf-8"))
    videos = list(source.get("videos", []))
    images = list(source.get("images", []))
    annotations = list(source.get("annotations", []))
    if not videos or not images:
        raise ValueError("annotation must contain videos and images")
    image_by_video: dict[int, list[dict[str, Any]]] = {}
    for image in images:
        image_by_video.setdefault(int(image["video_id"]), []).append(image)
    ann_by_image: dict[int, list[dict[str, Any]]] = {}
    for item in annotations:
        ann_by_image.setdefault(int(item["image_id"]), []).append(item)

    # Greedy bin packing by frame count is deterministic and keeps workers
    # close in duration while never splitting a video.
    ordered = sorted(
        ((_video_id(video), len(image_by_video.get(_video_id(video), []))) for video in videos),
        key=lambda pair: (-pair[1], pair[0]),
    )
    bins: list[list[int]] = [[] for _ in range(count)]
    loads = [0] * count
    for video_id, frame_count in ordered:
        index = min(range(count), key=lambda value: (loads[value], value))
        bins[index].append(video_id)
        loads[index] += frame_count

    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, video_ids in enumerate(bins):
        video_set = set(video_ids)
        shard_videos = [video for video in videos if _video_id(video) in video_set]
        shard_images = [image for image in images if int(image["video_id"]) in video_set]
        image_ids = {int(image["id"]) for image in shard_images}
        shard_annotations = [item for item in annotations if int(item["image_id"]) in image_ids]
        shard = dict(source)
        shard["videos"] = shard_videos
        shard["images"] = shard_images
        shard["annotations"] = shard_annotations
        path = output / f"shard_{index:02d}.json"
        path.write_text(json.dumps(shard, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        records.append(
            {
                "index": index,
                "path": str(path),
                "sha256": _sha256(path),
                "video_ids": sorted(video_set),
                "video_count": len(video_set),
                "frame_count": len(shard_images),
                "annotation_count": len(shard_annotations),
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact": "tempotrack_v10_complete_video_annotation_shards",
        "source": str(annotation.resolve()),
        "source_sha256": _sha256(annotation),
        "source_video_count": len(videos),
        "source_frame_count": len(images),
        "source_annotation_count": len(annotations),
        "shard_count": count,
        "complete_video_disjoint": True,
        "shards": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(build_shards(args.annotation.resolve(), args.output.resolve(), args.count), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
