"""Assemble disjoint complete-video frontend cache shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any

from tempotrack_v10.replay_cache import (
    FrontendReplayCacheReader,
    REPLAY_CACHE_ARTIFACT,
    REPLAY_CACHE_SCHEMA_VERSION,
    sha256_file,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    args = parser.parse_args()
    annotation = args.annotation.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty cache output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "videos").mkdir(parents=True, exist_ok=True)

    source = _read(annotation)
    source_images = list(source["images"])
    source_image_ids = [int(item["id"]) for item in source_images]
    source_by_video: dict[int, list[int]] = defaultdict(list)
    source_video_order: list[int] = []
    source_video_seen: set[int] = set()
    for item in source_images:
        video_id = int(item["video_id"])
        source_by_video[video_id].append(int(item["id"]))
        if video_id not in source_video_seen:
            source_video_seen.add(video_id)
            source_video_order.append(video_id)
    annotation_hash = sha256_file(annotation)

    common_provenance: dict[str, Any] | None = None
    ordered_videos: list[int | str] = []
    seen_videos: set[str] = set()
    seen_images: set[int] = set()
    global_video_items: list[dict[str, Any]] = []
    source_shards: list[dict[str, Any]] = []
    category_ids: list[int] | None = None

    for shard_root in args.shard:
        shard_root = shard_root.resolve()
        reader = FrontendReplayCacheReader(shard_root)
        manifest = reader.manifest
        provenance = dict(manifest.get("provenance", {}))
        if provenance.get("source_annotation_sha256") not in (None, annotation_hash):
            raise RuntimeError(f"annotation hash mismatch in cache shard: {shard_root}")
        if common_provenance is None:
            common_provenance = provenance
        else:
            for key in (
                "repo_branch",
                "repo_head",
                "external_cov_commit",
                "external_config_sha256",
                "external_checkpoint_sha256",
            ):
                if provenance.get(key) != common_provenance.get(key):
                    raise RuntimeError(f"provenance mismatch for {key}: {shard_root}")
        shard_category_ids = provenance.get("category_ids")
        if shard_category_ids is not None:
            values = [int(value) for value in shard_category_ids]
            if category_ids is None:
                category_ids = values
            elif category_ids != values:
                raise RuntimeError(f"category ontology mismatch in cache shard: {shard_root}")

        shard_videos = []
        for video_id, video_path, summary in reader.videos():
            video_key = str(video_id)
            if video_key in seen_videos:
                raise RuntimeError(f"duplicate video ownership in cache shards: {video_id}")
            seen_videos.add(video_key)
            if isinstance(video_id, int) and int(video_id) not in source_by_video:
                raise RuntimeError(f"cache video is outside fixed subset: {video_id}")
            records = list(reader.frames(video_path))
            image_ids = [int(record["image_id"]) for record in records]
            expected = source_by_video.get(int(video_id), [])
            if image_ids != expected:
                raise RuntimeError(
                    f"frame order mismatch for video {video_id}: "
                    f"cache={len(image_ids)} expected={len(expected)}"
                )
            if set(image_ids) & seen_images:
                raise RuntimeError(f"duplicate image ownership for video {video_id}")
            seen_images.update(image_ids)
            target = output / "videos" / video_key
            shutil.copytree(video_path, target)
            copied_summary = dict(summary)
            copied_summary["path"] = str(Path("videos") / video_key)
            copied_summary["frames_jsonl_sha256"] = sha256_file(target / "frames.jsonl")
            copied_summary["frame_count"] = len(records)
            copied_summary["image_ids"] = image_ids
            global_video_items.append(copied_summary)
            shard_videos.append(video_id)
        source_shards.append(
            {
                "root": str(shard_root),
                "manifest_sha256": sha256_file(shard_root / "manifest.json"),
                "video_ids": shard_videos,
            }
        )

    if seen_images != set(source_image_ids):
        raise RuntimeError(
            f"cache image union mismatch: expected {len(source_image_ids)}, got {len(seen_images)}"
        )
    if category_ids is None:
        raise RuntimeError("cache shards did not record the dataset category ontology")
    summary_by_video = {str(item["video_id"]): item for item in global_video_items}
    if set(summary_by_video) != {str(value) for value in source_video_order}:
        raise RuntimeError("cache video summaries do not cover the source video order")
    ordered_videos = [int(value) for value in source_video_order]
    global_video_items = [summary_by_video[str(value)] for value in ordered_videos]
    provenance = dict(common_provenance or {})
    provenance.update(
        {
            "source_annotation": str(annotation),
            "source_annotation_sha256": annotation_hash,
            "source_image_count": len(source_images),
            "source_video_count": len(source_by_video),
            "category_ids": category_ids,
            "source_shards": source_shards,
        }
    )
    manifest = {
        "schema_version": REPLAY_CACHE_SCHEMA_VERSION,
        "artifact": REPLAY_CACHE_ARTIFACT,
        "status": "PASS",
        "cache_boundary": "before_pinned_covtrack_tracker_match",
        "cached_fields": [
            "det_bboxes",
            "det_scores",
            "det_labels",
            "track_feats",
            "cls_feats",
            "video_id",
            "frame_id",
            "image_id",
            "filename",
            "method",
            "match_called",
        ],
        "forbidden_cached_state": [
            "final_ids",
            "memo_embeds",
            "memo_ids",
            "native_affinity",
            "track_ids",
            "tracklets",
        ],
        "frame_count": len(source_images),
        "video_count": len(ordered_videos),
        "ordered_video_ids": ordered_videos,
        "ordered_image_ids": source_image_ids,
        "videos": global_video_items,
        "provenance": provenance,
        "cache_size_bytes": _size(output),
    }
    _write(output / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", "frames": len(source_images), "videos": len(ordered_videos), "bytes": manifest["cache_size_bytes"]}))


if __name__ == "__main__":
    main()
