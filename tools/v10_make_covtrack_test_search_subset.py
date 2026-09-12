#!/usr/bin/env python3
"""Create one deterministic complete-video COVTrack Test search subset.

The subset is a Test-only development stream for the explicitly
``TEST_TUNED_MODEL_SPECIFIC`` V10.4 search.  It never truncates a video and
never changes detector observations or categories; only whole videos are
selected from the pinned Test annotation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("images"), list):
        raise ValueError(f"TAO annotation has no images list: {path}")
    return value


def _frame_key(image: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(image.get("frame_id", image.get("frame_index", 0))),
        int(image.get("frame_index", image.get("frame_id", 0))),
        int(image["id"]),
    )


def make_subset(
    source: str | Path,
    output: str | Path,
    manifest_output: str | Path,
    *,
    fraction: float = 0.22,
    seed: int = 20260912,
    video_ids: list[int] | None = None,
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    manifest_path = Path(manifest_output).resolve()
    if video_ids is None and not 0.20 <= float(fraction) <= 0.25:
        raise ValueError("V10.4 Test subset fraction must be in [0.20, 0.25]")
    data = _load(source_path)
    categories = {int(item["id"]): item for item in data.get("categories", [])}
    novel_ids = {cid for cid, item in categories.items() if item.get("frequency") == "r"}
    images_by_video: dict[int, list[dict[str, Any]]] = {}
    for image in data["images"]:
        images_by_video.setdefault(int(image["video_id"]), []).append(image)
    for images in images_by_video.values():
        images.sort(key=_frame_key)
    categories_by_image: dict[int, set[int]] = {}
    for annotation in data.get("annotations", []):
        if "image_id" in annotation and "category_id" in annotation:
            categories_by_image.setdefault(int(annotation["image_id"]), set()).add(
                int(annotation["category_id"])
            )

    videos: list[dict[str, Any]] = []
    for video_id, images in sorted(images_by_video.items()):
        image_ids = {int(image["id"]) for image in images}
        observed_categories = set().union(
            *(categories_by_image.get(image_id, set()) for image_id in image_ids)
        ) if image_ids else set()
        videos.append(
            {
                "video_id": int(video_id),
                "images": images,
                "frames": len(images),
                "categories": observed_categories,
                "novel": bool(observed_categories & novel_ids),
            }
        )
    total_frames = sum(int(item["frames"]) for item in videos)
    requested_video_ids = None if video_ids is None else {int(value) for value in video_ids}
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    covered_categories: set[int] = set()
    if requested_video_ids is not None:
        missing = sorted(requested_video_ids - {int(item["video_id"]) for item in videos})
        if missing:
            raise ValueError(f"requested video IDs are absent from Test annotation: {missing}")
        selected = [item for item in videos if int(item["video_id"]) in requested_video_ids]
        selected.sort(key=lambda item: int(item["video_id"]))
        selected_ids = {int(item["video_id"]) for item in selected}
        covered_categories = set().union(*(item["categories"] for item in selected)) if selected else set()
        target_frames = sum(int(item["frames"]) for item in selected)
    else:
        target_frames = int(round(total_frames * float(fraction)))
    if not videos or target_frames < 1:
        raise ValueError("empty Test annotation")

    lengths = sorted(int(item["frames"]) for item in videos)
    short_cut = lengths[max(0, int(round((len(lengths) - 1) * 0.33)))]
    medium_cut = lengths[max(0, int(round((len(lengths) - 1) * 0.66)))]

    def bucket(item: dict[str, Any]) -> str:
        length = int(item["frames"])
        if length <= short_cut:
            return "short"
        if length <= medium_cut:
            return "medium"
        return "long"

    for item in videos:
        item["length_bucket"] = bucket(item)

    rng = random.Random(int(seed))
    tie_break = {int(item["video_id"]): rng.random() for item in videos}
    def add(item: dict[str, Any]) -> None:
        selected.append(item)
        selected_ids.add(int(item["video_id"]))
        covered_categories.update(item["categories"])

    # Keep every video with a visible Novel category.  There are few such
    # videos in TAO Test, and this makes the Test-tuned subset informative
    # about the requested Novel association without using Novel metrics to
    # choose a parameter later.
    if requested_video_ids is None:
        for item in sorted((item for item in videos if item["novel"]), key=lambda x: int(x["video_id"])):
            add(item)

    # Ensure every length bucket is represented before the greedy fill.
    for name in ("short", "medium", "long") if requested_video_ids is None else ():
        candidates = [item for item in videos if item["length_bucket"] == name and int(item["video_id"]) not in selected_ids]
        if candidates:
            add(min(candidates, key=lambda item: (tie_break[int(item["video_id"])], int(item["video_id"]))))

    # The fill is deterministic and prioritizes category coverage while
    # maintaining the source's rough length distribution.  The final video is
    # never truncated, so the achieved fraction is explicitly recorded.
    desired_bucket_frames = {"short": 0.30, "medium": 0.64, "long": 0.06}
    while requested_video_ids is None and sum(int(item["frames"]) for item in selected) < target_frames:
        current_frames = sum(int(item["frames"]) for item in selected)
        current_by_bucket = {name: sum(int(item["frames"]) for item in selected if item["length_bucket"] == name) for name in desired_bucket_frames}

        def rank(item: dict[str, Any]) -> tuple[float, float, float, int]:
            new_categories = len(item["categories"] - covered_categories)
            bucket_share = current_by_bucket[item["length_bucket"]] / max(current_frames, 1)
            bucket_penalty = bucket_share - desired_bucket_frames[item["length_bucket"]]
            novel_bonus = 0.25 if item["novel"] else 0.0
            return (-float(new_categories) - novel_bonus, float(bucket_penalty), tie_break[int(item["video_id"])], int(item["video_id"]))

        remaining = [item for item in videos if int(item["video_id"]) not in selected_ids]
        if not remaining:
            break
        add(min(remaining, key=rank))

    selected.sort(key=lambda item: int(item["video_id"]))
    selected_image_ids = {int(image["id"]) for item in selected for image in item["images"]}
    subset = copy.deepcopy(data)
    subset["images"] = [image for item in selected for image in item["images"]]
    subset["annotations"] = [
        annotation for annotation in data.get("annotations", [])
        if int(annotation.get("image_id", -1)) in selected_image_ids
    ]
    selected_video_ids = {int(item["video_id"]) for item in selected}
    subset["videos"] = [
        video for video in data.get("videos", [])
        if int(video.get("id", -1)) in selected_video_ids
    ]
    subset["tracks"] = [
        track for track in data.get("tracks", [])
        if int(track.get("video_id", track.get("video", -1))) in selected_video_ids
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")

    selected_frames = len(subset["images"])
    selected_categories = set()
    for annotation in subset["annotations"]:
        if "category_id" in annotation:
            selected_categories.add(int(annotation["category_id"]))
    manifest = {
        "status": "PASS",
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "source": str(source_path),
        "source_annotation_sha256": _sha256(source_path),
        "output": str(output_path),
        "annotation_sha256": _sha256(output_path),
        "seed": int(seed),
        "requested_fraction": float(fraction),
        "achieved_fraction": float(selected_frames / max(total_frames, 1)),
        "source_videos": len(videos),
        "source_frames": total_frames,
        "videos": len(selected),
        "frames": selected_frames,
        "video_ids": [int(item["video_id"]) for item in selected],
        "frame_count_by_video": {str(item["video_id"]): int(item["frames"]) for item in selected},
        "length_bucket_cutoffs": {"short_max": int(short_cut), "medium_max": int(medium_cut)},
        "length_buckets": {name: sum(int(item["frames"]) for item in selected if item["length_bucket"] == name) for name in ("short", "medium", "long")},
        "novel_video_count": sum(bool(item["novel"]) for item in selected),
        "novel_category_ids": sorted(int(value) for value in novel_ids),
        "category_ids_observed": sorted(selected_categories),
        "category_coverage_count": len(selected_categories),
        "source_category_count": len(categories),
        "selection_mode": "explicit_video_ids" if requested_video_ids is not None else "stratified_fraction",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fraction", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--video-ids", nargs="+", type=int, help="explicit complete-video IDs for a real smoke subset")
    args = parser.parse_args()
    print(json.dumps(make_subset(args.source, args.output, args.manifest, fraction=args.fraction, seed=args.seed, video_ids=args.video_ids), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
