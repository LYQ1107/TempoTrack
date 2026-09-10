#!/usr/bin/env python3
"""Build the deterministic, GT-independent 128-video active-search subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-count", type=int, default=128)
    args = parser.parse_args()
    source = Path(args.annotation).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    videos = list(payload.get("videos", []))
    if not videos:
        raise ValueError(f"annotation has no videos: {source}")
    selected = sorted(
        videos,
        key=lambda item: hashlib.sha256(str(int(item["id"])).encode("utf-8")).hexdigest(),
    )[: int(args.video_count)]
    selected_ids = {int(item["id"]) for item in selected}
    images = [item for item in payload.get("images", []) if int(item.get("video_id", -1)) in selected_ids]
    image_ids = {int(item["id"]) for item in images}
    annotations = [item for item in payload.get("annotations", []) if int(item.get("image_id", -1)) in image_ids]
    result = dict(payload)
    result["videos"] = selected
    result["images"] = images
    result["annotations"] = annotations
    result["tempotrack_v9_subset"] = {
        "source_annotation": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection": "sorted sha256(str(video_id)) first N",
        "video_count": int(len(selected)),
        "video_ids_sha256": hashlib.sha256(",".join(str(int(item["id"])) for item in selected).encode("utf-8")).hexdigest(),
        "image_count": int(len(images)),
        "annotation_count": int(len(annotations)),
        "gt_used_for_selection": False,
    }
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(result["tempotrack_v9_subset"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
