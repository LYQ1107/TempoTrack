#!/usr/bin/env python3
"""Make an experiment-owned complete-video TAO annotation subset."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def make_subset(source: str | Path, output: str | Path, video_id: int, max_frames: int) -> dict:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    data = json.loads(source_path.read_text(encoding="utf-8"))
    selected = sorted(
        [item for item in data.get("images", []) if int(item["video_id"]) == int(video_id)],
        key=lambda item: (int(item.get("frame_id", item.get("frame_index", 0))), int(item["id"])),
    )[: int(max_frames)]
    if len(selected) < int(max_frames):
        raise ValueError(f"video {video_id} has only {len(selected)} frames")
    image_ids = {int(item["id"]) for item in selected}
    subset = copy.deepcopy(data)
    subset["images"] = selected
    subset["annotations"] = [
        item for item in data.get("annotations", []) if int(item.get("image_id", -1)) in image_ids
    ]
    subset["videos"] = [
        item for item in data.get("videos", []) if int(item.get("id", -1)) == int(video_id)
    ]
    subset["tracks"] = [
        item for item in data.get("tracks", [])
        if int(item.get("video_id", item.get("video", -1))) == int(video_id)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    return {
        "status": "PASS",
        "source": str(source_path),
        "output": str(output_path),
        "video_id": int(video_id),
        "frames": len(selected),
        "image_ids": [int(item["id"]) for item in selected],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-id", required=True, type=int)
    parser.add_argument("--max-frames", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(make_subset(args.source, args.output, args.video_id, args.max_frames), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
