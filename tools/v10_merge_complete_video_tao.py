#!/usr/bin/env python3
"""Merge complete-video TAO JSON shards without loading them into RAM.

Each input shard is formatted independently by the official TAO formatter, so
its local track IDs start at zero.  This utility validates that shard
annotations are an ordered, non-overlapping partition of the full annotation,
then applies one deterministic shard offset to ``track_id`` only.  Detection
boxes, scores, categories, image IDs, and row order are otherwise preserved.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
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


def image_keys(annotation: dict[str, Any]) -> list[tuple[int, int, int]]:
    return [
        (int(info["id"]), int(info["video_id"]), int(info["frame_id"]))
        for info in annotation["images"]
    ]


def iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Yield objects from a flat JSON array using bounded input memory."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        started = False
        finished = False
        while not finished:
            block = handle.read(1024 * 1024)
            if block:
                buffer += block
            elif not buffer:
                break
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    buffer = buffer[1:]
                    started = True
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == "]":
                    finished = True
                    buffer = buffer[1:]
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    buffer = buffer.lstrip()
                    if not buffer:
                        break
                try:
                    value, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object JSON item")
                yield value
                buffer = buffer[end:]
            if not block:
                if not finished:
                    raise ValueError(f"truncated JSON array: {path}")
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument(
        "--shard", action="append", nargs=2, metavar=("ANNOTATION", "PREDICTION"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    full_keys = image_keys(full_annotation)
    key_positions = {key: index for index, key in enumerate(full_keys)}
    if len(key_positions) != len(full_keys):
        raise ValueError("full annotation contains duplicate image keys")
    full_video_positions: dict[int, list[int]] = {}
    for index, (_, video_id, _) in enumerate(full_keys):
        full_video_positions.setdefault(video_id, []).append(index)

    shards: list[dict[str, Any]] = []
    for raw_annotation, raw_prediction in args.shard:
        annotation_path = Path(raw_annotation).resolve()
        prediction_path = Path(raw_prediction).resolve()
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        keys = image_keys(annotation)
        if not keys:
            raise ValueError(f"empty shard annotation: {annotation_path}")
        if len({key for key in keys}) != len(keys):
            raise ValueError(f"shard annotation contains duplicate image keys: {annotation_path}")
        shard_positions = [key_positions.get(key) for key in keys]
        if any(position is None for position in shard_positions):
            raise ValueError(f"shard contains an image absent from full annotation: {annotation_path}")
        shard_video_ids = {key[1] for key in keys}
        for video_id in shard_video_ids:
            positions = [
                key_positions[key]
                for key in keys
                if key[1] == video_id
            ]
            if positions != full_video_positions.get(video_id):
                raise ValueError(
                    "shard splits or reorders a complete video: "
                    f"{annotation_path} video={video_id}"
                )
        start = min(int(position) for position in shard_positions)
        shards.append(
            {
                "annotation": annotation_path,
                "prediction": prediction_path,
                "keys": keys,
                "start": start,
                "image_ids": {key[0] for key in keys},
            }
        )

    shards.sort(key=lambda item: int(item["start"]))
    covered: set[tuple[int, int, int]] = set()
    for shard in shards:
        shard_keys = set(shard["keys"])
        if covered.intersection(shard_keys):
            raise ValueError("shard annotations overlap")
        covered.update(shard_keys)
    if covered != set(full_keys):
        raise ValueError(
            "shard coverage is incomplete: "
            f"{len(covered)} != {len(full_keys)} images"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shard_receipts: list[dict[str, Any]] = []
    global_offset = 0
    rows_written = 0
    with args.output.open("w+", encoding="utf-8") as output:
        output.write("[\n")
        first = True
        for shard in shards:
            prediction_path = Path(shard["prediction"])
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            image_ids = shard["image_ids"]
            max_local_id = -1
            shard_rows = 0
            for row in iter_json_array(prediction_path):
                image_id = int(row["image_id"])
                if image_id not in image_ids:
                    raise ValueError(
                        f"prediction row image_id {image_id} is outside shard: {prediction_path}"
                    )
                if "track_id" not in row:
                    raise ValueError(f"prediction row lacks track_id: {prediction_path}")
                max_local_id = max(max_local_id, int(row["track_id"]))
                shard_rows += 1

            for row in iter_json_array(prediction_path):
                row["track_id"] = int(row["track_id"]) + global_offset
                if not first:
                    output.write(",\n")
                json.dump(row, output, separators=(",", ":"))
                first = False
                rows_written += 1
            if max_local_id >= 0:
                global_offset += max_local_id + 1
            shard_receipts.append(
                {
                    "annotation": str(shard["annotation"]),
                    "annotation_sha256": sha256_file(Path(shard["annotation"])),
                    "prediction": str(prediction_path),
                    "prediction_sha256": sha256_file(prediction_path),
                    "images": len(shard["keys"]),
                    "rows": shard_rows,
                    "track_offset_after": global_offset,
                }
            )
        output.write("\n]\n")

    manifest = {
        "status": "PASS",
        "semantics": "complete_video_shard_offset_track_id_only",
        "annotation": str(args.annotation.resolve()),
        "annotation_sha256": sha256_file(args.annotation.resolve()),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "images": len(full_keys),
        "rows": rows_written,
        "shards": shard_receipts,
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "images": len(full_keys), "rows": rows_written}))


if __name__ == "__main__":
    main()
