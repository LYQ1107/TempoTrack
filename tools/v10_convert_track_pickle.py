#!/usr/bin/env python3
"""Convert one pinned OVTrack result pickle to TAO JSON with bounded output memory.

The official TAO formatter keeps every result in memory and applies a global
majority vote.  This utility retains the already-produced pickle (the model is
never run here), but emits one complete video at a time and applies the same
per-video local-ID offset and per-track majority vote.  It is intended for
complete-video shards only, so the annotation order is checked before any
prediction is written.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xyxy_to_xywh(values: np.ndarray) -> list[float]:
    return [
        float(values[0]),
        float(values[1]),
        float(values[2] - values[0]),
        float(values[3] - values[1]),
    ]


def frame_rows(result: Any, image: dict[str, Any], category_ids: list[int]) -> list[dict[str, Any]]:
    if not isinstance(result, (list, tuple)):
        raise ValueError("track result for one image must be a list/tuple")
    use_cat_ids = len(result) == len(category_ids)
    rows: list[dict[str, Any]] = []
    for label, bboxes in enumerate(result):
        array = np.asarray(bboxes)
        if array.ndim != 2 or (array.shape[1] if array.ndim == 2 else 0) < 6:
            raise ValueError(f"unexpected track bbox shape for image {image['id']}: {array.shape}")
        category_id = int(category_ids[label]) if use_cat_ids else int(label + 1)
        for bbox in array:
            rows.append(
                {
                    "image_id": int(image["id"]),
                    "bbox": xyxy_to_xywh(np.asarray(bbox[1:5], dtype=np.float64)),
                    "score": float(bbox[-1]),
                    "category_id": category_id,
                    "video_id": int(image["video_id"]),
                    "local_track_id": int(bbox[0]),
                }
            )
    return rows


def class_category_ids(annotation: dict[str, Any], source: Path) -> list[int]:
    """Resolve the pin's dataset ``cat_ids`` from its literal class tuple."""

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    class_names: list[str] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "LVIS_CLASSES"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"LVIS_CLASSES in {source} is not a literal string sequence")
        class_names = list(value)
        break
    if class_names is None:
        raise ValueError(f"LVIS_CLASSES not found in audited source: {source}")
    by_name = {str(category["name"]): int(category["id"]) for category in annotation["categories"]}
    return [by_name[name] for name in class_names if name in by_name]


def write_video(
    frames: list[list[dict[str, Any]]],
    output: Any,
    max_track_id: int,
) -> int:
    if not frames:
        return max_track_id
    local_ids = [
        int(row["local_track_id"])
        for frame in frames
        for row in frame
        if int(row["local_track_id"]) >= 0
    ]
    offset = int(max_track_id)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        for row in frame:
            local_id = int(row.pop("local_track_id"))
            if local_id < 0:
                continue
            row["track_id"] = offset + local_id
            grouped[offset + local_id].append(row)

    for track_id in sorted(grouped):
        records = grouped[track_id]
        counts: dict[int, int] = defaultdict(int)
        for record in records:
            counts[int(record["category_id"])] += 1
        majority = min(
            category_id
            for category_id, count in counts.items()
            if count == max(counts.values())
        )
        for record in records:
            record["category_id"] = majority
            json.dump(record, output, separators=(",", ":"))
            output.write(",\n")

    if local_ids:
        max_track_id += max(local_ids) + 1
    return max_track_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--classes-source",
        type=Path,
        help="Audited source containing the exact LVIS_CLASSES tuple used by the runner",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    images = annotation.get("images")
    categories = annotation.get("categories")
    if not isinstance(images, list) or not isinstance(categories, list):
        raise ValueError("annotation must contain images and categories lists")
    category_ids = [int(category["id"]) for category in categories]
    classes_source = None if args.classes_source is None else args.classes_source.resolve()
    if classes_source is not None:
        if not classes_source.is_file():
            raise FileNotFoundError(classes_source)
        category_ids = class_category_ids(annotation, classes_source)

    with args.prediction.open("rb") as stream:
        prediction = pickle.load(stream, encoding="latin1")
    if not isinstance(prediction, dict) or "track_results" not in prediction:
        raise ValueError("prediction must be an OVTrack result dict with track_results")
    track_results = prediction["track_results"]
    if not isinstance(track_results, list) or len(track_results) != len(images):
        raise ValueError(
            f"prediction/image count mismatch: {len(track_results) if isinstance(track_results, list) else 'non-list'} != {len(images)}"
        )
    # Detection results are not needed for TAO tracking JSON and can otherwise
    # keep a second large object live during conversion.
    prediction.pop("bbox_results", None)
    del prediction

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    max_track_id = 0
    current_video: int | None = None
    frames: list[list[dict[str, Any]]] = []
    with args.output.open("w+", encoding="utf-8") as output:
        output.write("[\n")
        for index, (image, result) in enumerate(zip(images, track_results)):
            video_id = int(image["video_id"])
            rows = frame_rows(result, image, category_ids)
            if current_video is not None and video_id != current_video:
                before = output.tell()
                max_track_id = write_video(frames, output, max_track_id)
                rows_written += sum(len(frame) for frame in frames)
                frames = []
            current_video = video_id
            frames.append(rows)
            if index and index % 500 == 0:
                output.flush()
        if frames:
            max_track_id = write_video(frames, output, max_track_id)
            rows_written += sum(len(frame) for frame in frames)
        output.flush()
        output.seek(0, 2)
        end = output.tell()
        if end >= 2:
            output.seek(end - 2)
            if output.read(2) == ",\n":
                output.seek(end - 2)
                output.truncate()
        output.write("\n]\n")

    if args.manifest:
        manifest = {
            "status": "PASS",
            "annotation": str(args.annotation.resolve()),
            "annotation_sha256": sha256_file(args.annotation),
            "prediction": str(args.prediction.resolve()),
            "prediction_sha256": sha256_file(args.prediction),
            "output": str(args.output.resolve()),
            "output_sha256": sha256_file(args.output),
            "images": len(images),
            "rows": rows_written,
            "category_count": len(category_ids),
            "classes_source": None if classes_source is None else str(classes_source),
            "semantics": "official_TAO_track2json_video_offset_plus_majority_vote",
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "images": len(images), "rows": rows_written}))


if __name__ == "__main__":
    main()
