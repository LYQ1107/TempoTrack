"""Join replay-time candidate ranks with GT only after prediction is complete."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _iou_xyxy(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _xywh_to_xyxy(box: Iterable[float]) -> list[float]:
    value = [float(item) for item in box]
    return [value[0], value[1], value[0] + value[2], value[1] + value[3]]


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _summary(records: list[dict[str, Any]], *, total_legal: int) -> dict[str, Any]:
    if not records:
        return {
            "events_with_gt": 0,
            "events_with_positive_candidate": 0,
            "top1_recall": None,
            "top2_recall": None,
            "mrr": None,
            "mean_gt_rank": None,
            "median_gt_rank": None,
            "mean_positive_best_negative_margin": None,
            "legal_event_count": int(total_legal),
        }
    ranks = [int(item["rank"]) for item in records]
    margins = [float(item["margin"]) for item in records]
    return {
        "events_with_gt": len(records),
        "events_with_positive_candidate": len(records),
        "top1_recall": float(sum(rank <= 1 for rank in ranks) / len(ranks)),
        "top2_recall": float(sum(rank <= 2 for rank in ranks) / len(ranks)),
        "mrr": float(sum(1.0 / rank for rank in ranks) / len(ranks)),
        "mean_gt_rank": _mean([float(rank) for rank in ranks]),
        "median_gt_rank": float(sorted(ranks)[len(ranks) // 2]) if ranks else None,
        "mean_positive_best_negative_margin": _mean(margins),
        "legal_event_count": int(total_legal),
    }


def _load_prediction_manifest(prediction: Path) -> dict[str, Any]:
    path = prediction.parent / f"{prediction.stem}.manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("q1", "qdic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotation = json.loads(args.annotation.resolve().read_text(encoding="utf-8"))
    cache_manifest = json.loads(args.cache_manifest.resolve().read_text(encoding="utf-8"))
    prediction = json.loads(args.prediction.resolve().read_text(encoding="utf-8"))
    if not isinstance(prediction, list):
        raise ValueError("prediction must be a JSON list")
    category_ids = [int(value) for value in cache_manifest["provenance"]["category_ids"]]
    image_order = {
        str(value): index for index, value in enumerate(cache_manifest["ordered_image_ids"])
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    category_by_id = {
        int(item["id"]): item for item in annotation.get("categories", ())
    }
    for item in annotation.get("annotations", ()):
        annotations_by_image[int(item["image_id"])].append(item)
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction:
        predictions_by_image[int(row["image_id"])].append(row)
    prediction_manifest = _load_prediction_manifest(args.prediction.resolve())
    offsets = {
        str(key): int(value)
        for key, value in prediction_manifest.get("video_track_offsets", {}).items()
    }

    def gt_match(image_id: int, category_id: int, box_xyxy: list[float]) -> int | None:
        matches = []
        for ann in annotations_by_image.get(int(image_id), ()):
            if int(ann.get("category_id", -1)) != int(category_id):
                continue
            matches.append((_iou_xyxy(box_xyxy, _xywh_to_xyxy(ann["bbox"])), int(ann.get("track_id", -1))))
        if not matches:
            return None
        matches.sort(reverse=True)
        return matches[0][1] if matches[0][0] >= 0.5 and matches[0][1] >= 0 else None

    # ``track_to_gt`` is updated only with prediction rows from strictly prior
    # images, so the join cannot leak the current/future GT identity into the
    # ranking event.
    track_to_gt: dict[tuple[str, int], set[int]] = defaultdict(set)
    source_image_ids = [int(value) for value in cache_manifest["ordered_image_ids"]]
    next_image_index = 0

    def update_prior_images(until_index: int) -> None:
        nonlocal next_image_index
        while next_image_index < until_index:
            image_id = source_image_ids[next_image_index]
            for row in predictions_by_image.get(image_id, ()):
                category_id = int(row["category_id"])
                gt_id = gt_match(
                    image_id,
                    category_id,
                    _xywh_to_xyxy(row["bbox"]),
                )
                if gt_id is not None:
                    track_to_gt[(str(row["video_id"]), int(row["track_id"]))].add(gt_id)
            next_image_index += 1

    records_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gap_buckets = ((0, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 240), (240, 361))
    gap_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    legal_events = 0
    events_seen = 0
    with args.events.resolve().open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            events_seen += 1
            image_id = event.get("image_id")
            if image_id is None:
                raise RuntimeError("replay event lacks image_id")
            order = image_order.get(str(image_id))
            if order is None:
                raise RuntimeError(f"replay event image is outside cache: {image_id}")
            update_prior_images(order)
            memory_ids = [int(value) for value in event.get("candidate_memory_ids", ())]
            if not memory_ids:
                continue
            legal_events += 1
            category_index = int(event["observation_label"])
            if category_index < 0 or category_index >= len(category_ids):
                continue
            category_id = category_ids[category_index]
            gt_id = gt_match(
                int(image_id),
                category_id,
                [float(value) for value in event["observation_box_xyxy"]],
            )
            if gt_id is None:
                continue
            logits_key = "q1_logits" if args.mode == "q1" else "qdic_logits"
            logits = [None if value is None else float(value) for value in event.get(logits_key, ())]
            if len(logits) != len(memory_ids):
                continue
            valid = [(index, value) for index, value in enumerate(logits) if value is not None]
            if not valid:
                continue
            scored = sorted(valid, key=lambda item: (-item[1], memory_ids[item[0]]))
            ranks = {index: rank for rank, (index, _value) in enumerate(scored, 1)}
            offset = offsets.get(str(event["video_id"]))
            if offset is None:
                raise RuntimeError(f"prediction manifest lacks video offset: {event['video_id']}")
            positive = []
            for index, memory_id in enumerate(memory_ids):
                global_id = int(offset) + int(memory_id)
                if gt_id in track_to_gt.get((str(event["video_id"]), global_id), set()):
                    positive.append(index)
            if not positive:
                continue
            positive_index = min(positive, key=lambda index: ranks.get(index, 10**9))
            if positive_index not in ranks:
                continue
            rank = int(ranks[positive_index])
            positive_score = float(logits[positive_index])
            best_negative = max(
                (value for index, value in valid if index not in positive),
                default=float("-inf"),
            )
            margin = positive_score - best_negative if np.isfinite(best_negative) else positive_score
            record = {"rank": rank, "margin": float(margin)}
            category_group = "novel" if category_by_id.get(category_id, {}).get("frequency") == "r" else "base"
            records_by_group[category_group].append(record)
            for lower, upper in gap_buckets:
                gap = int(event.get("candidate_gaps", [0])[positive_index])
                if lower <= gap < upper:
                    gap_records[f"{lower}-{upper - 1}"].append(record)
                    break

    result = {
        "status": "PASS",
        "mode": args.mode,
        "events_seen": events_seen,
        "legal_event_count": legal_events,
        "category_groups": {
            group: _summary(records, total_legal=legal_events)
            for group, records in sorted(records_by_group.items())
        },
        "overall": _summary(
            [record for records in records_by_group.values() for record in records],
            total_legal=legal_events,
        ),
        "gap_buckets": {
            key: _summary(value, total_legal=legal_events)
            for key, value in sorted(gap_records.items())
        },
        "inputs": {
            "events": str(args.events.resolve()),
            "prediction": str(args.prediction.resolve()),
            "annotation": str(args.annotation.resolve()),
            "cache_manifest": str(args.cache_manifest.resolve()),
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
