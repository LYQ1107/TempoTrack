#!/usr/bin/env python3
"""Post-hoc diagnostics for the V11 structural Full-Test trials.

The detector/tracker never sees the annotation.  This tool joins the
experiment-owned replay event JSONL with the completed prediction parts and
the annotation only after inference has finished.  It reports candidate
supply, QDIC rank, accepted association, and temporal-gap statistics without
changing the official prediction or TETA evaluation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


RANK_POINTS = (1, 8, 16, 32, 64)
GAP_BINS = (
    ("gap_le_180", 0, 180),
    ("gap_181_360", 181, 360),
    ("gap_361_720", 361, 720),
    ("gap_gt_720", 721, None),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _xywh_to_xyxy(box: Iterable[Any]) -> tuple[float, float, float, float]:
    values = [float(value) for value in box]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid xywh box: {box!r}")
    x, y, width, height = values
    return x, y, x + max(width, 0.0), y + max(height, 0.0)


def _xyxy(value: Iterable[Any]) -> tuple[float, float, float, float]:
    values = [float(item) for item in value]
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"invalid xyxy box: {value!r}")
    return values[0], values[1], values[2], values[3]


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def _annotation(path: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]], dict[int, str], dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, Mapping):
        raise ValueError("annotation root must be a mapping")
    categories = [dict(item) for item in data.get("categories", [])]
    category_ids = sorted(int(item["id"]) for item in categories)
    category_frequency = {
        int(item["id"]): str(item.get("frequency", "f")) for item in categories
    }
    images_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    image_by_id: dict[int, dict[str, Any]] = {}
    for raw in data.get("images", []):
        image = dict(raw)
        image_id = int(image["id"])
        video_id = int(image["video_id"])
        image["id"] = image_id
        image["video_id"] = video_id
        image["frame_id"] = int(image.get("frame_id", image.get("frame_index", 0)))
        images_by_video[video_id].append(image)
        image_by_id[image_id] = image
    for items in images_by_video.values():
        items.sort(key=lambda item: (int(item["frame_id"]), int(item["id"])))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in data.get("annotations", []):
        item = dict(raw)
        image_id = int(item["image_id"])
        if image_id not in image_by_id:
            continue
        item["category_id"] = int(item["category_id"])
        item["track_id"] = int(item.get("track_id", item.get("instance_id", item["id"])))
        item["bbox_xyxy"] = _xywh_to_xyxy(item["bbox"])
        annotations_by_image[image_id].append(item)
    return (
        dict(images_by_video),
        dict(annotations_by_image),
        category_frequency,
        {
            "path": str(path),
            "sha256": _sha256(path),
            "category_count": len(category_ids),
            "category_ids_sorted": category_ids,
            "category_mapping": "sorted_annotation_category_ids[label]",
            "category_frequency_counts": {
                "base": sum(value != "r" for value in category_frequency.values()),
                "novel": sum(value == "r" for value in category_frequency.values()),
            },
        },
    )


def _load_parts(shard_dir: Path) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], dict[str, Any]]:
    """Load the pre-offset transport parts so event IDs remain local IDs."""
    parts = sorted((shard_dir / "stream" / "parts").glob("part_*.jsonl"))
    fallback = False
    if not parts:
        fallback_path = shard_dir / "stream" / "tao_track.json"
        if not fallback_path.is_file():
            return {}, {"status": "MISSING", "path": str(fallback_path), "rows": 0}
        fallback = True
        raw = _read_json(fallback_path)
        frames: list[dict[str, Any]] = []
        # This fallback is only for diagnostics of legacy artifacts.  The
        # structural run always writes parts with local_track_id.
        by_image: dict[int, dict[str, Any]] = {}
        for row in raw if isinstance(raw, list) else []:
            image_id = int(row["image_id"])
            frame = by_image.setdefault(
                image_id,
                {"image_id": image_id, "video_id": int(row["video_id"]), "rows": []},
            )
            frame["rows"].append(row)
        frames = list(by_image.values())
    else:
        frames = []
        for path in parts:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        frames.append(json.loads(line))
    result: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    rows = 0
    for frame in frames:
        video_id = int(frame["video_id"])
        image_id = int(frame["image_id"])
        for raw in frame.get("rows", []):
            if "local_track_id" in raw:
                local_id = int(raw["local_track_id"])
            else:
                local_id = int(raw.get("track_id", -1))
            try:
                box = _xywh_to_xyxy(raw["bbox"])
            except (KeyError, TypeError, ValueError):
                continue
            result[(video_id, image_id)].append(
                {
                    "local_track_id": local_id,
                    "category_id": int(raw["category_id"]),
                    "bbox_xyxy": box,
                    "score": float(raw.get("score", 0.0)),
                }
            )
            rows += 1
    return dict(result), {
        "status": "PASS",
        "paths": [str(path) for path in parts] if parts else [str(shard_dir / "stream" / "tao_track.json")],
        "rows": rows,
        "legacy_global_track_id_fallback": fallback,
    }


def _load_events(shard_dir: Path, receipt: Mapping[str, Any]) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], dict[str, Any]]:
    outputs = receipt.get("outputs", {})
    raw_path = outputs.get("event_diagnostics") if isinstance(outputs, Mapping) else None
    path = Path(str(raw_path)) if raw_path else shard_dir / "event_diagnostics.jsonl"
    if not path.is_file():
        return {}, {"status": "MISSING", "path": str(path), "rows": 0}
    result: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    rows = 0
    invalid = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                video_id = int(event["video_id"])
                image_id = event.get("image_id")
                if image_id is None:
                    invalid += 1
                    continue
                key = (video_id, int(image_id))
                result[key].append(event)
                rows += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid += 1
    return dict(result), {
        "status": "PASS" if invalid == 0 else "PASS_WITH_INVALID_ROWS",
        "path": str(path),
        "sha256": _sha256(path),
        "rows": rows,
        "invalid_rows": invalid,
    }


def _new_counts() -> dict[str, Any]:
    return {
        "events": 0,
        "events_with_gt": 0,
        "events_without_gt_match": 0,
        "events_without_prior_identity": 0,
        "association_events": 0,
        "positive_events": 0,
        "accepted_total": 0,
        "accepted_correct": 0,
        "false_merge": 0,
        "accepted_unresolved": 0,
        "rejected_true_association": 0,
        "prefilter_rank_values": [],
        "qdic_rank_values": [],
        "prefilter_recall_at": {str(point): 0 for point in RANK_POINTS},
        "qdic_recall_at": {str(point): 0 for point in RANK_POINTS},
    }


def _quantiles(values: list[int | float]) -> dict[str, float] | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    def percentile(percent: float) -> float:
        if len(finite) == 1:
            return finite[0]
        index = (len(finite) - 1) * percent / 100.0
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return finite[lower]
        fraction = index - lower
        return finite[lower] * (1.0 - fraction) + finite[upper] * fraction
    return {name: percentile(point) for name, point in (("p50", 50), ("p95", 95))}


def _finalize_counts(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    denominator = int(result["association_events"])
    accepted = int(result["accepted_total"])
    result["candidate_recall"] = (
        float(result["positive_events"]) / denominator if denominator else None
    )
    result["association_recall"] = (
        float(result["accepted_correct"]) / denominator if denominator else None
    )
    result["association_precision"] = (
        float(result["accepted_correct"]) / accepted if accepted else None
    )
    result["false_merge_rate"] = (
        float(result["false_merge"]) / accepted if accepted else None
    )
    result["prefilter_rank_quantiles"] = _quantiles(result.pop("prefilter_rank_values"))
    result["qdic_rank_quantiles"] = _quantiles(result.pop("qdic_rank_values"))
    return result


def _add_event_to_counts(
    counts: dict[str, Any],
    *,
    candidate_ids: list[int],
    prefilter_ranks: list[Any],
    qdic_ranks: list[Any],
    positive_indices: list[int],
    accepted: bool,
    assignment: int | None,
    track_to_gt: Mapping[tuple[int, int], set[tuple[int, int]]],
    video_id: int,
    target_gt: tuple[int, int],
) -> None:
    counts["association_events"] += 1
    if positive_indices:
        counts["positive_events"] += 1
        best_prefilter = min(
            (int(prefilter_ranks[index]) for index in positive_indices if index < len(prefilter_ranks) and prefilter_ranks[index] is not None),
            default=None,
        )
        best_qdic = min(
            (int(qdic_ranks[index]) for index in positive_indices if index < len(qdic_ranks) and qdic_ranks[index] is not None),
            default=None,
        )
        if best_prefilter is not None:
            counts["prefilter_rank_values"].append(best_prefilter)
            for point in RANK_POINTS:
                if best_prefilter <= point:
                    counts["prefilter_recall_at"][str(point)] += 1
        if best_qdic is not None:
            counts["qdic_rank_values"].append(best_qdic)
            for point in RANK_POINTS:
                if best_qdic <= point:
                    counts["qdic_recall_at"][str(point)] += 1
    if not accepted:
        if positive_indices:
            counts["rejected_true_association"] += 1
        return
    counts["accepted_total"] += 1
    assigned = None if assignment is None else track_to_gt.get((video_id, int(assignment)), set())
    if target_gt in (assigned or set()):
        counts["accepted_correct"] += 1
    elif assigned:
        counts["false_merge"] += 1
    else:
        counts["accepted_unresolved"] += 1


def _gap_bucket(gap: int) -> str | None:
    for name, low, high in GAP_BINS:
        if gap >= low and (high is None or gap <= high):
            return name
    return None


def diagnose(*, annotation_path: Path, candidate_root: Path, output: Path) -> dict[str, Any]:
    images_by_video, annotations_by_image, category_frequency, annotation_meta = _annotation(annotation_path)
    receipt_path = candidate_root / "receipt.json"
    receipt = _read_json(receipt_path) if receipt_path.is_file() else {}
    shard_records = receipt.get("shards", []) if isinstance(receipt, Mapping) else []
    if not isinstance(shard_records, list) or not shard_records:
        raise RuntimeError(f"candidate receipt has no completed shards: {receipt_path}")

    events_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    predictions_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    shard_meta: list[dict[str, Any]] = []
    for record in shard_records:
        shard_dir = Path(str(record["directory"])).resolve()
        shard_receipt_path = Path(str(record.get("receipt", shard_dir / "receipt.json")))
        shard_receipt = _read_json(shard_receipt_path) if shard_receipt_path.is_file() else {}
        shard_predictions, pred_meta = _load_parts(shard_dir)
        shard_events, event_meta = _load_events(shard_dir, shard_receipt)
        for key, rows in shard_predictions.items():
            predictions_by_key[key].extend(rows)
        for key, rows in shard_events.items():
            events_by_key[key].extend(rows)
        shard_meta.append(
            {
                "index": int(record.get("index", -1)),
                "directory": str(shard_dir),
                "prediction": pred_meta,
                "events": event_meta,
                "runtime_diagnostics": str(shard_dir / "diagnostics.json"),
            }
        )

    groups = {name: _new_counts() for name in ("overall", "base", "novel")}
    gaps = {name: _new_counts() for name, _low, _high in GAP_BINS}
    track_to_gt: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    last_gt_frame: dict[tuple[int, int], int] = {}
    event_keys = set(events_by_key)
    video_ids = sorted({video_id for video_id, _image_id in event_keys})
    event_source_fallback = 0
    unresolved_event_images = 0
    for video_id in video_ids:
        image_list = images_by_video.get(video_id, [])
        image_by_id = {int(item["id"]): item for item in image_list}
        for image in image_list:
            image_id = int(image["id"])
            key = (video_id, image_id)
            current_events = events_by_key.get(key, [])
            gt_rows = annotations_by_image.get(image_id, [])
            for event in current_events:
                # A missing event is expected for native branches that return
                # before the association hook; no synthetic denominator is
                # created here.
                groups["overall"]["events"] += 1
                category_id = event.get("observation_category_id")
                if category_id is None:
                    label = int(event.get("observation_label", -1))
                    category_ids = annotation_meta["category_ids_sorted"]
                    if 0 <= label < len(category_ids):
                        category_id = category_ids[label]
                        event_source_fallback += 1
                    else:
                        category_id = None
                matched_gt = None
                if category_id is not None:
                    try:
                        observation_box = _xyxy(event["observation_box_xyxy"])
                    except (KeyError, TypeError, ValueError):
                        observation_box = None
                    if observation_box is not None:
                        possible = [
                            row for row in gt_rows if int(row["category_id"]) == int(category_id)
                        ]
                        matched_gt = max(
                            possible,
                            key=lambda row: _iou(observation_box, row["bbox_xyxy"]),
                            default=None,
                        )
                        if matched_gt is not None and _iou(observation_box, matched_gt["bbox_xyxy"]) < 0.5:
                            matched_gt = None
                if matched_gt is None:
                    groups["overall"]["events_without_gt_match"] += 1
                    continue
                gt_key = (video_id, int(matched_gt["track_id"]))
                split = "novel" if category_frequency.get(int(matched_gt["category_id"]), "f") == "r" else "base"
                groups[split]["events"] += 1
                groups["overall"]["events_with_gt"] += 1
                groups[split]["events_with_gt"] += 1
                prior_frame = last_gt_frame.get(gt_key)
                if prior_frame is None:
                    groups["overall"]["events_without_prior_identity"] += 1
                    groups[split]["events_without_prior_identity"] += 1
                    continue
                groups["overall"]["association_events"] += 0
                candidate_ids = [int(value) for value in event.get("candidate_memory_ids", [])]
                prefilter_ranks = list(event.get("candidate_prefilter_ranks", []))
                qdic_ranks = list(event.get("candidate_logit_ranks", []))
                positive_indices = [
                    index
                    for index, candidate_id in enumerate(candidate_ids)
                    if gt_key in track_to_gt.get((video_id, candidate_id), set())
                ]
                accepted = bool(event.get("proposal_accepted", False))
                assignment_raw = event.get("proposal_assignment")
                assignment = None if assignment_raw is None else int(assignment_raw)
                for target_counts in (groups["overall"], groups[split]):
                    _add_event_to_counts(
                        target_counts,
                        candidate_ids=candidate_ids,
                        prefilter_ranks=prefilter_ranks,
                        qdic_ranks=qdic_ranks,
                        positive_indices=positive_indices,
                        accepted=accepted,
                        assignment=assignment,
                        track_to_gt=track_to_gt,
                        video_id=video_id,
                        target_gt=gt_key,
                    )
                gap = int(image["frame_id"]) - int(prior_frame)
                bucket = _gap_bucket(gap)
                if bucket is not None:
                    gaps[bucket]["events"] += 1
                    gaps[bucket]["events_with_gt"] += 1
                    _add_event_to_counts(
                        gaps[bucket],
                        candidate_ids=candidate_ids,
                        prefilter_ranks=prefilter_ranks,
                        qdic_ranks=qdic_ranks,
                        positive_indices=positive_indices,
                        accepted=accepted,
                        assignment=assignment,
                        track_to_gt=track_to_gt,
                        video_id=video_id,
                        target_gt=gt_key,
                    )
            # Only after every event on this image has been scored do current
            # predictions become available to later events.
            for prediction in predictions_by_key.get(key, []):
                possible = [
                    row for row in gt_rows
                    if int(row["category_id"]) == int(prediction["category_id"])
                ]
                match = max(
                    possible,
                    key=lambda row: _iou(prediction["bbox_xyxy"], row["bbox_xyxy"]),
                    default=None,
                )
                if match is None or _iou(prediction["bbox_xyxy"], match["bbox_xyxy"]) < 0.5:
                    continue
                target = (video_id, int(match["track_id"]))
                local_key = (video_id, int(prediction["local_track_id"]))
                track_to_gt[local_key].add(target)
                last_gt_frame[target] = int(image["frame_id"])
        # Events with an image id not in the complete shard are not silently
        # treated as valid; expose them as an integrity count.
        unresolved_event_images += sum(
            1 for event_key in events_by_key
            if event_key[0] == video_id and event_key[1] not in image_by_id
        )

    for value in groups.values():
        value["events_without_prior_identity"] = int(value["events_without_prior_identity"])
    finalized_groups = {name: _finalize_counts(value) for name, value in groups.items()}
    finalized_gaps = {name: _finalize_counts(value) for name, value in gaps.items()}
    output_value: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "v11_qdic_structure_posthoc_diagnostics",
        "status": "PASS" if event_keys else "UNAVAILABLE_NO_EVENT_ROWS",
        "inference_gt_used": False,
        "diagnostic_join": "posthoc_event_jsonl_prediction_parts_annotation",
        "annotation": annotation_meta,
        "candidate_root": str(candidate_root),
        "candidate_spec": receipt.get("spec", {}),
        "event_source": {
            "rows": sum(len(rows) for rows in events_by_key.values()),
            "keys": len(events_by_key),
            "category_id_fallback_rows": event_source_fallback,
            "unresolved_event_images": unresolved_event_images,
            "category_id_source": "runtime_dataset_cat_ids" if event_source_fallback == 0 else "runtime_dataset_cat_ids_with_sorted_annotation_fallback",
        },
        "groups": finalized_groups,
        "temporal_gap_bins": finalized_gaps,
        "shards": shard_meta,
        "definitions": {
            "association_events": "valid IoU>=0.5 same-category observation with the GT identity matched by a strictly prior prediction",
            "positive_events": "at least one causal candidate memory ID already mapped posthoc to that GT identity",
            "prefilter_rank": "candidate_prefilter_ranks emitted by the runtime event diagnostic",
            "qdic_rank": "candidate_logit_ranks emitted by the runtime event diagnostic",
            "false_merge": "accepted assignment mapped to a known different GT identity and not to the target",
            "accepted_unresolved": "accepted assignment had no posthoc known GT mapping",
            "gap": "current annotation frame_id minus the last prior prediction frame mapped to the same GT identity",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    diagnose(
        annotation_path=Path(args.annotation).resolve(),
        candidate_root=Path(args.candidate_root).resolve(),
        output=Path(args.output).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
