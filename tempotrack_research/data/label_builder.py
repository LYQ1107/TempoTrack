"""Ground-truth label construction kept separate from model inputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import json
import numpy as np


def load_coco_like(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Annotation root is not a mapping: {path}")
    return payload


def annotation_summary(path: str | Path) -> dict[str, Any]:
    payload = load_coco_like(path)
    videos = payload.get("videos", [])
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    tracks = payload.get("tracks", [])
    return {
        "path": str(path),
        "video_count": len(videos),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "track_count": len(tracks),
        "has_track_id": any("track_id" in item for item in annotations),
        "category_count": len(payload.get("categories", [])),
    }


def build_image_index(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["id"]): item for item in payload.get("images", []) if "id" in item}


def build_training_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Return labels indexed by annotation rows for explicit trainer use."""

    payload = load_coco_like(path)
    records = []
    for annotation in payload.get("annotations", []):
        image = annotation.get("image_id", -1)
        records.append(
            (
                int(image),
                int(annotation.get("track_id", -1)),
                int(annotation.get("category_id", -1)),
                float(annotation.get("iou", 1.0 if "track_id" in annotation else 0.0)),
            )
        )
    return {
        "image_ids": np.asarray([row[0] for row in records], dtype=np.int64),
        "gt_identity": np.asarray([row[1] for row in records], dtype=np.int64),
        "gt_category": np.asarray([row[2] for row in records], dtype=np.int64),
        "matched_gt_iou": np.asarray([row[3] for row in records], dtype=np.float32),
        "known_mask": np.asarray([row[1] >= 0 for row in records], dtype=bool),
    }


def direct_successor_labels(tracklets: Iterable[Mapping[str, Any]]) -> dict[tuple[int, int], int]:
    """Label only consecutive segments of an identity as positive edges."""

    by_identity: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for tracklet in tracklets:
        identity = tracklet.get("gt_identity")
        if identity is None or int(identity) < 0:
            continue
        key = (int(tracklet.get("video_id", -1)), int(identity))
        by_identity[key].append(tracklet)
    labels: dict[tuple[int, int], int] = {}
    for segments in by_identity.values():
        segments.sort(key=lambda item: (int(item.get("first_frame", 0)), int(item.get("last_frame", 0))))
        for left, right in zip(segments, segments[1:]):
            if int(left.get("local_id", -1)) >= 0 and int(right.get("local_id", -1)) >= 0:
                labels[(int(left["local_id"]), int(right["local_id"]))] = 1
    return labels


def pairwise_identity_f1(assignments: np.ndarray, identities: np.ndarray, known_mask: np.ndarray | None = None) -> float:
    """Contingency-count pairwise F1 used only as a training surrogate."""

    assignments = np.asarray(assignments, dtype=np.int64)
    identities = np.asarray(identities, dtype=np.int64)
    mask = np.ones_like(assignments, dtype=bool) if known_mask is None else np.asarray(known_mask, dtype=bool)
    mask &= identities >= 0
    assignments, identities = assignments[mask], identities[mask]
    if assignments.size < 2:
        return 1.0 if assignments.size else 0.0
    counts: dict[tuple[int, int], int] = defaultdict(int)
    pred_counts: dict[int, int] = defaultdict(int)
    gt_counts: dict[int, int] = defaultdict(int)
    for pred, gt in zip(assignments.tolist(), identities.tolist()):
        counts[(int(pred), int(gt))] += 1
        pred_counts[int(pred)] += 1
        gt_counts[int(gt)] += 1
    tp = sum(value * (value - 1) // 2 for value in counts.values())
    pred_pairs = sum(value * (value - 1) // 2 for value in pred_counts.values())
    gt_pairs = sum(value * (value - 1) // 2 for value in gt_counts.values())
    precision = tp / pred_pairs if pred_pairs else 0.0
    recall = tp / gt_pairs if gt_pairs else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
