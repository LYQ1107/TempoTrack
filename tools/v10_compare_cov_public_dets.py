#!/usr/bin/env python3
"""Compare two split-aware MASA public-detection trees exactly.

This is intentionally stricter than a numeric tolerance check: the expected
image set must be identical, each pickle must have the exact MASA keys and
array contracts, and boxes, scores, and labels are compared element by
element.  The command is used for the COV pre-filter-cache FAST PATH gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tempotrack_v10.cov_detection_export import masa_public_detection_relative_path


def _expected(annotation: Path, split: str) -> list[Path]:
    data = json.loads(annotation.read_text(encoding="utf-8"))
    paths: list[Path] = []
    seen: set[Path] = set()
    for image in data.get("images", []):
        filename = str(image["file_name"]).replace("\\", "/")
        if "data/tao/frames/" in filename:
            relative = masa_public_detection_relative_path(filename)
        else:
            relative = Path(filename[:-4] + ".pth") if filename.lower().endswith(".jpg") else Path(filename).with_suffix(".pth")
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != split:
            raise ValueError(f"unsafe or wrong-split annotation path: {filename!r} -> {relative}")
        if relative in seen:
            raise ValueError(f"duplicate annotation image path: {relative}")
        seen.add(relative)
        paths.append(relative)
    return paths


def _payload(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        value: Any = pickle.load(handle)
    if not isinstance(value, dict) or tuple(value.keys()) != ("det_labels", "det_bboxes"):
        raise ValueError(f"exact MASA keys required: {path}")
    labels = value["det_labels"]
    bboxes = value["det_bboxes"]
    if not isinstance(labels, np.ndarray) or not isinstance(bboxes, np.ndarray):
        raise ValueError(f"numpy arrays required: {path}")
    if labels.dtype != np.int64 or labels.ndim != 1:
        raise ValueError(f"label contract failed: {path}: {labels.dtype} {labels.shape}")
    if bboxes.dtype != np.float32 or bboxes.ndim != 2 or bboxes.shape[1:] != (5,):
        raise ValueError(f"bbox contract failed: {path}: {bboxes.dtype} {bboxes.shape}")
    if len(labels) != len(bboxes) or not np.isfinite(bboxes).all():
        raise ValueError(f"finite/alignment contract failed: {path}")
    return bboxes, labels


def compare(*, left: str | Path, right: str | Path, annotation: str | Path, split: str) -> dict[str, Any]:
    left_root = Path(left).resolve()
    right_root = Path(right).resolve()
    annotation_path = Path(annotation).resolve()
    expected = _expected(annotation_path, split)
    left_set = {path.relative_to(left_root) for path in (left_root / split).rglob("*.pth")} if (left_root / split).is_dir() else set()
    right_set = {path.relative_to(right_root) for path in (right_root / split).rglob("*.pth")} if (right_root / split).is_dir() else set()
    expected_set = set(expected)
    missing_left = sorted(expected_set - left_set, key=lambda p: p.as_posix())
    missing_right = sorted(expected_set - right_set, key=lambda p: p.as_posix())
    extra_left = sorted(left_set - expected_set, key=lambda p: p.as_posix())
    extra_right = sorted(right_set - expected_set, key=lambda p: p.as_posix())
    mismatches: list[dict[str, Any]] = []
    max_bbox_diff = 0.0
    max_score_diff = 0.0
    label_diff = 0
    compared = 0
    for relative in expected:
        left_path = left_root / relative
        right_path = right_root / relative
        if not left_path.is_file() or not right_path.is_file():
            continue
        try:
            left_boxes, left_labels = _payload(left_path)
            right_boxes, right_labels = _payload(right_path)
        except Exception as exc:
            mismatches.append({"path": relative.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
            continue
        compared += 1
        if left_boxes.shape != right_boxes.shape or left_labels.shape != right_labels.shape:
            mismatches.append({
                "path": relative.as_posix(),
                "error": "shape_mismatch",
                "left_bbox_shape": list(left_boxes.shape),
                "right_bbox_shape": list(right_boxes.shape),
                "left_label_shape": list(left_labels.shape),
                "right_label_shape": list(right_labels.shape),
            })
            continue
        if len(left_boxes):
            bbox_diff = np.abs(left_boxes[:, :4].astype(np.float64) - right_boxes[:, :4].astype(np.float64))
            score_diff = np.abs(left_boxes[:, 4].astype(np.float64) - right_boxes[:, 4].astype(np.float64))
            max_bbox_diff = max(max_bbox_diff, float(bbox_diff.max(initial=0.0)))
            max_score_diff = max(max_score_diff, float(score_diff.max(initial=0.0)))
        frame_label_diff = int(np.count_nonzero(left_labels != right_labels))
        label_diff += frame_label_diff
        if not np.array_equal(left_boxes, right_boxes) or not np.array_equal(left_labels, right_labels):
            mismatches.append({
                "path": relative.as_posix(),
                "error": "array_mismatch",
                "max_bbox_diff": float(np.abs(left_boxes[:, :4].astype(np.float64) - right_boxes[:, :4].astype(np.float64)).max(initial=0.0)) if len(left_boxes) else 0.0,
                "max_score_diff": float(np.abs(left_boxes[:, 4].astype(np.float64) - right_boxes[:, 4].astype(np.float64)).max(initial=0.0)) if len(left_boxes) else 0.0,
                "label_diff": frame_label_diff,
            })
    status = "PASS" if not (missing_left or missing_right or extra_left or extra_right or mismatches) and compared == len(expected) else "FAIL"
    return {
        "status": status,
        "protocol": "cov_prefilter_cache_offline_pinned_filter_vs_fresh_tempo_disabled_post_filter",
        "split": split,
        "annotation": str(annotation_path),
        "frames_expected": len(expected),
        "frames_compared": compared,
        "missing_left": [p.as_posix() for p in missing_left],
        "missing_right": [p.as_posix() for p in missing_right],
        "extra_left": [p.as_posix() for p in extra_left],
        "extra_right": [p.as_posix() for p in extra_right],
        "max_bbox_diff": max_bbox_diff,
        "max_score_diff": max_score_diff,
        "label_diff": label_diff,
        "mismatches": mismatches,
        "exact_frame_count": compared == len(expected),
        "exact_bbox_shape": not any(item.get("error") == "shape_mismatch" for item in mismatches),
        "exact_labels": label_diff == 0,
        "exact_bbox_values": max_bbox_diff == 0.0,
        "exact_score_values": max_score_diff == 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(left=args.left, right=args.right, annotation=args.annotation, split=args.split)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
