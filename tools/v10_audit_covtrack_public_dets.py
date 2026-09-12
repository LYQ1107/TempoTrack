#!/usr/bin/env python3
"""Audit COVTrack public detections against the exact MASA file contract.

The exporter writes one pickle for each COV frame.  This command is the
read-only, split-aware receipt builder used before a MASA run; it never uses
GT boxes or track IDs as detection data.  The annotation is used only to
define the expected image order and path set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tempotrack_v10.cov_detection_export import masa_public_detection_relative_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_annotation_array(digest: Any, relative: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))


def _expected_relative(file_name: str, split: str) -> Path:
    text = str(file_name).replace("\\", "/")
    if "data/tao/frames/" in text:
        result = masa_public_detection_relative_path(text)
    else:
        result = Path(text[:-4] + ".pth") if text.lower().endswith(".jpg") else Path(text).with_suffix(".pth")
    if result.is_absolute() or ".." in result.parts:
        raise ValueError(f"unsafe annotation image path: {file_name!r}")
    if not result.parts or result.parts[0] != split:
        raise ValueError(
            f"annotation image is not in requested {split!r} split: {file_name!r} -> {result}"
        )
    return result


def _load_annotation(path: Path, split: str) -> tuple[list[Path], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = data.get("images")
    if not isinstance(images, list):
        raise ValueError(f"annotation has no images list: {path}")
    expected: list[Path] = []
    seen: set[Path] = set()
    for image in images:
        relative = _expected_relative(str(image["file_name"]), split)
        if relative in seen:
            raise ValueError(f"duplicate annotation image path: {relative}")
        seen.add(relative)
        expected.append(relative)
    return expected, data


def audit_public_detections(
    *,
    root: str | Path,
    annotation: str | Path,
    split: str,
    source_checkpoint: str | Path,
    source_config: str | Path,
    output_manifest: str | Path,
    source_stage: str = "post_filter_post_mcf_pre_association",
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    annotation_path = Path(annotation).resolve()
    checkpoint_path = Path(source_checkpoint).resolve()
    config_path = Path(source_config).resolve()
    manifest_path = Path(output_manifest).resolve()
    expected, annotation_data = _load_annotation(annotation_path, split)
    expected_set = set(expected)
    split_root = root_path / split
    actual_set = (
        {path.relative_to(root_path) for path in split_root.rglob("*.pth")}
        if split_root.is_dir()
        else set()
    )
    missing = sorted(expected_set - actual_set, key=lambda p: p.as_posix())
    extra = sorted(actual_set - expected_set, key=lambda p: p.as_posix())

    frame_hash = hashlib.sha256()
    bbox_hash = hashlib.sha256()
    score_hash = hashlib.sha256()
    label_hash = hashlib.sha256()
    errors: list[str] = []
    detections = 0
    score_min = float("inf")
    score_max = float("-inf")

    for relative in expected:
        frame_hash.update(relative.as_posix().encode("utf-8"))
        frame_hash.update(b"\0")
        path = root_path / relative
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:  # pragma: no cover - exercised by real audit
            errors.append(f"pickle_load_failed:{relative}:{type(exc).__name__}:{exc}")
            continue
        if not isinstance(payload, dict) or tuple(payload.keys()) != (
            "det_labels",
            "det_bboxes",
        ):
            errors.append(f"keys_not_exact:{relative}:{list(payload) if isinstance(payload, dict) else type(payload).__name__}")
            continue
        labels = payload["det_labels"]
        bboxes = payload["det_bboxes"]
        if not isinstance(labels, np.ndarray) or not isinstance(bboxes, np.ndarray):
            errors.append(f"arrays_required:{relative}")
            continue
        if labels.dtype != np.dtype(np.int64) or labels.ndim != 1:
            errors.append(f"label_contract:{relative}:{labels.dtype}:{labels.shape}")
            continue
        if bboxes.dtype != np.dtype(np.float32) or bboxes.ndim != 2 or bboxes.shape[1:] != (5,):
            errors.append(f"bbox_contract:{relative}:{bboxes.dtype}:{bboxes.shape}")
            continue
        if len(labels) != len(bboxes) or not np.isfinite(bboxes).all():
            errors.append(f"finite_alignment_contract:{relative}")
            continue
        detections += int(len(labels))
        if len(bboxes):
            score_min = min(score_min, float(bboxes[:, 4].min()))
            score_max = max(score_max, float(bboxes[:, 4].max()))
        _sha256_annotation_array(bbox_hash, relative.as_posix(), bboxes[:, :4])
        _sha256_annotation_array(score_hash, relative.as_posix(), bboxes[:, 4])
        _sha256_annotation_array(label_hash, relative.as_posix(), labels)

    if not np.isfinite(score_min):
        score_min = None
        score_max = None
    if missing:
        errors.append(f"missing_files:{len(missing)}")
    if extra:
        errors.append(f"extra_files:{len(extra)}")
    if not checkpoint_path.is_file():
        errors.append(f"missing_source_checkpoint:{checkpoint_path}")
    if not config_path.is_file():
        errors.append(f"missing_source_config:{config_path}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "covtrack_public_detections_for_masa_manifest",
        "status": "PASS" if not errors else "FAIL",
        "source_frontend": "COVTrack",
        "usage": "MASA-R50 public detections",
        "source_stage": source_stage,
        "source_capture": "V10_COV_DET_EXPORT_ROOT before adapter.prepare",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": _sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
        "source_config": str(config_path),
        "source_config_sha256": _sha256_file(config_path) if config_path.is_file() else None,
        "annotation": str(annotation_path),
        "annotation_sha256": _sha256_file(annotation_path),
        "split": split,
        "frames": len(expected),
        "detections": detections,
        "missing_files": [path.as_posix() for path in missing],
        "extra_files": [path.as_posix() for path in extra],
        "bbox_format": "xyxy+score",
        "label_dtype": "int64",
        "bbox_dtype": "float32",
        "contains_track_id": False,
        "contains_gt": False,
        "contains_tempo_assignment": False,
        "score_min": score_min,
        "score_max": score_max,
        "frame_path_hash": frame_hash.hexdigest(),
        "bbox_hash": bbox_hash.hexdigest(),
        "score_hash": score_hash.hexdigest(),
        "label_hash": label_hash.hexdigest(),
        "errors": errors,
        "annotation_image_count": len(annotation_data.get("images", [])),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--source-stage", default="post_filter_post_mcf_pre_association")
    args = parser.parse_args()
    result = audit_public_detections(
        root=args.root,
        annotation=args.annotation,
        split=args.split,
        source_checkpoint=args.source_checkpoint,
        source_config=args.source_config,
        output_manifest=args.output_manifest,
        source_stage=args.source_stage,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
