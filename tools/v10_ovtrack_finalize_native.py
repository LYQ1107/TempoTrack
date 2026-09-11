#!/usr/bin/env python
"""Merge complete-video OVTrack shards and run the pinned official TETA evaluator.

This utility never runs the detector.  It refuses to merge until all requested
shard predictions exist and their annotation image order is exactly a partition
of the full annotation order.  The evaluator is the pinned OVTR/TETA source
used by OVTrack's TaoDataset.evaluate().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_keys(annotation: dict[str, Any]) -> list[tuple[int, int, int]]:
    return [
        (int(image["id"]), int(image["video_id"]), int(image["frame_id"]))
        for image in annotation["images"]
    ]


def load_prediction(path: Path) -> dict[str, list[Any]]:
    with path.open("rb") as handle:
        value = pickle.load(handle, encoding="latin1")
    if not isinstance(value, dict) or set(value) != {"bbox_results", "track_results"}:
        raise ValueError(f"unexpected OVTrack prediction structure: {path}")
    if not isinstance(value["bbox_results"], list) or not isinstance(
        value["track_results"], list
    ):
        raise ValueError(f"prediction fields must be lists: {path}")
    if len(value["bbox_results"]) != len(value["track_results"]):
        raise ValueError(f"bbox/track result lengths differ: {path}")
    return value


def evaluator_manifest(root: Path) -> tuple[dict[str, str], str]:
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    manifest = {str(path.relative_to(root)): sha256_file(path) for path in files}
    digest = hashlib.sha256(
        "".join(f"{name} {value}\n" for name, value in manifest.items()).encode()
    ).hexdigest()
    return manifest, digest


def mean_teta_rows(
    summary: dict[str, Any], names: Iterable[str]
) -> dict[str, float | None]:
    fields = ("TETA", "LocA", "AssocA", "ClsA")
    rows = []
    for name in names:
        record = summary.get(name)
        if not record or "TETA" not in record or 50 not in record["TETA"]:
            continue
        row = np.asarray(record["TETA"][50], dtype=np.float64)
        if row.size < 4:
            raise ValueError(f"official TETA row has fewer than four fields: {name}")
        rows.append(row[:4])
    if not rows:
        return {field: None for field in fields}
    mean = np.mean(np.stack(rows), axis=0)
    return {field: float(value) for field, value in zip(fields, mean)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-prefix", required=True)
    parser.add_argument("--shards", default="0,1,2,3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.int = int
    annotation_path = args.annotation.resolve()
    full_annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    shard_ids = [int(value) for value in args.shards.split(",") if value.strip()]
    all_expected = image_keys(full_annotation)
    merged_keys: list[tuple[int, int, int]] = []
    shard_predictions: list[tuple[int, Path, dict[str, list[Any]], str]] = []

    for shard_id in shard_ids:
        shard_dir = args.shard_root / args.split / f"shard_{shard_id}"
        shard_annotation_path = shard_dir / "annotation.json"
        prediction_path = shard_dir / "native_results.pkl"
        if not shard_annotation_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(
                f"split {args.split} is incomplete: {shard_annotation_path} or {prediction_path}"
            )
        shard_annotation = json.loads(shard_annotation_path.read_text(encoding="utf-8"))
        keys = image_keys(shard_annotation)
        prediction = load_prediction(prediction_path)
        if len(prediction["track_results"]) != len(keys):
            raise ValueError(
                f"prediction/image count mismatch for shard {shard_id}: "
                f"{len(prediction['track_results'])} != {len(keys)}"
            )
        merged_keys.extend(keys)
        shard_predictions.append(
            (shard_id, prediction_path, prediction, sha256_file(prediction_path))
        )

    if merged_keys != all_expected:
        raise ValueError(
            "shard annotation image order is not exactly the full annotation order; "
            "refusing to merge"
        )

    merged: defaultdict[str, list[Any]] = defaultdict(list)
    for _, _, prediction, _ in shard_predictions:
        for field in ("bbox_results", "track_results"):
            merged[field].extend(prediction[field])

    args.output_root.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_root / "native_results.pkl"
    with merged_path.open("wb") as handle:
        pickle.dump(dict(merged), handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.path.insert(0, str(args.upstream_root.resolve()))
    sys.path.insert(0, str(args.evaluator_root.resolve()))
    from mmcv import Config  # type: ignore
    from mmdet.datasets import build_dataset  # type: ignore
    from ovtrack.models.roi_heads.class_name import LVIS_CLASSES  # type: ignore

    import ovtrack.datasets  # noqa: F401  # type: ignore

    config = Config.fromfile(str(args.config.resolve()))
    config.data.test.ann_file = str(annotation_path)
    config.data.test.img_prefix = args.image_prefix
    config.data.test.classes = list(LVIS_CLASSES)
    config.data.test.test_mode = True
    config.data.test.ref_img_sampler = None
    dataset = build_dataset(config.data.test)

    evaluation_root = args.output_root / "official_teta"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    with merged_path.open("rb") as handle:
        merged_prediction = pickle.load(handle, encoding="latin1")
    dataset.evaluate(
        merged_prediction,
        metric=["track"],
        resfile_path=str(evaluation_root),
    )

    summary_path = evaluation_root / "OVTrack" / "teta_summary_results.pth"
    if not summary_path.is_file():
        raise FileNotFoundError(f"official evaluator did not write {summary_path}")
    with summary_path.open("rb") as handle:
        summary = pickle.load(handle, encoding="latin1")
    combined = summary.get("COMBINED_SEQ", summary)
    base_names = {
        str(category["name"])
        for category in full_annotation["categories"]
        if category.get("frequency") != "r"
    }
    novel_names = {
        str(category["name"])
        for category in full_annotation["categories"]
        if category.get("frequency") == "r"
    }

    evaluator_manifest_data, evaluator_manifest_sha = evaluator_manifest(
        args.evaluator_root.resolve()
    )
    result = {
        "status": "PASS",
        "split": args.split,
        "command": " ".join(sys.argv),
        "annotation": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "upstream_root": str(args.upstream_root.resolve()),
        "evaluator_root": str(args.evaluator_root.resolve()),
        "evaluator_manifest_sha256": evaluator_manifest_sha,
        "evaluator_manifest": evaluator_manifest_data,
        "coverage": {
            "videos": len(full_annotation["videos"]),
            "images": len(full_annotation["images"]),
            "annotations": len(full_annotation["annotations"]),
            "shards": shard_ids,
        },
        "shards": [
            {
                "shard": shard_id,
                "prediction": str(path),
                "prediction_sha256": prediction_sha,
            }
            for shard_id, path, _, prediction_sha in shard_predictions
        ],
        "merged_prediction": {
            "path": str(merged_path),
            "sha256": sha256_file(merged_path),
            "images": len(merged["track_results"]),
        },
        "official_summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "metrics_percent": {
            "base": mean_teta_rows(combined, base_names),
            "novel": mean_teta_rows(combined, novel_names),
        },
    }
    result_path = args.output_root / "official_evaluation.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["metrics_percent"], indent=2, sort_keys=True))
    print(f"WROTE {result_path}")


if __name__ == "__main__":
    main()
