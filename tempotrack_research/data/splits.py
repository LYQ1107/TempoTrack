"""Leakage-safe video and category split manifests."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..config import file_hash, object_hash


def group_videos(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("video_id", record.get("video", "unknown")))].append(record)
    return dict(groups)


def assert_disjoint_video_splits(splits: Mapping[str, Iterable[Any]]) -> None:
    seen: dict[Any, str] = {}
    for split, video_ids in splits.items():
        for video_id in video_ids:
            if video_id in seen:
                raise ValueError(f"video {video_id!r} appears in {seen[video_id]} and {split}")
            seen[video_id] = split


class SplitManifestBuilder:
    """Create deterministic train/internal-val manifests from real videos.

    The official validation annotation is recorded as a separate protocol
    split and is never sampled into train episodes.
    """

    def __init__(self, internal_val_fraction: float = 0.1):
        if not 0.0 < internal_val_fraction < 1.0:
            raise ValueError("internal_val_fraction must be between 0 and 1")
        self.internal_val_fraction = float(internal_val_fraction)

    def build(self, annotation: str | Path, category_protocol: str | Path | None, *, seed: int, output: str | Path) -> dict[str, Any]:
        annotation = Path(annotation)
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        videos = sorted((dict(item) for item in payload.get("videos", [])), key=lambda item: int(item["id"]))
        if not videos:
            raise ValueError(f"annotation has no videos: {annotation}")
        rng = random.Random(int(seed))
        shuffled = [int(item["id"]) for item in videos]
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * self.internal_val_fraction)))
        internal_val = sorted(shuffled[:val_count])
        train = sorted(shuffled[val_count:])
        assert_disjoint_video_splits({"train_base": train, "val_base_internal": internal_val})

        category_info: dict[str, Any] = {"protocol": "ordinary_cross_video", "verified": False}
        if category_protocol is not None:
            path = Path(category_protocol)
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                category_info = {
                    "protocol": str(raw.get("name", raw.get("protocol", path.name))),
                    "verified": bool(raw.get("verified", False)),
                    "path": str(path),
                    "hash": file_hash(path),
                    "base_categories": raw.get("base_categories"),
                    "novel_categories": raw.get("novel_categories"),
                }

        result = {
            "schema_version": 1,
            "seed": int(seed),
            "annotation": str(annotation),
            "annotation_hash": file_hash(annotation),
            "category_protocol": category_info,
            "video_count": len(videos),
            "splits": {
                "train_base": train,
                "val_base_internal": internal_val,
                "official_validation": [],
            },
            "video_names": {str(item["id"]): item.get("name") for item in videos},
            "leakage_check": {"disjoint": True, "identity_split": "video_then_track"},
        }
        result["manifest_hash"] = object_hash(result)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result


def add_official_validation(manifest: Mapping[str, Any], validation_annotation: str | Path, output: str | Path) -> dict[str, Any]:
    validation_annotation = Path(validation_annotation)
    payload = json.loads(validation_annotation.read_text(encoding="utf-8"))
    official = sorted(int(item["id"]) for item in payload.get("videos", []))
    splits = {key: list(value) for key, value in manifest.get("splits", {}).items()}
    splits["official_validation"] = official
    assert_disjoint_video_splits({"train_base": splits.get("train_base", []), "val_base_internal": splits.get("val_base_internal", [])})
    result = dict(manifest)
    result["splits"] = splits
    result["official_validation_annotation"] = str(validation_annotation)
    result["official_validation_hash"] = file_hash(validation_annotation)
    result.pop("manifest_hash", None)
    result["manifest_hash"] = object_hash(result)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = payload.get("manifest_hash")
    actual_payload = dict(payload)
    actual_payload.pop("manifest_hash", None)
    if expected and expected != object_hash(actual_payload):
        raise ValueError(f"split manifest hash mismatch: {path}")
    assert_disjoint_video_splits({
        "train_base": payload.get("splits", {}).get("train_base", []),
        "val_base_internal": payload.get("splits", {}).get("val_base_internal", []),
    })
    return payload
