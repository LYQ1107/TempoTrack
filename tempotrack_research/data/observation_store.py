"""Immutable, safe NPZ-backed observation ledger."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..config import file_hash, object_hash
from ..schemas import ObservationBatch, ObservationKey


class ObservationLedger:
    """A compact fact store; no GT fields are stored in its model-facing view."""

    schema_version = 1

    def __init__(self, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]):
        required = {"image_ids", "timestamps", "bboxes_xyxy", "scores", "category_ids", "appearance", "video_ids", "frame_indices", "detection_indices"}
        missing = required.difference(arrays)
        if missing:
            raise ValueError(f"Observation ledger missing arrays: {sorted(missing)}")
        n = len(arrays["image_ids"])
        if any(len(arrays[name]) != n for name in required):
            raise ValueError("Observation ledger arrays have inconsistent row counts")
        self.arrays = {name: np.asarray(value) for name, value in arrays.items()}
        self.metadata = dict(metadata)
        self.metadata.setdefault("schema_version", self.schema_version)
        self.metadata.setdefault("row_count", n)

    @property
    def row_count(self) -> int:
        return int(self.arrays["image_ids"].shape[0])

    @property
    def appearance_dim(self) -> int:
        return int(self.arrays["appearance"].shape[1])

    def keys(self) -> list[ObservationKey]:
        dataset_id = str(self.metadata.get("dataset_id", "unknown"))
        return [
            ObservationKey(dataset_id, int(video), int(frame), int(det))
            for video, frame, det in zip(
                self.arrays["video_ids"], self.arrays["frame_indices"], self.arrays["detection_indices"]
            )
        ]

    def model_batch(self, rows: Iterable[int] | None = None) -> ObservationBatch:
        indices = np.arange(self.row_count) if rows is None else np.asarray(list(rows), dtype=np.int64)
        keys = [self.keys()[int(i)] for i in indices]
        return ObservationBatch(
            keys=keys,
            image_ids=self.arrays["image_ids"][indices],
            timestamps=self.arrays["timestamps"][indices],
            bboxes_xyxy=self.arrays["bboxes_xyxy"][indices],
            scores=self.arrays["scores"][indices],
            category_ids=self.arrays["category_ids"][indices],
            appearance=self.arrays["appearance"][indices],
            original_payload=tuple(),
        )

    def content_hash(self) -> str:
        metadata = {key: value for key, value in self.metadata.items() if key not in {"content_hash", "npz_sha256"}}
        descriptor = {"metadata": metadata, "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in self.arrays.items()}}
        return object_hash(descriptor)

    def save(self, path: str | Path, payload: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays)
        meta_path = path.with_suffix(path.suffix + ".json")
        metadata = dict(self.metadata)
        metadata.update({"content_hash": self.content_hash(), "npz_sha256": file_hash(path)})
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if payload is not None:
            path.with_suffix(path.suffix + ".payload.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        return metadata

    @classmethod
    def load(cls, path: str | Path) -> "ObservationLedger":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        meta_path = path.with_suffix(path.suffix + ".json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        ledger = cls(arrays, metadata)
        expected = metadata.get("content_hash")
        if expected and expected != ledger.content_hash():
            raise ValueError(f"Observation ledger content hash mismatch: {path}")
        return ledger


def empty_observation_ledger(dataset_id: str, appearance_dim: int = 256) -> ObservationLedger:
    return ObservationLedger(
        {
            "image_ids": np.empty((0,), dtype=np.int64),
            "timestamps": np.empty((0,), dtype=np.float64),
            "bboxes_xyxy": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "category_ids": np.empty((0,), dtype=np.int64),
            "appearance": np.empty((0, appearance_dim), dtype=np.float32),
            "video_ids": np.empty((0,), dtype=np.int64),
            "frame_indices": np.empty((0,), dtype=np.int64),
            "detection_indices": np.empty((0,), dtype=np.int64),
        },
        {"dataset_id": dataset_id, "feature_status": "missing"},
    )
