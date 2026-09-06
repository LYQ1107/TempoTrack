"""Immutable, content-addressed observation ledgers.

The ledger is the boundary between frozen visual facts and every trainable
component.  Labels, identities and rewards live in separate shards.  A
legacy ``.pt`` cache may be imported only through the explicit legacy path;
it is never silently treated as a v2 train ledger.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..config import canonical_json, file_hash
from ..schemas import ObservationBatch, ObservationKey


REQUIRED_ARRAYS = {
    "video_ids",
    "frame_indices",
    "image_ids",
    "source_detection_indices",
    "bboxes_xyxy",
    "scores",
    "category_ids",
    "appearance",
    "image_widths",
    "image_heights",
    "frame_times",
}
PAYLOAD_ARRAYS = (
    "video_ids",
    "frame_indices",
    "image_ids",
    "source_detection_indices",
    "bboxes_xyxy",
    "scores",
    "category_ids",
    "image_widths",
    "image_heights",
    "frame_times",
)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()
    rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": list(self.errors), "rows": self.rows}


@dataclass
class FrameIndex:
    """Frame-level index, including frames with zero detections."""

    records: list[dict[str, Any]]
    metadata: dict[str, Any]

    def validate(self) -> ValidationResult:
        errors: list[str] = []
        seen: set[tuple[str, int, int]] = set()
        for item in self.records:
            required = {
                "dataset_id",
                "video_id",
                "frame_index",
                "image_id",
                "file_name",
                "width",
                "height",
                "frame_time",
                "time_unit",
            }
            missing = required.difference(item)
            if missing:
                errors.append(f"frame missing fields: {sorted(missing)}")
                continue
            key = (str(item["dataset_id"]), int(item["video_id"]), int(item["frame_index"]))
            if key in seen:
                errors.append(f"duplicate frame index: {key}")
            seen.add(key)
            if int(item["width"]) <= 0 or int(item["height"]) <= 0:
                errors.append(f"invalid frame size: {key}")
        return ValidationResult(not errors, tuple(errors[:50]), len(self.records))

    def save(self, path: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        result = self.validate()
        if not result.valid:
            raise ValueError("invalid FrameIndex: " + "; ".join(result.errors))
        payload = {
            "schema_version": 1,
            "metadata": self.metadata,
            "records": self.records,
        }
        _atomic_text(payload, path)
        return {"path": str(path), "record_count": len(self.records)}

    @classmethod
    def load(cls, path: str | Path) -> "FrameIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(list(payload.get("records", [])), dict(payload.get("metadata", {})))
        checked = result.validate()
        if not checked.valid:
            raise ValueError(f"invalid FrameIndex {path}: {checked.errors}")
        return result


def _atomic_text(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _stable_metadata(value: Any) -> Any:
    """Remove volatile/self references while retaining semantic provenance."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name in {"content_hash", "observation_payload_hash", "feature_hash", "npz_sha256", "generated_at", "created_at", "saved_at"}:
                continue
            result[name] = _stable_metadata(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_stable_metadata(item) for item in value]
    if isinstance(value, Path):
        return f"<absolute>/{value.name}" if value.is_absolute() else str(value)
    if isinstance(value, str) and value.startswith("/"):
        return f"<absolute>/{Path(value).name}"
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _update_hash(digest: Any, name: str, array: np.ndarray) -> None:
    value = np.ascontiguousarray(array)
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json(list(value.shape)).encode("ascii"))
    digest.update(b"\0")
    # Chunking avoids an additional full-size byte copy for large features.
    raw = memoryview(value).cast("B")
    for start in range(0, len(raw), 8 * 1024 * 1024):
        digest.update(raw[start : start + 8 * 1024 * 1024])


class ObservationLedger:
    schema_version = 2

    def __init__(self, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], *, readonly: bool = False):
        aliases = dict(arrays)
        if "timestamps" in aliases and "frame_times" not in aliases:
            aliases["frame_times"] = aliases.pop("timestamps")
        if "detection_indices" in aliases and "source_detection_indices" not in aliases:
            aliases["source_detection_indices"] = aliases.pop("detection_indices")
        if "image_widths" not in aliases:
            aliases["image_widths"] = np.zeros((len(aliases.get("image_ids", [])),), dtype=np.int32)
        if "image_heights" not in aliases:
            aliases["image_heights"] = np.zeros((len(aliases.get("image_ids", [])),), dtype=np.int32)
        missing = REQUIRED_ARRAYS.difference(aliases)
        if missing:
            raise ValueError(f"Observation ledger missing arrays: {sorted(missing)}")
        n = len(np.asarray(aliases["image_ids"]))
        errors: list[str] = []
        for name in REQUIRED_ARRAYS:
            value = np.asarray(aliases[name])
            if value.ndim == 0 or len(value) != n:
                errors.append(f"{name} has invalid row count/shape {value.shape}")
            aliases[name] = value
        if aliases["bboxes_xyxy"].ndim != 2 or aliases["bboxes_xyxy"].shape[-1] != 4:
            errors.append("bboxes_xyxy must be [N,4]")
        if aliases["appearance"].ndim != 2:
            errors.append("appearance must be [N,D]")
        if errors:
            raise ValueError("invalid observation ledger: " + "; ".join(errors))
        self.arrays = {name: aliases[name] for name in sorted(aliases)}
        self.metadata = dict(metadata)
        self.metadata.setdefault("schema_version", self.schema_version)
        self.metadata.setdefault("row_count", n)
        if int(self.metadata["schema_version"]) != self.schema_version:
            raise ValueError("legacy ledger must be imported explicitly with FeatureExportManager.import_legacy")
        if readonly:
            for value in self.arrays.values():
                value.setflags(write=False)

    @property
    def row_count(self) -> int:
        return int(self.arrays["image_ids"].shape[0])

    @property
    def appearance_dim(self) -> int:
        return int(self.arrays["appearance"].shape[1])

    def validate(self, *, full: bool = False) -> ValidationResult:
        errors: list[str] = []
        if self.metadata.get("schema_version") != self.schema_version:
            errors.append("schema_version is not 2")
        if self.metadata.get("row_count") != self.row_count:
            errors.append("metadata row_count mismatch")
        if np.any(~np.isfinite(self.arrays["bboxes_xyxy"])) or np.any(~np.isfinite(self.arrays["scores"])):
            errors.append("non-finite boxes or scores")
        if self.row_count and np.any(self.arrays["bboxes_xyxy"][:, 2:] < self.arrays["bboxes_xyxy"][:, :2]):
            errors.append("inverted bounding box")
        if np.any(self.arrays["image_widths"] <= 0) or np.any(self.arrays["image_heights"] <= 0):
            errors.append("missing/invalid image dimensions")
        if len(set(self.keys())) != self.row_count:
            errors.append("duplicate observation UID")
        if full:
            if not np.isfinite(self.arrays["appearance"]).all():
                errors.append("non-finite appearance")
            if not np.isfinite(self.arrays["frame_times"]).all():
                errors.append("non-finite frame times")
        return ValidationResult(not errors, tuple(errors[:50]), self.row_count)

    def keys(self, rows: Iterable[int] | np.ndarray | None = None) -> list[ObservationKey]:
        indices = np.arange(self.row_count, dtype=np.int64) if rows is None else np.asarray(list(rows) if not isinstance(rows, np.ndarray) else rows, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.row_count):
            raise IndexError("ledger row index out of bounds")
        dataset_id = str(self.metadata.get("dataset_id", "unknown"))
        video = self.arrays["video_ids"][indices]
        frames = self.arrays["frame_indices"][indices]
        detections = self.arrays["source_detection_indices"][indices]
        return [ObservationKey(dataset_id, int(v), int(f), int(d)) for v, f, d in zip(video, frames, detections)]

    def model_batch(self, rows: Iterable[int] | np.ndarray | None = None) -> ObservationBatch:
        indices = np.arange(self.row_count, dtype=np.int64) if rows is None else np.asarray(list(rows) if not isinstance(rows, np.ndarray) else rows, dtype=np.int64)
        keys = self.keys(indices)
        a = self.arrays
        return ObservationBatch(
            keys=keys,
            image_ids=a["image_ids"][indices],
            frame_indices=a["frame_indices"][indices],
            timestamps=a["frame_times"][indices],
            bboxes_xyxy=a["bboxes_xyxy"][indices],
            scores=a["scores"][indices],
            category_ids=a["category_ids"][indices],
            appearance=a["appearance"][indices],
            video_ids=a["video_ids"][indices],
            image_widths=a["image_widths"][indices],
            image_heights=a["image_heights"][indices],
            original_payload=tuple(),
            rows=indices.copy(),
        )

    def _hash(self, names: Sequence[str]) -> str:
        import hashlib

        digest = hashlib.sha256()
        digest.update(canonical_json(_stable_metadata(self.metadata)).encode("utf-8"))
        for name in sorted(names):
            _update_hash(digest, name, self.arrays[name])
        return digest.hexdigest()

    def content_hash(self) -> str:
        return self._hash(tuple(self.arrays))

    def observation_payload_hash(self) -> str:
        return self._hash(PAYLOAD_ARRAYS)

    def feature_hash(self) -> str:
        return self._hash(("appearance",))

    def save(self, path: str | Path, *, overwrite: bool = False, payload: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("v2 ledger path must end in .npz")
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        checked = self.validate(full=True)
        if not checked.valid:
            raise ValueError("cannot save invalid ledger: " + "; ".join(checked.errors))
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
        os.close(fd)
        meta_path = path.with_suffix(path.suffix + ".json")
        try:
            np.savez_compressed(temp_name, **self.arrays)
            with open(temp_name, "rb") as handle:
                os.fsync(handle.fileno())
            content = self.content_hash()
            metadata = dict(self.metadata)
            metadata.update({
                "schema_version": self.schema_version,
                "row_count": self.row_count,
                "content_hash": content,
                "observation_payload_hash": self.observation_payload_hash(),
                "feature_hash": self.feature_hash(),
                "npz_sha256": file_hash(temp_name),
            })
            _atomic_text(metadata, meta_path)
            os.replace(temp_name, path)
            if payload is not None:
                _atomic_text({"schema_version": 1, "records": payload}, path.with_suffix(path.suffix + ".payload.json"))
            return metadata
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    @classmethod
    def load(cls, path: str | Path, *, verify_hash: bool = True) -> "ObservationLedger":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        meta_path = path.with_suffix(path.suffix + ".json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if int(metadata.get("schema_version", 0)) != cls.schema_version:
            raise ValueError(f"legacy or unversioned ledger {path}; use explicit legacy import")
        if verify_hash:
            expected_file = metadata.get("npz_sha256")
            if not expected_file:
                raise ValueError(f"ledger has no npz_sha256 sidecar: {path}")
            actual_file = file_hash(path)
            if actual_file != expected_file:
                raise ValueError(f"ledger file hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        ledger = cls(arrays, metadata, readonly=True)
        if verify_hash:
            for key in ("content_hash", "observation_payload_hash", "feature_hash"):
                expected = metadata.get(key)
                actual = getattr(ledger, key)()
                if expected and expected != actual:
                    raise ValueError(f"ledger {key} mismatch: {path}")
        checked = ledger.validate(full=True)
        if not checked.valid:
            raise ValueError(f"invalid ledger {path}: {checked.errors}")
        return ledger


def empty_observation_ledger(dataset_id: str, appearance_dim: int = 256, *, split: str = "unknown") -> ObservationLedger:
    empty = {
        "video_ids": np.empty((0,), dtype=np.int64),
        "frame_indices": np.empty((0,), dtype=np.int64),
        "image_ids": np.empty((0,), dtype=np.int64),
        "source_detection_indices": np.empty((0,), dtype=np.int64),
        "bboxes_xyxy": np.empty((0, 4), dtype=np.float32),
        "scores": np.empty((0,), dtype=np.float32),
        "category_ids": np.empty((0,), dtype=np.int64),
        "appearance": np.empty((0, appearance_dim), dtype=np.float32),
        "image_widths": np.empty((0,), dtype=np.int32),
        "image_heights": np.empty((0,), dtype=np.int32),
        "frame_times": np.empty((0,), dtype=np.float64),
    }
    return ObservationLedger(empty, {"dataset_id": dataset_id, "split": split, "feature_status": "empty"})
