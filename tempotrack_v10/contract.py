"""Strict pre-association input contract for V10 frontend adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


class SnapshotContractError(ValueError):
    """Raised when an adapter supplies an invalid or unsafe snapshot."""


class FrameCollisionError(SnapshotContractError):
    """Raised when a commit would assign one frame to the same ID twice."""


_FORBIDDEN_METADATA_TOKENS = (
    "gt",
    "ground_truth",
    "oracle",
    "identity_label",
    "assigned_track_ids",
    "final_ids",
    "track_ids",
    "post_association",
    "association_committed",
)


def _contains_forbidden_key(value: Any, path: str = "metadata") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in _FORBIDDEN_METADATA_TOKENS):
                return f"{path}.{key}"
            result = _contains_forbidden_key(nested, f"{path}.{key}")
            if result is not None:
                return result
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            result = _contains_forbidden_key(nested, f"{path}[{index}]")
            if result is not None:
                return result
    return None


def _readonly_array(value: Any, *, dtype: np.dtype, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    if ndim is not None and array.ndim != ndim:
        raise SnapshotContractError(f"{name} must be rank {ndim}, got {array.shape}")
    if not np.isfinite(array).all():
        raise SnapshotContractError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PreAssociationSnapshot:
    """Frontend-neutral state immediately before native ID assignment.

    The object contains detector observations and the frontend's already
    finalized native affinity only. It deliberately has no GT or committed
    track IDs. All arrays are copied and made read-only at construction so a
    shared overlay cannot mutate detector output by aliasing.
    """

    video_id: int | str
    frame_id: int
    boxes_xyxy: np.ndarray
    det_scores: np.ndarray
    labels: np.ndarray
    observation_uids: Sequence[str]
    embeddings: np.ndarray
    native_affinity: np.ndarray
    memory_ids: Sequence[int] = field(default_factory=tuple)
    memory_embeddings: np.ndarray | None = None
    memory_last_frame: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.metadata, Mapping) is False:
            raise SnapshotContractError("metadata must be a mapping")
        forbidden = _contains_forbidden_key(self.metadata)
        if forbidden is not None:
            raise SnapshotContractError(
                f"BLOCKED_POST_ASSOCIATION_INPUT: forbidden field {forbidden}"
            )
        metadata = dict(self.metadata)
        if metadata.get("pre_association") is False:
            raise SnapshotContractError("BLOCKED_POST_ASSOCIATION_INPUT: pre_association=False")
        if metadata.get("association_stage") not in (None, "pre_association"):
            raise SnapshotContractError(
                "BLOCKED_POST_ASSOCIATION_INPUT: association_stage is not pre_association"
            )
        object.__setattr__(self, "metadata", metadata)

        boxes = _readonly_array(self.boxes_xyxy, dtype=np.float32, name="boxes_xyxy", ndim=2)
        if boxes.shape[1:] != (4,):
            raise SnapshotContractError(f"boxes_xyxy must be [N,4], got {boxes.shape}")
        scores = _readonly_array(self.det_scores, dtype=np.float32, name="det_scores", ndim=1)
        labels = _readonly_array(self.labels, dtype=np.int64, name="labels", ndim=1)
        embeddings = _readonly_array(self.embeddings, dtype=np.float32, name="embeddings", ndim=2)
        if not (len(boxes) == len(scores) == len(labels) == len(embeddings)):
            raise SnapshotContractError("observation arrays must have equal length")
        uids = tuple(str(value) for value in self.observation_uids)
        if len(uids) != len(boxes) or len(set(uids)) != len(uids) or any(not value for value in uids):
            raise SnapshotContractError("observation_uids must be non-empty and unique [N]")
        object.__setattr__(self, "boxes_xyxy", boxes)
        object.__setattr__(self, "det_scores", scores)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "observation_uids", uids)

        memory_ids = tuple(int(value) for value in self.memory_ids)
        if len(set(memory_ids)) != len(memory_ids):
            raise SnapshotContractError("memory_ids must be unique")
        object.__setattr__(self, "memory_ids", memory_ids)
        memory_count = len(memory_ids)
        affinity = _readonly_array(self.native_affinity, dtype=np.float32, name="native_affinity", ndim=2)
        if affinity.shape != (len(boxes), memory_count):
            raise SnapshotContractError(
                f"native_affinity must be [N,M]={len(boxes), memory_count}, got {affinity.shape}"
            )
        object.__setattr__(self, "native_affinity", affinity)

        if self.memory_embeddings is None:
            memory_embeddings = None
        else:
            memory_embeddings = _readonly_array(
                self.memory_embeddings, dtype=np.float32, name="memory_embeddings", ndim=2
            )
            if memory_embeddings.shape != (memory_count, embeddings.shape[1]):
                raise SnapshotContractError(
                    "memory_embeddings must have shape [M,D] matching memory_ids and embeddings"
                )
        object.__setattr__(self, "memory_embeddings", memory_embeddings)

        if self.memory_last_frame is None:
            if memory_count:
                raise SnapshotContractError("memory_last_frame is required when memory_ids are present")
            memory_last_frame = np.zeros((0,), dtype=np.int64)
        else:
            memory_last_frame = _readonly_array(
                self.memory_last_frame, dtype=np.int64, name="memory_last_frame", ndim=1
            )
            if len(memory_last_frame) != memory_count:
                raise SnapshotContractError("memory_last_frame must be [M]")
        if np.any(memory_last_frame >= int(self.frame_id)):
            raise SnapshotContractError(
                "BLOCKED_POST_ASSOCIATION_INPUT: memory contains a non-causal current/future frame"
            )
        object.__setattr__(self, "memory_last_frame", memory_last_frame)

    @property
    def observation_count(self) -> int:
        return int(len(self.boxes_xyxy))

    @property
    def feature_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def immutable_observation_hash(self) -> str:
        """Hash only detector-visible fields; track decisions are excluded."""
        digest = hashlib.sha256()
        for array in (self.boxes_xyxy, self.det_scores, self.labels, self.embeddings):
            digest.update(np.ascontiguousarray(array).tobytes(order="C"))
        digest.update(json.dumps(self.observation_uids, separators=(",", ":")).encode())
        return digest.hexdigest()
