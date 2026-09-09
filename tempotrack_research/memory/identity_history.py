"""Shared finite identity histories for Paper EMD and streaming recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..data.feature_export import iter_manifest_ledgers, load_dataset_manifest


@dataclass
class MemoryObservation:
    feature_raw: np.ndarray
    feature_norm: np.ndarray
    bbox_xyxy: np.ndarray
    frame_id: int
    image_width: int
    image_height: int
    aspect_ratio: float
    area_pixels: float
    area_norm: float
    det_score: float
    association_score: float | None
    association_margin: float | None
    fast_score: float | None
    slow_score: float | None
    is_birth: bool


@dataclass
class IdentityHistory:
    local_track_id: int
    video_id: int
    first_frame: int
    last_frame: int
    observations: list[MemoryObservation] = field(default_factory=list)


class IdentityHistoryStore:
    def __init__(self, capacity: int = 64, dedup_cosine: float = 0.95, dedup_scope: str = "all") -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0.0 <= dedup_cosine <= 1.0:
            raise ValueError("dedup_cosine must be in [0,1]")
        if dedup_scope not in {"all"}:
            raise ValueError("only dedup_scope='all' is supported by the paper contract")
        self.capacity = int(capacity)
        self.dedup_cosine = float(dedup_cosine)
        self.dedup_scope = dedup_scope
        self._histories: dict[int, IdentityHistory] = {}

    def add(self, track_id: int, observation: MemoryObservation) -> bool:
        history = self._histories.get(int(track_id))
        if history is None:
            history = IdentityHistory(
                local_track_id=int(track_id),
                video_id=-1,
                first_frame=int(observation.frame_id),
                last_frame=int(observation.frame_id),
                observations=[],
            )
            self._histories[int(track_id)] = history
        feature = np.asarray(observation.feature_norm, dtype=np.float32).reshape(-1)
        for previous in history.observations:
            if float(np.dot(feature, previous.feature_norm)) > self.dedup_cosine:
                return False
        history.observations.append(observation)
        if len(history.observations) > self.capacity:
            del history.observations[0]
        history.first_frame = int(history.observations[0].frame_id)
        history.last_frame = int(history.observations[-1].frame_id)
        return True

    def get(self, track_id: int) -> IdentityHistory:
        return self._histories[int(track_id)]

    def histories(self) -> list[IdentityHistory]:
        return list(self._histories.values())


def _history_observation(
    *,
    video_id: int,
    frame_id: int,
    box: np.ndarray,
    width: int,
    height: int,
    feature: np.ndarray,
    score: float,
    association_score: float | None,
    association_margin: float | None,
    fast_score: float | None,
    slow_score: float | None,
    is_birth: bool,
) -> MemoryObservation:
    raw = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = raw / max(float(np.linalg.norm(raw)), 1e-12)
    box = np.asarray(box, dtype=np.float32).reshape(4)
    bw = max(float(box[2] - box[0]), 1e-6)
    bh = max(float(box[3] - box[1]), 1e-6)
    area = bw * bh
    area_norm = area / max(float(width) * float(height), 1.0)
    return MemoryObservation(
        feature_raw=raw,
        feature_norm=norm.astype(np.float32),
        bbox_xyxy=box,
        frame_id=int(frame_id),
        image_width=int(width),
        image_height=int(height),
        aspect_ratio=float(bw / bh),
        area_pixels=float(area),
        area_norm=float(area_norm),
        det_score=float(score),
        association_score=None if association_score is None else float(association_score),
        association_margin=None if association_margin is None else float(association_margin),
        fast_score=None if fast_score is None else float(fast_score),
        slow_score=None if slow_score is None else float(slow_score),
        is_birth=bool(is_birth),
    )


def _prediction_records(prediction: Any) -> dict[str, int]:
    if isinstance(prediction, (str, Path)):
        prediction = __import__("json").loads(Path(prediction).read_text(encoding="utf-8"))
    if isinstance(prediction, Mapping):
        prediction = prediction.get("records", prediction.get("predictions", []))
    result: dict[str, int] = {}
    for item in prediction or []:
        if "observation_uid" in item and "track_id" in item:
            result[str(item["observation_uid"])] = int(item["track_id"])
    return result


def build_identity_histories(
    observation_manifest: str | Path | Mapping[str, Any],
    track_id_mapping: Any,
    *,
    capacity: int,
    dedup_cosine: float,
) -> dict[tuple[int, int], IdentityHistory]:
    """Build per-video histories from immutable cache rows and local IDs."""

    manifest = load_dataset_manifest(observation_manifest) if isinstance(observation_manifest, (str, Path)) else dict(observation_manifest)
    mapping = _prediction_records(track_id_mapping)
    output: dict[tuple[int, int], IdentityHistory] = {}
    stores: dict[int, IdentityHistoryStore] = {}
    seen: dict[tuple[int, int], int] = {}
    for video_id, ledger in iter_manifest_ledgers(manifest):
        store = stores.setdefault(int(video_id), IdentityHistoryStore(capacity, dedup_cosine, "all"))
        keys = ledger.keys()
        for row, key in enumerate(keys):
            uid = str(key.uid)
            if uid not in mapping:
                raise ValueError(f"frontend mapping misses native observation UID {uid}")
            local_id = int(mapping[uid])
            arrays = ledger.arrays
            observation = _history_observation(
                video_id=int(video_id),
                frame_id=int(arrays["frame_indices"][row]),
                box=arrays["bboxes_xyxy"][row],
                width=int(arrays["image_widths"][row]),
                height=int(arrays["image_heights"][row]),
                feature=arrays["appearance"][row],
                score=float(arrays["scores"][row]),
                association_score=None,
                association_margin=float(arrays["detection_margin"][row]) if "detection_margin" in arrays and np.isfinite(arrays["detection_margin"][row]) else None,
                fast_score=float(arrays["fast_score"][row]) if "fast_score" in arrays and np.isfinite(arrays["fast_score"][row]) else None,
                slow_score=float(arrays["slow_score"][row]) if "slow_score" in arrays and np.isfinite(arrays["slow_score"][row]) else None,
                is_birth=(local_id not in seen),
            )
            # Store carries video-scoped IDs.  The tuple key avoids collisions
            # when the native tracker restarts IDs for another video.
            if local_id not in store._histories:
                store._histories[local_id] = IdentityHistory(local_id, int(video_id), observation.frame_id, observation.frame_id, [])
            store.add(local_id, observation)
            seen[(int(video_id), local_id)] = 1
        for history in store.histories():
            history.video_id = int(video_id)
            output[(int(video_id), int(history.local_track_id))] = history
    return output


__all__ = [
    "MemoryObservation",
    "IdentityHistory",
    "IdentityHistoryStore",
    "build_identity_histories",
]
