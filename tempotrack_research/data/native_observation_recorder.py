"""Native MASA observation cache writer.

The recorder sits after the official detector, feature extractor, filtering,
and association call.  It never runs a model itself.  A video is stored as a
single compressed NPZ plus a JSON sidecar so replay can bind every row to the
same native observation UID.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from torch import Tensor

from ..config import object_hash


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return _sha256_bytes(contiguous.tobytes(order="C"))


@dataclass
class NativeFrameObservation:
    video_id: int
    frame_id: int
    image_id: int
    image_height: int
    image_width: int
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    embeddings_raw: np.ndarray
    assigned_track_ids: np.ndarray
    accepted_score: np.ndarray
    detection_margin: np.ndarray
    fast_score: np.ndarray | None = None
    slow_score: np.ndarray | None = None


class NativeObservationRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        rank: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.metadata = dict(metadata or {})
        self.current_video_id: int | None = None
        self._frames: list[NativeFrameObservation] = []

    @staticmethod
    def _to_numpy(value: Tensor | np.ndarray, dtype: np.dtype) -> np.ndarray:
        if isinstance(value, Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    def append(
        self,
        *,
        video_id: int,
        frame_id: int,
        image_id: int,
        image_hw: tuple[int, int],
        bboxes: Tensor,
        labels: Tensor,
        scores: Tensor,
        embeds: Tensor,
        trace: Any,
    ) -> None:
        video_id = int(video_id)
        if self.current_video_id is None:
            self.current_video_id = video_id
        elif self.current_video_id != video_id:
            self.flush_video()
            self.current_video_id = video_id
        count = int(bboxes.shape[0])
        if trace is None:
            ids = np.full((count,), -1, dtype=np.int64)
            accepted_score = np.full((count,), np.nan, dtype=np.float32)
            margin = np.full((count,), np.nan, dtype=np.float32)
            fast = slow = None
        else:
            ids = self._to_numpy(trace.ids, np.int64).reshape(-1)
            accepted_score = self._to_numpy(trace.accepted_score, np.float32).reshape(-1)
            margin = self._to_numpy(trace.detection_margin, np.float32).reshape(-1)
            fast = None if trace.fast_score is None else self._to_numpy(trace.fast_score, np.float32)
            slow = None if trace.slow_score is None else self._to_numpy(trace.slow_score, np.float32)
            # A full dual score matrix is a debug diagnostic.  The compact
            # cache stores the per-detection best prototype score so replay
            # remains [N]-aligned and bounded in size.
            if fast is not None and fast.ndim > 1:
                fast = np.max(fast, axis=1)
            if slow is not None and slow.ndim > 1:
                slow = np.max(slow, axis=1)
        if not (len(ids) == len(accepted_score) == len(margin) == count):
            raise ValueError("association trace is not aligned with native filtered observations")
        height, width = int(image_hw[0]), int(image_hw[1])
        self._frames.append(
            NativeFrameObservation(
                video_id=video_id,
                frame_id=int(frame_id),
                image_id=int(image_id),
                image_height=height,
                image_width=width,
                boxes_xyxy=self._to_numpy(bboxes, np.float32).reshape(-1, 4),
                scores=self._to_numpy(scores, np.float32).reshape(-1),
                labels=self._to_numpy(labels, np.int64).reshape(-1),
                embeddings_raw=self._to_numpy(embeds, np.float32).reshape(count, -1),
                assigned_track_ids=ids,
                accepted_score=accepted_score,
                detection_margin=margin,
                fast_score=fast,
                slow_score=slow,
            )
        )

    def flush_video(self) -> Path | None:
        if self.current_video_id is None:
            return None
        video_id = int(self.current_video_id)
        frames = self._frames
        if not frames:
            self.current_video_id = None
            return None
        dim = int(frames[0].embeddings_raw.shape[1]) if frames else 0
        arrays: dict[str, np.ndarray] = {
            "video_ids": np.concatenate([np.full(len(f.scores), f.video_id, dtype=np.int64) for f in frames]),
            "frame_indices": np.concatenate([np.full(len(f.scores), f.frame_id, dtype=np.int64) for f in frames]),
            "image_ids": np.concatenate([np.full(len(f.scores), f.image_id, dtype=np.int64) for f in frames]),
            "image_heights": np.concatenate([np.full(len(f.scores), f.image_height, dtype=np.int32) for f in frames]),
            "image_widths": np.concatenate([np.full(len(f.scores), f.image_width, dtype=np.int32) for f in frames]),
            "boxes_xyxy": np.concatenate([f.boxes_xyxy for f in frames], axis=0).astype(np.float32, copy=False),
            "scores": np.concatenate([f.scores for f in frames]).astype(np.float32, copy=False),
            "labels": np.concatenate([f.labels for f in frames]).astype(np.int64, copy=False),
            "embeddings_raw": np.concatenate([f.embeddings_raw for f in frames], axis=0).astype(np.float32, copy=False).reshape(-1, dim),
            "assigned_track_ids": np.concatenate([f.assigned_track_ids for f in frames]).astype(np.int64, copy=False),
            "accepted_score": np.concatenate([f.accepted_score for f in frames]).astype(np.float32, copy=False),
            "detection_margin": np.concatenate([f.detection_margin for f in frames]).astype(np.float32, copy=False),
        }
        if any(f.fast_score is not None for f in frames):
            arrays["fast_score"] = np.concatenate([
                f.fast_score if f.fast_score is not None else np.full(len(f.scores), np.nan, dtype=np.float32)
                for f in frames
            ]).astype(np.float32, copy=False)
        if any(f.slow_score is not None for f in frames):
            arrays["slow_score"] = np.concatenate([
                f.slow_score if f.slow_score is not None else np.full(len(f.scores), np.nan, dtype=np.float32)
                for f in frames
            ]).astype(np.float32, copy=False)

        path = self.output_dir / f"video_{video_id}.npz"
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        os.close(fd)
        try:
            with open(name, "wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

        array_hashes = {key: _array_hash(value) for key, value in arrays.items()}
        provenance = dict(self.metadata)
        provenance.setdefault("config_hash", self.metadata.get("config_hash"))
        provenance.setdefault("checkpoint_hash", self.metadata.get("checkpoint_hash"))
        provenance.setdefault("detector_init_weight_hash", self.metadata.get("detector_init_weight_hash"))
        provenance.setdefault("annotation_hash", self.metadata.get("annotation_hash"))
        try:
            provenance.setdefault("source_code_head", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
        except Exception:
            provenance.setdefault("source_code_head", None)
        sidecar = {
            "schema_version": 1,
            "artifact": "native_masa_observation_cache",
            "rank": self.rank,
            "video_id": video_id,
            "row_count": int(len(arrays["scores"])),
            "embedding_dim": dim,
            "frame_count": len(frames),
            "provenance": provenance,
            "array_sha256": array_hashes,
            "content_hash": object_hash({"array_sha256": array_hashes, "video_id": video_id}),
        }
        sidecar_path = path.with_suffix(path.suffix + ".json")
        sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._frames = []
        self.current_video_id = None
        return path

    def close(self) -> None:
        self.flush_video()


def load_native_cache_frame(path: str | Path) -> list[NativeFrameObservation]:
    """Load one native shard and reconstruct frame-level observations."""

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    frames: list[NativeFrameObservation] = []
    frame_ids = arrays["frame_indices"]
    for frame_id in np.unique(frame_ids):
        rows = np.flatnonzero(frame_ids == frame_id)
        first = int(rows[0]) if len(rows) else 0
        frames.append(
            NativeFrameObservation(
                video_id=int(arrays["video_ids"][first]) if len(rows) else -1,
                frame_id=int(frame_id),
                image_id=int(arrays["image_ids"][first]) if len(rows) else -1,
                image_height=int(arrays["image_heights"][first]) if len(rows) else 0,
                image_width=int(arrays["image_widths"][first]) if len(rows) else 0,
                boxes_xyxy=arrays["boxes_xyxy"][rows],
                scores=arrays["scores"][rows],
                labels=arrays["labels"][rows],
                embeddings_raw=arrays["embeddings_raw"][rows],
                assigned_track_ids=arrays["assigned_track_ids"][rows],
                accepted_score=arrays["accepted_score"][rows],
                detection_margin=arrays["detection_margin"][rows],
                fast_score=arrays["fast_score"][rows] if "fast_score" in arrays else None,
                slow_score=arrays["slow_score"][rows] if "slow_score" in arrays else None,
            )
        )
    return frames
