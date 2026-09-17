"""Disk-backed COV frontend replay cache.

The cache is deliberately placed before ``OVTrackerUncertainty.match``.  It
contains detector/ROI outputs and the non-state arguments passed to the pinned
native tracker, never native affinity, memo state, tracklets, or final IDs.

All writes are opt-in through the caller.  The normal V10 runtime does not
construct a writer and therefore does not touch the filesystem.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np


REPLAY_CACHE_SCHEMA_VERSION = 1
REPLAY_CACHE_ARTIFACT = "v11_covtrack_frontend_replay_cache"
CACHED_ARRAY_FIELDS = (
    "det_bboxes",
    "det_scores",
    "det_labels",
    "track_feats",
    "cls_feats",
)
FORBIDDEN_CACHE_FIELDS = {
    "native_affinity",
    "memo_ids",
    "memo_embeds",
    "tracklets",
    "final_ids",
    "track_ids",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _array(value: Any, *, name: str, dtype: np.dtype, ndim: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    current = value.detach() if hasattr(value, "detach") else value
    current = current.cpu() if hasattr(current, "cpu") else current
    result = np.asarray(current, dtype=dtype)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(result.copy())


def _normalise_bboxes(value: Any) -> np.ndarray:
    result = _array(value, name="det_bboxes", dtype=np.float32, ndim=2)
    if result is None:
        raise ValueError("det_bboxes cannot be None")
    if result.shape[1:] != (5,):
        raise ValueError(f"det_bboxes must be [N,5], got {result.shape}")
    return result


def _normalise_labels(value: Any, count: int) -> np.ndarray:
    result = _array(value, name="det_labels", dtype=np.int64, ndim=1)
    if result is None or len(result) != count:
        raise ValueError("det_labels must align with det_bboxes")
    return result


def _field_descriptor(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


class _VideoWriter:
    def __init__(self, root: Path, video_id: int | str) -> None:
        self.video_id = int(video_id) if str(video_id).lstrip("-").isdigit() else str(video_id)
        video_name = str(self.video_id)
        if not video_name or video_name in {".", ".."} or "/" in video_name or "\\" in video_name:
            raise ValueError(f"unsafe video_id for replay cache: {video_id!r}")
        self.directory = root / "videos" / video_name
        self.array_directory = self.directory / "arrays"
        self.array_directory.mkdir(parents=True, exist_ok=True)
        self.frames_path = self.directory / "frames.jsonl"
        self.handle = self.frames_path.open("x", encoding="utf-8")
        self.frame_count = 0
        self.frame_ids: list[int] = []
        self.image_ids: list[int | None] = []

    def capture(
        self,
        *,
        frame_id: int,
        image_id: int | None,
        filename: str,
        method: str,
        det_bboxes: Any,
        det_labels: Any,
        track_feats: Any,
        cls_feats: Any,
    ) -> None:
        bboxes = _normalise_bboxes(det_bboxes)
        labels = _normalise_labels(det_labels, len(bboxes))
        scores = np.ascontiguousarray(bboxes[:, 4].copy())
        tracks = _array(track_feats, name="track_feats", dtype=np.float32, ndim=2)
        classes = _array(cls_feats, name="cls_feats", dtype=np.float32, ndim=2)
        for name, value in (("track_feats", tracks), ("cls_feats", classes)):
            if value is not None and len(value) != len(bboxes):
                raise ValueError(f"{name} must align with det_bboxes")
        if tracks is not None and classes is None:
            raise ValueError("cls_feats must be present when track_feats is present")

        ordinal = self.frame_count
        array_name = f"frame_{ordinal:06d}.npz"
        array_path = self.array_directory / array_name
        if array_path.exists():
            raise FileExistsError(f"refusing to overwrite replay cache frame: {array_path}")
        arrays = {
            "det_bboxes": bboxes,
            "det_scores": scores,
            "det_labels": labels,
        }
        if tracks is not None:
            arrays["track_feats"] = tracks
        if classes is not None:
            arrays["cls_feats"] = classes
        # A file handle avoids numpy's automatic suffix handling for the
        # temporary filename and keeps the final replace atomic.
        temporary = array_path.with_name(f".{array_name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, array_path)

        frame = {
            "ordinal": ordinal,
            "video_id": self.video_id,
            "frame_id": int(frame_id),
            "image_id": None if image_id is None else int(image_id),
            "filename": str(filename),
            "method": str(method),
            "match_called": tracks is not None,
            "array_file": str(Path("arrays") / array_name),
            "array_sha256": sha256_file(array_path),
            "fields": {name: _field_descriptor(value) for name, value in arrays.items()},
        }
        self.handle.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.handle.flush()
        self.frame_count += 1
        self.frame_ids.append(int(frame_id))
        self.image_ids.append(None if image_id is None else int(image_id))

    def close(self) -> dict[str, Any]:
        if not self.handle.closed:
            self.handle.flush()
            self.handle.close()
        return {
            "video_id": self.video_id,
            "path": str(Path("videos") / str(self.video_id)),
            "frame_count": self.frame_count,
            "frame_ids": list(self.frame_ids),
            "image_ids": list(self.image_ids),
            "frames_jsonl_sha256": sha256_file(self.frames_path),
        }


class FrontendReplayCacheWriter:
    """Write one process-owned, complete-video frontend cache."""

    def __init__(self, root: str | Path, *, provenance: Mapping[str, Any] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_absolute():
            raise ValueError("replay cache root must be absolute")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "videos").mkdir(parents=True, exist_ok=True)
        existing = self.root / "manifest.json"
        if existing.is_file():
            try:
                current = json.loads(existing.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid existing replay cache manifest: {existing}") from exc
            if current.get("status") in {"PASS", "COMPLETED", "RUNNING"}:
                raise FileExistsError(f"replay cache root is already in use: {self.root}")
        self.provenance = dict(provenance or {})
        self.started_at_unix = time.time()
        self._videos: OrderedDict[str, _VideoWriter] = OrderedDict()
        self._closed_summaries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._last_video_key: str | None = None
        self._status = "RUNNING"

    @property
    def video_count(self) -> int:
        return len(self._closed_summaries) + len(self._videos)

    @property
    def frame_count(self) -> int:
        return sum(item.frame_count for item in self._videos.values()) + sum(
            int(item["frame_count"]) for item in self._closed_summaries.values()
        )

    def capture(self, *, video_id: int | str, **kwargs: Any) -> None:
        key = str(video_id)
        if any(token in kwargs for token in FORBIDDEN_CACHE_FIELDS):
            raise ValueError("frontend replay cache cannot receive native state or IDs")
        if self._last_video_key is not None and key != self._last_video_key and key in self._closed_summaries:
            raise ValueError(f"video {video_id} reappeared after a complete cache segment")
        writer = self._videos.get(key)
        if writer is None:
            writer = _VideoWriter(self.root, video_id)
            self._videos[key] = writer
        self._last_video_key = key
        writer.capture(**kwargs)

    def _close_open_videos(self) -> None:
        for key, writer in list(self._videos.items()):
            self._closed_summaries[key] = writer.close()
        self._videos.clear()

    def _manifest(self, status: str) -> dict[str, Any]:
        videos = list(self._closed_summaries.values())
        # These fields are a hard provenance contract for Official-Train
        # supervision.  The frontend writer itself never receives GT; the
        # values only describe the downstream artifact boundary and therefore
        # remain auditable in both shard and merged manifests.
        data_contract = {
            "input_source": self.provenance.get("input_source", "UNKNOWN"),
            "supervision_source": self.provenance.get("supervision_source", "UNKNOWN"),
            "oracle_features_used": bool(self.provenance.get("oracle_features_used", True)),
            "gt_boxes_used_as_model_input": bool(self.provenance.get("gt_boxes_used_as_model_input", True)),
            "gt_tracks_used_as_memory": bool(self.provenance.get("gt_tracks_used_as_memory", True)),
            "gt_used_only_for_supervision": bool(self.provenance.get("gt_used_only_for_supervision", False)),
        }
        return {
            "schema_version": REPLAY_CACHE_SCHEMA_VERSION,
            "artifact": REPLAY_CACHE_ARTIFACT,
            "status": status,
            "started_at_unix": self.started_at_unix,
            "ended_at_unix": time.time(),
            "cache_boundary": "before_pinned_covtrack_tracker_match",
            "cached_fields": list(CACHED_ARRAY_FIELDS) + [
                "video_id", "frame_id", "image_id", "filename", "method", "match_called"
            ],
            "forbidden_cached_state": sorted(FORBIDDEN_CACHE_FIELDS),
            "frame_count": sum(int(item["frame_count"]) for item in videos),
            "video_count": len(videos),
            "ordered_video_ids": [item["video_id"] for item in videos],
            "videos": videos,
            "provenance": _jsonable(self.provenance),
            **data_contract,
        }

    def finalize(self, status: str = "COMPLETED") -> Path:
        if self._status not in {"RUNNING", "PROCESS_EXIT"}:
            return self.root / "manifest.json"
        self._close_open_videos()
        self._status = str(status)
        manifest = self._manifest(self._status)
        temporary = self.root / f".manifest.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.root / "manifest.json")
        # Keep an explicitly named contract receipt next to the reader's
        # historical manifest.  It is intentionally identical, so no second
        # mutable source of truth is introduced.
        contract_path = self.root / "cache_manifest.json"
        contract_temporary = self.root / f".cache_manifest.{os.getpid()}.tmp"
        contract_temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(contract_temporary, contract_path)
        return self.root / "manifest.json"


class FrontendReplayCacheReader:
    """Stream a verified cache without loading all frames into RAM."""

    def __init__(self, root: str | Path, *, verify_hashes: bool = True) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("artifact") != REPLAY_CACHE_ARTIFACT:
            raise ValueError("replay cache artifact type mismatch")
        if int(self.manifest.get("schema_version", -1)) != REPLAY_CACHE_SCHEMA_VERSION:
            raise ValueError("replay cache schema mismatch")
        if self.manifest.get("status") not in {"PASS", "COMPLETED"}:
            raise ValueError(f"replay cache is not complete: {self.manifest.get('status')}")
        forbidden = set(self.manifest.get("forbidden_cached_state", ()))
        if forbidden != FORBIDDEN_CACHE_FIELDS:
            raise ValueError("replay cache forbidden-state contract mismatch")
        self.verify_hashes = bool(verify_hashes)

    def videos(self) -> Iterable[tuple[int | str, Path, dict[str, Any]]]:
        by_id = {str(item["video_id"]): item for item in self.manifest.get("videos", ())}
        for raw_video_id in self.manifest.get("ordered_video_ids", ()):
            item = by_id.get(str(raw_video_id))
            if item is None:
                raise ValueError(f"manifest video order references missing video: {raw_video_id}")
            path = self.root / str(item["path"])
            if not path.is_dir():
                raise FileNotFoundError(path)
            yield raw_video_id, path, item

    def frames(self, video_path: Path) -> Iterable[dict[str, Any]]:
        frames_path = video_path / "frames.jsonl"
        if self.verify_hashes:
            summary = next(
                item for item in self.manifest["videos"] if str(self.root / item["path"]) == str(video_path)
            )
            if sha256_file(frames_path) != summary["frames_jsonl_sha256"]:
                raise ValueError(f"replay cache frame manifest hash mismatch: {frames_path}")
        with frames_path.open(encoding="utf-8") as handle:
            expected_ordinal = 0
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record.get("ordinal", -1)) != expected_ordinal:
                    raise ValueError(f"replay cache frame order mismatch in {frames_path}")
                expected_ordinal += 1
                yield record

    def load_arrays(self, video_path: Path, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
        array_path = video_path / str(record["array_file"])
        if self.verify_hashes and sha256_file(array_path) != str(record["array_sha256"]):
            raise ValueError(f"replay cache array hash mismatch: {array_path}")
        with np.load(array_path, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
        expected_fields = set(record.get("fields", {}))
        if set(arrays) != expected_fields:
            raise ValueError(f"replay cache fields mismatch at {array_path}")
        for name, descriptor in record.get("fields", {}).items():
            if list(arrays[name].shape) != list(descriptor["shape"]) or str(arrays[name].dtype) != descriptor["dtype"]:
                raise ValueError(f"replay cache array descriptor mismatch at {array_path}:{name}")
        return arrays


__all__ = [
    "CACHED_ARRAY_FIELDS",
    "FORBIDDEN_CACHE_FIELDS",
    "FrontendReplayCacheReader",
    "FrontendReplayCacheWriter",
    "REPLAY_CACHE_ARTIFACT",
    "REPLAY_CACHE_SCHEMA_VERSION",
    "sha256_file",
]
