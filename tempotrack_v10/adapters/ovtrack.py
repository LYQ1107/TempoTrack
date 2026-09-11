"""OVTrack's thin pre-association adapter for the shared V10 core.

This module intentionally contains no matching, memory, or reranking logic.
It converts OVTrack's already-finalized tensors/state into the strict shared
``PreAssociationSnapshot`` and converts an overlay proposal into the native
tracker's pre-assigned-ID sentinel form.  The pinned OVTrack source remains
external and is not modified by this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..contract import PreAssociationSnapshot, SnapshotContractError
from ..overlay import OverlayProposal, TempoTrackConfig, TempoTrackOverlay


def _numpy(value: Any, *, dtype: np.dtype) -> np.ndarray:
    """Copy torch/NumPy-like values to a CPU NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).copy()


@dataclass(frozen=True)
class OVTrackTempoConfig:
    """Adapter-level config, including the required external core config path."""

    enabled: bool = False
    config_path: str = "configs/research/v10/tempo.yaml"
    overlay: TempoTrackConfig = TempoTrackConfig(enabled=False)


def _tempo_mapping(path: str | Path) -> tuple[dict[str, Any], Path]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - only used in a missing env
        raise RuntimeError("OVTrack tempo config requires PyYAML") from exc
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"tempo config must be a mapping: {config_path}")
    return dict(raw), config_path


def load_ovtrack_tempo_config(path: str | Path) -> OVTrackTempoConfig:
    """Load ``tempo.enabled`` and ``tempo.config_path`` without inventing defaults.

    The frontend file contains a ``tempo`` mapping and points to a second
    core-parameter YAML.  Values in the frontend mapping override values from
    the pointed file, so a run can disable the adapter without changing the
    shared core file.
    """

    frontend, frontend_path = _tempo_mapping(path)
    tempo = frontend.get("tempo", frontend)
    if not isinstance(tempo, Mapping):
        raise ValueError(f"tempo must be a mapping: {frontend_path}")
    tempo = dict(tempo)
    config_ref = tempo.get("config_path")
    if config_ref is None:
        raise ValueError("OVTrack config must provide tempo.config_path")
    core_path = Path(str(config_ref))
    if not core_path.is_absolute():
        core_path = (frontend_path.parent / core_path).resolve()
    core, _ = _tempo_mapping(core_path)
    merged = dict(core.get("tempo", core) if isinstance(core.get("tempo", core), Mapping) else {})
    merged.update({key: value for key, value in tempo.items() if key != "config_path"})

    allowed = {
        "enabled",
        "alpha_fast",
        "alpha_slow",
        "min_gap",
        "max_gap",
        "candidate_top_k",
        "top_r",
        "native_weight",
        "active_weight",
        "support_weight",
        "gap_penalty",
        "score_threshold",
        "margin_threshold",
        "reliability_weight",
        "memory_capacity",
    }
    overlay_kwargs = {key: value for key, value in merged.items() if key in allowed}
    overlay_kwargs["enabled"] = bool(merged.get("enabled", False))
    overlay = TempoTrackConfig(**overlay_kwargs)
    return OVTrackTempoConfig(
        enabled=bool(merged.get("enabled", False)),
        config_path=str(core_path),
        overlay=overlay,
    )


class OVTrackTempoAdapter:
    """Convert OVTrack pre-association state to/from the shared overlay.

    ``build_snapshot`` is intended to be called after OVTrack's line-200
    final native affinity and before its line-216 ID initialization/greedy
    assignment.  The caller then uses ``preassigned_ids`` before native
    bookkeeping and finally calls ``commit`` with the actual final IDs.
    """

    def __init__(
        self,
        config: OVTrackTempoConfig | None = None,
        *,
        overlay: TempoTrackOverlay | None = None,
    ) -> None:
        self.config = config or OVTrackTempoConfig()
        if overlay is not None and overlay.config != self.config.overlay:
            raise ValueError("overlay config does not match OVTrack adapter config")
        self.overlay = overlay or TempoTrackOverlay(self.config.overlay)

    @classmethod
    def from_config(cls, path: str | Path) -> "OVTrackTempoAdapter":
        return cls(load_ovtrack_tempo_config(path))

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.config.overlay.enabled)

    def reset(self, video_id: int | str | None = None) -> None:
        self.overlay.reset(video_id)

    @staticmethod
    def _memo_from_tracker(
        tracker: Any,
        *,
        explicit_ids: Sequence[int] | None,
        explicit_embeddings: Any | None,
        explicit_last_frame: Any | None,
    ) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
        """Read only the tracker's existing memo, never committed current IDs."""

        if tracker is not None:
            tracklets = getattr(tracker, "tracklets", {})
            if not bool(getattr(tracker, "empty", not bool(tracklets))):
                _, _, memo_embeddings, _, memo_ids = tracker.memo
                ids = tuple(int(value) for value in _numpy(memo_ids, dtype=np.int64).reshape(-1).tolist())
                embeddings = _numpy(memo_embeddings, dtype=np.float32).reshape(len(ids), -1)
                last_frame = []
                for memory_id in ids:
                    record = tracklets.get(memory_id)
                    frames = None if record is None else record.get("frame_ids")
                    if not frames:
                        raise SnapshotContractError(
                            "OVTrack memo entry lacks causal frame history"
                        )
                    last_frame.append(int(frames[-1]))
                return ids, embeddings, np.asarray(last_frame, dtype=np.int64)
            return tuple(), np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        ids = tuple(int(value) for value in (explicit_ids or ()))
        if explicit_embeddings is None:
            if ids:
                raise SnapshotContractError("memory_embeddings are required with explicit memory_ids")
            embeddings = np.zeros((0, 0), dtype=np.float32)
        else:
            embeddings = _numpy(explicit_embeddings, dtype=np.float32).reshape(len(ids), -1)
        if explicit_last_frame is None:
            if ids:
                raise SnapshotContractError("memory_last_frame is required with explicit memory_ids")
            last_frame = np.zeros((0,), dtype=np.int64)
        else:
            last_frame = _numpy(explicit_last_frame, dtype=np.int64).reshape(-1)
        if len(last_frame) != len(ids):
            raise SnapshotContractError("memory_last_frame must align with memory_ids")
        return ids, embeddings, last_frame

    def build_snapshot(
        self,
        *,
        video_id: int | str,
        frame_id: int,
        bboxes: Any,
        labels: Any,
        embeddings: Any,
        native_affinity: Any,
        tracker: Any | None = None,
        observation_uids: Sequence[str] | None = None,
        source_indices: Sequence[int] | None = None,
        memory_ids: Sequence[int] | None = None,
        memory_embeddings: Any | None = None,
        memory_last_frame: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        final_ids: Any | None = None,
    ) -> PreAssociationSnapshot:
        """Build a strict snapshot from OVTrack's pre-ID tensors/state.

        ``final_ids`` exists only as a hard safety tripwire.  Passing it means
        the caller is attempting to feed post-association state into the
        pre-association core and is rejected before the snapshot is built.
        """

        if final_ids is not None:
            raise SnapshotContractError(
                "BLOCKED_POST_ASSOCIATION_INPUT: OVTrack final IDs supplied to build_snapshot"
            )
        supplied_metadata = dict(metadata or {})
        unsafe_keys = {
            "ids",
            "native_ids",
            "assigned_ids",
            "assigned_track_ids",
            "final_ids",
            "track_ids",
        }
        if any(str(key).lower() in unsafe_keys for key in supplied_metadata):
            raise SnapshotContractError(
                "BLOCKED_POST_ASSOCIATION_INPUT: committed IDs in OVTrack metadata"
            )
        supplied_metadata.setdefault("association_stage", "pre_association")
        supplied_metadata.setdefault("native_method", "ovtrack-teta")

        raw_boxes = _numpy(bboxes, dtype=np.float32)
        if raw_boxes.ndim != 2 or raw_boxes.shape[1] not in (4, 5):
            raise SnapshotContractError("OVTrack bboxes must be [N,4] or [N,5] xyxy+score")
        if raw_boxes.shape[1] == 5:
            boxes = raw_boxes[:, :4]
            scores = raw_boxes[:, 4]
        else:
            boxes = raw_boxes
            if "det_scores" not in supplied_metadata:
                raise SnapshotContractError("OVTrack [N,4] boxes require det_scores metadata")
            scores = _numpy(supplied_metadata.pop("det_scores"), dtype=np.float32).reshape(-1)
        labels_array = _numpy(labels, dtype=np.int64).reshape(-1)
        embedding_array = _numpy(embeddings, dtype=np.float32)
        if embedding_array.ndim != 2:
            raise SnapshotContractError("OVTrack embeddings must be [N,D]")
        if not (len(boxes) == len(scores) == len(labels_array) == len(embedding_array)):
            raise SnapshotContractError("OVTrack observation fields have inconsistent row counts")

        if source_indices is None:
            rows = np.arange(len(boxes), dtype=np.int64)
        else:
            rows = _numpy(source_indices, dtype=np.int64).reshape(-1)
            if len(rows) != len(boxes):
                raise SnapshotContractError("source_indices must align with post-distractor rows")
        if observation_uids is None:
            uids = tuple(f"{video_id}:{int(frame_id)}:{int(row)}" for row in rows.tolist())
        else:
            uids = tuple(str(value) for value in observation_uids)

        ids, memo_embeddings, memo_last = self._memo_from_tracker(
            tracker,
            explicit_ids=memory_ids,
            explicit_embeddings=memory_embeddings,
            explicit_last_frame=memory_last_frame,
        )
        affinity = _numpy(native_affinity, dtype=np.float32)
        if affinity.ndim != 2 or affinity.shape != (len(boxes), len(ids)):
            raise SnapshotContractError(
                f"OVTrack native_affinity must be [{len(boxes)},{len(ids)}], got {affinity.shape}"
            )
        if len(ids) and memo_embeddings.shape[1] != embedding_array.shape[1]:
            raise SnapshotContractError("OVTrack memo/current embedding dimensions differ")
        supplied_metadata["post_distractor_subset"] = True
        supplied_metadata["native_row_indices"] = tuple(int(value) for value in rows.tolist())
        return PreAssociationSnapshot(
            video_id=video_id,
            frame_id=int(frame_id),
            boxes_xyxy=boxes,
            det_scores=scores,
            labels=labels_array,
            observation_uids=uids,
            embeddings=embedding_array,
            native_affinity=affinity,
            memory_ids=ids,
            memory_embeddings=memo_embeddings if len(ids) else None,
            memory_last_frame=memo_last,
            metadata=supplied_metadata,
        )

    def propose(self, snapshot: PreAssociationSnapshot) -> OverlayProposal:
        return self.overlay.propose(snapshot)

    @staticmethod
    def preassigned_ids(proposal: OverlayProposal, observation_count: int) -> np.ndarray:
        """Map accepted overlay assignments to OVTrack's pre-ID vector.

        The returned vector is meant to be applied before ``init_tracklets``
        and ``update_memo``.  Unaccepted observations stay at OVTrack's NEW
        sentinel ``-1``; the adapter never allocates IDs or updates memo.
        """

        if len(proposal.assignments) != int(observation_count):
            raise SnapshotContractError("overlay proposal is not aligned with OVTrack observations")
        output = np.full((int(observation_count),), -1, dtype=np.int64)
        for index, (accepted, assignment) in enumerate(zip(proposal.accepted, proposal.assignments)):
            if accepted:
                if assignment is None or int(assignment) < 0:
                    raise SnapshotContractError("accepted OVTrack proposal lacks an existing memory ID")
                output[index] = int(assignment)
        return output

    def commit(self, snapshot: PreAssociationSnapshot, final_ids: Any) -> np.ndarray:
        """Commit the actual native final IDs after OVTrack bookkeeping."""

        ids = _numpy(final_ids, dtype=np.int64).reshape(-1)
        self.overlay.commit(snapshot, ids)
        return ids


__all__ = ["OVTrackTempoAdapter", "OVTrackTempoConfig", "load_ovtrack_tempo_config"]
