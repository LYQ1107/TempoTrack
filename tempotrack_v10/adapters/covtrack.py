"""Thin COVTrack bridge at the native pre-association boundary.

The caller must invoke :meth:`COVTrackTempoAdapter.prepare` after COVTrack has
completed ``confused_features``/MCF fusion and finalized its native affinity,
but before COVTrack initializes or commits ``ids``.  The adapter owns no
detector, cue, fusion, embedding, or native memo implementation.

The external COVTrack checkout is intentionally not imported here.  This
keeps the adapter testable in the TempoTrack repository and lets the eventual
frontend patch pass the exact COV tensors through without changing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..contract import PreAssociationSnapshot, SnapshotContractError
from ..overlay import OverlayProposal, TempoTrackConfig, TempoTrackOverlay


def _numpy(value: Any, *, dtype: np.dtype) -> np.ndarray:
    """Materialize a CPU copy from numpy or a torch-like tensor."""

    current = value.detach() if hasattr(value, "detach") else value
    current = current.cpu() if hasattr(current, "cpu") else current
    return np.asarray(current, dtype=dtype)


@dataclass(frozen=True)
class COVTrackAssociationDecision:
    """Prepared proposal and native-ID seed for one COVTrack frame.

    ``native_id_seed`` contains existing memo IDs accepted by the overlay and
    ``-1`` for rejected/unchanged observations.  It is a proposal only; the
    COVTrack caller remains responsible for its final ID allocation and memo
    bookkeeping, then calls ``commit_after_native_ids`` with the actual IDs.
    """

    snapshot: PreAssociationSnapshot
    proposal: OverlayProposal
    native_id_seed: np.ndarray

    def __post_init__(self) -> None:
        seed = np.asarray(self.native_id_seed, dtype=np.int64).reshape(-1).copy()
        if len(seed) != self.snapshot.observation_count:
            raise SnapshotContractError("native_id_seed must be aligned with COV observations")
        seed.setflags(write=False)
        object.__setattr__(self, "native_id_seed", seed)


class COVTrackTempoAdapter:
    """Convert COVTrack's final native match state to the shared core contract.

    The adapter accepts the tensors that exist at the locator in
    ``OVTrackerUncertainty.match``:

    ``bboxes`` is COVTrack's post-filter ``[N,5]`` ``xyxy+score`` tensor,
    ``embeds`` is the already-fused COV association feature, and ``scores`` is
    the completed native bisoftmax/cosine affinity ``[N,M]``.  No value is
    recomputed or written back to those tensors.
    """

    def __init__(
        self,
        *,
        overlay: TempoTrackOverlay | None = None,
        config: TempoTrackConfig | None = None,
    ) -> None:
        if overlay is not None and config is not None:
            raise ValueError("provide overlay or config, not both")
        self.overlay = overlay or TempoTrackOverlay(config)

    @property
    def enabled(self) -> bool:
        return bool(self.overlay.config.enabled)

    def reset(self, video_id: int | str | None = None) -> None:
        """Reset only the adapter's shared-overlay state."""

        self.overlay.reset(video_id)

    def build_snapshot(
        self,
        *,
        video_id: int | str,
        frame_id: int,
        bboxes: Any,
        labels: Any,
        embeds: Any,
        scores: Any,
        memo_ids: Sequence[int] = (),
        memo_embeds: Any | None = None,
        memo_last_frame: Any | None = None,
        observation_uids: Sequence[str] | None = None,
        native_cls_embeds: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PreAssociationSnapshot:
        """Build an immutable snapshot at COVTrack's exact hook.

        ``memo_last_frame`` is required whenever ``memo_ids`` is non-empty.
        COVTrack's ``tracklets`` records provide that causal value; accepting a
        guessed timestamp would make the overlay unsafe, so the adapter fails
        closed instead.
        """

        bbox_array = _numpy(bboxes, dtype=np.float32)
        if bbox_array.ndim != 2 or bbox_array.shape[1] != 5:
            raise SnapshotContractError("COVTrack bboxes must be post-filter [N,5] xyxy+score")
        label_array = _numpy(labels, dtype=np.int64).reshape(-1)
        embed_array = _numpy(embeds, dtype=np.float32)
        affinity_array = _numpy(scores, dtype=np.float32)
        if label_array.shape[0] != bbox_array.shape[0]:
            raise SnapshotContractError("COVTrack labels must align with bboxes")
        if embed_array.ndim != 2 or embed_array.shape[0] != bbox_array.shape[0]:
            raise SnapshotContractError("COVTrack embeds must be [N,D] and align with bboxes")
        if native_cls_embeds is not None:
            cls_array = _numpy(native_cls_embeds, dtype=np.float32)
            if cls_array.ndim < 1 or cls_array.shape[0] != bbox_array.shape[0]:
                raise SnapshotContractError("COVTrack cls_embeds must align with bboxes")

        memory_id_tuple = tuple(int(value) for value in memo_ids)
        if memory_id_tuple and memo_embeds is None:
            raise SnapshotContractError("COVTrack memo_embeds is required for non-empty memo_ids")
        if memory_id_tuple and memo_last_frame is None:
            raise SnapshotContractError("COVTrack memo_last_frame is required for causal replay")

        if observation_uids is None:
            uid_tuple = tuple(f"{video_id}:{int(frame_id)}:{index}" for index in range(len(bbox_array)))
        else:
            uid_tuple = tuple(str(value) for value in observation_uids)

        snapshot_metadata = dict(metadata or {})
        snapshot_metadata.setdefault("frontend", "covtrack")
        snapshot_metadata.setdefault("association_stage", "pre_association")
        snapshot_metadata.setdefault("native_affinity_stage", "post_mcf_pre_id_commit")

        return PreAssociationSnapshot(
            video_id=video_id,
            frame_id=int(frame_id),
            boxes_xyxy=bbox_array[:, :4],
            det_scores=bbox_array[:, 4],
            labels=label_array,
            observation_uids=uid_tuple,
            embeddings=embed_array,
            native_affinity=affinity_array,
            memory_ids=memory_id_tuple,
            memory_embeddings=None if memo_embeds is None else _numpy(memo_embeds, dtype=np.float32),
            memory_last_frame=None
            if memo_last_frame is None
            else _numpy(memo_last_frame, dtype=np.int64).reshape(-1),
            metadata=snapshot_metadata,
        )

    def prepare(self, **kwargs: Any) -> COVTrackAssociationDecision:
        """Propose at the pre-ID hook and return a native-ID seed."""

        snapshot = self.build_snapshot(**kwargs)
        proposal = self.overlay.propose(snapshot)
        memory_index = set(snapshot.memory_ids)
        seed = np.full((snapshot.observation_count,), -1, dtype=np.int64)
        for index, assignment in enumerate(proposal.assignments):
            if assignment is None:
                continue
            assigned_id = int(assignment)
            if assigned_id not in memory_index:
                raise SnapshotContractError("overlay assignment is not a COVTrack memo ID")
            seed[index] = assigned_id
        return COVTrackAssociationDecision(snapshot, proposal, seed)

    @staticmethod
    def native_seed(decision: COVTrackAssociationDecision) -> np.ndarray:
        """Return a read-only numpy seed suitable for a native long tensor."""

        return decision.native_id_seed

    def commit_after_native_ids(self, decision: COVTrackAssociationDecision, final_ids: Any) -> None:
        """Commit overlay state only after COVTrack has committed final IDs."""

        self.overlay.commit(decision.snapshot, _numpy(final_ids, dtype=np.int64).reshape(-1))


__all__ = ["COVTrackAssociationDecision", "COVTrackTempoAdapter"]
