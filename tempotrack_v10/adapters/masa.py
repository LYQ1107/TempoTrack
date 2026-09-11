"""Thin MASA-R50 bridge into the shared V10 pre-association overlay.

This module owns only tensor/state conversion.  It deliberately does not
implement Dual memory, candidate search, reliability, reranking, competition,
or any other TempoTrack logic; those remain in ``tempotrack_v10.overlay``.
The bridge is called after MASA's native affinity matrix is complete and
before native IDs or memo state are committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tempotrack_v10 import OverlayProposal, PreAssociationSnapshot, TempoTrackOverlay


def _numpy(value: Any, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


@dataclass
class MasaPreAssociationTrace:
    """AssociationTrace-compatible diagnostics for overlay decisions."""

    frame_id: int
    ids: torch.Tensor
    accepted: torch.Tensor
    accepted_score: torch.Tensor
    detection_margin: torch.Tensor
    assigned_memo_index: torch.Tensor
    score_matrix: torch.Tensor | None = None
    fast_score: torch.Tensor | None = None
    slow_score: torch.Tensor | None = None


@dataclass
class MasaPreAssociationDecision:
    """Pending snapshot/proposal retained until native bookkeeping commits."""

    snapshot: PreAssociationSnapshot
    proposal: OverlayProposal
    ids: torch.Tensor
    trace: MasaPreAssociationTrace


class MasaTaoPreAssociationAdapter:
    """Map MASA tensors to/from the one shared ``TempoTrackOverlay``.

    ``decide`` is intentionally called with the already-computed native
    affinity.  When the overlay is disabled, it delegates to MASA's existing
    ``_assign_matches`` path, preserving native IDs and diagnostics exactly.
    """

    def __init__(self, overlay: TempoTrackOverlay, *, uid_prefix: str = "masa-r50") -> None:
        if not isinstance(overlay, TempoTrackOverlay):
            raise TypeError("overlay must be the shared TempoTrackOverlay")
        self.overlay = overlay
        self.uid_prefix = str(uid_prefix)

    @property
    def enabled(self) -> bool:
        return bool(self.overlay.config.enabled)

    def reset(self, video_id: int | str | None = None) -> None:
        """Reset overlay state when MASA starts a new video/sequence."""
        self.overlay.reset(video_id)

    @staticmethod
    def _memory_numpy(memory: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = _numpy(memory["ids"], dtype=np.int64).reshape(-1)
        embeds = _numpy(memory["embeds"], dtype=np.float32)
        frame_ids = _numpy(memory["frame_ids"], dtype=np.int64).reshape(-1)
        if embeds.ndim != 2 or len(ids) != len(embeds) or len(ids) != len(frame_ids):
            raise ValueError("MASA memory must contain aligned ids, embeds and frame_ids")
        return ids, embeds, frame_ids

    def build_snapshot(
        self,
        *,
        video_id: int | str,
        frame_id: int,
        bboxes: Any,
        labels: Any,
        scores: Any,
        embeds: Any,
        native_affinity: Any,
        memory: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        observation_uids: Sequence[str] | None = None,
    ) -> PreAssociationSnapshot:
        memory_ids, memory_embeds, memory_frames = self._memory_numpy(memory)
        boxes = _numpy(bboxes, dtype=np.float32)
        count = int(len(boxes))
        uids = tuple(observation_uids or (
            f"{self.uid_prefix}:{video_id}:{int(frame_id)}:{index}" for index in range(count)
        ))
        snapshot_metadata = {
            "association_stage": "pre_association",
            "pre_association": True,
            "frontend": "masa_r50",
            **dict(metadata or {}),
        }
        return PreAssociationSnapshot(
            video_id=video_id,
            frame_id=int(frame_id),
            boxes_xyxy=boxes,
            det_scores=_numpy(scores, dtype=np.float32).reshape(-1),
            labels=_numpy(labels, dtype=np.int64).reshape(-1),
            observation_uids=uids,
            embeddings=_numpy(embeds, dtype=np.float32),
            native_affinity=_numpy(native_affinity, dtype=np.float32),
            memory_ids=tuple(int(value) for value in memory_ids.tolist()),
            memory_embeddings=memory_embeds,
            memory_last_frame=memory_frames,
            metadata=snapshot_metadata,
        )

    @staticmethod
    def _overlay_trace(
        *,
        proposal: OverlayProposal,
        native_affinity: torch.Tensor,
        ids: torch.Tensor,
        assigned_memo_index: torch.Tensor,
    ) -> MasaPreAssociationTrace:
        dtype = native_affinity.dtype if native_affinity.is_floating_point() else torch.float32
        device = native_affinity.device
        accepted = torch.as_tensor(proposal.accepted, dtype=torch.bool, device=device)
        accepted_score = torch.as_tensor(proposal.scores, dtype=dtype, device=device)
        margins = torch.as_tensor(proposal.margins, dtype=dtype, device=device)
        return MasaPreAssociationTrace(
            frame_id=-1,
            ids=ids,
            accepted=accepted,
            accepted_score=accepted_score,
            detection_margin=margins,
            assigned_memo_index=assigned_memo_index,
            score_matrix=native_affinity.detach().clone() if native_affinity.numel() else None,
        )

    def decide(
        self,
        *,
        tracker: Any,
        video_id: int | str | None,
        frame_id: int,
        bboxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        embeds: torch.Tensor,
        native_affinity: torch.Tensor,
        memory: Mapping[str, Any],
    ) -> MasaPreAssociationDecision:
        if video_id is None:
            raise ValueError("MASA overlay requires video_id for causal state isolation")
        snapshot = self.build_snapshot(
            video_id=video_id,
            frame_id=frame_id,
            bboxes=bboxes,
            labels=labels,
            scores=scores,
            embeds=embeds,
            native_affinity=native_affinity,
            memory=memory,
        )
        proposal = self.overlay.propose(snapshot)

        # Disabled is an exact native no-op: do not reinterpret or replace
        # MASA's native greedy assignment.
        if not self.enabled:
            ids, trace = tracker._assign_matches(native_affinity, memory["ids"], scores)
            return MasaPreAssociationDecision(snapshot, proposal, ids, trace)

        ids = torch.full(
            (snapshot.observation_count,), -1, dtype=torch.long, device=bboxes.device
        )
        memory_ids = _numpy(memory["ids"], dtype=np.int64).reshape(-1)
        memo_index = {int(value): index for index, value in enumerate(memory_ids.tolist())}
        assigned = torch.full_like(ids, -1)
        for index, (accepted, assignment) in enumerate(zip(proposal.accepted, proposal.assignments)):
            if not accepted or assignment is None:
                continue
            ids[index] = int(assignment)
            # A dormant identity is not present in the current native memo;
            # keep the trace's memo index at -1 while preserving the logical
            # identity in ``ids``.  The frontend's ordinary update path then
            # reintroduces that identity causally.
            assigned[index] = int(memo_index.get(int(assignment), -1))
        trace = self._overlay_trace(
            proposal=proposal,
            native_affinity=native_affinity,
            ids=ids,
            assigned_memo_index=assigned,
        )
        return MasaPreAssociationDecision(snapshot, proposal, ids, trace)

    def commit(self, decision: MasaPreAssociationDecision, final_ids: Any) -> None:
        """Commit only after MASA has updated its native memo state."""
        if isinstance(final_ids, torch.Tensor):
            final_ids = final_ids.detach().cpu().numpy()
        self.overlay.commit(decision.snapshot, final_ids)


def install_masa_overlay(tracker: Any, overlay: TempoTrackOverlay) -> MasaTaoPreAssociationAdapter:
    """Attach the thin adapter without replacing the MASA tracker or core."""
    adapter = MasaTaoPreAssociationAdapter(overlay)
    setter = getattr(tracker, "set_pre_association_adapter", None)
    if setter is None:
        raise TypeError("tracker does not expose the MASA pre-association attachment point")
    setter(adapter)
    return adapter


__all__ = [
    "MasaPreAssociationDecision",
    "MasaPreAssociationTrace",
    "MasaTaoPreAssociationAdapter",
    "install_masa_overlay",
]
