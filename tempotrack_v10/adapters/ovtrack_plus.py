"""Thin OVTrack+ bridge for the pinned OVT-B ``OVSortTracker``.

The official frontend owns detection, embedding, motion, LAP, new-ID
allocation, and memo writes. This module only converts the tensors that exist
at ``ovsort_tracker.py`` line 101 into the shared V10 pre-association contract.
It intentionally contains no matching or memory algorithm of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..contract import PreAssociationSnapshot, SnapshotContractError
from ..overlay import OverlayProposal, TempoTrackConfig, TempoTrackOverlay


CORE_SHA_V10_FULL = "c1d4b685a4e8b0863260cb657cd3f5d746285f64"


def _numpy(value: Any, *, dtype: np.dtype) -> np.ndarray:
    current = value.detach() if hasattr(value, "detach") else value
    current = current.cpu() if hasattr(current, "cpu") else current
    return np.asarray(current, dtype=dtype).copy()


@dataclass(frozen=True)
class OVTrackPlusDecision:
    """A proposal kept until the native tracker has committed its IDs."""

    snapshot: PreAssociationSnapshot
    proposal: OverlayProposal


class OVTrackPlusTempoAdapter:
    """State conversion at the OVTrack+ final-affinity boundary.

    ``disabled=True`` is deliberately represented separately from the shared
    core's ``enabled`` flag. The pinned upstream tracker can therefore keep
    its native path when the adapter is disabled.
    """

    def __init__(
        self,
        *,
        disabled: bool = True,
        overlay: TempoTrackOverlay | None = None,
    ) -> None:
        if overlay is not None and bool(disabled) == bool(overlay.config.enabled):
            raise ValueError("disabled and overlay.enabled describe conflicting modes")
        self.disabled = bool(disabled)
        self.overlay = overlay or TempoTrackOverlay(
            TempoTrackConfig(enabled=not self.disabled)
        )
        if self.disabled and self.overlay.config.enabled:
            raise ValueError("disabled=True requires a disabled shared overlay")
        if not self.disabled and not self.overlay.config.enabled:
            raise ValueError("disabled=False requires an enabled shared overlay")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "OVTrackPlusTempoAdapter":
        """Build the thin adapter from an explicit ``tempo`` mapping."""

        raw = dict(mapping or {})
        tempo = raw.get("tempo", raw)
        if not isinstance(tempo, Mapping):
            raise ValueError("OVTrack+ tempo configuration must be a mapping")
        tempo = dict(tempo)
        if "disabled" in tempo:
            disabled = bool(tempo["disabled"])
        elif "enabled" in tempo:
            disabled = not bool(tempo["enabled"])
        else:
            disabled = True
        allowed = {
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
            "reranker_weight",
            "reranker_checkpoint",
            "reranker_source_root",
            "reranker_device",
        }
        values = {key: tempo[key] for key in allowed if key in tempo}
        values["enabled"] = not disabled
        return cls(
            disabled=disabled,
            overlay=TempoTrackOverlay(TempoTrackConfig(**values)),
        )

    @property
    def enabled(self) -> bool:
        return not self.disabled

    def reset(self, video_id: int | str | None = None) -> None:
        self.overlay.reset(video_id)

    def build_snapshot(
        self,
        *,
        video_id: int | str | None,
        frame_id: int,
        bboxes: Any,
        labels: Any,
        embeds: Any,
        native_affinity: Any,
        memory_ids: Sequence[int],
        memory_embeddings: Any,
        memory_last_frame: Any,
    ) -> PreAssociationSnapshot:
        """Build a snapshot from the exact post-distractor line-101 tensors."""

        if video_id is None:
            raise SnapshotContractError(
                "OVTrack+ enabled overlay requires img_metas[0]['video_id']"
            )
        bbox_array = _numpy(bboxes, dtype=np.float32)
        if bbox_array.ndim != 2 or bbox_array.shape[1] != 5:
            raise SnapshotContractError("OVTrack+ bboxes must be post-filter [N,5]")
        label_array = _numpy(labels, dtype=np.int64).reshape(-1)
        embed_array = _numpy(embeds, dtype=np.float32)
        ids = tuple(int(value) for value in memory_ids)
        memory_array = _numpy(memory_embeddings, dtype=np.float32)
        frame_array = _numpy(memory_last_frame, dtype=np.int64).reshape(-1)
        affinity_array = _numpy(native_affinity, dtype=np.float32)
        if label_array.shape != (len(bbox_array),):
            raise SnapshotContractError("OVTrack+ labels must align with bboxes")
        if embed_array.ndim != 2 or embed_array.shape[0] != len(bbox_array):
            raise SnapshotContractError("OVTrack+ embeds must be [N,D]")
        if memory_array.ndim != 2 or memory_array.shape != (len(ids), embed_array.shape[1]):
            raise SnapshotContractError("OVTrack+ memory embeddings must be [M,D]")
        if frame_array.shape != (len(ids),):
            raise SnapshotContractError("OVTrack+ memory_last_frame must be [M]")
        if affinity_array.shape != (len(bbox_array), len(ids)):
            raise SnapshotContractError(
                "OVTrack+ native affinity must align with post-distractor rows and active IDs"
            )
        uids = tuple(
            f"ovtrack-plus:{video_id}:{int(frame_id)}:{index}"
            for index in range(len(bbox_array))
        )
        return PreAssociationSnapshot(
            video_id=video_id,
            frame_id=int(frame_id),
            boxes_xyxy=bbox_array[:, :4],
            det_scores=bbox_array[:, 4],
            labels=label_array,
            observation_uids=uids,
            embeddings=embed_array,
            native_affinity=affinity_array,
            memory_ids=ids,
            memory_embeddings=memory_array,
            memory_last_frame=frame_array,
            metadata={
                "association_stage": "pre_association",
                "pre_association": True,
                "frontend": "ovtrack_plus",
                "native_affinity_stage": "ovsort_line_101_before_lap",
                "post_distractor_subset": True,
            },
        )

    def prepare(self, **kwargs: Any) -> OVTrackPlusDecision:
        if not self.enabled:
            raise RuntimeError("disabled OVTrack+ adapter must not replace native assignment")
        snapshot = self.build_snapshot(**kwargs)
        return OVTrackPlusDecision(snapshot=snapshot, proposal=self.overlay.propose(snapshot))

    @staticmethod
    def preassigned_ids(decision: OVTrackPlusDecision, *, device: torch.device) -> torch.Tensor:
        """Map accepted existing IDs; leave rejected rows for native new-ID allocation."""

        output = np.full((decision.snapshot.observation_count,), -1, dtype=np.int64)
        for index, (accepted, assignment) in enumerate(
            zip(decision.proposal.accepted, decision.proposal.assignments)
        ):
            if accepted:
                if assignment is None or int(assignment) < 0:
                    raise SnapshotContractError("accepted OVTrack+ proposal lacks an existing ID")
                output[index] = int(assignment)
        return torch.as_tensor(output, dtype=torch.long, device=device)

    def commit(self, decision: OVTrackPlusDecision, final_ids: Any) -> None:
        self.overlay.commit(decision.snapshot, _numpy(final_ids, dtype=np.int64).reshape(-1))


__all__ = [
    "CORE_SHA_V10_FULL",
    "OVTrackPlusDecision",
    "OVTrackPlusTempoAdapter",
]
