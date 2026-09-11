"""One shared, causal TempoTrack association overlay for V10 frontends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from tempotrack_research.memory.fixed_dual import FixedDualMemory
from tempotrack_research.memory.state import MemoryState

from .contract import FrameCollisionError, PreAssociationSnapshot, SnapshotContractError


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / max(norm, 1e-8)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_normalize(left), _normalize(right)))


@dataclass(frozen=True)
class TempoTrackConfig:
    """Frontend-independent defaults shared by every V10 adapter."""

    enabled: bool = True
    alpha_fast: float = 0.70
    alpha_slow: float = 0.15
    min_gap: int = 0
    max_gap: int = 60
    candidate_top_k: int = 8
    top_r: int = 3
    native_weight: float = 0.45
    active_weight: float = 0.35
    support_weight: float = 0.20
    gap_penalty: float = 0.02
    score_threshold: float = 0.60
    margin_threshold: float = 0.0
    reliability_weight: float = 0.0
    memory_capacity: int = 64

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha_slow <= self.alpha_fast < 1.0):
            raise ValueError("TempoTrack requires 0 <= alpha_slow <= alpha_fast < 1")
        if self.min_gap < 0 or self.max_gap < self.min_gap:
            raise ValueError("gap bounds must satisfy 0 <= min_gap <= max_gap")
        if self.candidate_top_k < 1 or self.top_r < 1 or self.memory_capacity < 1:
            raise ValueError("candidate_top_k, top_r and memory_capacity must be positive")
        if min(self.native_weight, self.active_weight, self.support_weight) < 0:
            raise ValueError("score weights must be non-negative")


@dataclass(frozen=True)
class OverlayProposal:
    """Pure association proposal; it contains no detector mutation."""

    video_id: int | str
    frame_id: int
    observation_uids: tuple[str, ...]
    assignments: tuple[int | None, ...]
    scores: tuple[float, ...]
    margins: tuple[float, ...]
    accepted: tuple[bool, ...]
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _MemoryRecord:
    state: MemoryState
    last_frame: int
    history: list[np.ndarray]


class TempoTrackOverlay:
    """Causal pre-association overlay shared by OVTrack/COVTrack/TRACT/MASA.

    The native frontend calls ``propose`` after its final affinity is formed
    and before it commits IDs/memo state. It then maps the proposal to its
    own assignment representation and calls ``commit`` with the actual final
    IDs. Detector boxes, scores, labels and feature tensors are never edited.
    """

    def __init__(self, config: TempoTrackConfig | None = None) -> None:
        self.config = config or TempoTrackConfig()
        self._dual = FixedDualMemory(
            mode="fixed_dual",
            alpha_fast=self.config.alpha_fast,
            alpha_slow=self.config.alpha_slow,
        )
        self._records: dict[tuple[str, int], _MemoryRecord] = {}
        self._pending: dict[tuple[str, int], tuple[str, OverlayProposal]] = {}

    @staticmethod
    def _video_key(video_id: int | str) -> str:
        return str(video_id)

    def reset(self, video_id: int | str | None = None) -> None:
        """Reset one video or all overlay state."""
        if video_id is None:
            self._records.clear()
            self._pending.clear()
            return
        key = self._video_key(video_id)
        for record_key in [item for item in self._records if item[0] == key]:
            self._records.pop(record_key, None)
        for pending_key in [item for item in self._pending if item[0] == key]:
            self._pending.pop(pending_key, None)

    def _ensure_snapshot_memory(self, snapshot: PreAssociationSnapshot) -> None:
        video = self._video_key(snapshot.video_id)
        if snapshot.memory_embeddings is None:
            return
        for index, memory_id in enumerate(snapshot.memory_ids):
            key = (video, int(memory_id))
            current = self._records.get(key)
            if current is None:
                prototype = torch.as_tensor(_normalize(snapshot.memory_embeddings[index]), dtype=torch.float32)
                state = self._dual.initialize(prototype, frame=int(snapshot.memory_last_frame[index]))
                self._records[key] = _MemoryRecord(
                    state=state,
                    last_frame=int(snapshot.memory_last_frame[index]),
                    history=[_normalize(snapshot.memory_embeddings[index])],
                )
            else:
                current.last_frame = int(snapshot.memory_last_frame[index])

    def _support_score(self, query: np.ndarray, record: _MemoryRecord) -> float:
        values = [_normalize(record.state.fast.detach().cpu().numpy()), _normalize(record.state.slow.detach().cpu().numpy())]
        values.extend(record.history[-self.config.memory_capacity :])
        scores = sorted((_cosine(query, value) for value in values), reverse=True)
        count = min(self.config.top_r, len(scores))
        return float(np.mean(scores[:count])) if count else float("-inf")

    def _candidate_score(
        self,
        snapshot: PreAssociationSnapshot,
        observation_index: int,
        memory_index: int,
        record: _MemoryRecord,
    ) -> float:
        query = snapshot.embeddings[observation_index]
        native = float(snapshot.native_affinity[observation_index, memory_index])
        active = 0.70 * _cosine(query, record.state.fast.detach().cpu().numpy()) + 0.30 * _cosine(
            query, record.state.slow.detach().cpu().numpy()
        )
        support = self._support_score(query, record)
        gap = int(snapshot.frame_id) - int(record.last_frame)
        score = (
            self.config.native_weight * native
            + self.config.active_weight * active
            + self.config.support_weight * support
            - self.config.gap_penalty * gap / max(float(self.config.max_gap), 1.0)
        )
        reliability = snapshot.metadata.get("memory_reliability")
        if reliability is not None:
            values = np.asarray(reliability, dtype=np.float32).reshape(-1)
            if len(values) != len(snapshot.memory_ids):
                raise SnapshotContractError("memory_reliability must be [M]")
            score += self.config.reliability_weight * float(np.log(max(float(values[memory_index]), 1e-6)))
        return float(score)

    def propose(self, snapshot: PreAssociationSnapshot) -> OverlayProposal:
        """Produce a deterministic candidate assignment before native IDs commit."""
        self._ensure_snapshot_memory(snapshot)
        count = snapshot.observation_count
        empty = tuple(None for _ in range(count))
        if not self.config.enabled:
            proposal = OverlayProposal(
                video_id=snapshot.video_id,
                frame_id=int(snapshot.frame_id),
                observation_uids=tuple(snapshot.observation_uids),
                assignments=empty,
                scores=tuple(float("nan") for _ in range(count)),
                margins=tuple(float("nan") for _ in range(count)),
                accepted=tuple(False for _ in range(count)),
                reasons=tuple("disabled_noop" for _ in range(count)),
                diagnostics={"enabled": False, "observation_hash": snapshot.immutable_observation_hash()},
            )
            self._pending[(self._video_key(snapshot.video_id), int(snapshot.frame_id))] = (
                snapshot.immutable_observation_hash(), proposal
            )
            return proposal

        video = self._video_key(snapshot.video_id)
        occupied = {int(value) for value in snapshot.metadata.get("occupied_ids", ())}
        all_candidates: list[list[tuple[float, int, int]]] = [[] for _ in range(count)]
        for observation_index in range(count):
            legal: list[tuple[float, int, int]] = []
            for memory_index, memory_id in enumerate(snapshot.memory_ids):
                key = (video, int(memory_id))
                record = self._records.get(key)
                if record is None:
                    continue
                gap = int(snapshot.frame_id) - int(record.last_frame)
                if not (int(record.last_frame) < int(snapshot.frame_id)):
                    continue
                if gap < self.config.min_gap or gap > self.config.max_gap:
                    continue
                if not np.isfinite(snapshot.native_affinity[observation_index, memory_index]):
                    continue
                prefilter = float(snapshot.native_affinity[observation_index, memory_index]) + _cosine(
                    snapshot.embeddings[observation_index], record.state.fast.detach().cpu().numpy()
                )
                legal.append((prefilter, int(memory_id), memory_index))
            legal.sort(key=lambda item: (-item[0], item[1]))
            for _, memory_id, memory_index in legal[: self.config.candidate_top_k]:
                score = self._candidate_score(snapshot, observation_index, memory_index, self._records[(video, memory_id)])
                all_candidates[observation_index].append((score, memory_id, memory_index))
            all_candidates[observation_index].sort(key=lambda item: (-item[0], item[1]))

        assignments: list[int | None] = [None] * count
        scores = [float("nan")] * count
        margins = [float("nan")] * count
        accepted = [False] * count
        reasons = ["no_legal_candidate"] * count
        proposals_by_memory: dict[int, list[int]] = {}
        for index, candidates in enumerate(all_candidates):
            if not candidates:
                continue
            best_score, best_memory, _ = candidates[0]
            second_score = candidates[1][0] if len(candidates) > 1 else float("-inf")
            margin = best_score - second_score if np.isfinite(second_score) else float("inf")
            scores[index] = float(best_score)
            margins[index] = float(margin)
            if best_memory in occupied:
                reasons[index] = "frame_collision"
                continue
            if best_score < self.config.score_threshold or margin < self.config.margin_threshold:
                reasons[index] = "threshold_or_margin"
                continue
            assignments[index] = int(best_memory)
            accepted[index] = True
            reasons[index] = "accepted"
            proposals_by_memory.setdefault(int(best_memory), []).append(index)

        competition_losers = 0
        for memory_id, indices in sorted(proposals_by_memory.items()):
            winner = min(indices, key=lambda index: (-scores[index], -margins[index], snapshot.observation_uids[index]))
            for index in indices:
                if index == winner:
                    continue
                assignments[index] = None
                accepted[index] = False
                reasons[index] = "competition_loser"
                competition_losers += 1

        proposal = OverlayProposal(
            video_id=snapshot.video_id,
            frame_id=int(snapshot.frame_id),
            observation_uids=tuple(snapshot.observation_uids),
            assignments=tuple(assignments),
            scores=tuple(float(value) for value in scores),
            margins=tuple(float(value) for value in margins),
            accepted=tuple(accepted),
            reasons=tuple(reasons),
            diagnostics={
                "enabled": True,
                "observation_hash": snapshot.immutable_observation_hash(),
                "candidate_top_k": int(self.config.candidate_top_k),
                "legal_candidate_count": int(sum(len(values) for values in all_candidates)),
                "competition_losers": int(competition_losers),
                "frame_collision_rejections": int(sum(value == "frame_collision" for value in reasons)),
            },
        )
        self._pending[(video, int(snapshot.frame_id))] = (snapshot.immutable_observation_hash(), proposal)
        return proposal

    def commit(self, snapshot: PreAssociationSnapshot, final_ids: Any) -> None:
        """Commit native final IDs and update causal memory after assignment."""
        key = (self._video_key(snapshot.video_id), int(snapshot.frame_id))
        pending = self._pending.get(key)
        if pending is None or pending[0] != snapshot.immutable_observation_hash():
            raise SnapshotContractError("commit must immediately follow propose for the same snapshot")
        ids = np.asarray(final_ids, dtype=np.int64).reshape(-1)
        if len(ids) != snapshot.observation_count:
            raise SnapshotContractError("final_ids must be aligned with observations")
        positive = [int(value) for value in ids if int(value) >= 0]
        if len(positive) != len(set(positive)):
            raise FrameCollisionError("frame collision: duplicate non-negative final ID")
        if not self.config.enabled:
            self._pending.pop(key, None)
            return
        video = self._video_key(snapshot.video_id)
        with torch.no_grad():
            for index, final_id in enumerate(ids.tolist()):
                if int(final_id) < 0:
                    continue
                memory_key = (video, int(final_id))
                current = self._records.get(memory_key)
                if current is None:
                    prototype = torch.as_tensor(_normalize(snapshot.embeddings[index]), dtype=torch.float32)
                    state = self._dual.initialize(prototype, frame=int(snapshot.frame_id))
                    history: list[np.ndarray] = []
                    current = _MemoryRecord(state=state, last_frame=int(snapshot.frame_id), history=history)
                state, _ = self._dual.update(
                    current.state,
                    torch.as_tensor(_normalize(snapshot.embeddings[index]), dtype=torch.float32),
                    confidence=float(snapshot.det_scores[index]),
                    frame=int(snapshot.frame_id),
                )
                current.state = state
                current.last_frame = int(snapshot.frame_id)
                current.history.append(_normalize(snapshot.embeddings[index]))
                current.history = current.history[-self.config.memory_capacity :]
                self._records[memory_key] = current
        self._pending.pop(key, None)


__all__ = ["OverlayProposal", "TempoTrackConfig", "TempoTrackOverlay"]
