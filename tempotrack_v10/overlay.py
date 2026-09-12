"""One causal, frontend-independent TempoTrack association overlay.

The overlay is called after a frontend has formed its native affinity and
before that frontend commits IDs.  Its candidate universe is the native
snapshot memory union the overlay's independent dormant records.  Dormant
records never receive a fabricated native affinity; they are ranked only by
causal memory evidence and, when configured, the exact V9 query-conditioned
reranker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tempotrack_research.memory.fixed_dual import FixedDualMemory
from tempotrack_research.memory.state import MemoryState

from .contract import FrameCollisionError, PreAssociationSnapshot, SnapshotContractError
from .reranker import load_exact_v9_reranker


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
    # A nonzero weight requires a verified V9 checkpoint. There is no
    # heuristic fallback under this switch.
    reranker_weight: float = 0.0
    reranker_checkpoint: str | None = None
    reranker_source_root: str | None = None
    reranker_device: str = "cpu"

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha_slow <= self.alpha_fast < 1.0):
            raise ValueError("TempoTrack requires 0 <= alpha_slow <= alpha_fast < 1")
        if self.min_gap < 0 or self.max_gap < self.min_gap:
            raise ValueError("gap bounds must satisfy 0 <= min_gap <= max_gap")
        if self.candidate_top_k < 1 or self.top_r < 1 or self.memory_capacity < 1:
            raise ValueError("candidate_top_k, top_r and memory_capacity must be positive")
        if min(self.native_weight, self.active_weight, self.support_weight) < 0:
            raise ValueError("score weights must be non-negative")
        if self.reranker_weight < 0:
            raise ValueError("reranker_weight must be non-negative")


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
    first_frame: int
    last_frame: int
    history: list[np.ndarray]
    evidence_history: list[np.ndarray] = field(default_factory=list)
    # This bank is independent of the heuristic ``history``.  It is the
    # checkpoint-contract bank: causal cosine deduplication and synchronized
    # evidence are applied before the checkpoint's capacity truncation.
    reranker_history: list[np.ndarray] = field(default_factory=list)
    reranker_evidence_history: list[np.ndarray] = field(default_factory=list)
    root_id: int = 0
    lineage: tuple[int, ...] = field(default_factory=tuple)
    last_box: np.ndarray | None = None
    # The Q1 reranker prefilter is defined by the last real observation, not
    # by the runtime-truncated heuristic history or a fast/slow prototype.
    last_embedding: np.ndarray | None = None


@dataclass(frozen=True)
class _Candidate:
    memory_id: int
    memory_index: int | None
    record: _MemoryRecord
    root_id: int
    is_native: bool


class TempoTrackOverlay:
    """Causal overlay shared by OVTrack/COVTrack/TRACT/MASA adapters.

    The exact V9 reranker is opt-in through ``reranker_checkpoint`` and
    ``reranker_weight``. If it is enabled, missing source, evidence, or
    checkpoint provenance is a hard error; the heuristic score is never
    relabeled as a FULL reranker implementation.
    """

    def __init__(self, config: TempoTrackConfig | None = None, *, reranker: Any | None = None) -> None:
        self.config = config or TempoTrackConfig()
        if self.config.reranker_weight > 0.0 and reranker is None:
            if not self.config.reranker_checkpoint:
                raise SnapshotContractError("BLOCKED_QUERY_RERANKER_CHECKPOINT_MISSING")
            reranker = load_exact_v9_reranker(
                self.config.reranker_checkpoint,
                source_root=self.config.reranker_source_root or "/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9",
                device=self.config.reranker_device,
            )
        if self.config.reranker_weight > 0.0 and reranker is not None:
            if not callable(getattr(reranker, "score_event", None)):
                raise SnapshotContractError("BLOCKED_QUERY_RERANKER_SOURCE_MISSING: score_event")
            provenance = getattr(reranker, "provenance", None)
            if not isinstance(provenance, Mapping) or provenance.get("status") != "EXACT_V9_MODEL_CODE_AND_WEIGHTS":
                raise SnapshotContractError("BLOCKED_QUERY_RERANKER_SOURCE_MISSING: exact provenance")
            feature_config = provenance.get("feature_config")
            if not isinstance(feature_config, Mapping):
                raise SnapshotContractError("BLOCKED_QUERY_RERANKER_FEATURE_CONFIG_MISSING")
            for key, expected in (("alpha_fast", self.config.alpha_fast), ("alpha_slow", self.config.alpha_slow)):
                if key not in feature_config or not np.isclose(
                    float(feature_config[key]), float(expected), rtol=0.0, atol=1e-8
                ):
                    raise SnapshotContractError(f"BLOCKED_QUERY_RERANKER_ALPHA_MISMATCH: {key}")
            if int(self.config.candidate_top_k) > int(feature_config["candidate_top_k"]):
                raise SnapshotContractError("BLOCKED_QUERY_RERANKER_CONTEXT_K_TOO_SMALL")
            self._reranker_feature_config = dict(feature_config)
        else:
            self._reranker_feature_config = {}
        self._reranker = reranker
        self._dual = FixedDualMemory(
            mode="fixed_dual",
            alpha_fast=self.config.alpha_fast,
            alpha_slow=self.config.alpha_slow,
        )
        self._records: dict[tuple[str, int], _MemoryRecord] = {}
        self._pending: dict[tuple[str, int], tuple[str, OverlayProposal]] = {}
        self._last_query_observations: int | None = None
        # A Q1 reranker may not silently bootstrap an identity from the
        # native frontend's EMA/memo embedding.  Keep this counter explicit so
        # runtime receipts can prove that the forbidden fallback was never
        # used.  The counter is incremented immediately before the
        # fail-closed exception and therefore remains zero for a valid run.
        self._reranker_native_memo_bootstrap_count = 0

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

    @staticmethod
    def _indexed_metadata(
        snapshot: PreAssociationSnapshot,
        name: str,
        index: int,
        key: int,
        expected_count: int,
    ) -> Any | None:
        value = snapshot.metadata.get(name)
        if value is None:
            return None
        if isinstance(value, Mapping):
            return value.get(key, value.get(str(key)))
        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == expected_count:
            return value[index]
        if isinstance(value, (list, tuple)) and len(value) == expected_count:
            return value[index]
        raise SnapshotContractError(f"{name} must be an aligned mapping or sequence")

    @staticmethod
    def _evidence_value(value: Any | None, name: str) -> list[np.ndarray]:
        if value is None:
            return []
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1 and array.shape == (7,):
            array = array[None, :]
        if array.ndim != 2 or array.shape[1:] != (7,) or not np.isfinite(array).all():
            raise SnapshotContractError(f"{name} must be finite [L,7]")
        return [np.asarray(row, dtype=np.float32).copy() for row in array]

    @staticmethod
    def _history_value(value: Any | None, dimension: int, name: str) -> list[np.ndarray]:
        if value is None:
            return []
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != dimension or not np.isfinite(array).all():
            raise SnapshotContractError(f"{name} must be finite [L,D]")
        return [_normalize(row) for row in array]

    @staticmethod
    def _raw_history_value(value: Any | None, dimension: int, name: str) -> list[np.ndarray]:
        if value is None:
            return []
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != dimension or not np.isfinite(array).all():
            raise SnapshotContractError(f"{name} must be finite [L,D]")
        return [np.asarray(row, dtype=np.float32).copy() for row in array]

    def _canonical_reranker_bank(
        self,
        history: Sequence[np.ndarray],
        evidence: Sequence[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Rebuild the causal V9 bank with synchronized feature/evidence rows."""
        if self._reranker is None or not history or not evidence:
            return [], []
        length = min(len(history), len(evidence))
        if length < 1:
            return [], []
        bank_features: list[np.ndarray] = []
        bank_evidence: list[np.ndarray] = []
        threshold = float(self._reranker_feature_config["memory_dedup_cos"])
        capacity = int(self._reranker_feature_config["memory_capacity"])
        for feature, row in zip(history[-length:], evidence[-length:]):
            feature_value = np.asarray(feature, dtype=np.float32).copy()
            evidence_value = np.asarray(row, dtype=np.float32).reshape(-1)
            if evidence_value.shape != (7,) or not np.isfinite(evidence_value).all():
                raise SnapshotContractError("reranker evidence rows must be finite [7]")
            if not bank_features or max(_cosine(feature_value, old) for old in bank_features) < threshold:
                bank_features.append(feature_value.copy())
                bank_evidence.append(evidence_value.copy())
        return bank_features[-capacity:], bank_evidence[-capacity:]

    def _append_reranker_rows(
        self,
        record: _MemoryRecord,
        feature: np.ndarray,
        evidence: Sequence[np.ndarray],
    ) -> None:
        """Append one causal observation and keep feature/evidence aligned."""
        if self._reranker is None or not evidence:
            return
        threshold = float(self._reranker_feature_config["memory_dedup_cos"])
        capacity = int(self._reranker_feature_config["memory_capacity"])
        feature_value = np.asarray(feature, dtype=np.float32).copy()
        for row in evidence:
            evidence_value = np.asarray(row, dtype=np.float32).reshape(-1)
            if evidence_value.shape != (7,) or not np.isfinite(evidence_value).all():
                raise SnapshotContractError("reranker evidence rows must be finite [7]")
            if not record.reranker_history or max(
                _cosine(feature_value, old) for old in record.reranker_history
            ) < threshold:
                record.reranker_history.append(feature_value.copy())
                record.reranker_evidence_history.append(evidence_value.copy())
        record.reranker_history = record.reranker_history[-capacity:]
        record.reranker_evidence_history = record.reranker_evidence_history[-capacity:]

    def _ensure_snapshot_memory(self, snapshot: PreAssociationSnapshot) -> None:
        video = self._video_key(snapshot.video_id)
        if snapshot.memory_embeddings is None:
            return
        for index, memory_id in enumerate(snapshot.memory_ids):
            key = (video, int(memory_id))
            explicit_history = self._indexed_metadata(
                snapshot, "memory_embedding_history", index, int(memory_id), len(snapshot.memory_ids)
            )
            raw_history = self._raw_history_value(
                explicit_history, snapshot.feature_dim, "memory_embedding_history"
            )
            current = self._records.get(key)
            if not raw_history:
                if (
                    current is None
                    and self._reranker is not None
                    and self.config.reranker_weight > 0.0
                ):
                    self._reranker_native_memo_bootstrap_count += 1
                    raise SnapshotContractError(
                        "BLOCKED_RERANKER_NATIVE_MEMO_BOOTSTRAP: FULL Q1 requires "
                        "the candidate's exact last real observation embedding; "
                        "native frontend memo/EMA embedding cannot initialize "
                        "last_embedding."
                    )
                raw_history = [np.asarray(snapshot.memory_embeddings[index], dtype=np.float32).copy()]
            history = [_normalize(row) for row in raw_history]
            evidence_value = self._indexed_metadata(
                snapshot, "memory_evidence", index, int(memory_id), len(snapshot.memory_ids)
            )
            if evidence_value is None:
                evidence_value = self._indexed_metadata(
                    snapshot, "evidence_by_memory", index, int(memory_id), len(snapshot.memory_ids)
                )
            evidence = self._evidence_value(evidence_value, "memory_evidence")
            root_value = self._indexed_metadata(
                snapshot, "memory_root_ids", index, int(memory_id), len(snapshot.memory_ids)
            )
            root_id = int(memory_id if root_value is None else root_value)
            lineage_value = self._indexed_metadata(
                snapshot, "memory_lineage", index, int(memory_id), len(snapshot.memory_ids)
            )
            lineage = tuple(int(value) for value in lineage_value) if lineage_value is not None else (root_id,)
            if current is None:
                prototype = torch.as_tensor(history[-1], dtype=torch.float32)
                state = self._dual.initialize(prototype, frame=int(snapshot.memory_last_frame[index]))
                reranker_history, reranker_evidence = self._canonical_reranker_bank(raw_history, evidence)
                self._records[key] = _MemoryRecord(
                    state=state,
                    first_frame=int(snapshot.memory_last_frame[index]),
                    last_frame=int(snapshot.memory_last_frame[index]),
                    history=history[-self.config.memory_capacity :],
                    evidence_history=evidence[-self.config.memory_capacity :],
                    reranker_history=reranker_history,
                    reranker_evidence_history=reranker_evidence,
                    root_id=root_id,
                    lineage=lineage,
                    last_embedding=_normalize(raw_history[-1]),
                )
            else:
                current.last_frame = int(snapshot.memory_last_frame[index])
                if explicit_history is not None:
                    current.history = history[-self.config.memory_capacity :]
                    current.last_embedding = _normalize(raw_history[-1])
                if evidence_value is not None:
                    current.evidence_history = evidence[-self.config.memory_capacity :]
                if self._reranker is not None and explicit_history is not None and evidence_value is not None:
                    (
                        current.reranker_history,
                        current.reranker_evidence_history,
                    ) = self._canonical_reranker_bank(raw_history, evidence)

    def _candidate_union(self, snapshot: PreAssociationSnapshot) -> list[_Candidate]:
        video = self._video_key(snapshot.video_id)
        native_index = {int(memory_id): index for index, memory_id in enumerate(snapshot.memory_ids)}
        candidates: list[_Candidate] = []
        seen: set[int] = set()
        for memory_id in snapshot.memory_ids:
            memory_id = int(memory_id)
            record = self._records.get((video, memory_id))
            if record is None:
                continue
            candidates.append(_Candidate(memory_id, native_index[memory_id], record, int(record.root_id), True))
            seen.add(memory_id)
        # This is deliberately independent of the native memory list. A
        # dormant ID can re-enter after the frontend has evicted it.
        for (record_video, memory_id), record in sorted(self._records.items(), key=lambda item: item[0][1]):
            if record_video != video or memory_id in seen:
                continue
            candidates.append(_Candidate(int(memory_id), None, record, int(record.root_id), False))
        return candidates

    def _reranker_prefilter_score(
        self,
        query: np.ndarray,
        candidate: _Candidate,
    ) -> float:
        """Exact Q1 prefilter matching V9 ``_rank_candidates``.

        The training-side prefilter is cosine(query, candidate's last real
        observation).  Native affinity, fast/slow prototypes, support, and
        reliability are deliberately excluded from this FULL-path ranking.
        """
        if self._reranker is None or self.config.reranker_weight <= 0.0:
            raise SnapshotContractError("reranker prefilter requested while reranker is disabled")
        expected_q = int(self._reranker_feature_config["query_observations"])
        if expected_q != 1:
            raise SnapshotContractError(
                f"BLOCKED_RERANKER_PREFILTER_Q_MISMATCH: expected Q={expected_q}"
            )
        if candidate.record.last_embedding is None:
            raise SnapshotContractError("BLOCKED_RERANKER_LAST_EMBEDDING_MISSING")
        return _cosine(query, candidate.record.last_embedding)

    def _support_score(self, query: np.ndarray, record: _MemoryRecord) -> float:
        values = [
            _normalize(record.state.fast.detach().cpu().numpy()),
            _normalize(record.state.slow.detach().cpu().numpy()),
        ]
        values.extend(record.history[-self.config.memory_capacity :])
        scores = sorted((_cosine(query, value) for value in values), reverse=True)
        count = min(self.config.top_r, len(scores))
        return float(np.mean(scores[:count])) if count else float("-inf")

    def _candidate_score(
        self,
        snapshot: PreAssociationSnapshot,
        observation_index: int,
        candidate: _Candidate,
        reranker_score: float | None,
    ) -> float:
        query = snapshot.embeddings[observation_index]
        active = 0.70 * _cosine(query, candidate.record.state.fast.detach().cpu().numpy()) + 0.30 * _cosine(
            query, candidate.record.state.slow.detach().cpu().numpy()
        )
        support = self._support_score(query, candidate.record)
        terms = [(self.config.active_weight, active), (self.config.support_weight, support)]
        if candidate.is_native:
            if candidate.memory_index is None:
                raise SnapshotContractError("native candidate is missing its affinity index")
            native = float(snapshot.native_affinity[observation_index, candidate.memory_index])
            terms.insert(0, (self.config.native_weight, native))
        weight_total = sum(weight for weight, _ in terms)
        score = sum(weight * value for weight, value in terms) / max(weight_total, 1e-8)
        gap = int(snapshot.frame_id) - int(candidate.record.last_frame)
        score -= self.config.gap_penalty * gap / max(float(self.config.max_gap), 1.0)
        if candidate.is_native:
            reliability = snapshot.metadata.get("memory_reliability")
            if reliability is not None:
                values = np.asarray(reliability, dtype=np.float32).reshape(-1)
                if len(values) != len(snapshot.memory_ids):
                    raise SnapshotContractError("memory_reliability must be [M]")
                score += self.config.reliability_weight * float(
                    np.log(max(float(values[candidate.memory_index]), 1e-6))
                )
        if self.config.reranker_weight > 0.0:
            if reranker_score is None:
                raise SnapshotContractError("BLOCKED_QUERY_RERANKER_EVIDENCE_MISSING")
            # V9's production replay uses the query-conditioned model logit as
            # the decision score.  A positive weight is an explicit opt-in to
            # that exact path; it must not silently blend the V9 model with
            # V10's heuristic score.
            return float(self.config.reranker_weight * float(reranker_score))
        return float(score)

    def _query_sequence(self, snapshot: PreAssociationSnapshot, observation_index: int) -> np.ndarray:
        expected = int(self._reranker_feature_config.get("query_observations", 1))
        value = snapshot.metadata.get("query_embeddings")
        if value is None:
            if expected > 1:
                raise SnapshotContractError(
                    "BLOCKED_QUERY_PROTOCOL_MISMATCH: checkpoint requires "
                    f"Q={expected} but snapshot has no query_embeddings"
                )
            result = snapshot.embeddings[observation_index : observation_index + 1]
            self._last_query_observations = int(result.shape[0])
            return result
        if isinstance(value, Mapping):
            value = value.get(snapshot.observation_uids[observation_index], value.get(str(observation_index)))
        else:
            value = np.asarray(value, dtype=np.float32)
            if value.ndim == 3 and value.shape[0] == snapshot.observation_count:
                value = value[observation_index]
            elif value.ndim == 2 and value.shape == snapshot.embeddings.shape:
                value = value[observation_index : observation_index + 1]
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != snapshot.feature_dim or not np.isfinite(array).all():
            raise SnapshotContractError("query_embeddings must be [N,L,D] or aligned [N,D]")
        if array.shape[0] != expected:
            raise SnapshotContractError(
                f"BLOCKED_QUERY_PROTOCOL_MISMATCH: expected Q={expected}, got Q={array.shape[0]}"
            )
        self._last_query_observations = int(array.shape[0])
        return array

    def _reranker_scores(
        self,
        snapshot: PreAssociationSnapshot,
        observation_index: int,
        context_candidates: Sequence[tuple[float, _Candidate]],
        decision_candidates: Sequence[tuple[float, _Candidate]],
    ) -> tuple[np.ndarray | None, int]:
        if self.config.reranker_weight <= 0.0:
            return None, 0
        if self._reranker is None:
            raise SnapshotContractError("BLOCKED_QUERY_RERANKER_SOURCE_MISSING")
        if not context_candidates:
            return None, 0
        query = self._query_sequence(snapshot, observation_index)
        query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
        payload: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        candidate_ids: list[int] = []
        missing = 0
        for rank, (_, candidate) in enumerate(context_candidates, start=1):
            history = candidate.record.reranker_history
            evidence = candidate.record.reranker_evidence_history
            length = min(len(history), len(evidence))
            if length < 1:
                missing += 1
                continue
            memory = np.stack(history[-length:], axis=0)
            memory = memory / np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), 1e-8)
            evidence_array = np.stack(evidence[-length:], axis=0).astype(np.float32)
            payload.append((query @ memory.T, evidence_array, int(snapshot.frame_id - candidate.record.last_frame), rank))
            candidate_ids.append(int(candidate.memory_id))
        if not payload:
            return None, missing
        logits, _ = self._reranker.score_event(payload)
        by_memory_id = {memory_id: float(logit) for memory_id, logit in zip(candidate_ids, logits)}
        aligned = np.full(len(decision_candidates), np.nan, dtype=np.float32)
        for index, (_, candidate) in enumerate(decision_candidates):
            if int(candidate.memory_id) in by_memory_id:
                aligned[index] = by_memory_id[int(candidate.memory_id)]
        return aligned, missing

    def propose(self, snapshot: PreAssociationSnapshot) -> OverlayProposal:
        """Produce a deterministic candidate assignment before native IDs commit."""
        count = snapshot.observation_count
        empty = tuple(None for _ in range(count))
        observation_hash = snapshot.immutable_observation_hash()
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
                diagnostics={"enabled": False, "observation_hash": observation_hash},
            )
            self._pending[(self._video_key(snapshot.video_id), int(snapshot.frame_id))] = (observation_hash, proposal)
            return proposal

        # Validate the checkpoint query protocol before touching any native
        # memory bootstrap.  A Q>1 checkpoint without explicit query history
        # must report the protocol mismatch, never be shadowed by a separate
        # Q1 memo-bootstrap failure.
        if self.config.reranker_weight > 0.0:
            for observation_index in range(count):
                self._query_sequence(snapshot, observation_index)
        self._ensure_snapshot_memory(snapshot)
        video = self._video_key(snapshot.video_id)
        candidates = self._candidate_union(snapshot)
        occupied = {int(value) for value in snapshot.metadata.get("occupied_ids", ())}
        occupied_roots = {int(value) for value in snapshot.metadata.get("occupied_root_ids", ())}
        all_candidates: list[list[tuple[float, int, int]]] = [[] for _ in range(count)]
        reranker_missing_total = 0
        for observation_index in range(count):
            # Validate the checkpoint's query protocol before candidate
            # filtering. A Q>1 checkpoint must fail closed even when this
            # frame has no legal memory candidate.
            legal: list[tuple[float, _Candidate]] = []
            for candidate in candidates:
                gap = int(snapshot.frame_id) - int(candidate.record.last_frame)
                if not (int(candidate.record.last_frame) < int(snapshot.frame_id)):
                    continue
                if gap < self.config.min_gap or gap > self.config.max_gap:
                    continue
                query = snapshot.embeddings[observation_index]
                if self.config.reranker_weight > 0.0:
                    # FULL Q1 reranker path: preserve the exact training
                    # candidate rank before constructing the Top-K context.
                    prefilter = self._reranker_prefilter_score(query, candidate)
                else:
                    active = 0.70 * _cosine(
                        query, candidate.record.state.fast.detach().cpu().numpy()
                    ) + 0.30 * _cosine(
                        query, candidate.record.state.slow.detach().cpu().numpy()
                    )
                    if candidate.is_native:
                        if candidate.memory_index is None:
                            continue
                        native = float(snapshot.native_affinity[observation_index, candidate.memory_index])
                        if not np.isfinite(native):
                            continue
                        prefilter = native + active
                    else:
                        # No zero/one/mean native affinity is manufactured for
                        # a dormant ID: it enters through causal memory evidence.
                        prefilter = active + self._support_score(query, candidate.record)
                legal.append((float(prefilter), candidate))
            legal.sort(key=lambda item: (-item[0], item[1].memory_id, item[1].root_id))
            selected = legal[: self.config.candidate_top_k]
            context = legal[: int(self._reranker_feature_config.get(
                "candidate_top_k", self.config.candidate_top_k
            ))] if self.config.reranker_weight > 0.0 else selected
            logits, missing = self._reranker_scores(
                snapshot, observation_index, context, selected
            )
            reranker_missing_total += missing
            for rank, (_, candidate) in enumerate(selected, start=1):
                if self.config.reranker_weight > 0.0:
                    if logits is None or rank - 1 >= len(logits) or not np.isfinite(logits[rank - 1]):
                        continue
                    reranker_score = float(logits[rank - 1])
                else:
                    reranker_score = None
                score = self._candidate_score(snapshot, observation_index, candidate, reranker_score)
                all_candidates[observation_index].append((score, candidate.memory_id, candidate.root_id))
            all_candidates[observation_index].sort(key=lambda item: (-item[0], item[1], item[2]))

        assignments: list[int | None] = [None] * count
        scores = [float("nan")] * count
        margins = [float("nan")] * count
        accepted = [False] * count
        reasons = ["no_legal_candidate"] * count
        proposals_by_root: dict[int, list[int]] = {}
        for index, candidates_for_observation in enumerate(all_candidates):
            if not candidates_for_observation:
                if self.config.reranker_weight > 0.0 and reranker_missing_total:
                    reasons[index] = "reranker_evidence_missing"
                continue
            best_score, best_memory, best_root = candidates_for_observation[0]
            second_score = candidates_for_observation[1][0] if len(candidates_for_observation) > 1 else float("-inf")
            margin = best_score - second_score if np.isfinite(second_score) else float("inf")
            scores[index] = float(best_score)
            margins[index] = float(margin)
            if best_memory in occupied or best_root in occupied_roots:
                reasons[index] = "frame_collision"
                continue
            if best_score < self.config.score_threshold or margin < self.config.margin_threshold:
                reasons[index] = "threshold_or_margin"
                continue
            assignments[index] = int(best_memory)
            accepted[index] = True
            reasons[index] = "accepted"
            proposals_by_root.setdefault(int(best_root), []).append(index)

        # Competition is event-local. A losing proposal is not allowed to
        # fall through to its second candidate in this event.
        competition_losers = 0
        for root_id, indices in sorted(proposals_by_root.items()):
            winner = min(indices, key=lambda index: (-scores[index], -margins[index], snapshot.observation_uids[index]))
            for index in indices:
                if index == winner:
                    continue
                assignments[index] = None
                accepted[index] = False
                reasons[index] = "competition_loser"
                competition_losers += 1

        native_count = sum(1 for candidate in candidates if candidate.is_native)
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
                "observation_hash": observation_hash,
                "candidate_top_k": int(self.config.candidate_top_k),
                "native_candidate_count": int(native_count),
                "dormant_candidate_count": int(len(candidates) - native_count),
                "legal_candidate_count": int(sum(len(values) for values in all_candidates)),
                "reranker_context_candidate_top_k": int(self._reranker_feature_config.get(
                    "candidate_top_k", self.config.candidate_top_k
                )),
                "reranker_decision_candidate_top_k": int(self.config.candidate_top_k),
                "reranker_feature_config": dict(self._reranker_feature_config),
                "reranker_expected_query_observations": int(
                    self._reranker_feature_config.get("query_observations", 1)
                ),
                "reranker_actual_query_observations": self._last_query_observations,
                "reranker_native_memo_bootstrap_count": int(
                    self._reranker_native_memo_bootstrap_count
                ),
                "competition_losers": int(competition_losers),
                "frame_collision_rejections": int(sum(value == "frame_collision" for value in reasons)),
                "reranker_status": self._reranker.provenance if self._reranker is not None else "DISABLED_NOT_FULL",
                "reranker_missing_evidence": int(reranker_missing_total),
                "full_capability_status": (
                    "FULL_Q1_RERANKER_RUNTIME_ACTIVE"
                    if self._reranker is not None and self.config.reranker_weight > 0
                    else "OVERLAY_WITHOUT_RERANKER"
                ),
            },
        )
        self._pending[(video, int(snapshot.frame_id))] = (observation_hash, proposal)
        return proposal

    @staticmethod
    def _observation_value(snapshot: PreAssociationSnapshot, name: str, index: int) -> Any | None:
        value = snapshot.metadata.get(name)
        if value is None:
            return None
        if isinstance(value, Mapping):
            return value.get(snapshot.observation_uids[index], value.get(str(index)))
        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == snapshot.observation_count:
            return value[index]
        if isinstance(value, (list, tuple)) and len(value) == snapshot.observation_count:
            return value[index]
        raise SnapshotContractError(f"{name} must be aligned with observations")

    @staticmethod
    def _observation_value_present(snapshot: PreAssociationSnapshot, name: str, index: int) -> bool:
        """Distinguish an explicit ``None`` evidence row from an absent field."""
        value = snapshot.metadata.get(name)
        if value is None:
            return False
        if isinstance(value, Mapping):
            uid = snapshot.observation_uids[index]
            return uid in value or str(index) in value
        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == snapshot.observation_count:
            return True
        if isinstance(value, (list, tuple)) and len(value) == snapshot.observation_count:
            return True
        raise SnapshotContractError(f"{name} must be aligned with observations")

    def _causal_observation_evidence(
        self,
        snapshot: PreAssociationSnapshot,
        observation_index: int,
        record: _MemoryRecord | None,
    ) -> np.ndarray:
        """Emit the exact seven fields before updating an identity state.

        This is the online counterpart of V9's
        ``build_anchor_evidence_sequence``.  Evidence is attached to the
        logical identity at commit time, so a dormant candidate can later be
        scored without reading post-association IDs or future observations.
        """
        frame = int(snapshot.frame_id)
        score = float(snapshot.det_scores[observation_index])
        if record is None:
            return np.asarray([score, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        current = np.asarray(snapshot.embeddings[observation_index], dtype=np.float32)
        fast = record.state.fast.detach().cpu().numpy()
        slow = record.state.slow.detach().cpu().numpy()
        box = np.asarray(snapshot.boxes_xyxy[observation_index], dtype=np.float32)
        old_box = record.last_box
        if old_box is None:
            area_ratio = 0.0
        else:
            area = max(float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])), 1e-6)
            old_area = max(float(max(0.0, old_box[2] - old_box[0]) * max(0.0, old_box[3] - old_box[1])), 1e-6)
            area_ratio = float(np.clip(np.log(area / old_area), -2.0, 2.0) / 2.0)
        feature_max_gap = int(self._reranker_feature_config.get("max_gap", self.config.max_gap))
        return np.asarray(
            [
                score,
                _cosine(current, fast),
                _cosine(current, slow),
                _cosine(fast, slow),
                min(1.0, max(0, frame - int(record.first_frame)) / 100.0),
                min(1.0, max(0, frame - int(record.last_frame)) / max(feature_max_gap, 1)),
                area_ratio,
            ],
            dtype=np.float32,
        )

    def commit(self, snapshot: PreAssociationSnapshot, final_ids: Any) -> None:
        """Commit native final IDs and update causal memory after assignment."""
        key = (self._video_key(snapshot.video_id), int(snapshot.frame_id))
        pending = self._pending.get(key)
        if pending is None or pending[0] != snapshot.immutable_observation_hash():
            raise SnapshotContractError("commit must immediately follow propose for the same snapshot")
        proposal = pending[1]
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
                proposal_id = proposal.assignments[index]
                proposal_record = self._records.get((video, int(proposal_id))) if proposal_id is not None else None
                was_new = current is None
                if current is None:
                    current_embedding = _normalize(snapshot.embeddings[index])
                    prototype = torch.as_tensor(current_embedding, dtype=torch.float32)
                    state = self._dual.initialize(prototype, frame=int(snapshot.frame_id))
                    source_root_value = self._observation_value(snapshot, "observation_root_ids", index)
                    if proposal_record is not None and proposal_id == int(final_id):
                        root_id = int(proposal_record.root_id)
                        lineage = tuple(proposal_record.lineage)
                    else:
                        root_id = int(final_id if source_root_value is None else source_root_value)
                        lineage = (root_id,)
                    current = _MemoryRecord(
                        state=state,
                        first_frame=int(snapshot.frame_id),
                        last_frame=int(snapshot.frame_id),
                        history=[],
                        evidence_history=[],
                        root_id=root_id,
                        lineage=lineage,
                        last_embedding=current_embedding.copy(),
                    )
                evidence_value = self._observation_value(snapshot, "observation_evidence", index)
                evidence_explicit = self._observation_value_present(snapshot, "observation_evidence", index)
                if evidence_value is None and not evidence_explicit:
                    evidence_value = self._observation_value(snapshot, "evidence_by_observation", index)
                    evidence_explicit = self._observation_value_present(snapshot, "evidence_by_observation", index)
                if evidence_value is None and not evidence_explicit:
                    evidence_value = self._causal_observation_evidence(
                        snapshot, index, None if was_new else current
                    )
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
                evidence = self._evidence_value(evidence_value, "observation_evidence")
                if evidence:
                    current.evidence_history.extend(evidence)
                    current.evidence_history = current.evidence_history[-self.config.memory_capacity :]
                self._append_reranker_rows(
                    current,
                    snapshot.embeddings[index],
                    evidence,
                )
                current.last_box = np.asarray(snapshot.boxes_xyxy[index], dtype=np.float32).copy()
                current.last_embedding = _normalize(snapshot.embeddings[index]).copy()
                self._records[memory_key] = current
        self._pending.pop(key, None)


__all__ = ["OverlayProposal", "TempoTrackConfig", "TempoTrackOverlay"]
