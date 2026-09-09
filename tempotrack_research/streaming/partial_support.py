"""Streaming PSMR primitives used by V7 C9/C10."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..analysis.partial_support import PartialSupportConfig, PartialSupportScorer
from ..models.memory_reliability import MemoryReliabilityCalibrator


@dataclass
class MemoryAnchor:
    fragment_id: str
    root_id: int
    video_id: int
    first_frame: int
    last_frame: int
    features: np.ndarray
    evidence: np.ndarray
    row_indices: list[int] = field(default_factory=list)


@dataclass
class ReactivationDecision:
    fragment_id: str
    candidate_id: str | None
    score: float
    margin: float
    accepted: bool
    reason: str


@dataclass
class ReactivationDiagnostics:
    candidate_pairs: int = 0
    scorer_calls: int = 0
    finite_scores: int = 0
    changed_observation_ids: int = 0
    accepted: int = 0
    rejected: int = 0
    scorer_batches: int = 0
    decisions: list[ReactivationDecision] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_pairs": int(self.candidate_pairs),
            "scorer_calls": int(self.scorer_calls),
            "finite_scores": int(self.finite_scores),
            "changed_observation_ids": int(self.changed_observation_ids),
            "accepted": int(self.accepted),
            "rejected": int(self.rejected),
            "scorer_batches": int(self.scorer_batches),
            "decisions": [item.__dict__ for item in self.decisions],
        }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-6)
    return float(np.dot(a, b) / denom)


def build_anchor_evidence(
    features: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    frames: np.ndarray,
    *,
    max_gap: int = 60,
    previous: MemoryAnchor | None = None,
) -> np.ndarray:
    """Build the fixed seven-dimensional, causal anchor evidence vector.

    The first anchor follows the task-book birth contract exactly.  Later
    anchors use only an already-observed prior anchor; no future rows or GT
    labels are consulted.
    """
    if len(features) == 0:
        return np.zeros(7, dtype=np.float32)
    current = np.asarray(features[0], dtype=np.float32)
    box = np.asarray(boxes_xyxy[0], dtype=np.float32)
    det_score = float(np.asarray(scores)[0])
    if previous is None:
        return np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    fast = np.mean(previous.features[-min(4, len(previous.features)):], axis=0)
    slow = np.mean(previous.features, axis=0)
    prev_box = previous._last_box if hasattr(previous, "_last_box") else box
    area = max(float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])), 1e-6)
    old_area = max(float(max(0.0, prev_box[2] - prev_box[0]) * max(0.0, prev_box[3] - prev_box[1])), 1e-6)
    area_change = float(np.clip(np.log(area / old_area), -2.0, 2.0) / 2.0)
    gap = max(0, int(frames[0]) - int(previous.last_frame))
    return np.asarray([
        det_score,
        _cosine(current, fast),
        _cosine(current, slow),
        _cosine(fast, slow),
        min(1.0, len(previous.features) / 100.0),
        min(1.0, gap / max(int(max_gap), 1)),
        area_change,
    ], dtype=np.float32)


def _split_fragments(records: Sequence[Mapping[str, Any]], embeddings: np.ndarray) -> list[MemoryAnchor]:
    grouped: dict[tuple[int, int, int], list[int]] = {}
    last_by_id: dict[tuple[int, int], int] = {}
    fragment_counter: dict[tuple[int, int], int] = {}
    frame_values: dict[int, list[int]] = {}
    for item in records:
        video = int(item["video_id"]); frame_values.setdefault(video, []).append(int(item["frame_index"]))
    frame_rank = {video: {frame: rank for rank, frame in enumerate(sorted(set(values)))} for video, values in frame_values.items()}
    ordered = sorted(range(len(records)), key=lambda i: (int(records[i]["video_id"]), int(records[i]["frame_index"]), i))
    for index in ordered:
        row = records[index]
        key = (int(row["video_id"]), int(row["track_id"]))
        frame = int(row["frame_index"])
        rank = frame_rank[key[0]][frame]
        if key not in last_by_id or rank > last_by_id[key] + 1:
            fragment_counter[key] = fragment_counter.get(key, -1) + 1
        frag_key = (key[0], key[1], fragment_counter[key])
        grouped.setdefault(frag_key, []).append(index)
        last_by_id[key] = rank
    output: list[MemoryAnchor] = []
    for (video_id, local_id, serial), indices in grouped.items():
        indices.sort(key=lambda i: (int(records[i]["frame_index"]), i))
        feats = np.asarray(embeddings[indices], dtype=np.float32)
        boxes = np.asarray([records[i]["_box_xyxy"] for i in indices], dtype=np.float32)
        scores = np.asarray([records[i]["score"] for i in indices], dtype=np.float32)
        frames = np.asarray([records[i]["frame_index"] for i in indices], dtype=np.int64)
        # Store a compact recent bank.  Deduplication is performed against the
        # existing bank, never by averaging the bank before top-r scoring.
        bank: list[np.ndarray] = []
        for feature in feats:
            if not bank or max(_cosine(feature, old) for old in bank) < 0.95:
                bank.append(feature)
        if not bank:
            bank = [feats[0]]
        bank = bank[-64:]
        anchor = MemoryAnchor(
            fragment_id=f"{video_id}:{local_id}:{serial}", root_id=int(local_id), video_id=int(video_id),
            first_frame=int(frames[0]), last_frame=int(frames[-1]), features=np.asarray(bank, dtype=np.float32),
            evidence=build_anchor_evidence(feats, boxes, scores, frames), row_indices=indices,
        )
        anchor._last_box = boxes[-1]  # type: ignore[attr-defined]
        output.append(anchor)
    return sorted(output, key=lambda item: (item.video_id, item.first_frame, item.fragment_id))


def _causal_candidates(buckets: dict[int, list[MemoryAnchor]], fragment: MemoryAnchor, max_gap: int) -> list[MemoryAnchor]:
    """Retrieve exactly the legal past anchors without an O(N²) scan."""
    first = int(fragment.first_frame)
    lower = first - int(max_gap)
    return [item for frame in range(lower, first) for item in buckets.get(frame, []) if item.last_frame < first]


def _frame_occupancy(records: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], dict[int, int]]:
    """Count current track IDs per video/frame for the official uniqueness contract."""
    occupancy: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for row in records:
        key = (int(row["video_id"]), int(row["frame_index"]))
        track_id = int(row["track_id"])
        counts = occupancy[key]
        counts[track_id] = counts.get(track_id, 0) + 1
    return occupancy


def _has_frame_collision(
    occupancy: Mapping[tuple[int, int], Mapping[int, int]],
    result: Sequence[Mapping[str, Any]],
    fragment: MemoryAnchor,
    target: int,
) -> bool:
    """Reject a causal merge if another current-frame row already owns target."""
    own_target_counts: dict[tuple[int, int], int] = defaultdict(int)
    for row_index in fragment.row_indices:
        row = result[row_index]
        key = (int(row["video_id"]), int(row["frame_index"]))
        if int(row["track_id"]) == int(target):
            own_target_counts[key] += 1
    for key in {(int(result[i]["video_id"]), int(result[i]["frame_index"])) for i in fragment.row_indices}:
        total = int(occupancy.get(key, {}).get(int(target), 0))
        if total - own_target_counts.get(key, 0) > 0:
            return True
    return False


def _apply_fragment_target(
    occupancy: dict[tuple[int, int], dict[int, int]],
    result: list[dict[str, Any]],
    fragment: MemoryAnchor,
    source: int,
    target: int,
) -> int:
    """Rewrite source rows and keep per-frame counts synchronized."""
    changed = 0
    for row_index in fragment.row_indices:
        row = result[row_index]
        if int(row["track_id"]) != int(source):
            continue
        key = (int(row["video_id"]), int(row["frame_index"]))
        counts = occupancy[key]
        counts[source] = counts.get(source, 0) - 1
        if counts[source] <= 0:
            counts.pop(source, None)
        counts[target] = counts.get(target, 0) + 1
        row["track_id"] = int(target)
        changed += 1
    return changed


class StreamingReactivationEngine:
    """Causal partial-support reactivation with a single competition gate."""

    def __init__(
        self,
        config: PartialSupportConfig | None = None,
        *,
        query_observations: int = 1,
        score_threshold: float = 0.60,
        margin_threshold: float = 0.0,
        use_reliability: bool = False,
        reliability_model: MemoryReliabilityCalibrator | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or PartialSupportConfig(query_observations=query_observations)
        self.score_threshold = float(score_threshold)
        self.margin_threshold = float(margin_threshold)
        self.use_reliability = bool(use_reliability)
        self.reliability_model = reliability_model
        self.device = device
        self.scorer = PartialSupportScorer(self.config, beta=0.25 if use_reliability else 0.0).to(device)

    def process_video(self, records: Sequence[Mapping[str, Any]], embeddings: np.ndarray) -> tuple[list[dict[str, Any]], ReactivationDiagnostics]:
        if not records:
            return [], ReactivationDiagnostics()
        fragments = _split_fragments(records, embeddings)
        result = [dict(row) for row in records]
        diagnostics = ReactivationDiagnostics()
        accepted_by_root: dict[tuple[int, int], tuple[float, str]] = {}
        occupancy = _frame_occupancy(result)
        prior_buckets: dict[int, dict[int, list[MemoryAnchor]]] = {}
        root_for_fragment: dict[str, int] = {}
        for fragment in fragments:
            same_video = _causal_candidates(prior_buckets.setdefault(fragment.video_id, {}), fragment, self.config.max_gap)
            if not same_video:
                root_for_fragment[fragment.fragment_id] = fragment.root_id
                prior_buckets[fragment.video_id].setdefault(fragment.last_frame, []).append(fragment)
                continue
            q_indices = fragment.row_indices[: self.config.query_observations]
            query = torch.as_tensor(embeddings[q_indices], dtype=torch.float32, device=self.device)
            # top-k is only a prefilter.  Scoring calls are made on the actual
            # candidate memory, and no candidate is replaced by a GT row.
            query_features = np.asarray(embeddings[q_indices], dtype=np.float32)
            query_features = query_features / np.maximum(np.linalg.norm(query_features, axis=1, keepdims=True), self.config.eps)
            last_features = np.asarray([item.features[-1] for item in same_video], dtype=np.float32)
            last_features = last_features / np.maximum(np.linalg.norm(last_features, axis=1, keepdims=True), self.config.eps)
            pre_values = (query_features @ last_features.T).mean(axis=0)
            pre_scores = [(float(value), item) for value, item in zip(pre_values.tolist(), same_video)]
            candidates = [item for _, item in sorted(pre_scores, key=lambda pair: (-pair[0], pair[1].fragment_id))[: self.config.candidate_top_k]]
            diagnostics.candidate_pairs += len(candidates)
            # Batch the actual formal scorer calls for this fragment.  The
            # memory banks remain individual padded rows with a mask; this is
            # only a launch/CPU-overhead optimization and does not average a
            # bank before its per-query top-r operation.
            max_memory = max(len(candidate.features) for candidate in candidates)
            dim = int(query.shape[-1])
            memory_batch = torch.zeros((len(candidates), max_memory, dim), dtype=torch.float32, device=self.device)
            memory_mask = torch.zeros((len(candidates), max_memory), dtype=torch.bool, device=self.device)
            reliability_batch = None
            if self.use_reliability and self.reliability_model is not None:
                reliability_batch = torch.zeros((len(candidates), max_memory), dtype=torch.float32, device=self.device)
            for candidate_index, candidate in enumerate(candidates):
                count = len(candidate.features)
                memory_batch[candidate_index, :count] = torch.as_tensor(candidate.features, dtype=torch.float32, device=self.device)
                memory_mask[candidate_index, :count] = True
                if reliability_batch is not None:
                    with torch.no_grad():
                        value = self.reliability_model.reliability(torch.as_tensor(candidate.evidence, dtype=torch.float32, device=self.device).reshape(1, 7)).reshape(())
                    reliability_batch[candidate_index, :count] = value
            with torch.no_grad():
                evidence = self.scorer(query.unsqueeze(0).expand(len(candidates), -1, -1), memory_batch, memory_mask=memory_mask, memory_reliability=reliability_batch)
            diagnostics.scorer_calls += len(candidates)
            scored: list[tuple[float, MemoryAnchor]] = []
            values = evidence.score.detach().cpu().numpy().reshape(-1)
            for value, candidate in zip(values.tolist(), candidates):
                if np.isfinite(value):
                    diagnostics.finite_scores += 1
                    scored.append((float(value), candidate))
            scored.sort(key=lambda pair: (-pair[0], pair[1].fragment_id))
            best = scored[0] if scored else (float("-inf"), None)
            second = scored[1][0] if len(scored) > 1 else float("-inf")
            margin = best[0] - second if np.isfinite(best[0]) and np.isfinite(second) else (float("inf") if np.isfinite(best[0]) else float("-inf"))
            accept = best[1] is not None and best[0] >= self.score_threshold and margin >= self.margin_threshold
            reason = "accepted" if accept else ("no_candidate" if best[1] is None else "threshold_or_margin")
            decision = ReactivationDecision(fragment.fragment_id, None if best[1] is None else best[1].fragment_id, float(best[0]), float(margin), bool(accept), reason)
            diagnostics.decisions.append(decision)
            target = fragment.root_id
            if accept and best[1] is not None:
                candidate = best[1]
                root = root_for_fragment.get(candidate.fragment_id, candidate.root_id)
                key = (fragment.video_id, root)
                old = accepted_by_root.get(key)
                # A losing fragment is rejected, not sent to its second best.
                if old is not None and (best[0], fragment.fragment_id) <= (old[0], old[1]):
                    accept = False
                    decision.accepted = False
                    decision.reason = "competition_loser"
                elif root != fragment.root_id and _has_frame_collision(occupancy, result, fragment, root):
                    accept = False
                    decision.accepted = False
                    decision.reason = "frame_collision"
                else:
                    accepted_by_root[key] = (best[0], fragment.fragment_id)
                    target = root
            root_for_fragment[fragment.fragment_id] = target
            if target != fragment.root_id:
                diagnostics.accepted += 1
                diagnostics.changed_observation_ids += _apply_fragment_target(occupancy, result, fragment, fragment.root_id, target)
            else:
                diagnostics.rejected += 1 if best[1] is not None else 0
            prior_buckets[fragment.video_id].setdefault(fragment.last_frame, []).append(fragment)
        return result, diagnostics

    def process_video_batched(self, records: Sequence[Mapping[str, Any]], embeddings: np.ndarray, *, pair_batch_size: int = 2048) -> tuple[list[dict[str, Any]], ReactivationDiagnostics]:
        """Causal replay with globally batched formal scorer calls.

        Candidate lists are formed in chronological order before any score is
        computed, so every memory row is still strictly in the past.  Scores
        are then evaluated in padded batches with explicit query/memory masks;
        the sequential root/competition decision is applied only after those
        scores are available.  This preserves the online decision order while
        avoiding hundreds of thousands of tiny GPU launches on official V6.
        """
        if not records:
            return [], ReactivationDiagnostics()
        fragments = _split_fragments(records, embeddings)
        result = [dict(row) for row in records]
        diagnostics = ReactivationDiagnostics()
        occupancy = _frame_occupancy(result)
        prior_buckets: dict[int, dict[int, list[MemoryAnchor]]] = {}
        candidate_lists: list[list[MemoryAnchor]] = []
        pair_refs: list[tuple[int, MemoryAnchor]] = []
        for fragment_index, fragment in enumerate(fragments):
            same_video = _causal_candidates(prior_buckets.setdefault(fragment.video_id, {}), fragment, self.config.max_gap)
            if same_video:
                q_indices = fragment.row_indices[: self.config.query_observations]
                query_features = np.asarray(embeddings[q_indices], dtype=np.float32)
                query_features = query_features / np.maximum(np.linalg.norm(query_features, axis=1, keepdims=True), self.config.eps)
                last_features = np.asarray([item.features[-1] for item in same_video], dtype=np.float32)
                last_features = last_features / np.maximum(np.linalg.norm(last_features, axis=1, keepdims=True), self.config.eps)
                pre_values = (query_features @ last_features.T).mean(axis=0)
                scored_prefilter = [(float(value), item) for value, item in zip(pre_values.tolist(), same_video)]
                candidates = [item for _, item in sorted(scored_prefilter, key=lambda pair: (-pair[0], pair[1].fragment_id))[: self.config.candidate_top_k]]
            else:
                candidates = []
            candidate_lists.append(candidates)
            diagnostics.candidate_pairs += len(candidates)
            pair_refs.extend((fragment_index, candidate) for candidate in candidates)
            prior_buckets[fragment.video_id].setdefault(fragment.last_frame, []).append(fragment)

        scored_by_fragment: dict[int, list[tuple[float, MemoryAnchor]]] = defaultdict(list)
        max_query = max(1, int(self.config.query_observations))
        for start in range(0, len(pair_refs), int(pair_batch_size)):
            chunk = pair_refs[start:start + int(pair_batch_size)]
            if not chunk:
                continue
            queries = []
            query_masks = []
            memories = []
            memory_masks = []
            evidences = []
            max_memory = max(len(candidate.features) for _, candidate in chunk)
            dim = int(embeddings.shape[1])
            for fragment_index, candidate in chunk:
                fragment = fragments[fragment_index]
                q_indices = fragment.row_indices[:max_query]
                q = np.asarray(embeddings[q_indices], dtype=np.float32)
                qmask = np.zeros(max_query, dtype=bool); qmask[:len(q)] = True
                qpad = np.zeros((max_query, dim), dtype=np.float32); qpad[:len(q)] = q
                mem = np.asarray(candidate.features, dtype=np.float32)
                mmask = np.zeros(max_memory, dtype=bool); mmask[:len(mem)] = True
                mpad = np.zeros((max_memory, dim), dtype=np.float32); mpad[:len(mem)] = mem
                queries.append(qpad); query_masks.append(qmask); memories.append(mpad); memory_masks.append(mmask); evidences.append(candidate.evidence)
            query_tensor = torch.as_tensor(np.asarray(queries), dtype=torch.float32, device=self.device)
            query_mask_tensor = torch.as_tensor(np.asarray(query_masks), dtype=torch.bool, device=self.device)
            memory_tensor = torch.as_tensor(np.asarray(memories), dtype=torch.float32, device=self.device)
            memory_mask_tensor = torch.as_tensor(np.asarray(memory_masks), dtype=torch.bool, device=self.device)
            reliability_tensor = None
            if self.use_reliability and self.reliability_model is not None:
                with torch.no_grad():
                    rel = self.reliability_model.reliability(torch.as_tensor(np.asarray(evidences), dtype=torch.float32, device=self.device)).reshape(-1, 1)
                reliability_tensor = rel.expand(-1, max_memory)
            with torch.no_grad():
                scores = self.scorer(query_tensor, memory_tensor, query_mask=query_mask_tensor, memory_mask=memory_mask_tensor, memory_reliability=reliability_tensor).score
            diagnostics.scorer_batches += 1
            values = scores.detach().cpu().numpy().reshape(-1)
            diagnostics.scorer_calls += len(chunk)
            for (fragment_index, candidate), value in zip(chunk, values.tolist()):
                if np.isfinite(value):
                    diagnostics.finite_scores += 1
                    scored_by_fragment[fragment_index].append((float(value), candidate))

        accepted_by_root: dict[tuple[int, int], tuple[float, str]] = {}
        root_for_fragment: dict[str, int] = {}
        # Apply only after all formal score batches.  The iteration is still
        # chronological and therefore preserves one-winner competition.
        for fragment_index, fragment in enumerate(fragments):
            scored = sorted(scored_by_fragment.get(fragment_index, []), key=lambda pair: (-pair[0], pair[1].fragment_id))
            best = scored[0] if scored else (float("-inf"), None)
            second = scored[1][0] if len(scored) > 1 else float("-inf")
            margin = best[0] - second if np.isfinite(best[0]) and np.isfinite(second) else (float("inf") if np.isfinite(best[0]) else float("-inf"))
            accept = best[1] is not None and best[0] >= self.score_threshold and margin >= self.margin_threshold
            reason = "accepted" if accept else ("no_candidate" if best[1] is None else "threshold_or_margin")
            decision = ReactivationDecision(fragment.fragment_id, None if best[1] is None else best[1].fragment_id, float(best[0]), float(margin), bool(accept), reason)
            diagnostics.decisions.append(decision)
            target = fragment.root_id
            if accept and best[1] is not None:
                candidate = best[1]; root = root_for_fragment.get(candidate.fragment_id, candidate.root_id); key = (fragment.video_id, root)
                old = accepted_by_root.get(key)
                if old is not None and (best[0], fragment.fragment_id) <= (old[0], old[1]):
                    accept = False; decision.accepted = False; decision.reason = "competition_loser"
                elif root != fragment.root_id and _has_frame_collision(occupancy, result, fragment, root):
                    accept = False; decision.accepted = False; decision.reason = "frame_collision"
                else:
                    accepted_by_root[key] = (best[0], fragment.fragment_id); target = root
            root_for_fragment[fragment.fragment_id] = target
            if target != fragment.root_id:
                diagnostics.accepted += 1
                diagnostics.changed_observation_ids += _apply_fragment_target(occupancy, result, fragment, fragment.root_id, target)
            else:
                diagnostics.rejected += 1 if best[1] is not None else 0
        return result, diagnostics


__all__ = ["MemoryAnchor", "ReactivationDecision", "ReactivationDiagnostics", "StreamingReactivationEngine", "build_anchor_evidence"]
