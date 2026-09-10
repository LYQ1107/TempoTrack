"""Streaming PSMR primitives used by V8 per-anchor C9/C10."""

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
    # ``row_indices`` is the retained memory bank.  The full fragment rows
    # remain separate so a successful reactivation rewrites every original
    # observation, including observations removed by feature deduplication.
    fragment_rows: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        self.evidence = np.asarray(self.evidence, dtype=np.float32)
        if self.features.ndim != 2:
            raise ValueError(f"MemoryAnchor.features must be [K,D], got {self.features.shape}")
        if self.evidence.ndim != 2 or self.evidence.shape[-1] != 7:
            raise ValueError(f"MemoryAnchor.evidence must be [K,7], got {self.evidence.shape}")
        if len(self.features) != len(self.evidence) or len(self.features) != len(self.row_indices):
            raise ValueError("MemoryAnchor feature/evidence/row_indices lengths must agree")
        if not self.fragment_rows:
            self.fragment_rows = list(self.row_indices)


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
    fragments_total: int = 0
    fragments_with_legal_dormant: int = 0
    candidate_pairs_before_topk: int = 0
    candidate_pairs_after_topk: int = 0
    score_gate_pass: int = 0
    margin_gate_pass: int = 0
    competition_loser: int = 0
    frame_collision: int = 0
    gap_bins: dict[str, int] = field(default_factory=dict)
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
            "fragments_total": int(self.fragments_total),
            "fragments_with_legal_dormant": int(self.fragments_with_legal_dormant),
            "candidate_pairs_before_topk": int(self.candidate_pairs_before_topk),
            "candidate_pairs_after_topk": int(self.candidate_pairs_after_topk),
            "score_gate_pass": int(self.score_gate_pass),
            "margin_gate_pass": int(self.margin_gate_pass),
            "competition_loser": int(self.competition_loser),
            "frame_collision": int(self.frame_collision),
            "gap_bins": dict(self.gap_bins),
            "decisions": [item.__dict__ for item in self.decisions],
        }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-6)
    return float(np.dot(a, b) / denom)


def build_anchor_evidence_sequence(
    features: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    frames: np.ndarray,
    *,
    alpha_fast: float = 0.70,
    alpha_slow: float = 0.15,
    max_gap: int = 60,
) -> np.ndarray:
    """Build one causal seven-dimensional evidence vector per observation.

    Evidence for observation k is emitted before z_k updates either memory
    state.  This makes the sequence causal and gives every retained anchor a
    distinct evidence row instead of assigning one fragment-level vector to
    the whole bank.
    """
    feats = np.asarray(features, dtype=np.float32)
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    det_scores = np.asarray(scores, dtype=np.float32)
    frame_values = np.asarray(frames, dtype=np.int64)
    if feats.ndim != 2 or boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("features must be [N,D] and boxes_xyxy must be [N,4]")
    n = len(feats)
    if len(boxes) != n or len(det_scores) != n or len(frame_values) != n:
        raise ValueError("anchor evidence inputs must have the same length")
    if n == 0:
        return np.zeros((0, 7), dtype=np.float32)
    if not (0.0 <= float(alpha_fast) <= 1.0 and 0.0 <= float(alpha_slow) <= 1.0):
        raise ValueError("alpha_fast and alpha_slow must be in [0,1]")

    def normalize(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        return value / max(float(np.linalg.norm(value)), 1e-6)

    first_frame = int(frame_values[0])
    fast = normalize(feats[0])
    slow = normalize(feats[0])
    previous_box = boxes[0]
    previous_frame = int(frame_values[0])
    evidence = np.zeros((n, 7), dtype=np.float32)
    for index in range(n):
        current = feats[index]
        box = boxes[index]
        frame = int(frame_values[index])
        if index == 0:
            # Birth contract: detection score is retained and the initial
            # fast/slow states are identical, so their agreement is 1.
            evidence[index] = np.asarray([det_scores[index], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            area = max(float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])), 1e-6)
            old_area = max(float(max(0.0, previous_box[2] - previous_box[0]) * max(0.0, previous_box[3] - previous_box[1])), 1e-6)
            evidence[index] = np.asarray([
                det_scores[index],
                _cosine(current, fast),
                _cosine(current, slow),
                _cosine(fast, slow),
                min(1.0, max(0, frame - first_frame) / 100.0),
                min(1.0, max(0, frame - previous_frame) / max(int(max_gap), 1)),
                float(np.clip(np.log(area / old_area), -2.0, 2.0) / 2.0),
            ], dtype=np.float32)
        # The state update happens only after evidence has been recorded.
        z = normalize(current)
        fast = normalize((1.0 - float(alpha_fast)) * fast + float(alpha_fast) * z)
        slow = normalize((1.0 - float(alpha_slow)) * slow + float(alpha_slow) * z)
        previous_box = box
        previous_frame = frame
    return evidence


def build_anchor_evidence(
    features: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    frames: np.ndarray,
    *,
    max_gap: int = 60,
) -> np.ndarray:
    """Backward-compatible single-row helper; new code uses the sequence."""
    sequence = build_anchor_evidence_sequence(features, boxes_xyxy, scores, frames, max_gap=max_gap)
    return sequence[-1] if len(sequence) else np.zeros(7, dtype=np.float32)


def build_memory_anchor(
    *,
    fragment_id: str,
    root_id: int,
    video_id: int,
    rows: Sequence[int],
    features: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    frames: np.ndarray,
    dedup_cos: float = 0.95,
    capacity: int = 64,
    max_gap: int = 60,
) -> MemoryAnchor:
    """Build the canonical synchronized memory bank used by train/inference."""
    all_rows = [int(row) for row in rows]
    if not all_rows:
        raise ValueError("a memory fragment must contain at least one row")
    local_features = np.asarray(features[all_rows], dtype=np.float32)
    local_boxes = np.asarray(boxes_xyxy[all_rows], dtype=np.float32)
    local_scores = np.asarray(scores[all_rows], dtype=np.float32)
    local_frames = np.asarray(frames[all_rows], dtype=np.int64)
    sequence = build_anchor_evidence_sequence(local_features, local_boxes, local_scores, local_frames, max_gap=max_gap)
    bank_features: list[np.ndarray] = []
    bank_evidence: list[np.ndarray] = []
    bank_rows: list[int] = []
    for row, feature, evidence in zip(all_rows, local_features, sequence):
        if not bank_features or max(_cosine(feature, old) for old in bank_features) < float(dedup_cos):
            bank_features.append(np.asarray(feature, dtype=np.float32))
            bank_evidence.append(np.asarray(evidence, dtype=np.float32))
            bank_rows.append(int(row))
    # The bank is chronological after causal deduplication.  Retain the
    # most recent anchors and slice all three synchronized arrays together;
    # ``[:capacity]`` would silently keep the oldest observations.
    retain = max(0, int(capacity))
    if retain == 0:
        raise ValueError("memory capacity must be positive")
    bank_features = bank_features[-retain:]
    bank_evidence = bank_evidence[-retain:]
    bank_rows = bank_rows[-retain:]
    return MemoryAnchor(
        fragment_id=str(fragment_id), root_id=int(root_id), video_id=int(video_id),
        first_frame=int(local_frames[0]), last_frame=int(local_frames[-1]),
        features=np.asarray(bank_features, dtype=np.float32),
        evidence=np.asarray(bank_evidence, dtype=np.float32),
        row_indices=bank_rows, fragment_rows=all_rows,
    )


def _split_fragments(
    records: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    *,
    capacity: int = 64,
    max_gap: int = 60,
) -> list[MemoryAnchor]:
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
        anchor = build_memory_anchor(
            fragment_id=f"{video_id}:{local_id}:{serial}", root_id=int(local_id), video_id=int(video_id),
            rows=indices, features=embeddings, boxes_xyxy=np.asarray([records[i]["_box_xyxy"] for i in range(len(records))], dtype=np.float32),
            scores=np.asarray([records[i]["score"] for i in range(len(records))], dtype=np.float32),
            frames=np.asarray([records[i]["frame_index"] for i in range(len(records))], dtype=np.int64),
            capacity=int(capacity), max_gap=int(max_gap),
        )
        output.append(anchor)
    return sorted(output, key=lambda item: (item.video_id, item.first_frame, item.fragment_id))


def _causal_candidates(
    buckets: dict[int, list[MemoryAnchor]],
    fragment: MemoryAnchor,
    max_gap: int,
    min_gap: int = 0,
) -> list[MemoryAnchor]:
    """Retrieve exactly the legal past anchors without an O(N²) scan."""
    first = int(fragment.first_frame)
    lower = first - int(max_gap)
    upper = first - int(min_gap)
    if upper < lower:
        return []
    return [
        item
        for frame in range(lower, upper + 1)
        for item in buckets.get(frame, [])
        if item.last_frame < first and int(min_gap) <= first - int(item.last_frame) <= int(max_gap)
    ]


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
    for row_index in fragment.fragment_rows:
        row = result[row_index]
        key = (int(row["video_id"]), int(row["frame_index"]))
        if int(row["track_id"]) == int(target):
            own_target_counts[key] += 1
    for key in {(int(result[i]["video_id"]), int(result[i]["frame_index"])) for i in fragment.fragment_rows}:
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
    for row_index in fragment.fragment_rows:
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
        self.scorer = PartialSupportScorer(self.config, beta=0.0).to(device)
        if self.reliability_model is not None:
            self.reliability_model = self.reliability_model.to(device)

    def process_video(self, records: Sequence[Mapping[str, Any]], embeddings: np.ndarray) -> tuple[list[dict[str, Any]], ReactivationDiagnostics]:
        # Keep one implementation of candidate timing, scoring and event
        # competition.  The old scalar path had a separate persistent root
        # table and could disagree with batched replay after a reappearance.
        return self.process_video_batched(records, embeddings, pair_batch_size=max(1, len(records)))

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
        fragments = _split_fragments(
            records,
            embeddings,
            capacity=int(self.config.memory_capacity),
            max_gap=int(self.config.max_gap),
        )
        result = [dict(row) for row in records]
        diagnostics = ReactivationDiagnostics()
        diagnostics.fragments_total = len(fragments)
        occupancy = _frame_occupancy(result)
        prior_buckets: dict[int, dict[int, list[MemoryAnchor]]] = {}
        candidate_lists: list[list[MemoryAnchor]] = []
        pair_refs: list[tuple[int, MemoryAnchor]] = []
        for fragment_index, fragment in enumerate(fragments):
            same_video = _causal_candidates(
                prior_buckets.setdefault(fragment.video_id, {}),
                fragment,
                self.config.max_gap,
                self.config.min_dormant_gap,
            )
            if same_video:
                diagnostics.fragments_with_legal_dormant += 1
            diagnostics.candidate_pairs_before_topk += len(same_video)
            if same_video:
                q_indices = fragment.fragment_rows[: self.config.query_observations]
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
            diagnostics.candidate_pairs_after_topk += len(candidates)
            diagnostics.candidate_pairs += len(candidates)
            for candidate in candidates:
                gap = int(fragment.first_frame) - int(candidate.last_frame)
                if gap <= 10:
                    bucket = "0-10"
                elif gap <= 30:
                    bucket = "10-30"
                elif gap <= 60:
                    bucket = "30-60"
                elif gap <= 90:
                    bucket = "60-90"
                elif gap <= 120:
                    bucket = "90-120"
                elif gap <= 180:
                    bucket = "120-180"
                elif gap <= 240:
                    bucket = "180-240"
                else:
                    bucket = "240-360"
                diagnostics.gap_bins[bucket] = diagnostics.gap_bins.get(bucket, 0) + 1
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
                q_indices = fragment.fragment_rows[:max_query]
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
                reliability_tensor = torch.zeros((len(chunk), max_memory), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    for index, evidence_rows in enumerate(evidences):
                        values = self.reliability_model.reliability(torch.as_tensor(evidence_rows, dtype=torch.float32, device=self.device)).reshape(-1)
                        reliability_tensor[index, :len(values)] = values
            with torch.no_grad():
                beta = self.reliability_model.reliability_scale if self.use_reliability and self.reliability_model is not None else None
                scores = self.scorer(query_tensor, memory_tensor, query_mask=query_mask_tensor, memory_mask=memory_mask_tensor, memory_reliability=reliability_tensor, reliability_scale=beta).score
            diagnostics.scorer_batches += 1
            values = scores.detach().cpu().numpy().reshape(-1)
            diagnostics.scorer_calls += len(chunk)
            for (fragment_index, candidate), value in zip(chunk, values.tolist()):
                if np.isfinite(value):
                    diagnostics.finite_scores += 1
                    scored_by_fragment[fragment_index].append((float(value), candidate))

        root_for_fragment: dict[str, int] = {}
        # Competition is local to one decision frame.  In particular, a root
        # that won an event is not reserved for the rest of the video; a later
        # disappearance may reactivate it again.  We still apply the official
        # frame-collision check against the evolving output.
        event_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for fragment_index, fragment in enumerate(fragments):
            query_count = min(max(1, int(self.config.query_observations)), len(fragment.fragment_rows))
            decision_frame = int(records[fragment.fragment_rows[query_count - 1]]["frame_index"])
            event_groups[(int(fragment.video_id), decision_frame)].append(fragment_index)

        for event_key in sorted(event_groups):
            proposals: list[dict[str, Any]] = []
            for fragment_index in sorted(event_groups[event_key]):
                fragment = fragments[fragment_index]
                scored = sorted(scored_by_fragment.get(fragment_index, []), key=lambda pair: (-pair[0], pair[1].fragment_id))
                best = scored[0] if scored else (float("-inf"), None)
                second = scored[1][0] if len(scored) > 1 else float("-inf")
                margin = best[0] - second if np.isfinite(best[0]) and np.isfinite(second) else (float("inf") if np.isfinite(best[0]) else float("-inf"))
                score_pass = best[1] is not None and best[0] >= self.score_threshold
                margin_pass = best[1] is not None and margin >= self.margin_threshold
                if score_pass:
                    diagnostics.score_gate_pass += 1
                if margin_pass:
                    diagnostics.margin_gate_pass += 1
                accept = bool(score_pass and margin_pass)
                reason = "accepted" if accept else ("no_candidate" if best[1] is None else "threshold_or_margin")
                decision = ReactivationDecision(
                    fragment.fragment_id,
                    None if best[1] is None else best[1].fragment_id,
                    float(best[0]),
                    float(margin),
                    accept,
                    reason,
                )
                diagnostics.decisions.append(decision)
                root = None
                if accept and best[1] is not None:
                    root = root_for_fragment.get(best[1].fragment_id, best[1].root_id)
                proposals.append({
                    "fragment_index": fragment_index,
                    "fragment": fragment,
                    "best": best,
                    "margin": float(margin),
                    "decision": decision,
                    "root": root,
                    "accept": accept,
                })

            winners: dict[int, dict[str, Any]] = {}
            for proposal in proposals:
                if not proposal["accept"] or proposal["root"] is None:
                    continue
                root = int(proposal["root"])
                current = winners.get(root)
                candidate_key = (-float(proposal["best"][0]), -float(proposal["margin"]), str(proposal["fragment"].fragment_id))
                current_key = None if current is None else (-float(current["best"][0]), -float(current["margin"]), str(current["fragment"].fragment_id))
                if current is None or candidate_key < current_key:
                    winners[root] = proposal

            for proposal in proposals:
                fragment = proposal["fragment"]
                decision = proposal["decision"]
                target = int(fragment.root_id)
                if proposal["accept"] and proposal["root"] is not None:
                    root = int(proposal["root"])
                    if winners.get(root) is not proposal:
                        proposal["accept"] = False
                        decision.accepted = False
                        decision.reason = "competition_loser"
                        diagnostics.competition_loser += 1
                    elif root != fragment.root_id and _has_frame_collision(occupancy, result, fragment, root):
                        proposal["accept"] = False
                        decision.accepted = False
                        decision.reason = "frame_collision"
                        diagnostics.frame_collision += 1
                    else:
                        target = root
                root_for_fragment[fragment.fragment_id] = target
                if target != fragment.root_id:
                    diagnostics.accepted += 1
                    diagnostics.changed_observation_ids += _apply_fragment_target(
                        occupancy, result, fragment, fragment.root_id, target
                    )
                elif proposal["best"][1] is not None:
                    diagnostics.rejected += 1
        return result, diagnostics


__all__ = ["MemoryAnchor", "ReactivationDecision", "ReactivationDiagnostics", "StreamingReactivationEngine", "build_anchor_evidence", "build_anchor_evidence_sequence", "build_memory_anchor"]
