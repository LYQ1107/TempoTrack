from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..association.serialization import VideoLocalIdMap
from ..memory.identity_history import IdentityHistory, MemoryObservation
from .retrieval import DormantCandidate, build_dormant_candidates, topk_memory_score
from .state import DormantIdentity, TentativeTrack
from .transport import (
    balanced_sinkhorn,
    observation_reliability,
    streaming_ground_cost,
    unbalanced_sinkhorn,
)


@dataclass
class StreamingConfig:
    tentative_observations: int = 2
    dormant_horizon: int = 30
    candidate_top_k: int = 8
    top_r: int = 3
    transport_mode: str = "uot"
    epsilon: float = 0.05
    tau: float = 0.5
    sinkhorn_iters: int = 50
    lambda_mass: float = 0.2
    lambda_time: float = 0.1
    tau_evidence: float = 0.0
    tau_competition: float = 0.05
    tau_mass: float = 0.20
    tau_topk: float = 0.60
    tau_margin_topk: float = 0.05
    lambda_app: float = 0.80
    lambda_geo: float = 0.20
    margin_temperature: float = 0.10
    agreement_temperature: float = 0.10
    reliability_weighted: bool = False


@dataclass
class RecoveryEvidence:
    tentative_id: int
    dormant_id: int
    transport_cost: float
    transport_mass: float
    gap: int
    evidence_score: float
    competition_margin: float = float("nan")
    accepted: bool = False


def _as_observation(frame: Any, row: int) -> MemoryObservation:
    box = np.asarray(frame.boxes_xyxy[row], dtype=np.float32)
    raw = np.asarray(frame.embeddings_raw[row], dtype=np.float32)
    norm = raw / max(float(np.linalg.norm(raw)), 1e-12)
    width, height = int(frame.image_width), int(frame.image_height)
    bw = max(float(box[2] - box[0]), 1e-6)
    bh = max(float(box[3] - box[1]), 1e-6)
    area = bw * bh
    def _finite(array, index):
        if array is None:
            return None
        value = float(array[index])
        return value if np.isfinite(value) else None
    return MemoryObservation(
        feature_raw=raw,
        feature_norm=norm.astype(np.float32),
        bbox_xyxy=box,
        frame_id=int(frame.frame_id),
        image_width=width,
        image_height=height,
        aspect_ratio=float(bw / bh),
        area_pixels=float(area),
        area_norm=float(area / max(width * height, 1)),
        det_score=float(frame.scores[row]),
        association_score=_finite(getattr(frame, "accepted_score", None), row),
        association_margin=_finite(getattr(frame, "detection_margin", None), row),
        fast_score=_finite(getattr(frame, "fast_score", None), row),
        slow_score=_finite(getattr(frame, "slow_score", None), row),
        is_birth=False,
    )


class StreamingIdentityRecovery:
    def __init__(self, cfg: StreamingConfig):
        if cfg.transport_mode not in {"topk", "balanced", "uot", "rg-smt"}:
            raise ValueError(f"unknown streaming transport mode: {cfg.transport_mode}")
        self.cfg = cfg
        self.recovery_events: list[dict[str, Any]] = []
        self._diagnostics: dict[str, Any] = {}

    def _score_candidate(self, tentative: TentativeTrack, dormant: DormantIdentity, candidate: DormantCandidate) -> RecoveryEvidence:
        started = perf_counter()
        if self.cfg.transport_mode == "topk":
            score = topk_memory_score(tentative, dormant, top_r=self.cfg.top_r)
            return RecoveryEvidence(tentative.frontend_track_id, dormant.identity_id, -score, 1.0, candidate.gap, score)
        cost = streaming_ground_cost(tentative.observations, dormant.history.observations, lambda_app=self.cfg.lambda_app, lambda_geo=self.cfg.lambda_geo)
        n = cost.shape[0]
        m = cost.shape[1]
        if self.cfg.transport_mode == "balanced":
            result = balanced_sinkhorn(cost, epsilon=self.cfg.epsilon, iterations=self.cfg.sinkhorn_iters)
        else:
            if self.cfg.transport_mode == "rg-smt" or self.cfg.reliability_weighted:
                left_weights = torch.as_tensor([observation_reliability(item, cfg=self.cfg) for item in tentative.observations], dtype=torch.float32)
                right_weights = torch.as_tensor([observation_reliability(item, cfg=self.cfg) for item in dormant.history.observations], dtype=torch.float32)
            else:
                left_weights = torch.full((n,), 1.0 / max(n, 1))
                right_weights = torch.full((m,), 1.0 / max(m, 1))
            result = unbalanced_sinkhorn(cost, left_weights, right_weights, epsilon=self.cfg.epsilon, tau=self.cfg.tau, iterations=self.cfg.sinkhorn_iters)
        evidence = -float(result.normalized_cost) + self.cfg.lambda_mass * float(np.log(result.transport_mass + 1e-8)) - self.cfg.lambda_time * candidate.gap / max(float(self.cfg.dormant_horizon), 1.0)
        return RecoveryEvidence(tentative.frontend_track_id, dormant.identity_id, float(result.normalized_cost), float(result.transport_mass), candidate.gap, evidence, accepted=False)

    def _score_transport_batch(self, entries: list[tuple[TentativeTrack, DormantIdentity, DormantCandidate]]) -> list[RecoveryEvidence]:
        """Score one decision frame's transport candidates in one tensor batch.

        The per-candidate objective is unchanged from ``_score_candidate``.
        Padding is masked in log space, and every candidate still receives its
        own normalized cost, mass, and evidence threshold.  Batching avoids
        launching thousands of tiny 50-iteration torch kernels serially for a
        long video while retaining the V6 Sinkhorn contract.
        """
        if not entries:
            return []
        costs: list[torch.Tensor] = []
        sizes: list[tuple[int, int]] = []
        for tentative, dormant, _ in entries:
            cost = streaming_ground_cost(
                tentative.observations,
                dormant.history.observations,
                lambda_app=self.cfg.lambda_app,
                lambda_geo=self.cfg.lambda_geo,
            )
            if cost.numel() == 0 or not torch.isfinite(cost).all():
                # Preserve the existing explicit invalid-cost behaviour for a
                # malformed candidate rather than allowing padding to hide it.
                costs.append(cost)
            else:
                costs.append(cost.to(dtype=torch.float32))
            sizes.append((int(cost.shape[0]), int(cost.shape[1])))
        if any(cost.numel() == 0 or not torch.isfinite(cost).all() for cost in costs):
            return [self._score_candidate(tentative, dormant, candidate) for tentative, dormant, candidate in entries]

        batch_size = len(entries)
        max_n = max(size[0] for size in sizes)
        max_m = max(size[1] for size in sizes)
        device = costs[0].device
        batch_cost = torch.zeros((batch_size, max_n, max_m), dtype=torch.float32, device=device)
        valid = torch.zeros((batch_size, max_n, max_m), dtype=torch.bool, device=device)
        row_valid = torch.zeros((batch_size, max_n), dtype=torch.bool, device=device)
        col_valid = torch.zeros((batch_size, max_m), dtype=torch.bool, device=device)
        left = torch.zeros((batch_size, max_n), dtype=torch.float32, device=device)
        right = torch.zeros((batch_size, max_m), dtype=torch.float32, device=device)
        for index, ((tentative, dormant, _), cost, (n, m)) in enumerate(zip(entries, costs, sizes)):
            batch_cost[index, :n, :m] = cost
            valid[index, :n, :m] = True
            row_valid[index, :n] = True
            col_valid[index, :m] = True
            if self.cfg.transport_mode == "rg-smt" or self.cfg.reliability_weighted:
                left_values = [observation_reliability(item, cfg=self.cfg) for item in tentative.observations]
                right_values = [observation_reliability(item, cfg=self.cfg) for item in dormant.history.observations]
                left[index, :n] = torch.as_tensor(left_values, dtype=torch.float32, device=device)
                right[index, :m] = torch.as_tensor(right_values, dtype=torch.float32, device=device)
            else:
                left[index, :n] = 1.0 / max(n, 1)
                right[index, :m] = 1.0 / max(m, 1)
        left = left / left.sum(dim=1, keepdim=True).clamp_min(1e-12)
        right = right / right.sum(dim=1, keepdim=True).clamp_min(1e-12)
        neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
        log_k = torch.where(valid, -batch_cost / float(self.cfg.epsilon), neg_inf)
        log_a = torch.where(row_valid, left.clamp_min(1e-12).log(), neg_inf)
        log_b = torch.where(col_valid, right.clamp_min(1e-12).log(), neg_inf)
        log_u = torch.where(row_valid, torch.zeros_like(log_a), neg_inf)
        log_v = torch.where(col_valid, torch.zeros_like(log_b), neg_inf)
        if self.cfg.transport_mode == "balanced":
            eta = 1.0
        else:
            eta = float(self.cfg.tau) / (float(self.cfg.tau) + float(self.cfg.epsilon))
        with torch.no_grad():
            for _ in range(int(self.cfg.sinkhorn_iters)):
                new_u = eta * (log_a - torch.logsumexp(log_k + log_v[:, None, :], dim=2))
                # Mask before the column update: padded rows otherwise form
                # the indeterminate expression (-inf)-(-inf), contaminating
                # valid columns with NaN.
                log_u = torch.where(row_valid, new_u, neg_inf)
                new_v = eta * (log_b - torch.logsumexp(log_k.transpose(1, 2) + log_u[:, None, :], dim=2))
                log_v = torch.where(col_valid, new_v, neg_inf)
            gamma = torch.exp(log_u[:, :, None] + log_k + log_v[:, None, :])

        results: list[RecoveryEvidence] = []
        for index, ((tentative, dormant, candidate), (n, m)) in enumerate(zip(entries, sizes)):
            matrix = gamma[index, :n, :m]
            cost = costs[index]
            mass_tensor = matrix.sum()
            mass = float(mass_tensor.detach().cpu())
            finite = bool(torch.isfinite(matrix).all() and np.isfinite(mass) and mass > 1e-8)
            normalized = float((matrix * cost).sum().detach().cpu() / max(mass, 1e-8)) if finite else float("inf")
            if not np.isfinite(normalized):
                finite = False
            evidence = -normalized + self.cfg.lambda_mass * float(np.log(mass + 1e-8)) - self.cfg.lambda_time * candidate.gap / max(float(self.cfg.dormant_horizon), 1.0)
            results.append(RecoveryEvidence(tentative.frontend_track_id, dormant.identity_id, normalized, mass, candidate.gap, evidence, accepted=False))
        return results

    def _decide(self, ready: list[TentativeTrack], dormant: dict[int, DormantIdentity], decision_frame: int) -> dict[int, int]:
        if not ready:
            return {}
        candidate_map = build_dormant_candidates(ready, list(dormant.values()), max_gap=self.cfg.dormant_horizon, top_k=self.cfg.candidate_top_k)
        all_ids = sorted({candidate.dormant_id for values in candidate_map.values() for candidate in values})
        evidence_by_row: dict[int, list[RecoveryEvidence]] = {}
        score_matrix = np.full((len(ready), len(all_ids) + len(ready)), -1e9, dtype=np.float64)
        batched_entries: list[tuple[TentativeTrack, DormantIdentity, DormantCandidate]] = []
        for row, tentative in enumerate(ready):
            values = []
            for candidate in candidate_map.get(tentative.frontend_track_id, []):
                if self.cfg.transport_mode == "topk":
                    values.append(self._score_candidate(tentative, dormant[candidate.dormant_id], candidate))
                else:
                    batched_entries.append((tentative, dormant[candidate.dormant_id], candidate))
            evidence_by_row[tentative.frontend_track_id] = values
            # Explicit unmatched/new-identity competition column.
            score_matrix[row, len(all_ids) + row] = 0.0
        if batched_entries:
            for evidence in self._score_transport_batch(batched_entries):
                evidence_by_row.setdefault(evidence.tentative_id, []).append(evidence)
        row_by_id = {int(tentative.frontend_track_id): row for row, tentative in enumerate(ready)}
        for tentative_id, values in evidence_by_row.items():
            row = row_by_id[int(tentative_id)]
            for evidence in values:
                score_matrix[row, all_ids.index(int(evidence.dormant_id))] = evidence.evidence_score
        assignments = {}
        if score_matrix.size:
            rows, cols = linear_sum_assignment(-score_matrix)
            for row, col in zip(rows.tolist(), cols.tolist()):
                tentative = ready[row]
                values = sorted(evidence_by_row.get(tentative.frontend_track_id, []), key=lambda item: item.evidence_score, reverse=True)
                best = values[0] if values else None
                second = values[1].evidence_score if len(values) > 1 else 0.0
                margin = (best.evidence_score - second) if best is not None else float("-inf")
                accepted = False
                selected: RecoveryEvidence | None = None
                if col < len(all_ids):
                    selected_id = all_ids[col]
                    selected = next((item for item in values if item.dormant_id == selected_id), None)
                    if selected is not None:
                        selected.competition_margin = margin
                        if self.cfg.transport_mode == "topk":
                            accepted = selected.evidence_score >= self.cfg.tau_topk and margin >= self.cfg.tau_margin_topk
                        else:
                            accepted = selected.evidence_score >= self.cfg.tau_evidence and margin >= self.cfg.tau_competition and selected.transport_mass >= self.cfg.tau_mass
                        # A frontend integer ID can be reused after a gap.
                        # Reject a recovered fragment if it would occupy a
                        # frame already assigned to the target identity. The
                        # dormant key is the causal/root identity, not the
                        # frontend fragment ID, so this also rejects a root
                        # that is still active under another frontend ID.
                        occupied = getattr(self, "_occupied_frames", {}).get(int(selected.dormant_id), set())
                        candidate_frames = {int(item.frame_id) for item in tentative.observations}
                        if accepted and candidate_frames.intersection(occupied):
                            accepted = False
                        active_roots = getattr(self, "_active_root_ids", set())
                        if accepted and int(selected.dormant_id) in active_roots:
                            accepted = False
                        # A dormant integer can also be reused by a new
                        # frontend fragment in the current frame.  It cannot
                        # be the target of a different fragment without an
                        # immediate same-frame collision.
                        current_frontend_ids = getattr(self, "_current_frontend_ids", set())
                        if accepted and int(selected.dormant_id) in current_frontend_ids and int(selected.dormant_id) != int(tentative.frontend_track_id):
                            accepted = False
                        selected.accepted = bool(accepted)
                if selected is not None:
                    event = {
                        "tentative_frontend_id": int(tentative.frontend_track_id),
                        "candidate_dormant_id": int(selected.dormant_id),
                        "first_frame": int(tentative.first_frame),
                        "decision_frame": int(decision_frame),
                        "gap": int(selected.gap),
                        "transport_cost": float(selected.transport_cost),
                        "transport_mass": float(selected.transport_mass),
                        "evidence_score": float(selected.evidence_score),
                        "competition_margin": float(margin),
                        "accepted": bool(accepted),
                    }
                    self.recovery_events.append(event)
                    if accepted:
                        assignments[int(tentative.frontend_track_id)] = int(selected.dormant_id)
                        dormant.pop(int(selected.dormant_id), None)
                else:
                    self.recovery_events.append({
                        "tentative_frontend_id": int(tentative.frontend_track_id),
                        "candidate_dormant_id": None,
                        "first_frame": int(tentative.first_frame),
                        "decision_frame": int(decision_frame),
                        "gap": None,
                        "transport_cost": None,
                        "transport_mass": 0.0,
                        "evidence_score": 0.0,
                        "competition_margin": float(margin),
                        "accepted": False,
                    })
        return assignments

    def process_video(self, observations, frontend_track_ids=None, association_traces=None) -> VideoLocalIdMap:
        frames = sorted(list(observations), key=lambda value: int(value.frame_id))
        if not frames:
            return VideoLocalIdMap(-1)
        video_id = int(frames[0].video_id)
        self.recovery_events = []
        self._occupied_frames: dict[int, set[int]] = {}
        self._current_frontend_ids: set[int] = set()
        active: dict[int, list[MemoryObservation]] = {}
        active_root: dict[int, int] = {}
        dormant: dict[int, DormantIdentity] = {}
        pending: dict[int, TentativeTrack] = {}
        observation_to_root: dict[str, int] = {}
        last_seen: dict[int, int] = {}
        decision_delays: list[int] = []
        attempts = 0
        accepted_count = 0
        next_root_id = max((int(value) for value in np.asarray(frames[0].assigned_track_ids).reshape(-1)), default=0) + 1

        def _fresh_root(current_ids: set[int]) -> int:
            nonlocal next_root_id
            occupied_roots = set(int(value) for value in active_root.values()) | set(int(value) for value in current_ids)
            while next_root_id in occupied_roots or next_root_id in self._occupied_frames:
                next_root_id += 1
            value = int(next_root_id)
            next_root_id += 1
            return value

        def _effective_root(
            frontend_id: int,
            proposed: int,
            current_ids: set[int],
            observation_frames: Iterable[int],
        ) -> int:
            proposed = int(proposed)
            # If a recovered root is already active in this frame, keep the
            # new fragment separate with a fresh causal ID. This check is
            # deliberately on the root value, not only on equality with the
            # frontend ID; otherwise two different frontend fragments could
            # inherit one active root and collide in a later frame.
            active_roots = set(int(value) for value in active_root.values())
            if proposed in active_roots:
                return _fresh_root(current_ids)
            # A fragment can also be finalized after the previous owner has
            # gone dormant.  In that case the active-root check above is
            # empty, but reusing the dormant integer would still rewrite two
            # observations on the same frame to one identity.  Accepted
            # dormant candidates are checked in ``_decide``; this guard covers
            # the no-candidate/fallback path as well.
            occupied = self._occupied_frames.get(proposed, set())
            if occupied and set(int(value) for value in observation_frames).intersection(occupied):
                return _fresh_root(current_ids)
            return proposed

        transport_started = perf_counter()
        for frame_index, frame in enumerate(frames):
            current_ids = set(int(value) for value in (frontend_track_ids[frame_index] if frontend_track_ids is not None else frame.assigned_track_ids))
            ready: list[TentativeTrack] = []
            # An identity that was absent before this frame is now dormant;
            # this decision uses no observation after the current frame.
            for track_id, values in list(active.items()):
                if track_id not in current_ids and last_seen.get(track_id, -1) < int(frame.frame_id):
                    root = int(active_root.get(track_id, track_id))
                    # A root must have one active owner. If an older invalid
                    # state ever contains two owners, retain the active one
                    # and do not manufacture a duplicate dormant candidate.
                    if root not in active_root.values() or all(int(owner) == int(track_id) for owner, value in active_root.items() if int(value) == root):
                        history = IdentityHistory(root, video_id, values[0].frame_id, values[-1].frame_id, values[:])
                        dormant[root] = DormantIdentity(root, video_id, history, values[-1].frame_id, values[-1].bbox_xyxy, values[-1].feature_norm)
                    active.pop(track_id, None)
                    active_root.pop(track_id, None)
            for track_id, tentative in list(pending.items()):
                if track_id not in current_ids and tentative.observations:
                    ready.append(tentative)
            for row, track_id in enumerate(current_ids):
                # Keep row ordering deterministic even if the tracker order is
                # not sorted in the source cache.
                row_indices = np.flatnonzero(frame.assigned_track_ids == track_id)
                for row_index in row_indices.tolist():
                    observation = _as_observation(frame, int(row_index))
                    observation_uid = f"native_v6:{video_id}:{int(frame.frame_id)}:{int(row_index)}"
                    if track_id in active:
                        active[track_id].append(observation)
                        root = int(active_root.get(track_id, track_id))
                        self._occupied_frames.setdefault(root, set()).add(int(frame.frame_id))
                        if root != int(track_id):
                            observation_to_root[observation_uid] = root
                    else:
                        tentative = pending.setdefault(track_id, TentativeTrack(track_id, video_id, first_frame=int(frame.frame_id), last_frame=int(frame.frame_id)))
                        tentative.observations.append(observation)
                        tentative.rows.append(int(row_index))
                        tentative.observation_uids.append(observation_uid)
                        tentative.first_frame = int(tentative.first_frame if tentative.first_frame >= 0 else frame.frame_id)
                        tentative.last_frame = int(frame.frame_id)
                        if len(tentative.observations) >= int(self.cfg.tentative_observations):
                            if tentative not in ready:
                                ready.append(tentative)
                    last_seen[track_id] = int(frame.frame_id)
            if ready:
                attempts += len(ready)
                self._current_frontend_ids = set(int(value) for value in current_ids)
                self._active_root_ids = set(int(value) for value in active_root.values())
                assignments = self._decide(ready, dormant, int(frame.frame_id))
                for tentative in ready:
                    target = _effective_root(
                        tentative.frontend_track_id,
                        assignments.get(tentative.frontend_track_id, tentative.frontend_track_id),
                        current_ids,
                        (item.frame_id for item in tentative.observations),
                    )
                    target = int(target)
                    self._occupied_frames.setdefault(target, set()).update(int(item.frame_id) for item in tentative.observations)
                    if target != int(tentative.frontend_track_id):
                        observation_to_root.update({str(uid): target for uid in tentative.observation_uids})
                    active[int(tentative.frontend_track_id)] = list(tentative.observations)
                    active_root[int(tentative.frontend_track_id)] = target
                    decision_delays.append(int(frame.frame_id - tentative.first_frame))
                    if target != tentative.frontend_track_id:
                        accepted_count += 1
                    pending.pop(int(tentative.frontend_track_id), None)
        # A fragment that reaches the end of the stream is finalized using its
        # buffered observations only.
        if pending:
            ready = list(pending.values())
            attempts += len(ready)
            self._current_frontend_ids = set(int(value) for value in frontend_track_ids[-1]) if frontend_track_ids is not None else set(int(value) for value in frames[-1].assigned_track_ids)
            self._active_root_ids = set(int(value) for value in active_root.values())
            assignments = self._decide(ready, dormant, int(frames[-1].frame_id))
            for tentative in ready:
                target = _effective_root(
                    tentative.frontend_track_id,
                    assignments.get(tentative.frontend_track_id, tentative.frontend_track_id),
                    self._current_frontend_ids,
                    (item.frame_id for item in tentative.observations),
                )
                target = int(target)
                self._occupied_frames.setdefault(target, set()).update(int(item.frame_id) for item in tentative.observations)
                if target != int(tentative.frontend_track_id):
                    observation_to_root.update({str(uid): target for uid in tentative.observation_uids})
                decision_delays.append(int(frames[-1].frame_id - tentative.first_frame))
                if target != tentative.frontend_track_id:
                    accepted_count += 1
        self._diagnostics = {
            "mean_decision_delay": float(np.mean(decision_delays)) if decision_delays else 0.0,
            "p95_decision_delay": float(np.percentile(decision_delays, 95)) if decision_delays else 0.0,
            "recovery_attempts": int(attempts),
            "accepted_recoveries": int(accepted_count),
            "rejected_recoveries": int(max(0, attempts - accepted_count)),
            "candidate_count": len(self.recovery_events),
            "transport_runtime": float(perf_counter() - transport_started),
        }
        return VideoLocalIdMap(video_id, {}, observation_to_root)

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)
