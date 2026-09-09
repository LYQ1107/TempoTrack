"""A standalone implementation of the historical Paper EMD protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..memory.identity_history import IdentityHistory, MemoryObservation


@dataclass(frozen=True)
class PaperEMDConfig:
    bank_size: int = 64
    dedup_cosine: float = 0.95
    boundary_k: int = 3
    representative_size: int = 16
    max_gap: int = 30
    min_tracklet_length: int = 3
    theta_emd: float = 0.35
    sinkhorn_iters: int = 20
    sinkhorn_eps: float = 0.05
    lambda_app: float = 0.70
    lambda_geo: float = 0.20
    lambda_time: float = 0.10
    beta_area: float = 0.50


@dataclass
class RepresentativeSet:
    features: torch.Tensor
    aspect_ratios: torch.Tensor
    log_areas: torch.Tensor
    frame_times: torch.Tensor
    weights: torch.Tensor
    source_types: list[str]


@dataclass(frozen=True)
class PaperMerge:
    source_id: int
    target_id: int
    distance: float
    gap: int


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    return value / max(float(np.linalg.norm(value)), 1e-12)


class PaperRepresentativeExtractor:
    def __init__(self, cfg: PaperEMDConfig):
        self.cfg = cfg

    def extract(self, history: IdentityHistory) -> RepresentativeSet:
        observations = list(history.observations[-self.cfg.bank_size:])
        if not observations:
            return RepresentativeSet(
                torch.empty((0, 0)), torch.empty((0,)), torch.empty((0,)), torch.empty((0,)), torch.empty((0,)), []
            )
        k = min(self.cfg.boundary_k, len(observations))
        seed_specs: list[tuple[list[MemoryObservation], str]] = [(observations[:k], "earliest_boundary")]
        if len(observations) > k:
            seed_specs.append((observations[-k:], "latest_boundary"))
        candidates: list[tuple[MemoryObservation, str]] = [(obs, kind) for values, kind in seed_specs for obs in values]
        candidates.extend((obs, "diversity") for obs in observations)

        selected: list[tuple[MemoryObservation, str]] = []
        for obs, kind in candidates:
            feature = _unit(obs.feature_raw)
            if any(float(np.dot(feature, _unit(existing.feature_raw))) > self.cfg.dedup_cosine for existing, _ in selected):
                continue
            selected.append((obs, kind))
        # Boundary seeds are mandatory where they survive de-duplication;
        # then use standard farthest-point selection over all bank samples.
        if len(selected) > self.cfg.representative_size:
            mandatory: list[tuple[MemoryObservation, str]] = []
            for obs, kind in selected:
                if kind in {"earliest_boundary", "latest_boundary"}:
                    mandatory.append((obs, kind))
            mandatory = mandatory[: self.cfg.representative_size]
            selected = mandatory[:]
            remaining = [(obs, "diversity") for obs in observations if not any(obs is x[0] for x in selected)]
            while len(selected) < self.cfg.representative_size and remaining:
                distances = []
                for index, (obs, _) in enumerate(remaining):
                    feature = _unit(obs.feature_raw)
                    distance = min(1.0 - float(np.dot(feature, _unit(chosen.feature_raw))) for chosen, _ in selected)
                    distances.append((distance, index))
                _, best_index = max(distances, key=lambda value: (value[0], -value[1]))
                selected.append(remaining.pop(best_index))
        selected = selected[: self.cfg.representative_size]
        features = torch.as_tensor(np.stack([_unit(obs.feature_raw) for obs, _ in selected]), dtype=torch.float32)
        aspect = torch.as_tensor([float(obs.aspect_ratio) for obs, _ in selected], dtype=torch.float32)
        log_areas = torch.as_tensor([float(np.log(max(obs.area_norm, 1e-12))) for obs, _ in selected], dtype=torch.float32)
        times = torch.as_tensor([float(obs.frame_id) for obs, _ in selected], dtype=torch.float32)
        weights = torch.full((len(selected),), 1.0 / max(len(selected), 1), dtype=torch.float32)
        return RepresentativeSet(features, aspect, log_areas, times, weights, [kind for _, kind in selected])


def paper_ground_cost(
    left: RepresentativeSet,
    right: RepresentativeSet,
    *,
    pair_gap: float,
    cfg: PaperEMDConfig,
) -> torch.Tensor:
    if left.features.numel() == 0 or right.features.numel() == 0:
        return left.features.new_empty((left.features.shape[0], right.features.shape[0]))
    c_app = 1.0 - left.features @ right.features.t()
    c_geo = (
        (left.aspect_ratios[:, None] - right.aspect_ratios[None, :]).abs()
        + cfg.beta_area * (left.log_areas[:, None] - right.log_areas[None, :]).abs()
    )
    c_time = max(float(pair_gap), 0.0) / max(float(cfg.max_gap), 1.0)
    return cfg.lambda_app * c_app + cfg.lambda_geo * c_geo + cfg.lambda_time * c_time


def paper_sinkhorn_distance(
    cost: torch.Tensor,
    weights_left: torch.Tensor,
    weights_right: torch.Tensor,
    *,
    eps: float,
    iters: int,
) -> dict:
    if cost.numel() == 0 or not torch.isfinite(cost).all():
        return {"distance": float("inf"), "transport_mass": 0.0, "row_residual": float("inf"), "col_residual": float("inf"), "finite": False, "iterations": 0}
    a = weights_left.to(cost).clamp_min(1e-12)
    b = weights_right.to(cost).clamp_min(1e-12)
    a = a / a.sum()
    b = b / b.sum()
    log_k = -cost / float(eps)
    log_a = a.log()
    log_b = b.log()
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for iteration in range(int(iters)):
        log_u = log_a - torch.logsumexp(log_k + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_k.t() + log_u[None, :], dim=1)
    gamma = torch.exp(log_u[:, None] + log_k + log_v[None, :])
    row = gamma.sum(dim=1)
    col = gamma.sum(dim=0)
    row_residual = float((row - a).abs().max().detach().cpu())
    col_residual = float((col - b).abs().max().detach().cpu())
    mass = float(gamma.sum().detach().cpu())
    finite = bool(torch.isfinite(gamma).all() and row_residual <= 1e-3 and col_residual <= 1e-3)
    return {
        "distance": float((gamma * cost).sum().detach().cpu()) if finite else float("inf"),
        "transport_mass": mass,
        "row_residual": row_residual,
        "col_residual": col_residual,
        "finite": finite,
        "iterations": int(iters),
    }


def build_paper_candidates(
    histories: Mapping[tuple[int, int], IdentityHistory] | Sequence[IdentityHistory],
    cfg: PaperEMDConfig,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    values = list(histories.values()) if isinstance(histories, Mapping) else list(histories)
    candidates = []
    for left in values:
        for right in values:
            if left is right or left.video_id != right.video_id:
                continue
            if left.last_frame >= right.first_frame:
                continue
            gap = int(right.first_frame - left.last_frame)
            if gap > cfg.max_gap or len(left.observations) < cfg.min_tracklet_length or len(right.observations) < cfg.min_tracklet_length:
                continue
            candidates.append(((int(left.video_id), int(left.local_track_id)), (int(right.video_id), int(right.local_track_id))))
    return candidates


def paper_mnn_merge_plan(
    histories: Mapping[tuple[int, int], IdentityHistory] | Sequence[IdentityHistory],
    cfg: PaperEMDConfig,
) -> tuple[list[PaperMerge], dict]:
    if isinstance(histories, Mapping):
        by_key = dict(histories)
    else:
        by_key = {(int(value.video_id), int(value.local_track_id)): value for value in histories}
    extractor = PaperRepresentativeExtractor(cfg)
    reps = {key: extractor.extract(value) for key, value in by_key.items()}
    candidates = build_paper_candidates(by_key, cfg)
    diagnostics = {"candidate_count": len(candidates), "valid_transport_count": 0, "accepted_mnn_count": 0, "rejected_threshold_count": 0, "rejected_conflict_count": 0, "distances": [], "gaps": []}
    distances: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for left_key, right_key in candidates:
        left, right = by_key[left_key], by_key[right_key]
        result = paper_sinkhorn_distance(
            paper_ground_cost(reps[left_key], reps[right_key], pair_gap=right.first_frame - left.last_frame, cfg=cfg),
            reps[left_key].weights,
            reps[right_key].weights,
            eps=cfg.sinkhorn_eps,
            iters=cfg.sinkhorn_iters,
        )
        if not result["finite"]:
            continue
        diagnostics["valid_transport_count"] += 1
        distance = float(result["distance"])
        distances[(left_key, right_key)] = distance
        diagnostics["distances"].append(distance)
        diagnostics["gaps"].append(int(right.first_frame - left.last_frame))
    nearest_successor: dict[tuple[int, int], tuple[tuple[int, int], float]] = {}
    nearest_predecessor: dict[tuple[int, int], tuple[tuple[int, int], float]] = {}
    for (left, right), distance in distances.items():
        if distance > cfg.theta_emd:
            continue
        if left not in nearest_successor or distance < nearest_successor[left][1]:
            nearest_successor[left] = (right, distance)
        if right not in nearest_predecessor or distance < nearest_predecessor[right][1]:
            nearest_predecessor[right] = (left, distance)
    plan: list[PaperMerge] = []
    for source, (target, distance) in nearest_successor.items():
        if nearest_predecessor.get(target, (None, None))[0] != source:
            continue
        left, right = by_key[source], by_key[target]
        plan.append(PaperMerge(int(source[1]), int(target[1]), float(distance), int(right.first_frame - left.last_frame)))
    plan.sort(key=lambda value: (value.distance, value.source_id, value.target_id))
    used: set[int] = set()
    accepted: list[PaperMerge] = []
    for merge in plan:
        if merge.source_id in used or merge.target_id in used:
            diagnostics["rejected_conflict_count"] += 1
            continue
        accepted.append(merge)
        used.update((merge.source_id, merge.target_id))
    diagnostics["accepted_mnn_count"] = len(accepted)
    diagnostics["rejected_threshold_count"] = max(0, diagnostics["valid_transport_count"] - len(plan))
    diagnostics["mean_distance"] = float(np.mean(diagnostics["distances"])) if diagnostics["distances"] else None
    diagnostics["mean_gap"] = float(np.mean(diagnostics["gaps"])) if diagnostics["gaps"] else None
    diagnostics["plan"] = [merge.__dict__ for merge in accepted]
    return accepted, diagnostics


__all__ = [
    "PaperEMDConfig",
    "RepresentativeSet",
    "PaperRepresentativeExtractor",
    "PaperMerge",
    "paper_ground_cost",
    "paper_sinkhorn_distance",
    "build_paper_candidates",
    "paper_mnn_merge_plan",
]

