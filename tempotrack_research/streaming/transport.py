from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, log
from typing import Any, Sequence

import numpy as np
import torch

from ..memory.identity_history import MemoryObservation


@dataclass
class TransportResult:
    normalized_cost: float
    transport_mass: float
    row_mass: np.ndarray
    col_mass: np.ndarray
    row_residual: float
    col_residual: float
    iterations: int
    converged: bool
    valid: bool
    diagnostics: dict = field(default_factory=dict)


def streaming_ground_cost(
    tentative_obs: list[MemoryObservation],
    dormant_obs: list[MemoryObservation],
    *,
    lambda_app: float = 0.80,
    lambda_geo: float = 0.20,
) -> torch.Tensor:
    left = torch.as_tensor(np.stack([item.feature_norm for item in tentative_obs]), dtype=torch.float32)
    right = torch.as_tensor(np.stack([item.feature_norm for item in dormant_obs]), dtype=torch.float32)
    left = left / left.norm(dim=1, keepdim=True).clamp_min(1e-12)
    right = right / right.norm(dim=1, keepdim=True).clamp_min(1e-12)
    c_app = 1.0 - left @ right.t()
    ar_left = torch.as_tensor([np.log(max(item.aspect_ratio, 1e-12)) for item in tentative_obs])
    ar_right = torch.as_tensor([np.log(max(item.aspect_ratio, 1e-12)) for item in dormant_obs])
    area_left = torch.as_tensor([np.log(max(item.area_norm, 1e-12)) for item in tentative_obs])
    area_right = torch.as_tensor([np.log(max(item.area_norm, 1e-12)) for item in dormant_obs])
    c_geo = (ar_left[:, None] - ar_right[None, :]).abs() + 0.5 * (area_left[:, None] - area_right[None, :]).abs()
    return float(lambda_app) * c_app + float(lambda_geo) * c_geo


def _result_from_gamma(gamma: torch.Tensor, cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor, iterations: int, converged: bool, diagnostics: dict | None = None) -> TransportResult:
    row = gamma.sum(dim=1)
    col = gamma.sum(dim=0)
    mass = float(gamma.sum().detach().cpu())
    row_residual = float((row - a).abs().max().detach().cpu()) if len(a) else 0.0
    col_residual = float((col - b).abs().max().detach().cpu()) if len(b) else 0.0
    normalized = float((gamma * cost).sum().detach().cpu() / max(mass, 1e-8)) if torch.isfinite(gamma).all() and mass > 1e-8 else float("inf")
    finite = bool(torch.isfinite(gamma).all() and isfinite(normalized))
    return TransportResult(normalized, mass, row.detach().cpu().numpy(), col.detach().cpu().numpy(), row_residual, col_residual, int(iterations), bool(converged), bool(finite and mass >= 1e-4), dict(diagnostics or {}))


def balanced_sinkhorn(cost: torch.Tensor, source_weights: torch.Tensor | None = None, target_weights: torch.Tensor | None = None, *, epsilon: float = 0.05, iterations: int = 50, tolerance: float = 1e-4) -> TransportResult:
    if cost.numel() == 0 or not torch.isfinite(cost).all():
        return TransportResult(float("inf"), 0.0, np.empty(0), np.empty(0), float("inf"), float("inf"), 0, False, False, {"reason": "nonfinite_or_empty_cost"})
    n, m = cost.shape
    a = (source_weights if source_weights is not None else torch.full((n,), 1.0 / n)).to(cost).clamp_min(1e-12)
    b = (target_weights if target_weights is not None else torch.full((m,), 1.0 / m)).to(cost).clamp_min(1e-12)
    a, b = a / a.sum(), b / b.sum()
    log_k = -cost / float(epsilon)
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    converged = False
    for iteration in range(1, int(iterations) + 1):
        old_u, old_v = log_u, log_v
        log_u = a.log() - torch.logsumexp(log_k + log_v[None, :], dim=1)
        log_v = b.log() - torch.logsumexp(log_k.t() + log_u[None, :], dim=1)
        if max(float((log_u - old_u).abs().max()), float((log_v - old_v).abs().max())) <= tolerance:
            converged = True
            break
    gamma = torch.exp(log_u[:, None] + log_k + log_v[None, :])
    return _result_from_gamma(gamma, cost, a, b, iteration, converged, {"mode": "balanced", "epsilon": epsilon})


def unbalanced_sinkhorn(
    cost: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    epsilon: float,
    tau: float,
    iterations: int,
    tolerance: float = 1e-4,
) -> TransportResult:
    # V7 transport sanity requires a float32 log-domain update even when a
    # caller hands us AMP tensors.  Keeping the normalization in probability
    # space is what caused the previous UOT underflow at small epsilon.
    cost = cost.float()
    if cost.numel() == 0 or not torch.isfinite(cost).all():
        return TransportResult(float("inf"), 0.0, np.empty(0), np.empty(0), float("inf"), float("inf"), 0, False, False, {"reason": "nonfinite_or_empty_cost"})
    a = source_weights.to(device=cost.device, dtype=torch.float32).clamp_min(1e-12)
    b = target_weights.to(device=cost.device, dtype=torch.float32).clamp_min(1e-12)
    a, b = a / a.sum(), b / b.sum()
    eta = float(tau) / (float(tau) + float(epsilon))
    log_k = -cost / float(epsilon)
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    converged = False
    for iteration in range(1, int(iterations) + 1):
        old_u, old_v = log_u, log_v
        log_u = eta * (a.log() - torch.logsumexp(log_k + log_v[None, :], dim=1))
        log_v = eta * (b.log() - torch.logsumexp(log_k.t() + log_u[None, :], dim=1))
        delta = max(float((log_u - old_u).abs().max()), float((log_v - old_v).abs().max()))
        if delta <= tolerance:
            converged = True
            break
    gamma = torch.exp(log_u[:, None] + log_k + log_v[None, :])
    result = _result_from_gamma(gamma, cost, a, b, iteration, converged, {"mode": "uot", "epsilon": epsilon, "tau": tau})
    if result.transport_mass < 1e-4:
        result.valid = False
    return result


def observation_reliability(obs: MemoryObservation, *, cfg: Any) -> float:
    det = float(np.clip(obs.det_score, 0.0, 1.0))
    if obs.association_margin is not None and np.isfinite(obs.association_margin):
        temperature = float(getattr(cfg, "margin_temperature", 0.10))
        margin = 1.0 / (1.0 + exp(-float(obs.association_margin) / max(temperature, 1e-6)))
    else:
        margin = 1.0
    if obs.fast_score is not None and obs.slow_score is not None and np.isfinite(obs.fast_score) and np.isfinite(obs.slow_score):
        temperature = float(getattr(cfg, "agreement_temperature", 0.10))
        agree = exp(-abs(float(obs.fast_score) - float(obs.slow_score)) / max(temperature, 1e-6))
    else:
        agree = 1.0
    return max(1e-3, det * margin * agree)
