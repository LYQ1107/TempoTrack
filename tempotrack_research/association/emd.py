"""Traceable legacy EMD and repaired stable Sinkhorn EMD."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor
import torch.nn.functional as F


def _as_points(tracklet: Mapping[str, Any], representative: str = "all") -> tuple[Tensor, Tensor]:
    embeddings = torch.as_tensor(tracklet.get("appearance", tracklet.get("embeds", [])), dtype=torch.float32)
    if embeddings.ndim == 1 and embeddings.numel():
        embeddings = embeddings.unsqueeze(0)
    boxes = torch.as_tensor(tracklet.get("bboxes", tracklet.get("bbox", [])), dtype=torch.float32)
    if boxes.ndim == 1 and boxes.numel():
        boxes = boxes.unsqueeze(0)
    if not embeddings.numel() or not boxes.numel():
        return embeddings.reshape(0, -1), boxes.reshape(0, 4)
    if boxes.shape[-1] >= 5:
        boxes = boxes[:, :4]
    if representative == "boundary":
        embeddings, boxes = embeddings[[-1]], boxes[[-1]]
    elif representative == "mean":
        embeddings, boxes = embeddings.mean(0, keepdim=True), boxes.mean(0, keepdim=True)
    return F.normalize(embeddings, dim=-1), boxes


def _cost_matrix(left: Tensor, right: Tensor, left_boxes: Tensor, right_boxes: Tensor, lam_app: float, lam_shape: float, lam_scale: float) -> Tensor:
    app = 1.0 - left @ right.t()
    lc = (left_boxes[:, :2] + left_boxes[:, 2:]) / 2
    rc = (right_boxes[:, :2] + right_boxes[:, 2:]) / 2
    center = torch.cdist(lc, rc)
    lwh = (left_boxes[:, 2:] - left_boxes[:, :2]).clamp_min(1e-6)
    rwh = (right_boxes[:, 2:] - right_boxes[:, :2]).clamp_min(1e-6)
    shape = torch.cdist(torch.log(lwh), torch.log(rwh))
    scale = (torch.log(lwh.prod(-1))[:, None] - torch.log(rwh.prod(-1))[None, :]).abs()
    return lam_app * app + lam_shape * shape + lam_scale * scale + 0.01 * center


def _empty_result(reason: str) -> dict[str, Any]:
    return {"edge_score": float("inf"), "transport_mass": 0.0, "marginal_residual": float("inf"), "valid": False, "reason": reason}


def legacy_emd(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Original-style unregularized proxy retained only as a traceable control."""

    a, ab = _as_points(left)
    b, bb = _as_points(right)
    if not a.numel() or not b.numel():
        return _empty_result("empty_tracklet")
    cost = _cost_matrix(a, b, ab, bb, 1.0, 0.1, 0.05)
    distance = cost.mean()
    return {"edge_score": float(distance), "transport_mass": 1.0, "marginal_residual": 0.0, "valid": bool(torch.isfinite(distance)), "solver": "legacy_mean_transport"}


def stable_emd(left: Mapping[str, Any], right: Mapping[str, Any], sink_eps: float = 0.05, sink_iters: int = 100, time_gap: float = 0.0, max_gap: float = 90.0, lam_app: float = 1.0, lam_shape: float = 0.1, lam_scale: float = 0.05, lam_time: float = 0.1) -> dict[str, Any]:
    """Stable entropic transport with explicit mass and residual diagnostics."""

    a, ab = _as_points(left)
    b, bb = _as_points(right)
    if not a.numel() or not b.numel():
        return _empty_result("empty_tracklet")
    if sink_eps <= 0 or sink_iters < 1:
        raise ValueError("sink_eps must be positive and sink_iters >= 1")
    cost = _cost_matrix(a, b, ab, bb, lam_app, lam_shape, lam_scale)
    cost = cost + lam_time * min(max(float(time_gap), 0.0) / max(float(max_gap), 1.0), 1.0)
    if not torch.isfinite(cost).all():
        return _empty_result("nonfinite_cost")
    # For two one-token segments the transport polytope has exactly one
    # feasible plan.  Returning that plan's cost is the exact Sinkhorn result
    # (mass=1, zero marginal residual) and avoids 100 Python iterations for
    # the dominant short-tracklet case in a TAO candidate graph.
    if a.shape[0] == 1 and b.shape[0] == 1:
        distance = cost[0, 0]
        return {"edge_score": float(distance), "transport_mass": 1.0, "marginal_residual": 0.0, "valid": bool(torch.isfinite(distance)), "solver": "log_sinkhorn_degenerate", "iterations": 0}
    weights_a = torch.full((a.shape[0],), 1.0 / a.shape[0], dtype=cost.dtype, device=cost.device)
    weights_b = torch.full((b.shape[0],), 1.0 / b.shape[0], dtype=cost.dtype, device=cost.device)
    log_k = (-cost / sink_eps).clamp_min(-80.0)
    log_u = torch.zeros_like(weights_a)
    log_v = torch.zeros_like(weights_b)
    log_a, log_b = weights_a.log(), weights_b.log()
    for _ in range(sink_iters):
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0)
    transport = (log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)).exp()
    row_residual = (transport.sum(1) - weights_a).abs().sum()
    col_residual = (transport.sum(0) - weights_b).abs().sum()
    distance = (transport * cost).sum()
    valid = bool(torch.isfinite(distance) and torch.isfinite(transport).all())
    return {"edge_score": float(distance) if valid else float("inf"), "transport_mass": float(transport.sum()), "marginal_residual": float(row_residual + col_residual), "valid": valid, "solver": "log_sinkhorn", "iterations": sink_iters}
