"""Traceable legacy EMD and repaired stable Sinkhorn EMD."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

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
    # Center distance is not part of the V3 EMD cost: it is an unnormalised
    # pixel-space term and made the control depend on image resolution.  The
    # legal temporal gap is added separately by ``stable_emd``.
    del center
    return lam_app * app + lam_shape * shape + lam_scale * scale


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
    return {"edge_score": float(distance), "transport_mass": 1.0, "marginal_residual": 0.0, "valid": bool(torch.isfinite(distance)), "solver": "legacy_mean_cost_control", "cost_semantics": "unregularized_mean_pair_cost"}


def stable_emd(left: Mapping[str, Any], right: Mapping[str, Any], sink_eps: float = 0.05, sink_iters: int = 100, time_gap: float = 0.0, max_gap: float = 90.0, lam_app: float = 1.0, lam_shape: float = 0.1, lam_scale: float = 0.05, lam_time: float = 0.1, sink_tol: float = 1e-4) -> dict[str, Any]:
    """Stable entropic transport with explicit mass and residual diagnostics."""

    a, ab = _as_points(left)
    b, bb = _as_points(right)
    if not a.numel() or not b.numel():
        return _empty_result("empty_tracklet")
    if float(time_gap) < 0.0 or float(time_gap) > float(max_gap):
        return _empty_result("illegal_temporal_gap")
    if sink_eps <= 0 or sink_iters < 1 or sink_tol <= 0:
        raise ValueError("sink_eps and sink_tol must be positive and sink_iters >= 1")
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
        return {"edge_score": float(distance), "transport_mass": 1.0, "marginal_residual": 0.0, "valid": bool(torch.isfinite(distance)), "solver": "stable_sinkhorn_degenerate", "iterations": 0, "converged": True, "cost_semantics": "uniform_entropic_transport_cost"}
    weights_a = torch.full((a.shape[0],), 1.0 / a.shape[0], dtype=cost.dtype, device=cost.device)
    weights_b = torch.full((b.shape[0],), 1.0 / b.shape[0], dtype=cost.dtype, device=cost.device)
    # Do not clamp the log kernel: clipping changes the transport objective.
    # logsumexp remains finite for very negative entries and is the declared
    # numerical stabilization for this solver.
    log_k = -cost / sink_eps
    log_u = torch.zeros_like(weights_a)
    log_v = torch.zeros_like(weights_b)
    log_a, log_b = weights_a.log(), weights_b.log()
    iterations = int(sink_iters)
    for iteration in range(sink_iters):
        previous_u, previous_v = log_u, log_v
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0)
        dual_delta = torch.maximum((log_u - previous_u).abs().max(), (log_v - previous_v).abs().max())
        if bool(torch.isfinite(dual_delta) and dual_delta <= sink_tol):
            iterations = iteration + 1
            break
    transport = (log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)).exp()
    row_residual = (transport.sum(1) - weights_a).abs().sum()
    col_residual = (transport.sum(0) - weights_b).abs().sum()
    distance = (transport * cost).sum()
    residual = float(row_residual + col_residual)
    converged = residual <= 1e-3
    valid = bool(torch.isfinite(distance) and torch.isfinite(transport).all() and converged)
    return {"edge_score": float(distance) if valid else float("inf"), "transport_mass": float(transport.sum()), "marginal_residual": residual, "valid": valid, "solver": "stable_sinkhorn", "iterations": iterations, "converged": converged, "tolerance": float(sink_tol), "cost_semantics": "uniform_entropic_transport_cost"}


@torch.no_grad()
def stable_emd_batch(
    lefts: Sequence[Mapping[str, Any]],
    rights: Sequence[Mapping[str, Any]],
    time_gaps: Sequence[float],
    *,
    sink_eps: float = 0.05,
    sink_iters: int = 100,
    max_gap: float = 90.0,
    lam_app: float = 1.0,
    lam_shape: float = 0.1,
    lam_scale: float = 0.05,
    lam_time: float = 0.1,
    sink_tol: float = 1e-4,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    """Evaluate the same all-observation Sinkhorn objective in edge batches.

    The scalar solver is intentionally retained as the reference path.  The
    frontend initial graph, however, evaluates the identical objective for
    thousands of legal candidate edges.  Padding only within a small batch
    and masking padded transport mass keeps every real observation in the
    transport plan while removing the Python/torch-call overhead that made a
    large TAO graph effectively unfinishable.
    """
    if len(lefts) != len(rights) or len(lefts) != len(time_gaps):
        raise ValueError("stable_emd_batch inputs must have equal lengths")
    if sink_eps <= 0 or sink_iters < 1 or sink_tol <= 0 or batch_size < 1:
        raise ValueError("invalid stable_emd_batch solver settings")
    results: list[dict[str, Any] | None] = [None] * len(lefts)
    pending: list[int] = []
    singleton_values: list[tuple[int, Tensor, Tensor, Tensor, Tensor, float]] = []
    for index, (left, right, gap) in enumerate(zip(lefts, rights, time_gaps)):
        left_app, left_box = _as_points(left)
        right_app, right_box = _as_points(right)
        if not left_app.numel() or not right_app.numel():
            results[index] = _empty_result("empty_tracklet")
            continue
        if float(gap) < 0.0 or float(gap) > float(max_gap):
            results[index] = _empty_result("illegal_temporal_gap")
            continue
        if left_app.shape[1] != right_app.shape[1]:
            raise ValueError("stable_emd_batch appearance dimensions disagree")
        # A singleton transport polytope is exact without a Sinkhorn loop.
        if left_app.shape[0] == 1 and right_app.shape[0] == 1:
            singleton_values.append((index, left_app, right_app, left_box, right_box, float(gap)))
            continue
        pending.append(index)

    if singleton_values:
        left_app = torch.cat([value[1] for value in singleton_values], dim=0)
        right_app = torch.cat([value[2] for value in singleton_values], dim=0)
        left_box = torch.cat([value[3] for value in singleton_values], dim=0)
        right_box = torch.cat([value[4] for value in singleton_values], dim=0)
        app = 1.0 - (left_app * right_app).sum(dim=1)
        left_wh = (left_box[:, 2:] - left_box[:, :2]).clamp_min(1e-6)
        right_wh = (right_box[:, 2:] - right_box[:, :2]).clamp_min(1e-6)
        shape = torch.linalg.vector_norm(torch.log(left_wh) - torch.log(right_wh), dim=1)
        scale = (torch.log(left_wh.prod(-1)) - torch.log(right_wh.prod(-1))).abs()
        gaps = torch.as_tensor([value[5] for value in singleton_values], dtype=torch.float32)
        distance = lam_app * app + lam_shape * shape + lam_scale * scale
        distance = distance + lam_time * (gaps.clamp(0.0, float(max_gap)) / max(float(max_gap), 1.0))
        finite = torch.isfinite(distance)
        for row, value in enumerate(singleton_values):
            index = value[0]
            results[index] = {
                "edge_score": float(distance[row]) if bool(finite[row]) else float("inf"),
                "transport_mass": 1.0,
                "marginal_residual": 0.0,
                "valid": bool(finite[row]),
                "reason": None if bool(finite[row]) else "nonfinite_cost",
                "solver": "stable_sinkhorn_degenerate",
                "iterations": 0,
                "converged": bool(finite[row]),
                "tolerance": float(sink_tol),
                "cost_semantics": "uniform_entropic_transport_cost",
                "batched": True,
            }

    for start in range(0, len(pending), int(batch_size)):
        indices = pending[start : start + int(batch_size)]
        left_values = [_as_points(lefts[index]) for index in indices]
        right_values = [_as_points(rights[index]) for index in indices]
        left_len = [int(value[0].shape[0]) for value in left_values]
        right_len = [int(value[0].shape[0]) for value in right_values]
        left_max, right_max = max(left_len), max(right_len)
        dimension = int(left_values[0][0].shape[1])
        if any(int(value[0].shape[1]) != dimension for value in left_values + right_values):
            raise ValueError("stable_emd_batch appearance dimensions disagree")
        count = len(indices)
        left_app = torch.zeros((count, left_max, dimension), dtype=torch.float32)
        right_app = torch.zeros((count, right_max, dimension), dtype=torch.float32)
        left_box = torch.zeros((count, left_max, 4), dtype=torch.float32)
        right_box = torch.zeros((count, right_max, 4), dtype=torch.float32)
        left_mask = torch.zeros((count, left_max), dtype=torch.bool)
        right_mask = torch.zeros((count, right_max), dtype=torch.bool)
        for row, ((left_value, left_boxes), (right_value, right_boxes)) in enumerate(zip(left_values, right_values)):
            left_app[row, : left_value.shape[0]] = left_value
            right_app[row, : right_value.shape[0]] = right_value
            left_box[row, : left_boxes.shape[0]] = left_boxes
            right_box[row, : right_boxes.shape[0]] = right_boxes
            left_mask[row, : left_value.shape[0]] = True
            right_mask[row, : right_value.shape[0]] = True
        app = 1.0 - torch.bmm(left_app, right_app.transpose(1, 2))
        left_center = (left_box[:, :, :2] + left_box[:, :, 2:]) / 2.0
        right_center = (right_box[:, :, :2] + right_box[:, :, 2:]) / 2.0
        del left_center, right_center  # unnormalised pixel center is not in V3 cost
        left_wh = (left_box[:, :, 2:] - left_box[:, :, :2]).clamp_min(1e-6)
        right_wh = (right_box[:, :, 2:] - right_box[:, :, :2]).clamp_min(1e-6)
        shape = torch.cdist(torch.log(left_wh), torch.log(right_wh))
        left_area = torch.log(left_wh.prod(-1))
        right_area = torch.log(right_wh.prod(-1))
        scale = (left_area.unsqueeze(2) - right_area.unsqueeze(1)).abs()
        gap = torch.as_tensor([float(time_gaps[index]) for index in indices], dtype=torch.float32)
        cost = lam_app * app + lam_shape * shape + lam_scale * scale
        cost = cost + lam_time * (gap.clamp(0.0, float(max_gap)) / max(float(max_gap), 1.0)).reshape(-1, 1, 1)
        valid_cells = left_mask.unsqueeze(2) & right_mask.unsqueeze(1)
        log_k = torch.where(valid_cells, -cost / sink_eps, torch.full_like(cost, float("-inf")))
        log_a = torch.where(left_mask, -torch.log(torch.as_tensor(left_len, dtype=torch.float32)).unsqueeze(1), torch.zeros_like(left_mask, dtype=torch.float32))
        log_b = torch.where(right_mask, -torch.log(torch.as_tensor(right_len, dtype=torch.float32)).unsqueeze(1), torch.zeros_like(right_mask, dtype=torch.float32))
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        done = torch.zeros((count,), dtype=torch.bool)
        iteration_count = torch.full((count,), int(sink_iters), dtype=torch.int64)
        for iteration in range(int(sink_iters)):
            next_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(1), dim=2)
            # Padded rows have an all -inf kernel.  Mask them before feeding
            # the dual potential into the opposite update; otherwise +inf +
            # -inf can create NaNs in an otherwise valid batch.
            next_u = torch.where(left_mask, next_u, torch.zeros_like(next_u))
            next_v = log_b - torch.logsumexp(log_k + next_u.unsqueeze(2), dim=1)
            next_v = torch.where(right_mask, next_v, torch.zeros_like(next_v))
            delta_u = (next_u - log_u).abs().masked_fill(~left_mask, 0.0).amax(dim=1)
            delta_v = (next_v - log_v).abs().masked_fill(~right_mask, 0.0).amax(dim=1)
            delta = torch.maximum(delta_u, delta_v)
            next_done = (~done) & torch.isfinite(delta) & (delta <= float(sink_tol))
            log_u = torch.where(done.unsqueeze(1), log_u, next_u)
            log_v = torch.where(done.unsqueeze(1), log_v, next_v)
            iteration_count[next_done] = int(iteration + 1)
            done |= next_done
            if bool(done.all()):
                break
        transport = (log_u.unsqueeze(2) + log_k + log_v.unsqueeze(1)).exp()
        row_mass = transport.sum(dim=2)
        col_mass = transport.sum(dim=1)
        expected_a = log_a.exp()
        expected_b = log_b.exp()
        residual = ((row_mass - expected_a).abs() * left_mask).sum(dim=1) + ((col_mass - expected_b).abs() * right_mask).sum(dim=1)
        distance = (transport * cost).sum(dim=(1, 2))
        mass = transport.sum(dim=(1, 2))
        finite = torch.isfinite(distance) & torch.isfinite(transport).all(dim=2).all(dim=1)
        converged = residual <= 1e-3
        valid = finite & converged
        for row, index in enumerate(indices):
            results[index] = {
                "edge_score": float(distance[row]) if bool(valid[row]) else float("inf"),
                "transport_mass": float(mass[row]),
                "marginal_residual": float(residual[row]),
                "valid": bool(valid[row]),
                "reason": None if bool(valid[row]) else ("nonfinite_cost" if not bool(finite[row]) else "sinkhorn_not_converged"),
                "solver": "stable_sinkhorn",
                "iterations": int(iteration_count[row]),
                "converged": bool(converged[row]),
                "tolerance": float(sink_tol),
                "cost_semantics": "uniform_entropic_transport_cost",
                "batched": True,
            }
    return [value if value is not None else _empty_result("internal_batch_error") for value in results]
