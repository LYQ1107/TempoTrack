"""Shared M0/M1 frame scoring and unmatched assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _masked_softmax(scores: Tensor, mask: Tensor, dim: int) -> Tensor:
    if scores.shape != mask.shape:
        raise ValueError("score/legal mask shape mismatch")
    finite = torch.finfo(scores.dtype).min
    values = scores.masked_fill(~mask, finite)
    any_valid = mask.any(dim=dim, keepdim=True)
    output = torch.softmax(values, dim=dim)
    return torch.where(any_valid, output * mask.to(output.dtype), torch.zeros_like(output))


def score_prototype(query: Tensor, prototype: Tensor, *, logit_scale: float, legal_mask: Tensor | None = None) -> Tensor:
    query = F.normalize(query.float(), dim=-1)
    prototype = F.normalize(prototype.float(), dim=-1)
    cosine = query @ prototype.t()
    mask = torch.ones_like(cosine, dtype=torch.bool) if legal_mask is None else legal_mask.bool()
    logits = float(logit_scale) * cosine
    row = _masked_softmax(logits, mask, dim=1)
    col = _masked_softmax(logits, mask, dim=0)
    bisoftmax = 0.5 * (row + col)
    return 0.5 * (bisoftmax + cosine).masked_fill(~mask, -torch.inf)


def fuse_dual(fast_scores: Tensor, slow_scores: Tensor, *, fast_accept_threshold: float) -> Tensor:
    if fast_scores.shape != slow_scores.shape:
        raise ValueError("fast and slow scores have different shapes")
    return torch.where(fast_scores >= float(fast_accept_threshold), fast_scores, torch.maximum(fast_scores, slow_scores))


@dataclass(frozen=True)
class FrameAssignment:
    detection_indices: np.ndarray
    track_indices: np.ndarray
    accepted: np.ndarray
    accepted_margin: np.ndarray
    accepted_score: np.ndarray
    score_matrix: np.ndarray


def assign_with_unmatched(scores: Tensor, legal_mask: Tensor, *, match_threshold: float) -> FrameAssignment:
    if scores.ndim != 2 or legal_mask.shape != scores.shape:
        raise ValueError("assignment scores/legal_mask must be [detections,tracks]")
    n, m = scores.shape
    if n == 0:
        return FrameAssignment(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=bool), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), scores.detach().cpu().numpy())
    values = scores.detach().cpu().numpy().astype(np.float64)
    legal = legal_mask.detach().cpu().numpy().astype(bool)
    # Each detection has an explicit dummy column.  Threshold is a net
    # benefit against dummy and is applied before the solver.
    from scipy.optimize import linear_sum_assignment
    benefit = values - float(match_threshold)
    cost = np.zeros((n, m + n), dtype=np.float64)
    cost[:, :m] = np.where(legal & np.isfinite(benefit) & (benefit > 0), -benefit, 1e9)
    rows, cols = linear_sum_assignment(cost)
    dets: list[int] = []
    tracks: list[int] = []
    accepted: list[bool] = []
    margins: list[float] = []
    accepted_scores: list[float] = []
    for det, col in zip(rows.tolist(), cols.tolist()):
        if col >= m or not legal[det, col] or not np.isfinite(values[det, col]) or values[det, col] < match_threshold:
            continue
        alternatives = [float(v) for index, v in enumerate(values[det].tolist()) if legal[det, index] and index != col and np.isfinite(v)]
        other = max(alternatives + [float(match_threshold)])
        dets.append(det); tracks.append(col); accepted.append(True)
        accepted_scores.append(float(values[det, col]))
        margins.append(float(values[det, col] - other))
    return FrameAssignment(np.asarray(dets, dtype=np.int64), np.asarray(tracks, dtype=np.int64), np.asarray(accepted, dtype=bool), np.asarray(margins, dtype=np.float32), np.asarray(accepted_scores, dtype=np.float32), values)


__all__ = ["score_prototype", "fuse_dual", "assign_with_unmatched", "FrameAssignment"]
