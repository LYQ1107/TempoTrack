"""Losses for the distributional identity ablation and candidate-set models.

The functions in this module deliberately operate on an event at a time (or
on a padded batch of events).  A label of ``-1`` and an explicit ``mask`` are
both treated as invalid, so Novel/unknown rows cannot accidentally contribute
to a Base-only optimizer.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _batched_inputs(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize loss inputs to ``[B,K]`` and validate their alignment."""
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels, device=logits.device)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    if logits.ndim != 2 or labels.shape != logits.shape:
        raise ValueError(
            f"ranking logits/labels must be aligned [B,K], got {tuple(logits.shape)} "
            f"and {tuple(labels.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("ranking logits must be finite")
    if mask is None:
        valid_mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        valid_mask = torch.as_tensor(mask, device=logits.device, dtype=torch.bool)
        if valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != logits.shape:
            raise ValueError(
                f"ranking mask must match logits [B,K], got {tuple(valid_mask.shape)}"
            )
    valid = valid_mask & (labels >= 0)
    return logits, labels, valid


def _ranking_masks(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    logits, labels, valid = _batched_inputs(logits, labels, mask)
    positive = valid & (labels == 1)
    negative = valid & (labels == 0)
    usable = positive.any(dim=1) & negative.any(dim=1)
    if not bool(usable.any()):
        raise ValueError("rank loss requires a positive and a labeled negative per event")
    return logits[usable], positive[usable], negative[usable], valid[usable]


def listwise_group_loss(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Event-level multi-positive listwise ranking loss.

    For each usable event this is
    ``logsumexp(all valid) - logsumexp(all positive)``.  Multiple positives
    are retained in the denominator of the positive term; no arbitrary
    positive winner is selected.
    """
    logits, positive, _negative, valid = _ranking_masks(logits, labels, mask)
    all_logsumexp = torch.logsumexp(logits.masked_fill(~valid, -torch.inf), dim=1)
    positive_logsumexp = torch.logsumexp(
        logits.masked_fill(~positive, -torch.inf), dim=1
    )
    return (all_logsumexp - positive_logsumexp).mean()


def hard_negative_margin_loss(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor | None = None,
    *,
    margin: float = 0.2,
) -> Tensor:
    """Hardest-negative margin term for each usable event."""
    if not torch.isfinite(torch.as_tensor(margin, dtype=torch.float32)):
        raise ValueError("margin must be finite")
    logits, positive, negative, _valid = _ranking_masks(logits, labels, mask)
    hardest_negative = logits.masked_fill(~negative, -torch.inf).max(dim=1).values
    strongest_positive = logits.masked_fill(~positive, -torch.inf).max(dim=1).values
    return torch.relu(
        torch.as_tensor(float(margin), dtype=logits.dtype, device=logits.device)
        + hardest_negative
        - strongest_positive
    ).mean()


def distributional_ranking_loss(
    distribution_logits: Tensor,
    labels: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Distributional Identity Ranking Loss (DIRL).

    This is intentionally a separate public name even though its numerical
    definition is the same multi-positive listwise objective.  It makes the
    optimizer receipt and reports explicit about which branch was supervised.
    """
    return listwise_group_loss(distribution_logits, labels, mask)


def masked_candidate_softmax(
    logits: Tensor,
    mask: Tensor | None = None,
    *,
    dim: int = -1,
) -> Tensor:
    """Softmax over eligible candidates without turning padded rows into NaN.

    Rows with no eligible candidates are returned as all zeros.  This is used
    by the DSSL consistency term and is deliberately separate from the
    ranking loss, which still rejects unusable events.
    """
    values = torch.as_tensor(logits)
    if values.ndim == 0:
        raise ValueError("candidate logits must have at least one dimension")
    if mask is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    else:
        valid = torch.as_tensor(mask, device=values.device, dtype=torch.bool)
        if valid.shape != values.shape:
            raise ValueError("candidate softmax mask must match logits")
    masked = values.masked_fill(~valid, -torch.inf)
    result = torch.softmax(masked, dim=dim)
    return torch.where(valid, result, torch.zeros_like(result))


def structured_branch_ranking_loss(
    recent_logits: Tensor,
    long_logits: Tensor,
    labels: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Average listwise ranking loss for the independent Recent/Long branches."""
    recent = listwise_group_loss(recent_logits, labels, mask)
    long = listwise_group_loss(long_logits, labels, mask)
    return 0.5 * (recent + long)


def gate_aware_temporal_js_consistency(
    recent_logits: Tensor,
    long_logits: Tensor,
    gate: Tensor,
    mask: Tensor | None = None,
    *,
    detach_gate: bool = True,
) -> Tensor:
    """Gate-weighted Recent/Long Jensen-Shannon consistency.

    The gate is detached by default: consistency calibrates the two temporal
    distributions without allowing the consistency objective to directly
    collapse the gate itself.
    """
    recent = torch.as_tensor(recent_logits)
    long = torch.as_tensor(long_logits, device=recent.device, dtype=recent.dtype)
    if recent.shape != long.shape:
        raise ValueError("Recent and Long logits must be aligned")
    valid = torch.ones_like(recent, dtype=torch.bool) if mask is None else torch.as_tensor(
        mask, device=recent.device, dtype=torch.bool
    )
    if valid.shape != recent.shape:
        raise ValueError("temporal consistency mask must match logits")
    p = masked_candidate_softmax(recent, valid)
    q = masked_candidate_softmax(long, valid)
    midpoint = 0.5 * (p + q)
    eps = torch.finfo(recent.dtype).eps
    js = 0.5 * (
        (p * (torch.log(p.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps)))).sum(dim=-1)
        + (q * (torch.log(q.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps)))).sum(dim=-1)
    )
    gate_value = torch.as_tensor(gate, device=recent.device, dtype=recent.dtype)
    if detach_gate:
        gate_value = gate_value.detach()
    if gate_value.ndim == recent.ndim - 1:
        weight = gate_value
    else:
        weight = gate_value.mean(dim=-1) if gate_value.ndim == recent.ndim else gate_value
    weight = weight.clamp(0.0, 1.0)
    usable = valid.any(dim=-1)
    if not bool(usable.any()):
        return recent.new_zeros(())
    return (js[usable] * weight[usable]).mean()


def temporal_conflict_hard_negative_mask(
    labels: Tensor,
    temporal_conflict: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Keep every positive and only conflict-marked negatives for hard mining."""
    labels = torch.as_tensor(labels)
    conflict = torch.as_tensor(temporal_conflict, device=labels.device, dtype=torch.bool)
    if conflict.shape != labels.shape:
        raise ValueError("temporal_conflict must match labels")
    valid = torch.ones_like(labels, dtype=torch.bool) if mask is None else torch.as_tensor(
        mask, device=labels.device, dtype=torch.bool
    )
    if valid.shape != labels.shape:
        raise ValueError("temporal conflict mask must match labels")
    return valid & ((labels == 1) | ((labels == 0) & conflict))


def total_candidate_aware_loss(
    final_logits: Tensor,
    distribution_logits: Tensor | None,
    labels: Tensor,
    mask: Tensor | None = None,
    *,
    lambda_dist: float = 1.0,
    lambda_hard: float = 0.2,
    hard_margin: float = 0.2,
) -> dict[str, Tensor]:
    """Return all auditable components of the requested total objective."""
    final = listwise_group_loss(final_logits, labels, mask)
    hard = hard_negative_margin_loss(
        final_logits, labels, mask, margin=hard_margin
    )
    if distribution_logits is None:
        dist = final_logits.new_zeros(())
    else:
        dist = distributional_ranking_loss(distribution_logits, labels, mask)
    total = final + float(lambda_dist) * dist + float(lambda_hard) * hard
    return {"total": total, "final": final, "distributional": dist, "hard": hard}


def total_dssl_loss(
    final_logits: Tensor,
    recent_logits: Tensor,
    long_logits: Tensor,
    labels: Tensor,
    gate: Tensor,
    mask: Tensor | None = None,
    *,
    lambda_struct: float = 0.25,
    lambda_cons: float = 0.0,
    lambda_hard: float = 0.2,
    hard_margin: float = 0.2,
    temporal_conflict: Tensor | None = None,
) -> dict[str, Tensor]:
    """DSSL objective with exactly three explicit weights."""
    final = listwise_group_loss(final_logits, labels, mask)
    branch = structured_branch_ranking_loss(recent_logits, long_logits, labels, mask)
    consistency = gate_aware_temporal_js_consistency(
        recent_logits, long_logits, gate, mask
    )
    hard_mask = mask
    if temporal_conflict is not None:
        candidate_mask = temporal_conflict_hard_negative_mask(labels, temporal_conflict, mask)
        # If an event has no conflict negative, retain the ordinary negative
        # set for that event; this keeps H1 well-defined on sparse data.
        ordinary = torch.ones_like(candidate_mask, dtype=torch.bool) if mask is None else torch.as_tensor(mask, device=labels.device, dtype=torch.bool)
        has_conflict_negative = (candidate_mask & (labels == 0)).any(dim=-1, keepdim=True)
        hard_mask = torch.where(has_conflict_negative, candidate_mask, ordinary)
    hard = hard_negative_margin_loss(final_logits, labels, hard_mask, margin=hard_margin)
    total = final + float(lambda_struct) * branch + float(lambda_cons) * consistency + float(lambda_hard) * hard
    return {
        "total": total,
        "final": final,
        "structured": branch,
        "consistency": consistency,
        "hard": hard,
    }


__all__ = [
    "distributional_ranking_loss",
    "gate_aware_temporal_js_consistency",
    "hard_negative_margin_loss",
    "listwise_group_loss",
    "masked_candidate_softmax",
    "structured_branch_ranking_loss",
    "temporal_conflict_hard_negative_mask",
    "total_candidate_aware_loss",
    "total_dssl_loss",
]
