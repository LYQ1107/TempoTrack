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


__all__ = [
    "distributional_ranking_loss",
    "hard_negative_margin_loss",
    "listwise_group_loss",
    "total_candidate_aware_loss",
]
