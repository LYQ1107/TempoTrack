"""M1 retrieval, reliability and explicit rate-prior losses."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F



def _assert_rate_parameterisation(rates: dict[str, Tensor]) -> Tensor:
    """Validate structural rate constraints without creating a fake loss."""

    fast, slow = rates["alpha_fast"], rates["alpha_slow"]
    if (
        not torch.isfinite(fast).all()
        or not torch.isfinite(slow).all()
        or not bool((slow <= fast + 1e-7).all())
        or not bool(((fast >= 0) & (fast < 1)).all())
    ):
        raise ValueError("M1 rate parameterization violated 0 <= slow <= fast < 1")
    return fast.new_zeros(())


def _multi_positive(scores: Tensor, positive: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
    if scores.ndim == 1:
        scores = scores.unsqueeze(-1)
    if positive.ndim == 1:
        positive = positive.unsqueeze(-1)
    if known.ndim == 1:
        known = known.unsqueeze(-1)
    rows = positive.any(dim=-1) & known.any(dim=-1)
    if not rows.any():
        return scores.new_zeros(()), rows.sum()
    masked = scores.masked_fill(~known.bool(), torch.finfo(scores.dtype).min)
    numerator = torch.logsumexp(masked.masked_fill(~positive.bool(), torch.finfo(scores.dtype).min), dim=-1)
    denominator = torch.logsumexp(masked, dim=-1)
    return (-(numerator - denominator)[rows]).mean(), rows.sum()


def future_retrieval_loss(fast: Tensor, slow: Tensor, future_embedding: Tensor, same_identity: Tensor, candidate_known: Tensor | None = None, temperature: float = 0.07) -> tuple[Tensor, Tensor]:
    future = F.normalize(future_embedding, dim=-1)
    fast = F.normalize(fast, dim=-1)
    slow = F.normalize(slow, dim=-1)
    if future.ndim == 2:
        fast_score = (fast * future).sum(-1) / max(float(temperature), 1e-4)
        slow_score = (slow * future).sum(-1) / max(float(temperature), 1e-4)
        positive = same_identity.bool()
        known = torch.ones_like(positive)
        valid = positive
        if valid.any():
            loss = (F.softplus(-fast_score[positive]) + F.softplus(-slow_score[positive])).mean()
            return loss, valid.sum()
        return fast.new_zeros(()), valid.sum()
    scores_fast = torch.einsum("bd,bkd->bk", fast, future) / max(float(temperature), 1e-4)
    scores_slow = torch.einsum("bd,bkd->bk", slow, future) / max(float(temperature), 1e-4)
    positive = same_identity.bool()
    known = torch.ones_like(positive) if candidate_known is None else candidate_known.bool()
    loss_fast, count = _multi_positive(scores_fast, positive, known)
    loss_slow, _ = _multi_positive(scores_slow, positive, known)
    return loss_fast + loss_slow, count


def predictive_memory_loss(fast: Tensor, slow: Tensor, future_embedding: Tensor, same_identity: Tensor, rates: dict[str, Tensor] | None = None, reliability: Tensor | None = None, reliability_known: Tensor | None = None, candidate_known: Tensor | None = None, weights: dict[str, float] | None = None) -> dict[str, Tensor]:
    weights = {"future_identity": 1.0, "reliability_bce": 0.2, "rate_prior": 0.01, **(weights or {})}
    future, query_count = future_retrieval_loss(fast, slow, future_embedding, same_identity, candidate_known)
    total = weights["future_identity"] * future
    output: dict[str, Tensor] = {"future_identity": future.detach(), "valid_queries": query_count.detach()}
    if rates is not None and reliability is not None:
        known = torch.ones_like(reliability, dtype=torch.bool) if reliability_known is None else reliability_known.bool()
        if known.any():
            reliability_loss = F.binary_cross_entropy_with_logits(rates["reliability_logit"][known], reliability.to(rates["reliability_logit"].dtype)[known])
            total = total + weights["reliability_bce"] * reliability_loss
            output["reliability_bce"] = reliability_loss.detach()
            output["reliability_count"] = known.sum().detach()
    if rates is not None:
        # Raises if the structural parameterization is corrupted.  It is not
        # a fake gradient-producing regularizer.
        output["rate_constraint_diagnostic"] = _assert_rate_parameterisation(rates).detach()
    output["total"] = total
    return output


def counterfactual_utility_loss(predicted_utility: Tensor, utility_target: Tensor) -> Tensor:
    return F.smooth_l1_loss(predicted_utility, utility_target.to(predicted_utility.dtype))
