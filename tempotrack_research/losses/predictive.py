"""M1 future retrieval and counterfactual utility objectives."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from ..memory.predictive_dual import constrained_rate_penalty


def future_retrieval_loss(fast: Tensor, slow: Tensor, future_embedding: Tensor, same_identity: Tensor, temperature: float = 0.07) -> Tensor:
    fast_score = F.cosine_similarity(fast, future_embedding, dim=-1) / temperature
    slow_score = F.cosine_similarity(slow, future_embedding, dim=-1) / temperature
    target = same_identity.to(fast_score.dtype)
    return F.binary_cross_entropy_with_logits(fast_score, target) + F.binary_cross_entropy_with_logits(slow_score, target)


def predictive_memory_loss(fast: Tensor, slow: Tensor, future_embedding: Tensor, same_identity: Tensor, rates: dict[str, Tensor] | None = None, reliability_logit: Tensor | None = None, reliability: Tensor | None = None, weights: dict[str, float] | None = None) -> dict[str, Tensor]:
    weights = {"future_identity": 1.0, "reliability_bce": 0.2, "rate_regularizer": 0.01, **(weights or {})}
    future = future_retrieval_loss(fast, slow, future_embedding, same_identity)
    total = weights["future_identity"] * future
    output = {"future_identity": future.detach()}
    if reliability_logit is not None and reliability is not None:
        reliability_loss = F.binary_cross_entropy_with_logits(reliability_logit, reliability.to(reliability_logit.dtype))
        total = total + weights["reliability_bce"] * reliability_loss
        output["reliability_bce"] = reliability_loss.detach()
    if rates is not None:
        regularizer = constrained_rate_penalty(rates)
        total = total + weights["rate_regularizer"] * regularizer
        output["rate_regularizer"] = regularizer.detach()
    output["total"] = total
    return output


def counterfactual_utility_loss(predicted_utility: Tensor, utility_target: Tensor) -> Tensor:
    return F.smooth_l1_loss(predicted_utility, utility_target.to(predicted_utility.dtype))
