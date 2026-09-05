"""M1 training objective wrappers."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from ..losses.predictive import predictive_memory_loss
from ..losses.predictive import counterfactual_utility_loss


def memory_loss(memory_output: Mapping[str, Tensor], future_embedding: Tensor, same_identity: Tensor, reliability: Tensor | None = None) -> dict[str, Tensor]:
    return predictive_memory_loss(memory_output["fast"], memory_output["slow"], future_embedding, same_identity, memory_output.get("rates"), memory_output.get("reliability_logit"), reliability)


def utility_loss(predicted_utility: Tensor, branch_states: Mapping[str, Tensor], future_embedding: Tensor) -> dict[str, Tensor]:
    """Train action-utility predictions from the same-state counterfactuals."""
    future = future_embedding / future_embedding.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scores = torch.stack([(value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8) * future).sum(-1) for value in branch_states.values()], dim=-1)
    target = scores - scores[..., :1]
    loss = counterfactual_utility_loss(predicted_utility, target)
    return {"total": loss, "utility": loss.detach(), "utility_target": target.detach()}
