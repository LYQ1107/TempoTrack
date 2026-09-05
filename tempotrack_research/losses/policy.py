"""Masked BC and PPO losses for S5."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def masked_behavior_cloning_loss(logits: Tensor, actions: Tensor, action_mask: Tensor) -> Tensor:
    masked = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
    return F.cross_entropy(masked, actions.long())


def ppo_loss(new_logprob: Tensor, old_logprob: Tensor, advantage: Tensor, value: Tensor, returns: Tensor, entropy: Tensor, clip_ratio: float = 0.2, entropy_weight: float = 0.01, value_weight: float = 0.5) -> dict[str, Tensor]:
    ratio = (new_logprob - old_logprob).exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    policy = -torch.minimum(unclipped, clipped).mean()
    value_loss = 0.5 * (value - returns).square().mean()
    total = policy + value_weight * value_loss - entropy_weight * entropy.mean()
    return {"total": total, "policy": policy.detach(), "value": value_loss.detach(), "entropy": entropy.mean().detach(), "ratio_mean": ratio.mean().detach()}
