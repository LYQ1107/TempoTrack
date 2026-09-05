"""S5 BC and PPO update helpers with explicit terminal/truncation fields."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from ..losses.policy import masked_behavior_cloning_loss, ppo_loss


def behavior_cloning_step(policy: nn.Module, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    output = policy(batch["node_features"], batch["edge_features"], batch["edge_index"], batch["graph_state"], batch["action_mask"])
    loss = masked_behavior_cloning_loss(output["logits"], batch["actions"], batch["action_mask"])
    return {"total": loss, "bc": loss.detach()}


def ppo_step(policy: nn.Module, batch: Mapping[str, Tensor], clip_ratio: float = 0.2, entropy_weight: float = 0.01, value_weight: float = 0.5) -> dict[str, Tensor]:
    output = policy(batch["node_features"], batch["edge_features"], batch["edge_index"], batch["graph_state"], batch["action_mask"])
    distribution = policy.distribution(output)
    new_logprob = distribution.log_prob(batch["actions"])
    entropy = distribution.entropy()
    return ppo_loss(new_logprob, batch["old_logprob"], batch["advantage"], output["value"], batch["returns"], entropy, clip_ratio, entropy_weight, value_weight)
