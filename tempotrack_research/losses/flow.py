"""Flow matching loss with explicit valid masks."""

from __future__ import annotations

from torch import Tensor


def flow_matching_loss(predicted: Tensor, target: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    value = (predicted - target).square().mean(dim=-1)
    if valid_mask is None:
        return value.mean()
    mask = valid_mask.to(value.dtype)
    if mask.sum() <= 0:
        raise ValueError("flow matching has no valid target")
    return (value * mask).sum() / mask.sum()
