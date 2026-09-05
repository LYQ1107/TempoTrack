"""Diffusion epsilon objective with fixed masks."""

from __future__ import annotations

from torch import Tensor


def epsilon_loss(predicted: Tensor, noise: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    value = (predicted - noise).square()
    if valid_mask is None:
        return value.mean()
    mask = valid_mask.to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    if mask.sum() <= 0:
        raise ValueError("diffusion loss has no valid edge")
    return (value * mask).sum() / mask.sum().clamp_min(1.0)
