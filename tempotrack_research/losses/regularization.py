"""VICReg-style regularization on pre-normalization representations."""

from __future__ import annotations

import torch
from torch import Tensor


def vicreg_regularization(representations: Tensor, variance_target: float = 1.0, eps: float = 1e-4) -> dict[str, Tensor]:
    if representations.ndim != 2:
        representations = representations.reshape(-1, representations.shape[-1])
    centered = representations - representations.mean(dim=0)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    variance = torch.relu(variance_target - std).mean()
    covariance = centered.t() @ centered / max(centered.shape[0] - 1, 1)
    off_diag = covariance.flatten()[:-1].view(covariance.shape[0] - 1, covariance.shape[1] + 1)[:, 1:].flatten() if covariance.numel() else covariance
    covariance_loss = off_diag.square().mean() if off_diag.numel() else representations.new_zeros(())
    return {"variance": variance, "covariance": covariance_loss, "total": variance + covariance_loss, "mean_std": std.mean().detach()}
