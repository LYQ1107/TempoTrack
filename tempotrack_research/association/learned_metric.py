"""Ordinary metric scoring adapter; it has no prediction loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class OrdinaryMetric(nn.Module):
    def __init__(self, appearance_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(appearance_dim + 5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, appearance: Tensor, geometry: Tensor, time_offsets: Tensor) -> Tensor:
        if appearance.ndim != 3:
            raise ValueError("ordinary metric expects [B,L,D]")
        pooled = torch.cat((appearance.mean(-2), geometry.mean(-2), time_offsets.mean(-1, keepdim=True)), dim=-1)
        return F.normalize(self.encoder(pooled), dim=-1)

    def compute_loss(self, left: Tensor, right: Tensor, same_identity: Tensor) -> dict[str, Tensor]:
        score = (left * right).sum(-1)
        loss = F.binary_cross_entropy_with_logits(score, same_identity.to(score.dtype))
        return {"total": loss, "metric_bce": loss.detach()}
