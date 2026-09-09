"""The per-anchor V8 reliability calibrator architecture."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MemoryReliabilityCalibrator(nn.Module):
    """7 -> 32 -> 16 -> 1 logit model with an independent learned scale."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
        self.log_rel_scale = nn.Parameter(torch.tensor(-2.970195, dtype=torch.float32))

    def forward(self, evidence: Tensor) -> Tensor:
        if evidence.shape[-1] != 7:
            raise ValueError(f"reliability evidence must end in 7, got {tuple(evidence.shape)}")
        return self.net(evidence.float()).squeeze(-1)

    @property
    def reliability_scale(self) -> Tensor:
        return F.softplus(self.log_rel_scale)

    def reliability(self, evidence: Tensor) -> Tensor:
        # Keep r in (0, 1).  The learned beta is deliberately not folded into
        # r; PartialSupportScorer applies beta*log(r+eps) exactly once.
        return torch.sigmoid(self.forward(evidence))
