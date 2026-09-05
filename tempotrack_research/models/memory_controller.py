"""Constrained causal controller for M1 predictive dual memory."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MemoryController(nn.Module):
    """Two-layer MLP returning rates with ``0 <= slow <= fast < 1``."""

    def __init__(self, history_dim: int, observation_dim: int, evidence_dim: int, hidden_dim: int = 128, alpha_min: float = 1e-4, alpha_max: float = 0.99):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        input_dim = int(history_dim + observation_dim + evidence_dim)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))

    def forward(self, history_state: Tensor, accepted_observation: Tensor, causal_evidence: Tensor) -> dict[str, Tensor]:
        features = torch.cat((history_state, accepted_observation, causal_evidence), dim=-1)
        q_logit, fast_logit, slow_ratio_logit, reliability_logit = self.net(features).unbind(dim=-1)
        q = torch.sigmoid(q_logit)
        fast_raw = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(fast_logit)
        alpha_fast = q * fast_raw
        alpha_slow = alpha_fast * torch.sigmoid(slow_ratio_logit)
        return {
            "q": q,
            "alpha_fast": alpha_fast,
            "alpha_slow": alpha_slow,
            "write_weight": q,
            "alpha_fast_raw": fast_raw,
            "reliability_logit": reliability_logit,
        }
