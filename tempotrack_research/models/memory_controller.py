"""Causal M1 controller with explicit rate parameterisation."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


class MemoryController(nn.Module):
    """Two GELU layers returning q, two rates and reliability logits."""

    def __init__(self, history_dim: int, observation_dim: int, evidence_dim: int, hidden_dim: int = 128, alpha_min: float = 1e-4, alpha_max: float = 0.99, *, initial_alpha_fast: float = 0.7, initial_alpha_slow: float = 0.02):
        super().__init__()
        if history_dim != 2 * observation_dim:
            raise ValueError("M1 history_state must concatenate fast and slow (2D)")
        if not 0.0 < alpha_min < alpha_max < 1.0:
            raise ValueError("invalid M1 alpha bounds")
        if not 0.0 <= alpha_min <= initial_alpha_slow <= initial_alpha_fast < 1.0:
            raise ValueError("initial M1 rates must satisfy 0 <= slow <= fast < 1")
        self.history_dim = int(history_dim)
        self.observation_dim = int(observation_dim)
        self.evidence_dim = int(evidence_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.initial_alpha_fast = float(initial_alpha_fast)
        self.initial_alpha_slow = float(initial_alpha_slow)
        input_dim = history_dim + observation_dim + evidence_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))
        # Biases reproduce the declared M0 initial rates.  q is chosen above
        # the initial fast rate so the effective alpha does not silently start
        # at a new baseline.
        with torch.no_grad():
            final = self.net[-1]
            q0 = min(0.99, max(initial_alpha_fast + 0.05, 0.8))
            raw0 = min(0.98, max(initial_alpha_fast / q0, alpha_min + 1e-4))
            ratio0 = min(0.98, max(initial_alpha_slow / max(initial_alpha_fast, 1e-6), 1e-4))
            final.bias.copy_(torch.tensor([_inverse_sigmoid(q0), _inverse_sigmoid((raw0 - alpha_min) / (alpha_max - alpha_min)), _inverse_sigmoid(ratio0), 0.0]))
        self.initialization = {"q0": q0, "alpha_fast0": initial_alpha_fast, "alpha_slow0": initial_alpha_slow, "alpha_min": alpha_min, "alpha_max": alpha_max}

    def forward(self, history_state: Tensor, accepted_observation: Tensor, causal_evidence: Tensor) -> dict[str, Tensor]:
        if history_state.shape[-1] != self.history_dim or accepted_observation.shape[-1] != self.observation_dim or causal_evidence.shape[-1] != self.evidence_dim:
            raise ValueError(f"M1 input contract mismatch: history={history_state.shape}, observation={accepted_observation.shape}, evidence={causal_evidence.shape}")
        features = torch.cat((history_state, accepted_observation, causal_evidence), dim=-1)
        q_logit, fast_logit, slow_ratio_logit, reliability_logit = self.net(features).unbind(dim=-1)
        q = torch.sigmoid(q_logit)
        fast_raw = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(fast_logit)
        alpha_fast = q * fast_raw
        alpha_slow = alpha_fast * torch.sigmoid(slow_ratio_logit)
        return {
            "q_logit": q_logit,
            "fast_logit": fast_logit,
            "slow_ratio_logit": slow_ratio_logit,
            "reliability_logit": reliability_logit,
            "q": q,
            "alpha_fast_raw": fast_raw,
            "alpha_fast": alpha_fast,
            "alpha_slow": alpha_slow,
            "write_weight": q,
        }
