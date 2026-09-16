"""DeepSets candidate-aware QDIC (A1)."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .candidate_aware_qdic import CandidateAwareQDICBase, _mask_max, _mask_mean


class DeepSetQDIC(CandidateAwareQDICBase):
    """Permutation-equivariant set-context residual on top of parent QDIC."""

    architecture_name = "A1-DS-QDIC"

    def __init__(self, parent_qdic: nn.Module) -> None:
        super().__init__(parent_qdic)
        self.set_residual_head = nn.Sequential(
            nn.Linear(320, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.set_residual_head[-1].weight)
        nn.init.zeros_(self.set_residual_head[-1].bias)

    def _interaction(
        self,
        candidate: Tensor,
        distribution_token: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del distribution_token
        mean = _mask_mean(candidate, valid)
        maximum = _mask_max(candidate, valid)
        repeated_mean = mean.unsqueeze(1).expand_as(candidate)
        repeated_maximum = maximum.unsqueeze(1).expand_as(candidate)
        representation = torch.cat(
            (
                candidate,
                repeated_mean,
                repeated_maximum,
                candidate - repeated_mean,
                candidate - repeated_maximum,
            ),
            dim=-1,
        )
        delta = self.set_residual_head(representation).squeeze(-1)
        delta = delta.masked_fill(~valid, 0.0)
        return delta, representation


__all__ = ["DeepSetQDIC"]
