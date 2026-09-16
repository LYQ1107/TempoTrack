"""Distribution-Guided Set-Attention QDIC (A2)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .candidate_aware_qdic import CandidateAwareQDICBase


class DistributionGuidedSetAttentionQDIC(CandidateAwareQDICBase):
    """One lightweight, mask-aware, permutation-equivariant attention block."""

    architecture_name = "A2-DGSA-QDIC"

    def __init__(self, parent_qdic: nn.Module) -> None:
        super().__init__(parent_qdic)
        self.query_projection = nn.Linear(32, 64)
        self.key_projection = nn.Linear(32, 64)
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(64)
        self.feed_forward = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )
        self.feed_forward_norm = nn.LayerNorm(64)
        self.attention_residual_head = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.attention_residual_head[-1].weight)
        nn.init.zeros_(self.attention_residual_head[-1].bias)

    def _interaction(
        self,
        candidate: Tensor,
        distribution_token: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # A fully masked row is made safe for MHA, then its output is masked
        # back out.  Real training/inference events always have at least one
        # valid candidate, but this keeps padded unit-test batches finite.
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        if bool(empty.any()):
            safe_valid[empty, 0] = True
        query = self.query_projection(distribution_token)
        key = self.key_projection(distribution_token)
        attended, _ = self.attention(
            query,
            key,
            candidate,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        attended = self.attention_norm(attended + candidate)
        attended = self.feed_forward_norm(attended + self.feed_forward(attended))
        attended = attended.masked_fill(~valid.unsqueeze(-1), 0.0)
        projected_distribution = self.query_projection(distribution_token)
        representation = torch.cat((attended, candidate, projected_distribution), dim=-1)
        delta = self.attention_residual_head(representation).squeeze(-1)
        delta = delta.masked_fill(~valid, 0.0)
        return delta, attended


__all__ = ["DistributionGuidedSetAttentionQDIC"]
