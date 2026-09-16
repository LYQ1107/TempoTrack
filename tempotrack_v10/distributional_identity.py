"""Small learned components for temporal distribution identity evidence."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DistributionEvidenceEncoder(nn.Module):
    """Encode the controlled nine-dimensional distributional evidence."""

    input_dim = 9
    token_dim = 32

    def __init__(self, *, hidden_dim: int = 32) -> None:
        super().__init__()
        if int(hidden_dim) != self.token_dim:
            raise ValueError("the first distributional branch fixes hidden_dim=32")
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.token_dim),
            nn.ReLU(),
            nn.Linear(self.token_dim, self.token_dim),
            nn.ReLU(),
        )

    def forward(self, evidence: Tensor) -> Tensor:
        values = torch.as_tensor(evidence, dtype=self.network[0].weight.dtype,
                                 device=self.network[0].weight.device)
        if values.ndim < 1 or values.shape[-1] != self.input_dim:
            raise ValueError(
                f"distributional evidence must end in [{self.input_dim}], "
                f"got {tuple(values.shape)}"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("distributional evidence must be finite")
        return self.network(values)


class DistributionIdentityHead(nn.Module):
    """Score distribution evidence plus the parent's structured diagnostics."""

    token_dim = 32
    input_dim = 34

    def __init__(self, *, hidden_dim: int = 32) -> None:
        super().__init__()
        if int(hidden_dim) != self.token_dim:
            raise ValueError("the first distributional head fixes hidden_dim=32")
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.token_dim),
            nn.ReLU(),
            nn.Linear(self.token_dim, 1),
        )

    def forward(
        self,
        distribution_token: Tensor,
        alpha: Tensor,
        structured_score: Tensor,
    ) -> Tensor:
        token = torch.as_tensor(
            distribution_token,
            dtype=self.network[0].weight.dtype,
            device=self.network[0].weight.device,
        )
        if token.ndim < 1 or token.shape[-1] != self.token_dim:
            raise ValueError(
                f"distribution token must end in [{self.token_dim}], got {tuple(token.shape)}"
            )
        alpha = torch.as_tensor(alpha, dtype=token.dtype, device=token.device)
        structured_score = torch.as_tensor(
            structured_score, dtype=token.dtype, device=token.device
        )
        if alpha.shape != token.shape[:-1] or structured_score.shape != token.shape[:-1]:
            raise ValueError("alpha/structured_score must align with distribution token")
        values = torch.cat((token, alpha.unsqueeze(-1), structured_score.unsqueeze(-1)), dim=-1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("distribution identity inputs must be finite")
        return self.network(values).squeeze(-1)


__all__ = ["DistributionEvidenceEncoder", "DistributionIdentityHead"]
