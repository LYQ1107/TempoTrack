"""Non-oracle reranker for complete legal graph candidates."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GraphReranker(nn.Module):
    def __init__(self, node_dim: int, edge_feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(node_dim + edge_feature_dim + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, node_summary: Tensor, edge_features: Tensor, selected_edges: Tensor, edge_valid: Tensor | None = None) -> Tensor:
        # selected_edges is a [B,E] binary graph state; graph-level statistics
        # make this a structured score rather than an independent edge MLP.
        density = selected_edges.mean(dim=-1, keepdim=True)
        degree_proxy = selected_edges.sum(dim=-1, keepdim=True)
        summary = node_summary.mean(dim=-2) if node_summary.ndim == 3 else node_summary
        edge_summary = (edge_features * selected_edges.unsqueeze(-1)).sum(dim=-2) / selected_edges.sum(dim=-1, keepdim=True).clamp_min(1.0)
        features = torch.cat((summary, edge_summary, density, degree_proxy, torch.log1p(degree_proxy)), dim=-1)
        return self.net(features).squeeze(-1)
