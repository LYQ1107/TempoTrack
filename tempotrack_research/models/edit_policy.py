"""Masked graph-edit policy used by S5 BC and PPO."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class EditPolicy(nn.Module):
    ACTION_TYPES = ("ADD", "REMOVE", "REWIRE", "STOP")

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge = nn.Sequential(nn.Linear(edge_dim + hidden_dim * 2 + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.action_head = nn.Linear(hidden_dim, len(self.ACTION_TYPES))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, graph_state: Tensor, action_mask: Tensor | None = None) -> dict[str, Tensor]:
        node = self.node(node_features)
        batch = torch.arange(node.shape[0], device=node.device).view(-1, 1)
        src = edge_index[:, 0].clamp_min(0)
        dst = edge_index[:, 1].clamp_min(0)
        edge_context = self.edge(torch.cat((edge_features, node[batch, src], node[batch, dst], graph_state.unsqueeze(-1).to(edge_features.dtype)), dim=-1))
        pooled = node.mean(dim=-2) + edge_context.mean(dim=-2)
        logits = self.action_head(pooled)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return {"logits": logits, "value": self.value_head(pooled).squeeze(-1), "edge_context": edge_context}

    def distribution(self, output: dict[str, Tensor]) -> Categorical:
        return Categorical(logits=output["logits"])
