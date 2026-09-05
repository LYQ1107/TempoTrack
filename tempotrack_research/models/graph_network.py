"""Sparse edge-list graph network without PyG/DGL dependencies."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SparseGraphNetwork(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, layers: int = 4):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden_dim)
        self.edge_in = nn.Linear(edge_dim, hidden_dim)
        self.message = nn.ModuleList(nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers))
        self.node_update = nn.ModuleList(nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers))
        self.edge_update = nn.ModuleList(nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers))

    def forward(self, nodes: Tensor, edges: Tensor, edge_index: Tensor, node_valid: Tensor | None = None, edge_valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if nodes.ndim != 3 or edges.ndim != 3 or edge_index.ndim != 3:
            raise ValueError("graph tensors must be [B,N,F], [B,E,C], [B,2,E]")
        node = self.node_in(nodes)
        edge = self.edge_in(edges)
        batch = torch.arange(nodes.shape[0], device=nodes.device).view(-1, 1)
        for message_net, node_net, edge_net in zip(self.message, self.node_update, self.edge_update):
            src = edge_index[:, 0].clamp_min(0)
            dst = edge_index[:, 1].clamp_min(0)
            source = node[batch, src]
            target = node[batch, dst]
            message = message_net(torch.cat((source, target, edge), dim=-1))
            if edge_valid is not None:
                message = message * edge_valid.to(message.dtype).unsqueeze(-1)
            aggregate = torch.zeros_like(node)
            aggregate.scatter_add_(1, dst.unsqueeze(-1).expand_as(message), message)
            degree = torch.zeros((node.shape[0], node.shape[1], 1), device=node.device, dtype=node.dtype)
            degree.scatter_add_(1, dst.unsqueeze(-1), torch.ones_like(message[..., :1]))
            node = node + node_net(torch.cat((node, aggregate / degree.clamp_min(1.0)), dim=-1))
            edge = edge + edge_net(torch.cat((source, target, edge), dim=-1))
            if node_valid is not None:
                node = node * node_valid.to(node.dtype).unsqueeze(-1)
        return node, edge
