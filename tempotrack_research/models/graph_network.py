"""Dependency-free sparse graph network with strict validity semantics."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SparseGraphNetwork(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, layers: int = 4):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden_dim)
        self.edge_in = nn.Linear(edge_dim, hidden_dim)
        self.message = nn.ModuleList(nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers))
        self.node_update = nn.ModuleList(nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers))
        self.edge_update = nn.ModuleList(nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers))

    @staticmethod
    def _valid_indices(edge_index: Tensor, nodes: Tensor, node_valid: Tensor | None, edge_valid: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        if edge_index.ndim != 3 or edge_index.shape[1] != 2:
            raise ValueError("edge_index must have shape [B,2,E]")
        batch_size, _, edges = edge_index.shape
        raw_src, raw_dst = edge_index[:, 0], edge_index[:, 1]
        endpoint_valid = (raw_src >= 0) & (raw_src < nodes.shape[1]) & (raw_dst >= 0) & (raw_dst < nodes.shape[1])
        valid = endpoint_valid
        if edge_valid is not None:
            if edge_valid.shape != (batch_size, edges):
                raise ValueError("edge_valid must have shape [B,E]")
            valid = valid & edge_valid.bool()
        if node_valid is not None:
            if node_valid.shape != (batch_size, nodes.shape[1]):
                raise ValueError("node_valid must have shape [B,N]")
            safe_src = raw_src.clamp(0, max(nodes.shape[1] - 1, 0))
            safe_dst = raw_dst.clamp(0, max(nodes.shape[1] - 1, 0))
            valid = valid & node_valid.gather(1, safe_src) & node_valid.gather(1, safe_dst)
        safe_src = raw_src.masked_fill(~endpoint_valid, 0).long()
        safe_dst = raw_dst.masked_fill(~endpoint_valid, 0).long()
        return safe_src, safe_dst, valid

    def forward(self, nodes: Tensor, edges: Tensor, edge_index: Tensor, node_valid: Tensor | None = None, edge_valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if nodes.ndim != 3 or edges.ndim != 3 or edge_index.ndim != 3:
            raise ValueError("graph tensors must be [B,N,F], [B,E,C], [B,2,E]")
        if edges.shape[1] != edge_index.shape[-1]:
            raise ValueError("edge feature/index lengths disagree")
        node = self.node_in(nodes)
        edge = self.edge_in(edges)
        batch = torch.arange(nodes.shape[0], device=nodes.device).view(-1, 1)
        safe_src, safe_dst, valid = self._valid_indices(edge_index, nodes, node_valid, edge_valid)
        valid_f = valid.to(node.dtype).unsqueeze(-1)
        node_mask = torch.ones(node.shape[:2], dtype=node.dtype, device=node.device) if node_valid is None else node_valid.to(node.dtype)
        for message_net, node_net, edge_net in zip(self.message, self.node_update, self.edge_update):
            source = node[batch, safe_src]
            target = node[batch, safe_dst]
            message = message_net(torch.cat((source, target, edge), dim=-1)) * valid_f
            out_aggregate = torch.zeros_like(node)
            in_aggregate = torch.zeros_like(node)
            out_aggregate.scatter_add_(1, safe_src.unsqueeze(-1).expand_as(message), message)
            in_aggregate.scatter_add_(1, safe_dst.unsqueeze(-1).expand_as(message), message)
            out_degree = torch.zeros((node.shape[0], node.shape[1], 1), dtype=node.dtype, device=node.device)
            in_degree = torch.zeros_like(out_degree)
            out_degree.scatter_add_(1, safe_src.unsqueeze(-1), valid_f)
            in_degree.scatter_add_(1, safe_dst.unsqueeze(-1), valid_f)
            aggregate = out_aggregate / out_degree.clamp_min(1.0) + in_aggregate / in_degree.clamp_min(1.0)
            node = (node + node_net(torch.cat((node, aggregate, node * node_mask.unsqueeze(-1)), dim=-1))) * node_mask.unsqueeze(-1)
            edge = (edge + edge_net(torch.cat((source, target, edge), dim=-1)) * valid_f) * valid_f
        return node, edge
