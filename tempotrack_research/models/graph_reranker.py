"""Path-aware reranker for complete legal graph candidates."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class GraphReranker(nn.Module):
    """Score a graph from its ordered paths, not from edge density.

    Path extraction is discrete because the path-cover projection is
    discrete.  Node/edge encoders and the pooling remain differentiable for
    the reranker training task.
    """

    def __init__(self, node_dim: int, edge_feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.node = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge = nn.Sequential(nn.Linear(edge_feature_dim + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.path = nn.Sequential(nn.Linear(hidden_dim * 2 + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.graph = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    @staticmethod
    def _paths(num_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
        outgoing: dict[int, int] = {}
        incoming: set[int] = set()
        for source, target in edges:
            if source not in outgoing:
                outgoing[source] = target
                incoming.add(target)
        starts = [node for node in range(num_nodes) if node not in incoming]
        paths: list[list[int]] = []
        visited: set[int] = set()
        for start in starts + [node for node in range(num_nodes) if node not in starts]:
            if start in visited:
                continue
            path = [start]
            visited.add(start)
            while path[-1] in outgoing and outgoing[path[-1]] not in visited:
                path.append(outgoing[path[-1]])
                visited.add(path[-1])
            paths.append(path)
        return paths

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor | None = None,
        selected_edges: Tensor | None = None,
        node_valid: Tensor | None = None,
        edge_valid: Tensor | None = None,
        node_times: Tensor | None = None,
    ) -> Tensor:
        # Old callers passed ``(node_summary, edge_features, selected_edges)``.
        # Refuse that ambiguous form instead of silently producing a density
        # score; the complete graph contract is required for deployment.
        if edge_index is None or selected_edges is None:
            raise ValueError("GraphReranker requires edge_index and selected_edges")
        if node_features.ndim != 3 or edge_features.ndim != 3 or edge_index.ndim != 3 or selected_edges.ndim != 2:
            raise ValueError("reranker inputs must be [B,N,F], [B,E,C], [B,2,E], [B,E]")
        batch_size, nodes, _ = node_features.shape
        edges = edge_features.shape[1]
        if edge_index.shape != (batch_size, 2, edges) or selected_edges.shape != (batch_size, edges):
            raise ValueError("reranker graph dimensions disagree")
        node_mask = torch.ones((batch_size, nodes), dtype=torch.bool, device=node_features.device) if node_valid is None else node_valid.bool()
        edge_mask = torch.ones((batch_size, edges), dtype=torch.bool, device=node_features.device) if edge_valid is None else edge_valid.bool()
        if node_mask.shape != (batch_size, nodes) or edge_mask.shape != (batch_size, edges):
            raise ValueError("reranker validity masks have wrong shape")
        times = torch.zeros((batch_size, nodes), dtype=node_features.dtype, device=node_features.device) if node_times is None else node_times.to(node_features.dtype)
        if times.shape != (batch_size, nodes):
            raise ValueError("node_times must have shape [B,N]")
        node_encoded = self.node(node_features) * node_mask.unsqueeze(-1).to(node_features.dtype)
        scores: list[Tensor] = []
        for batch in range(batch_size):
            chosen = []
            for edge in range(edges):
                source, target = (int(edge_index[batch, 0, edge]), int(edge_index[batch, 1, edge]))
                if bool(selected_edges[batch, edge]) and bool(edge_mask[batch, edge]) and 0 <= source < nodes and 0 <= target < nodes and bool(node_mask[batch, source]) and bool(node_mask[batch, target]):
                    chosen.append((source, target))
            paths = self._paths(nodes, chosen)
            path_values: list[Tensor] = []
            for path in paths:
                path_nodes = torch.stack([node_encoded[batch, index] for index in path], dim=0)
                node_value = path_nodes.mean(dim=0)
                edge_values: list[Tensor] = []
                gaps: list[Tensor] = []
                for left, right in zip(path, path[1:]):
                    match = next((edge for edge, pair in enumerate(chosen) if pair == (left, right)), None)
                    if match is None:
                        continue
                    # Recover the corresponding original edge row; duplicates
                    # are rejected by candidate validation before this point.
                    original = next(edge for edge in range(edges) if int(edge_index[batch, 0, edge]) == left and int(edge_index[batch, 1, edge]) == right and bool(selected_edges[batch, edge]))
                    edge_values.append(edge_features[batch, original])
                    gaps.append((times[batch, right] - times[batch, left]).reshape(1))
                if edge_values:
                    edge_tensor = torch.stack(edge_values)
                    edge_summary = edge_tensor.mean(dim=0)
                    gap_summary = torch.stack(gaps).mean(dim=0)
                    length_value = node_features.new_tensor([float(len(edge_values))])
                else:
                    edge_summary = edge_features.new_zeros((self.edge_feature_dim,))
                    gap_summary = node_features.new_zeros((1,))
                    length_value = node_features.new_zeros((1,))
                path_values.append(self.path(torch.cat((node_value, self._edge_project(edge_summary), gap_summary, length_value), dim=-1)))
            if path_values:
                paths_tensor = torch.stack(path_values)
                mean = paths_tensor.mean(0)
                maximum = paths_tensor.max(0).values
            else:
                mean = node_features.new_zeros((self.hidden_dim,))
                maximum = mean
            global_node = node_encoded[batch, node_mask[batch]].mean(0) if bool(node_mask[batch].any()) else node_features.new_zeros((self.hidden_dim,))
            scores.append(self.graph(torch.cat((mean, maximum, global_node), dim=-1)).squeeze(-1))
        return torch.stack(scores)

    def _edge_project(self, edge_summary: Tensor) -> Tensor:
        # ``self.edge`` expects raw edge features plus two graph-local scalar
        # fields; use zero placeholders because path length/gap are supplied
        # separately to ``self.path``.
        zeros = edge_summary.new_zeros((2,))
        return self.edge(torch.cat((edge_summary, zeros), dim=-1))
