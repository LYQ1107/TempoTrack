"""S3 joint graph conditional flow matching."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .graph_network import SparseGraphNetwork


class GraphFlowMatcher(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, layers: int = 4):
        super().__init__()
        self.graph = SparseGraphNetwork(node_dim, edge_dim + 2, hidden_dim, layers)
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def vector_field(self, x_s: Tensor, time: Tensor, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, node_valid: Tensor | None = None, edge_valid: Tensor | None = None, condition_graph: Tensor | None = None) -> Tensor:
        if condition_graph is None:
            condition_graph = torch.zeros_like(x_s)
        graph_edges = torch.cat((edge_features, x_s.unsqueeze(-1), condition_graph.unsqueeze(-1)), dim=-1)
        _, edges = self.graph(node_features, graph_edges, edge_index, node_valid, edge_valid)
        time_features = self.time(time.reshape(-1, 1)).unsqueeze(1).expand(-1, edges.shape[1], -1)
        return self.output(torch.cat((edges, time_features), dim=-1)).squeeze(-1)

    def forward(self, x_s: Tensor, time: Tensor, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, node_valid: Tensor | None = None, edge_valid: Tensor | None = None) -> Tensor:
        return self.vector_field(x_s, time, node_features, edge_features, edge_index, node_valid, edge_valid)

    def compute_loss(self, target_graph: Tensor, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, edge_valid: Tensor, node_valid: Tensor | None = None, initial_graph: Tensor | None = None, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        if initial_graph is None:
            initial_graph = torch.zeros_like(target_graph)
        x0 = torch.randn(target_graph.shape, device=target_graph.device, dtype=target_graph.dtype, generator=generator)
        time = torch.rand(target_graph.shape[0], device=target_graph.device, dtype=target_graph.dtype, generator=generator)
        x1 = 2.0 * target_graph - 1.0
        xs = (1 - time[:, None]) * x0 + time[:, None] * x1
        target = x1 - x0
        predicted = self.vector_field(xs, time, node_features, edge_features, edge_index, node_valid, edge_valid, initial_graph)
        mask = edge_valid.to(predicted.dtype)
        loss = ((predicted - target).square() * mask).sum() / mask.sum().clamp_min(1.0)
        return {"total": loss, "flow": loss.detach(), "valid_edges": mask.sum().detach()}

    @torch.no_grad()
    def sample(self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, edge_valid: Tensor, node_valid: Tensor | None = None, samples: int = 4, steps: int = 32, generator: torch.Generator | None = None) -> Tensor:
        if samples < 1:
            raise ValueError("samples must be positive")
        batch = node_features.shape[0]
        outputs = []
        for _ in range(samples):
            state = torch.randn((batch, edge_features.shape[1]), device=node_features.device, dtype=node_features.dtype, generator=generator)
            dt = 1.0 / steps
            for index in range(steps):
                t = torch.full((batch,), index / steps, device=state.device, dtype=state.dtype)
                t_next = torch.full_like(t, (index + 1) / steps)
                k1 = self.vector_field(state, t, node_features, edge_features, edge_index, node_valid, edge_valid)
                trial = state + dt * k1
                k2 = self.vector_field(trial, t_next, node_features, edge_features, edge_index, node_valid, edge_valid)
                state = state + 0.5 * dt * (k1 + k2)
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def propose_graphs(self, tracklets: Tensor, candidate_graph: Mapping[str, Tensor], generator: torch.Generator | None = None, num_samples: int = 4) -> Tensor:
        """Return real-valued edge states; legal projection is explicit later."""
        samples = self.sample(tracklets, candidate_graph["edge_features"], candidate_graph["edge_index"], candidate_graph["edge_valid"], candidate_graph.get("node_valid"), num_samples, 32, generator)
        return samples.sigmoid()
