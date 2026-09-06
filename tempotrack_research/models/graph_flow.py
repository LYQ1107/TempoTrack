"""S3 joint graph conditional flow matching."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .graph_network import SparseGraphNetwork
from .graph_reranker import GraphReranker


class GraphFlowMatcher(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, layers: int = 4):
        super().__init__()
        self.graph = SparseGraphNetwork(node_dim, edge_dim + 2, hidden_dim, layers)
        self.reranker = GraphReranker(node_dim, edge_dim, hidden_dim)
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def vector_field(self, x_s: Tensor, time: Tensor, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, node_valid: Tensor | None = None, edge_valid: Tensor | None = None, condition_graph: Tensor | None = None) -> Tensor:
        if x_s.ndim != 2 or edge_features.ndim != 3 or edge_index.ndim != 3:
            raise ValueError("S3 x_s/edge/index tensors have shapes [B,E], [B,E,C], [B,2,E]")
        if x_s.shape[:2] != edge_features.shape[:2] or edge_index.shape[0] != x_s.shape[0] or edge_index.shape[-1] != x_s.shape[-1]:
            raise ValueError("S3 graph tensor dimensions disagree")
        valid = torch.ones_like(x_s, dtype=torch.bool) if edge_valid is None else edge_valid.bool()
        if valid.shape != x_s.shape:
            raise ValueError("edge_valid must have shape [B,E]")
        if condition_graph is None:
            condition_graph = torch.zeros_like(x_s)
        if condition_graph.shape != x_s.shape:
            raise ValueError("initial_graph condition must have shape [B,E]")
        graph_edges = torch.cat((edge_features, x_s.unsqueeze(-1), condition_graph.unsqueeze(-1)), dim=-1)
        _, edges = self.graph(node_features, graph_edges, edge_index, node_valid, edge_valid)
        time_value = time.reshape(-1, 1)
        if time_value.shape[0] != x_s.shape[0]:
            raise ValueError("flow time must have one value per graph")
        time_features = self.time(time_value).unsqueeze(1).expand(-1, edges.shape[1], -1)
        return self.output(torch.cat((edges, time_features), dim=-1)).squeeze(-1) * valid.to(x_s.dtype)

    def forward(self, x_s: Tensor, time: Tensor, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, node_valid: Tensor | None = None, edge_valid: Tensor | None = None) -> Tensor:
        return self.vector_field(x_s, time, node_features, edge_features, edge_index, node_valid, edge_valid)

    def compute_loss(self, target_graph: Tensor, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, edge_valid: Tensor, node_valid: Tensor | None = None, initial_graph: Tensor | None = None, generator: torch.Generator | None = None, loss_edge_mask: Tensor | None = None) -> dict[str, Tensor]:
        if target_graph.ndim != 2:
            raise ValueError("target_graph must have shape [B,E]")
        if edge_valid.shape != target_graph.shape:
            raise ValueError("edge_valid must have shape [B,E]")
        if initial_graph is None:
            initial_graph = torch.zeros_like(target_graph)
        if initial_graph.shape != target_graph.shape:
            raise ValueError("initial_graph must have shape [B,E]")
        x0 = torch.randn(target_graph.shape, device=target_graph.device, dtype=target_graph.dtype, generator=generator)
        time = torch.rand(target_graph.shape[0], device=target_graph.device, dtype=target_graph.dtype, generator=generator)
        x1 = 2.0 * target_graph - 1.0
        xs = (1 - time[:, None]) * x0 + time[:, None] * x1
        target = x1 - x0
        predicted = self.vector_field(xs, time, node_features, edge_features, edge_index, node_valid, edge_valid, initial_graph)
        mask = edge_valid.bool()
        if loss_edge_mask is not None:
            if loss_edge_mask.shape != mask.shape:
                raise ValueError("loss_edge_mask must have shape [B,E]")
            mask = mask & loss_edge_mask.bool()
        mask_value = mask.to(predicted.dtype)
        loss = ((predicted - target).square() * mask_value).sum() / mask_value.sum().clamp_min(1.0)
        # Train the path-aware graph scorer on the same GT target graph, while
        # keeping the scorer's graph projection discrete and explicit.
        target_score = self.reranker(node_features, edge_features, edge_index, target_graph > 0.5, node_valid, edge_valid)
        initial_score = self.reranker(node_features, edge_features, edge_index, initial_graph > 0.5, node_valid, edge_valid)
        rank = F.softplus(-(target_score - initial_score)).mean()
        total = loss + 0.1 * rank
        return {"total": total, "flow": loss.detach(), "reranker": rank.detach(), "valid_edges": mask_value.sum().detach()}

    @torch.no_grad()
    def sample(self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, edge_valid: Tensor, node_valid: Tensor | None = None, initial_graph: Tensor | None = None, *, samples: int = 4, steps: int = 32, generator: torch.Generator | None = None) -> Tensor:
        if samples < 1:
            raise ValueError("samples must be positive")
        if steps < 1:
            raise ValueError("S3 requires a positive number of integration steps")
        batch = node_features.shape[0]
        if initial_graph is None:
            initial_graph = torch.zeros((batch, edge_features.shape[1]), dtype=node_features.dtype, device=node_features.device)
        if initial_graph.shape != (batch, edge_features.shape[1]):
            raise ValueError("initial_graph must have shape [B,E]")
        valid = edge_valid.bool()
        outputs = []
        for _ in range(samples):
            state = torch.randn((batch, edge_features.shape[1]), device=node_features.device, dtype=node_features.dtype, generator=generator)
            state = state * valid.to(state.dtype)
            dt = 1.0 / steps
            for index in range(steps):
                t = torch.full((batch,), index / steps, device=state.device, dtype=state.dtype)
                t_next = torch.full_like(t, (index + 1) / steps)
                k1 = self.vector_field(state, t, node_features, edge_features, edge_index, node_valid, edge_valid, initial_graph)
                trial = state + dt * k1
                trial = trial * valid.to(trial.dtype)
                k2 = self.vector_field(trial, t_next, node_features, edge_features, edge_index, node_valid, edge_valid, initial_graph)
                state = state + 0.5 * dt * (k1 + k2)
                state = state * valid.to(state.dtype)
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def propose_graphs(self, tracklets: Tensor, candidate_graph: Mapping[str, Tensor], generator: torch.Generator | None = None, num_samples: int = 4, steps: int = 32) -> Tensor:
        """Return real-valued edge states; legal projection is explicit later."""
        return self.sample(
            tracklets,
            candidate_graph["edge_features"],
            candidate_graph["edge_index"],
            candidate_graph["edge_valid"],
            candidate_graph.get("node_valid"),
            candidate_graph.get("initial_graph"),
            samples=num_samples,
            steps=steps,
            generator=generator,
        )
