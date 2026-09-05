"""S4 VP-DDPM epsilon prediction and DDIM sampling."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .graph_network import SparseGraphNetwork


def cosine_beta_schedule(steps: int = 1000, offset: float = 0.008) -> Tensor:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((x / steps) + offset) / (1 + offset) * math.pi / 2).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


class GraphDiffusionMatcher(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, layers: int = 4, diffusion_steps: int = 1000):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        betas = cosine_beta_schedule(diffusion_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0))
        self.graph = SparseGraphNetwork(node_dim, edge_dim + 2, hidden_dim, layers)
        self.time = nn.Embedding(diffusion_steps, hidden_dim)
        self.output = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def predict_epsilon(self, x_t: Tensor, timestep: Tensor, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, edge_valid: Tensor, node_valid: Tensor | None = None, condition_graph: Tensor | None = None) -> Tensor:
        condition_graph = torch.zeros_like(x_t) if condition_graph is None else condition_graph
        _, edges = self.graph(node_features, torch.cat((edge_features, x_t.unsqueeze(-1), condition_graph.unsqueeze(-1)), dim=-1), edge_index, node_valid, edge_valid)
        time = self.time(timestep.long()).unsqueeze(1).expand(-1, edges.shape[1], -1)
        return self.output(torch.cat((edges, time), dim=-1)).squeeze(-1)

    def compute_loss(self, target_graph: Tensor, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, edge_valid: Tensor, condition_graph: Tensor, node_valid: Tensor | None = None, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        x0 = 2.0 * target_graph - 1.0
        timestep = torch.randint(0, self.diffusion_steps, (target_graph.shape[0],), device=target_graph.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        abar = self.alpha_bar[timestep].to(x0.dtype).unsqueeze(-1)
        xt = abar.sqrt() * x0 + (1 - abar).sqrt() * noise
        predicted = self.predict_epsilon(xt, timestep, node_features, edge_features, edge_index, edge_valid, node_valid, condition_graph)
        mask = edge_valid.to(predicted.dtype)
        loss = ((predicted - noise).square() * mask).sum() / mask.sum().clamp_min(1.0)
        return {"total": loss, "epsilon": loss.detach(), "valid_edges": mask.sum().detach(), "timestep_mean": timestep.float().mean().detach()}

    @torch.no_grad()
    def sample(self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor, edge_valid: Tensor, condition_graph: Tensor, node_valid: Tensor | None = None, samples: int = 4, steps: int = 50, generator: torch.Generator | None = None) -> Tensor:
        selected = torch.linspace(self.diffusion_steps - 1, 0, steps, device=node_features.device).long()
        result = []
        for _ in range(samples):
            x = torch.randn((node_features.shape[0], edge_features.shape[1]), device=node_features.device, dtype=node_features.dtype, generator=generator)
            for index, timestep in enumerate(selected):
                t = timestep.expand(node_features.shape[0])
                abar = self.alpha_bar[t].to(x.dtype).unsqueeze(-1)
                eps = self.predict_epsilon(x, t, node_features, edge_features, edge_index, edge_valid, node_valid, condition_graph)
                x0 = (x - (1 - abar).sqrt() * eps) / abar.sqrt().clamp_min(1e-5)
                if index == len(selected) - 1:
                    x = x0
                    continue
                next_t = selected[index + 1].expand(node_features.shape[0])
                next_abar = self.alpha_bar[next_t].to(x.dtype).unsqueeze(-1)
                x = next_abar.sqrt() * x0 + (1 - next_abar).sqrt() * eps
            result.append(x)
        return torch.stack(result, dim=1)

    @torch.no_grad()
    def propose_graphs(self, tracklets: Tensor, candidate_graph: dict[str, Tensor], generator: torch.Generator | None = None, num_samples: int = 4) -> Tensor:
        states = self.sample(tracklets, candidate_graph["edge_features"], candidate_graph["edge_index"], candidate_graph["edge_valid"], candidate_graph["initial_graph"], candidate_graph.get("node_valid"), num_samples, 50, generator)
        return states.sigmoid()
