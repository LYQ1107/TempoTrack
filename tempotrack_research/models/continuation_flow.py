"""S2 conditional successor-state flow matching."""

from __future__ import annotations

import hashlib
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FrozenTrajectoryProjection(nn.Module):
    """Common state projection frozen before S2 controls are compared."""

    def __init__(self, input_dim: int, latent_dim: int = 64, mean: Tensor | None = None, std: Tensor | None = None):
        super().__init__()
        self.projection = nn.Linear(input_dim, latent_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.register_buffer("mean", torch.zeros(input_dim) if mean is None else mean.detach().clone())
        self.register_buffer("std", torch.ones(input_dim) if std is None else std.detach().clone().clamp_min(1e-6))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def fit_standardization(self, values: Tensor) -> None:
        if values.numel() == 0:
            return
        self.mean.copy_(values.detach().mean(dim=0))
        self.std.copy_(values.detach().std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, values: Tensor) -> Tensor:
        return self.projection((values - self.mean) / self.std)

    def snapshot_hash(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.projection.weight, self.mean, self.std):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


class _ResidualField(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int = 256, layers: int = 4):
        super().__init__()
        blocks: list[nn.Module] = []
        in_dim = latent_dim + condition_dim + 1
        for _ in range(layers):
            blocks.extend((nn.Linear(in_dim, hidden_dim), nn.GELU()))
            in_dim = hidden_dim
        blocks.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        while time.ndim < state.ndim:
            time = time.unsqueeze(-1)
        if condition.ndim == state.ndim - 1:
            condition = condition.unsqueeze(-2).expand(*state.shape[:-1], condition.shape[-1])
        return self.net(torch.cat((state, time.expand(*state.shape[:-1], 1), condition), dim=-1))


class ContinuationFlowModel(nn.Module):
    def __init__(self, condition_dim: int, latent_dim: int = 64, hidden_dim: int = 256, layers: int = 4):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.field = _ResidualField(latent_dim, condition_dim, hidden_dim, layers)
        self.exists_head = nn.Sequential(nn.Linear(condition_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def vector_field(self, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        return self.field(state, time, condition)

    def forward(self, state: Tensor, time: Tensor, condition: Tensor) -> dict[str, Tensor]:
        return {"velocity": self.vector_field(state, time, condition), "existence_logit": self.exists_head(condition).squeeze(-1)}

    def compute_loss(self, source_state: Tensor, target_state: Tensor, condition: Tensor, exists: Tensor | None = None, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        noise = torch.randn(source_state.shape, device=source_state.device, dtype=source_state.dtype, generator=generator)
        t = torch.rand(source_state.shape[:-1], device=source_state.device, dtype=source_state.dtype, generator=generator)
        state = (1 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * target_state
        target = target_state - noise
        predicted = self.vector_field(state, t, condition)
        flow = F.mse_loss(predicted, target)
        output = {"total": flow, "flow": flow.detach()}
        if exists is not None:
            exists_loss = F.binary_cross_entropy_with_logits(self.exists_head(condition).squeeze(-1), exists.to(source_state.dtype))
            output["existence"] = exists_loss.detach()
            output["total"] = flow + 0.2 * exists_loss
        return output


@torch.no_grad()
def heun_integrate(field: Callable[[Tensor, Tensor], Tensor], initial: Tensor, steps: int = 32) -> Tensor:
    if steps < 1:
        raise ValueError("Heun integration requires at least one step")
    value = initial
    dt = 1.0 / steps
    for index in range(steps):
        t = torch.full(value.shape[:-1], index / steps, device=value.device, dtype=value.dtype)
        t_next = torch.full(value.shape[:-1], (index + 1) / steps, device=value.device, dtype=value.dtype)
        k1 = field(value, t)
        predictor = value + dt * k1
        k2 = field(predictor, t_next)
        value = value + 0.5 * dt * (k1 + k2)
    return value


def normalized_log_mean_kernel_support(samples: Tensor, candidate: Tensor, bandwidth: float = 0.5) -> Tensor:
    distances = (samples - candidate.unsqueeze(-2)).square().mean(dim=-1)
    log_kernel = -distances / (2 * bandwidth**2) - samples.shape[-1] * torch.log(torch.as_tensor(bandwidth, device=samples.device, dtype=samples.dtype))
    return torch.logsumexp(log_kernel, dim=-1) - torch.log(torch.as_tensor(samples.shape[-2], device=samples.device, dtype=samples.dtype))
