"""M1 predictive dual-speed memory and its causal update order."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .state import MemoryState, initialize_state, safe_normalize
from ..models.memory_controller import MemoryController


class PredictiveDualMemory(nn.Module):
    def __init__(self, history_dim: int, observation_dim: int, evidence_dim: int, hidden_dim: int = 128, alpha_min: float = 1e-4, alpha_max: float = 0.99, controller: nn.Module | None = None):
        super().__init__()
        self.controller = controller or MemoryController(history_dim, observation_dim, evidence_dim, hidden_dim, alpha_min, alpha_max)
        self.utility_head = nn.Sequential(nn.Linear(history_dim + observation_dim + evidence_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def initialize(self, prototype: Tensor, frame: Tensor | int = 0) -> MemoryState:
        return initialize_state(prototype, frame)

    def causal_evidence(self, state: MemoryState, observation: Tensor, gap: Tensor, score_margin: Tensor, geometry_delta: Tensor, age: Tensor) -> Tensor:
        consistency = (state.fast * state.slow).sum(dim=-1, keepdim=True)
        if state.fast.ndim == 1:
            fields = [consistency.reshape(-1), gap.reshape(-1), score_margin.reshape(-1), geometry_delta.reshape(-1), age.reshape(-1)]
        else:
            batch = state.fast.shape[0]

            def column(value: Tensor) -> Tensor:
                value = value.to(state.fast.device)
                if value.ndim == 0:
                    value = value.expand(batch)
                if value.shape[0] != batch:
                    value = value.expand(batch, *value.shape[1:])
                return value.reshape(batch, -1)

            fields = [column(consistency), column(gap), column(score_margin), column(geometry_delta), column(age)]
        return torch.cat(fields, dim=-1)

    def update(self, state: MemoryState, observation: Tensor, history_state: Tensor, causal_evidence: Tensor, frame: Tensor | int | None = None) -> tuple[MemoryState, dict[str, Tensor]]:
        """Read old state, match externally, then write the accepted observation."""

        z = safe_normalize(observation, state.fast)
        rates = self.controller(history_state, z, causal_evidence)
        fast = safe_normalize((1 - rates["alpha_fast"].unsqueeze(-1)) * state.fast + rates["alpha_fast"].unsqueeze(-1) * z, state.fast)
        slow = safe_normalize((1 - rates["alpha_slow"].unsqueeze(-1)) * state.slow + rates["alpha_slow"].unsqueeze(-1) * z, state.slow)
        last_seen = state.last_seen if frame is None else torch.as_tensor(frame, device=z.device)
        old_count = state.write_count if state.write_count is not None else torch.zeros_like(rates["q"])
        new_state = MemoryState(fast, slow, last_seen, old_count + rates["q"], dict(state.diagnostics))
        diagnostics = dict(rates)
        diagnostics.update({"prototype_norm_fast": fast.norm(dim=-1), "prototype_norm_slow": slow.norm(dim=-1), "write_rate": (rates["q"] > 0.5).to(z.dtype)})
        return new_state, diagnostics

    def update_from_match(self, state: MemoryState, observation: Tensor, history_state: Tensor, gap: Tensor, score_margin: Tensor, geometry_delta: Tensor, age: Tensor, frame: Tensor | int | None = None) -> tuple[MemoryState, dict[str, Tensor]]:
        evidence = self.causal_evidence(state, observation, gap, score_margin, geometry_delta, age)
        return self.update(state, observation, history_state, evidence, frame)

    def predict_utility(self, history_state: Tensor, accepted_observation: Tensor, causal_evidence: Tensor) -> Tensor:
        return self.utility_head(torch.cat((history_state, accepted_observation, causal_evidence), dim=-1))

    def counterfactual_branches(self, state: MemoryState, observation: Tensor, alpha_fast: Tensor, alpha_slow: Tensor) -> dict[str, MemoryState]:
        """Return hold/recent-only/recent-and-anchor branches for utility labels."""
        z = safe_normalize(observation, state.fast)
        fast = safe_normalize((1 - alpha_fast.unsqueeze(-1)) * state.fast + alpha_fast.unsqueeze(-1) * z, state.fast)
        slow = safe_normalize((1 - alpha_slow.unsqueeze(-1)) * state.slow + alpha_slow.unsqueeze(-1) * z, state.slow)
        return {
            "hold": state,
            "recent_only": MemoryState(fast, state.slow, state.last_seen, state.write_count, dict(state.diagnostics)),
            "recent_and_anchor": MemoryState(fast, slow, state.last_seen, state.write_count, dict(state.diagnostics)),
        }


def constrained_rate_penalty(rates: dict[str, Tensor]) -> Tensor:
    fast, slow = rates["alpha_fast"], rates["alpha_slow"]
    return (slow - fast).clamp_min(0).square().mean() + (fast.clamp_min(0) - 1).clamp_min(0).square().mean()
