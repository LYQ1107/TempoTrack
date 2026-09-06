"""M0 fixed memory controls with a shared matching implementation."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F

from .state import MemoryState, initialize_state, safe_normalize


MemoryMode = Literal["single_ema", "fixed_dual", "confidence_gated_dual"]


class FixedDualMemory:
    """Causal EMA memory used by all M0 controls.

    ``fast`` and ``slow`` are state variables, while matching normalization,
    temperature, BiSoftmax, candidates, and archive policies live outside the
    mode switch and are therefore identical across controls.
    """

    def __init__(
        self,
        mode: MemoryMode = "fixed_dual",
        alpha_fast: float = 0.7,
        alpha_slow: float = 0.02,
        single_alpha: float = 0.8,
        confidence_threshold: float = 0.55,
        logit_scale: float = 10.0,
    ) -> None:
        if mode not in {"single_ema", "fixed_dual", "confidence_gated_dual"}:
            raise ValueError(f"Unsupported M0 memory mode: {mode}")
        if not (0 <= alpha_slow <= alpha_fast < 1):
            raise ValueError("M0 requires 0 <= alpha_slow <= alpha_fast < 1")
        self.mode = mode
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow = float(alpha_slow)
        self.single_alpha = float(single_alpha)
        self.confidence_threshold = float(confidence_threshold)
        self.logit_scale = float(logit_scale)

    def initialize(self, prototype: Tensor, frame: Tensor | int = 0) -> MemoryState:
        return initialize_state(prototype, frame)

    def update(
        self,
        state: MemoryState,
        observation: Tensor,
        confidence: Tensor | float = 1.0,
        frame: Tensor | int | None = None,
    ) -> tuple[MemoryState, dict[str, Tensor]]:
        z = safe_normalize(observation, state.fast)
        conf = torch.as_tensor(confidence, dtype=z.dtype, device=z.device)
        while conf.ndim < z.ndim:
            conf = conf.unsqueeze(-1)
        if self.mode == "single_ema":
            alpha_fast = torch.full_like(conf, self.single_alpha)
            alpha_slow = alpha_fast
        elif self.mode == "fixed_dual":
            alpha_fast = torch.full_like(conf, self.alpha_fast)
            alpha_slow = torch.full_like(conf, self.alpha_slow)
        else:
            gate = (conf >= self.confidence_threshold).to(z.dtype)
            alpha_fast = gate * self.alpha_fast
            alpha_slow = gate * self.alpha_slow
        fast = safe_normalize((1 - alpha_fast) * state.fast + alpha_fast * z, state.fast)
        slow = safe_normalize((1 - alpha_slow) * state.slow + alpha_slow * z, state.slow)
        last_seen = state.last_seen if frame is None else torch.as_tensor(frame, device=z.device)
        count = (state.write_count if state.write_count is not None else torch.zeros_like(conf.squeeze(-1))) + (alpha_fast.squeeze(-1) > 0).to(z.dtype)
        updated = MemoryState(
            fast=fast,
            slow=slow,
            last_seen=last_seen,
            write_count=count,
            birth_time=state.birth_time,
            last_bbox=state.last_bbox,
            diagnostics=dict(state.diagnostics),
        )
        diagnostics = {
            "alpha_fast": alpha_fast.squeeze(-1),
            "alpha_slow": alpha_slow.squeeze(-1),
            "write_weight": (alpha_fast / max(self.alpha_fast, 1e-8)).squeeze(-1),
            "prototype_norm_fast": fast.norm(dim=-1),
            "prototype_norm_slow": slow.norm(dim=-1),
        }
        return updated, diagnostics

    def prototypes(self, state: MemoryState) -> Tensor:
        if self.mode == "single_ema":
            return state.fast
        return torch.stack((state.fast, state.slow), dim=-2)

    def logits(self, query: Tensor, state: MemoryState, metric: str = "bisoftmax") -> Tensor:
        query = F.normalize(query, dim=-1)
        prototypes = self.prototypes(state)
        if prototypes.ndim == query.ndim:
            return self.logit_scale * (query * F.normalize(prototypes, dim=-1)).sum(-1)
        scores = self.logit_scale * torch.einsum("...d,...kd->...k", query, F.normalize(prototypes, dim=-1))
        if metric == "cosine":
            return scores.mean(dim=-1)
        if metric == "softmax":
            return scores.softmax(dim=-1).max(dim=-1).values
        if metric != "bisoftmax":
            raise ValueError(f"Unknown shared matching metric: {metric}")
        row = scores.softmax(dim=-1)
        col = scores.softmax(dim=-2)
        return (row * col).sum(dim=-1)
