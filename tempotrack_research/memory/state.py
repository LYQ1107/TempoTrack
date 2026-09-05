"""Numerically safe compact identity memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


def safe_normalize(value: Tensor, fallback: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    """Normalize vectors and explicitly fall back for near-zero inputs."""

    norm = value.norm(dim=-1, keepdim=True)
    normalized = value / norm.clamp_min(eps)
    if fallback is None:
        fallback = torch.zeros_like(value)
    fallback_norm = fallback.norm(dim=-1, keepdim=True)
    fallback = fallback / fallback_norm.clamp_min(eps)
    return torch.where(norm > eps, normalized, fallback)


@dataclass
class MemoryState:
    fast: Tensor
    slow: Tensor
    last_seen: Tensor
    write_count: Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def detach(self) -> "MemoryState":
        return MemoryState(
            self.fast.detach(),
            self.slow.detach(),
            self.last_seen.detach(),
            None if self.write_count is None else self.write_count.detach(),
            dict(self.diagnostics),
        )


def initialize_state(prototype: Tensor, frame: Tensor | int = 0) -> MemoryState:
    prototype = safe_normalize(prototype, prototype)
    return MemoryState(prototype.clone(), prototype.clone(), torch.as_tensor(frame, device=prototype.device), torch.zeros(prototype.shape[:-1], device=prototype.device))
