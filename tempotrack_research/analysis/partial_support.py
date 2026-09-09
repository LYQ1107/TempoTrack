"""Partial-support scoring.

The implementation deliberately keeps the query observations separate until
after the per-query top-r operation.  Averaging the memory first is a subtly
different statistic and was the source of an earlier V6 control mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PartialSupportConfig:
    query_observations: int = 1
    top_r: int = 3
    memory_capacity: int = 64
    dedup_cos: float = 0.95
    max_gap: int = 60
    candidate_top_k: int = 8
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.query_observations < 1 or self.top_r < 1:
            raise ValueError("query_observations and top_r must be positive")
        if self.memory_capacity < self.top_r:
            raise ValueError("memory_capacity must be at least top_r")


@dataclass
class PartialSupportEvidence:
    score: Tensor
    query_scores: Tensor
    top_values: Tensor
    top_indices: Tensor
    cosine: Tensor
    query_mask: Tensor
    memory_mask: Tensor


class PartialSupportScorer(nn.Module):
    """Compute mean_q mean_top-r cosine support with optional reliability.

    ``query`` is ``[Q,D]`` or ``[B,Q,D]`` and ``memory`` is ``[K,D]`` or
    ``[B,K,D]``.  A batch dimension is retained in the returned scalar scores.
    Invalid padded memory entries never enter top-r; an all-invalid query
    yields ``-inf`` and is marked invalid rather than silently becoming zero.
    """

    def __init__(self, config: PartialSupportConfig | None = None, *, beta: float = 0.0) -> None:
        super().__init__()
        self.config = config or PartialSupportConfig()
        self.beta = float(beta)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        *,
        query_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        memory_reliability: Optional[Tensor] = None,
        reliability_scale: Optional[Tensor | float] = None,
        beta: Optional[float] = None,
    ) -> PartialSupportEvidence:
        q, m, squeeze = self._canonical(query, memory)
        q = F.normalize(q.float(), dim=-1, eps=self.config.eps)
        m = F.normalize(m.float(), dim=-1, eps=self.config.eps)
        cosine = torch.einsum("bqd,bkd->bqk", q, m)
        bsz, q_count, mem_count = cosine.shape
        qmask = self._mask(query_mask, bsz, q_count, cosine.device, default=True)
        mmask = self._mask(memory_mask, bsz, mem_count, cosine.device, default=True)
        values = cosine
        rel_beta = self.beta if beta is None else float(beta)
        if memory_reliability is not None and rel_beta != 0.0:
            rel = self._canonical_reliability(memory_reliability, bsz, mem_count, cosine.device)
            scale = 1.0 if reliability_scale is None else reliability_scale
            if not isinstance(scale, Tensor):
                scale = torch.as_tensor(float(scale), device=cosine.device)
            rel = F.softplus(scale.float()) * rel if scale.numel() == 1 else rel * scale.float()
            values = values + rel.clamp_min(self.config.eps).log().unsqueeze(1) * rel_beta
        values = values.masked_fill(~mmask.unsqueeze(1), float("-inf"))
        k = min(int(self.config.top_r), mem_count)
        top_values, top_indices = torch.topk(values, k=k, dim=-1)
        top_valid = torch.isfinite(top_values)
        q_counts = top_valid.sum(dim=-1).clamp_min(1)
        query_scores = top_values.masked_fill(~top_valid, 0.0).sum(dim=-1) / q_counts
        query_scores = query_scores.masked_fill(~qmask, float("-inf"))
        valid_q = qmask & torch.isfinite(query_scores)
        denom = valid_q.sum(dim=-1).clamp_min(1)
        score = query_scores.masked_fill(~valid_q, 0.0).sum(dim=-1) / denom
        score = score.masked_fill(valid_q.sum(dim=-1) == 0, float("-inf"))
        if squeeze:
            return PartialSupportEvidence(score[0], query_scores[0], top_values[0], top_indices[0], cosine[0], qmask[0], mmask[0])
        return PartialSupportEvidence(score, query_scores, top_values, top_indices, cosine, qmask, mmask)

    @staticmethod
    def _canonical(query: Tensor, memory: Tensor) -> tuple[Tensor, Tensor, bool]:
        if query.ndim == 2 and memory.ndim == 2:
            return query.unsqueeze(0), memory.unsqueeze(0), True
        if query.ndim == 3 and memory.ndim == 3:
            if query.shape[0] != memory.shape[0]:
                raise ValueError("query and memory batch dimensions differ")
            return query, memory, False
        raise ValueError("query/memory must both be [Q,D]/[K,D] or batched [B,Q,D]/[B,K,D]")

    @staticmethod
    def _mask(mask: Optional[Tensor], batch: int, count: int, device: torch.device, *, default: bool) -> Tensor:
        if mask is None:
            return torch.full((batch, count), default, device=device, dtype=torch.bool)
        result = torch.as_tensor(mask, device=device, dtype=torch.bool)
        if result.ndim == 1:
            result = result.unsqueeze(0).expand(batch, -1)
        if tuple(result.shape) != (batch, count):
            raise ValueError(f"mask shape {tuple(result.shape)} does not match {(batch, count)}")
        return result

    @staticmethod
    def _canonical_reliability(value: Tensor, batch: int, count: int, device: torch.device) -> Tensor:
        result = torch.as_tensor(value, device=device, dtype=torch.float32)
        if result.ndim == 1:
            result = result.unsqueeze(0).expand(batch, -1)
        if tuple(result.shape) != (batch, count):
            raise ValueError(f"reliability shape {tuple(result.shape)} does not match {(batch, count)}")
        return result


def score_numpy(query, memory, *, config: PartialSupportConfig | None = None, memory_mask=None):
    """Small CPU helper used by the diagnostics and JSONL data builder."""
    with torch.no_grad():
        evidence = PartialSupportScorer(config)(torch.as_tensor(query), torch.as_tensor(memory), memory_mask=memory_mask)
    return float(evidence.score.detach().cpu()), evidence

