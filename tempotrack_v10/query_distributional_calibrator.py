"""Structured Query-Conditioned Distributional Identity Calibration (QDIC).

QDIC-MO keeps the projected Gaussian-Moment/MGF-inspired evidence explicit.
The learned
part only chooses the fast/slow mixture and calibrates the resulting identity
score with a residual network; it is not a replacement black-box reranker.
The projected MO values are evidence features, not a Gaussian likelihood or a
fixed positive identity reward; the residual learns complex competition and
MO interactions left by the structured score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .qdic_features import (
    QDIC_DISTRIBUTIONAL_FEATURE_NAMES,
    QDIC_FEATURE_NAMES,
    QDIC_RAW_DIM,
    build_qdic_candidate_features,
    build_qdic_event_features,
)


GATE_INPUT_DIM = 9
FEATURE_INDEX = {name: index for index, name in enumerate(QDIC_FEATURE_NAMES)}
GATE_FEATURE_NAMES = (
    "query_fast_cosine",
    "query_slow_cosine",
    "fast_slow_cosine",
    "projected_fast_mean",
    "projected_fast_variance",
    "projected_slow_mean",
    "projected_slow_variance",
    "gap_over_max_gap",
    "memory_length_over_64",
)
GATE_FEATURE_INDICES = tuple(FEATURE_INDEX[name] for name in GATE_FEATURE_NAMES)
assert len(GATE_FEATURE_INDICES) == GATE_INPUT_DIM
RESIDUAL_INPUT_DIM = QDIC_RAW_DIM + 2


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return float(np.log(np.expm1(value)))


def _as_numpy_features(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] != QDIC_RAW_DIM:
        raise ValueError(f"QDIC features must end in [{QDIC_RAW_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("QDIC features must be finite")
    return array


class QueryDistributionalCalibrator(nn.Module):
    """Fast/slow gate plus a structured-score residual calibrator.

    The public ``forward`` accepts raw 33-D rows.  ``score_event`` additionally
    accepts candidate payloads and constructs the same 33-D rows used by the
    offline feature builder, which keeps the online boundary small and
    frontend-independent.  Projected MO is evidence, not a Gaussian likelihood
    or fixed positive identity reward; the residual learns complex
    competition/MO interactions.
    """

    feature_names = tuple(QDIC_FEATURE_NAMES)
    distributional_feature_names = tuple(QDIC_DISTRIBUTIONAL_FEATURE_NAMES)

    def __init__(
        self,
        input_dim: int = QDIC_RAW_DIM,
        *,
        gate_hidden: tuple[int, int] = (16, 8),
        residual_hidden: tuple[int, int] = (64, 32),
        dropout: float = 0.1,
        structured_branch_mode: str = "legacy",
    ) -> None:
        super().__init__()
        if int(input_dim) != QDIC_RAW_DIM:
            raise ValueError(f"QDIC V11 requires input_dim={QDIC_RAW_DIM}")
        if len(gate_hidden) != 2 or len(residual_hidden) != 2:
            raise ValueError("gate_hidden and residual_hidden must contain two widths")
        if structured_branch_mode not in {"legacy", "dssl"}:
            raise ValueError("structured_branch_mode must be 'legacy' or 'dssl'")
        self.input_dim = QDIC_RAW_DIM
        self.structured_branch_mode = str(structured_branch_mode)
        self.register_buffer("feature_mean", torch.zeros(QDIC_RAW_DIM))
        self.register_buffer("feature_scale", torch.ones(QDIC_RAW_DIM))
        self.gate = nn.Sequential(
            nn.Linear(GATE_INPUT_DIM, int(gate_hidden[0])),
            nn.ReLU(),
            nn.Linear(int(gate_hidden[0]), int(gate_hidden[1])),
            nn.ReLU(),
            nn.Linear(int(gate_hidden[1]), 1),
            nn.Sigmoid(),
        )
        self.residual_calibrator = nn.Sequential(
            nn.Linear(RESIDUAL_INPUT_DIM, int(residual_hidden[0])),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(residual_hidden[0]), int(residual_hidden[1])),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(residual_hidden[1]), 1),
        )
        # softplus(structured_scale) starts at one, so the initial model is a
        # structured identity scorer with a zero residual, not a random MLP.
        self.structured_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        # A positive, learnable uncertainty penalty keeps projected variance
        # from acting as an unconditional reward.  The initial penalty is 0.1.
        self.variance_penalty_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(0.1), dtype=torch.float32)
        )
        if self.structured_branch_mode == "dssl":
            # DSSL keeps recent and long support distributions independent.
            # Each softmax weighs three observable supports within its own
            # temporal branch; no branch receives the other branch's logits.
            self.recent_support = nn.Linear(GATE_INPUT_DIM, 3)
            self.long_support = nn.Linear(GATE_INPUT_DIM, 3)
            self.recent_variance_penalty_raw = nn.Parameter(
                torch.tensor(_inverse_softplus(0.1), dtype=torch.float32)
            )
            self.long_variance_penalty_raw = nn.Parameter(
                torch.tensor(_inverse_softplus(0.1), dtype=torch.float32)
            )
            nn.init.zeros_(self.recent_support.weight)
            nn.init.zeros_(self.recent_support.bias)
            nn.init.zeros_(self.long_support.weight)
            nn.init.zeros_(self.long_support.bias)
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)
        nn.init.zeros_(self.residual_calibrator[-1].weight)
        nn.init.zeros_(self.residual_calibrator[-1].bias)

    def set_normalization(self, mean: Any, scale: Any) -> None:
        """Install training-video-only normalization statistics."""
        mean_array = _as_numpy_features(mean).reshape(-1)
        scale_array = _as_numpy_features(scale).reshape(-1)
        if mean_array.shape != (QDIC_RAW_DIM,) or scale_array.shape != (QDIC_RAW_DIM,):
            raise ValueError("QDIC normalization statistics must be [33]")
        if np.any(scale_array <= 0.0) or not np.isfinite(scale_array).all():
            raise ValueError("QDIC feature scale must be finite and positive")
        self.feature_mean.copy_(torch.from_numpy(mean_array).to(self.feature_mean))
        self.feature_scale.copy_(torch.from_numpy(scale_array).to(self.feature_scale))

    def forward(self, features: Tensor, *, return_diagnostics: bool = False) -> Tensor | dict[str, Tensor]:
        values = torch.as_tensor(features, dtype=self.feature_mean.dtype, device=self.feature_mean.device)
        if values.ndim < 1 or values.shape[-1] != QDIC_RAW_DIM:
            raise ValueError(f"QDIC features must end in [{QDIC_RAW_DIM}], got {tuple(values.shape)}")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("QDIC features must be finite")
        leading = values.shape[:-1]
        flat = values.reshape(-1, QDIC_RAW_DIM)
        normalized = (flat - self.feature_mean) / self.feature_scale
        gate_values = flat[:, GATE_FEATURE_INDICES]
        alpha = self.gate(gate_values).squeeze(-1)
        q_fast = flat[:, FEATURE_INDEX["query_fast_cosine"]]
        q_slow = flat[:, FEATURE_INDEX["query_slow_cosine"]]
        fast_mean = flat[:, FEATURE_INDEX["projected_fast_mean"]]
        fast_variance = flat[:, FEATURE_INDEX["projected_fast_variance"]]
        slow_mean = flat[:, FEATURE_INDEX["projected_slow_mean"]]
        slow_variance = flat[:, FEATURE_INDEX["projected_slow_variance"]]
        if self.structured_branch_mode == "legacy":
            # Keep this branch byte-for-byte equivalent in its arithmetic to
            # the pre-DSSL V11 implementation.  Existing checkpoints have no
            # DSSL parameters and are loaded into this branch by default.
            variance_penalty = F.softplus(self.variance_penalty_raw)
            fast_branch = 0.5 * (q_fast + fast_mean) - variance_penalty * fast_variance
            slow_branch = 0.5 * (q_slow + slow_mean) - variance_penalty * slow_variance
            structured_score = alpha * fast_branch + (1.0 - alpha) * slow_branch
            recent_support_weights = None
            long_support_weights = None
            recent_variance_penalty = variance_penalty
            long_variance_penalty = variance_penalty
        else:
            fast_supports = torch.stack(
                (q_fast, fast_mean, flat[:, FEATURE_INDEX["projected_fast_mo"]]), dim=-1
            )
            slow_supports = torch.stack(
                (q_slow, slow_mean, flat[:, FEATURE_INDEX["projected_slow_mo"]]), dim=-1
            )
            recent_support_weights = F.softmax(self.recent_support(gate_values), dim=-1)
            long_support_weights = F.softmax(self.long_support(gate_values), dim=-1)
            recent_support = (recent_support_weights * fast_supports).sum(dim=-1)
            long_support = (long_support_weights * slow_supports).sum(dim=-1)
            recent_variance_penalty = F.softplus(self.recent_variance_penalty_raw)
            long_variance_penalty = F.softplus(self.long_variance_penalty_raw)
            fast_branch = recent_support - recent_variance_penalty * fast_variance
            slow_branch = long_support - long_variance_penalty * slow_variance
            structured_score = alpha * fast_branch + (1.0 - alpha) * slow_branch
        residual_input = torch.cat((normalized, alpha[:, None], structured_score[:, None]), dim=-1)
        delta = self.residual_calibrator(residual_input).squeeze(-1)
        logit = F.softplus(self.structured_scale) * structured_score + delta
        if not return_diagnostics:
            return logit.reshape(leading)
        diagnostics = {
            "logit": logit.reshape(leading),
            "alpha": alpha.reshape(leading),
            "structured_score": structured_score.reshape(leading),
            "fast_branch": fast_branch.reshape(leading),
            "slow_branch": slow_branch.reshape(leading),
            "variance_penalty": recent_variance_penalty.expand_as(alpha).reshape(leading),
            "recent_variance_penalty": recent_variance_penalty.expand_as(alpha).reshape(leading),
            "long_variance_penalty": long_variance_penalty.expand_as(alpha).reshape(leading),
            "residual": delta.reshape(leading),
        }
        if recent_support_weights is not None and long_support_weights is not None:
            diagnostics["recent_support_weights"] = recent_support_weights.reshape(*leading, 3)
            diagnostics["long_support_weights"] = long_support_weights.reshape(*leading, 3)
        return diagnostics

    @staticmethod
    def _candidate_mapping(candidate: Any) -> Mapping[str, Any]:
        if isinstance(candidate, Mapping):
            return candidate
        if hasattr(candidate, "__dict__"):
            return vars(candidate)
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            if len(candidate) != 7:
                raise ValueError(
                    "QDIC candidate tuples must be "
                    "(cosine, evidence, gap, rank, q_fast, q_slow, fast_slow)"
                )
            return {
                "cosine": candidate[0],
                "evidence": candidate[1],
                "gap": candidate[2],
                "rank": candidate[3],
                "query_fast_cosine": candidate[4],
                "query_slow_cosine": candidate[5],
                "fast_slow_cosine": candidate[6],
            }
        raise TypeError("unsupported QDIC candidate payload")

    def build_event_features(self, candidates: Sequence[Any]) -> np.ndarray:
        """Convert a context candidate payload to the controlled 33-D rows."""
        if not candidates:
            raise ValueError("QDIC event requires at least one candidate")
        rows = []
        for candidate in candidates:
            item = self._candidate_mapping(candidate)
            direct = {
                key: item[key]
                for key in ("query_fast_cosine", "query_slow_cosine", "fast_slow_cosine")
            }
            raw = build_qdic_candidate_features(
                item["cosine"],
                item["evidence"],
                int(item["gap"]),
                int(item["rank"]),
                **direct,
                recent_k=int(item.get("recent_k", 8)),
                top_r=int(item.get("top_r", 3)),
                max_gap=max(int(item.get("max_gap", 360)), 1),
            )
            rows.append({
                "base_features": raw[:19],
                "distributional_features": raw[19:],
            })
        return build_qdic_event_features(rows)

    def score_event(
        self,
        candidates: Sequence[Any] | np.ndarray | Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[np.ndarray, Any]:
        """Score a QDIC event, returning logits and its constructed features.

        Passing an ``[N,33]`` array is useful for parity tests and training
        tools.  A sequence of mappings/7-tuples is the online representation.
        With ``return_diagnostics=True`` the second item is the bounded model
        diagnostic dictionary instead of the raw feature matrix.
        """
        if isinstance(candidates, (np.ndarray, torch.Tensor)):
            raw = _as_numpy_features(candidates.detach().cpu().numpy() if isinstance(candidates, Tensor) else candidates)
            if raw.ndim == 1:
                raw = raw[None, :]
        else:
            raw = self.build_event_features(candidates)
        with torch.inference_mode():
            diagnostics = self.forward(torch.from_numpy(raw).to(self.feature_mean.device), return_diagnostics=True)
        assert isinstance(diagnostics, dict)
        logits = diagnostics["logit"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        if not np.isfinite(logits).all():
            raise FloatingPointError("QDIC logits are non-finite")
        if return_diagnostics:
            return logits, {
                key: value.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                for key, value in diagnostics.items()
            }
        return logits, raw


__all__ = [
    "FEATURE_INDEX",
    "GATE_FEATURE_NAMES",
    "GATE_FEATURE_INDICES",
    "GATE_INPUT_DIM",
    "RESIDUAL_INPUT_DIM",
    "QueryDistributionalCalibrator",
]
