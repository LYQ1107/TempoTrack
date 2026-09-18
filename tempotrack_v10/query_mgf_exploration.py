"""Exploration-only beta-bank calibrators for V12.

The formal V12 model is intentionally kept in ``query_mgf_calibrator.py``.
This module owns the 49-D exploratory schema and cannot be used to produce a
paper-valid/final checkpoint: its beta-bank and adaptive branches are for the
explicit ``TEST_TUNED_EXPLORATION`` study only.
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
    QDIC_MGF_EXPLORATION_BETAS,
    QDIC_MGF_EXPLORATION_FEATURE_NAMES,
    QDIC_MGF_EXPLORATION_RAW_DIM,
    QDIC_RAW_DIM,
    build_qdic_mgf_exploration_candidate_features,
    build_qdic_mgf_exploration_event_features,
)
from .query_distributional_calibrator import GATE_FEATURE_NAMES


EXPLORATION_GATE_INPUT_DIM = len(GATE_FEATURE_NAMES)
EXPLORATION_BETA_COUNT = len(QDIC_MGF_EXPLORATION_BETAS)
EXPLORATION_RESIDUAL_INPUT_DIM = QDIC_MGF_EXPLORATION_RAW_DIM + 2
EXPLORATION_FEATURE_INDEX = {
    name: index for index, name in enumerate(QDIC_MGF_EXPLORATION_FEATURE_NAMES)
}
EXPLORATION_GATE_FEATURE_INDICES = tuple(
    EXPLORATION_FEATURE_INDEX[name] for name in GATE_FEATURE_NAMES
)
EXPLORATION_FAST_BANK_SLICE = slice(QDIC_RAW_DIM, QDIC_RAW_DIM + EXPLORATION_BETA_COUNT)
EXPLORATION_SLOW_BANK_SLICE = slice(QDIC_RAW_DIM + EXPLORATION_BETA_COUNT, QDIC_MGF_EXPLORATION_RAW_DIM)


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return float(np.log(np.expm1(value)))


def _as_features(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] != QDIC_MGF_EXPLORATION_RAW_DIM:
        raise ValueError(
            "QDIC exploration features must end in "
            f"[{QDIC_MGF_EXPLORATION_RAW_DIM}], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("QDIC exploration features must be finite")
    return array


class QueryAdaptiveMGFCalibrator(nn.Module):
    """Adaptive beta-bank scorer for the registered 49-D exploration cache."""

    feature_names = tuple(QDIC_MGF_EXPLORATION_FEATURE_NAMES)
    mgf_betas = tuple(float(value) for value in QDIC_MGF_EXPLORATION_BETAS)

    def __init__(
        self,
        input_dim: int = QDIC_MGF_EXPLORATION_RAW_DIM,
        *,
        mgf_mode: str = "core",
        gate_hidden: tuple[int, int] = (16, 8),
        beta_hidden: int = 16,
        residual_hidden: tuple[int, int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(input_dim) != QDIC_MGF_EXPLORATION_RAW_DIM:
            raise ValueError(
                "exploration adaptive MGF requires "
                f"input_dim={QDIC_MGF_EXPLORATION_RAW_DIM}"
            )
        if mgf_mode not in {"core", "fused"}:
            raise ValueError("mgf_mode must be 'core' or 'fused'")
        if len(gate_hidden) != 2 or len(residual_hidden) != 2:
            raise ValueError("gate_hidden and residual_hidden must contain two widths")
        self.input_dim = QDIC_MGF_EXPLORATION_RAW_DIM
        self.mgf_mode = str(mgf_mode)
        self.register_buffer(
            "feature_mean", torch.zeros(QDIC_MGF_EXPLORATION_RAW_DIM)
        )
        self.register_buffer(
            "feature_scale", torch.ones(QDIC_MGF_EXPLORATION_RAW_DIM)
        )
        self.gate = nn.Sequential(
            nn.Linear(EXPLORATION_GATE_INPUT_DIM, int(gate_hidden[0])),
            nn.ReLU(),
            nn.Linear(int(gate_hidden[0]), int(gate_hidden[1])),
            nn.ReLU(),
            nn.Linear(int(gate_hidden[1]), 1),
            nn.Sigmoid(),
        )
        self.fast_beta_head = nn.Sequential(
            nn.Linear(EXPLORATION_GATE_INPUT_DIM, int(beta_hidden)),
            nn.ReLU(),
            nn.Linear(int(beta_hidden), EXPLORATION_BETA_COUNT),
        )
        self.slow_beta_head = nn.Sequential(
            nn.Linear(EXPLORATION_GATE_INPUT_DIM, int(beta_hidden)),
            nn.ReLU(),
            nn.Linear(int(beta_hidden), EXPLORATION_BETA_COUNT),
        )
        self.residual_calibrator = nn.Sequential(
            nn.Linear(EXPLORATION_RESIDUAL_INPUT_DIM, int(residual_hidden[0])),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(residual_hidden[0]), int(residual_hidden[1])),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(residual_hidden[1]), 1),
        )
        self.structured_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)
        # Start the adaptive bank uniformly so the untrained model is not
        # biased toward any beta in the pre-registered list.
        for head in (self.fast_beta_head, self.slow_beta_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.residual_calibrator[-1].weight)
        nn.init.zeros_(self.residual_calibrator[-1].bias)

    def set_normalization(self, mean: Any, scale: Any) -> None:
        mean_array = _as_features(mean).reshape(-1)
        scale_array = _as_features(scale).reshape(-1)
        if mean_array.shape != (QDIC_MGF_EXPLORATION_RAW_DIM,) or scale_array.shape != (
            QDIC_MGF_EXPLORATION_RAW_DIM,
        ):
            raise ValueError("exploration normalization statistics must be [49]")
        if np.any(scale_array <= 0.0) or not np.isfinite(scale_array).all():
            raise ValueError("exploration feature scale must be finite and positive")
        self.feature_mean.copy_(torch.from_numpy(mean_array).to(self.feature_mean))
        self.feature_scale.copy_(torch.from_numpy(scale_array).to(self.feature_scale))

    def forward(
        self,
        features: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | dict[str, Tensor | float | str]:
        values = torch.as_tensor(
            features, dtype=self.feature_mean.dtype, device=self.feature_mean.device
        )
        if values.ndim < 1 or values.shape[-1] != QDIC_MGF_EXPLORATION_RAW_DIM:
            raise ValueError(
                "exploration features must end in "
                f"[{QDIC_MGF_EXPLORATION_RAW_DIM}], got {tuple(values.shape)}"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("exploration features must be finite")
        leading = values.shape[:-1]
        flat = values.reshape(-1, QDIC_MGF_EXPLORATION_RAW_DIM)
        normalized = (flat - self.feature_mean) / self.feature_scale
        gate_values = flat[:, EXPLORATION_GATE_FEATURE_INDICES]
        alpha = self.gate(gate_values).squeeze(-1)
        fast_weights = torch.softmax(self.fast_beta_head(gate_values), dim=-1)
        slow_weights = torch.softmax(self.slow_beta_head(gate_values), dim=-1)
        fast_bank = flat[:, EXPLORATION_FAST_BANK_SLICE]
        slow_bank = flat[:, EXPLORATION_SLOW_BANK_SLICE]
        fast_lmgf = (fast_weights * fast_bank).sum(dim=-1)
        slow_lmgf = (slow_weights * slow_bank).sum(dim=-1)
        query_fast = flat[:, EXPLORATION_FEATURE_INDEX["query_fast_cosine"]]
        query_slow = flat[:, EXPLORATION_FEATURE_INDEX["query_slow_cosine"]]
        if self.mgf_mode == "core":
            fast_branch = fast_lmgf
            slow_branch = slow_lmgf
        else:
            fast_branch = 0.5 * (query_fast + fast_lmgf)
            slow_branch = 0.5 * (query_slow + slow_lmgf)
        structured_score = alpha * fast_branch + (1.0 - alpha) * slow_branch
        residual_input = torch.cat(
            (normalized, alpha[:, None], structured_score[:, None]), dim=-1
        )
        delta = self.residual_calibrator(residual_input).squeeze(-1)
        logit = F.softplus(self.structured_scale) * structured_score + delta
        if not return_diagnostics:
            return logit.reshape(leading)
        diagnostics: dict[str, Tensor | float | str] = {
            "logit": logit.reshape(leading),
            "alpha": alpha.reshape(leading),
            "structured_score": structured_score.reshape(leading),
            "fast_branch": fast_branch.reshape(leading),
            "slow_branch": slow_branch.reshape(leading),
            "residual": delta.reshape(leading),
            "fast_beta_weights": fast_weights.reshape(*leading, EXPLORATION_BETA_COUNT),
            "slow_beta_weights": slow_weights.reshape(*leading, EXPLORATION_BETA_COUNT),
            "mgf_betas": self.mgf_betas,
            "mgf_mode": self.mgf_mode,
        }
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
                    "exploration candidate tuples must be "
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
        raise TypeError("unsupported exploration candidate payload")

    def build_event_features(self, candidates: Sequence[Any]) -> np.ndarray:
        if not candidates:
            raise ValueError("exploration MGF event requires at least one candidate")
        rows = []
        for candidate in candidates:
            item = self._candidate_mapping(candidate)
            raw = build_qdic_mgf_exploration_candidate_features(
                item["cosine"],
                item["evidence"],
                int(item["gap"]),
                int(item["rank"]),
                query_fast_cosine=item["query_fast_cosine"],
                query_slow_cosine=item["query_slow_cosine"],
                fast_slow_cosine=item["fast_slow_cosine"],
                recent_k=int(item.get("recent_k", 8)),
                top_r=int(item.get("top_r", 3)),
                max_gap=max(int(item.get("max_gap", 360)), 1),
            )
            rows.append({"base_features": raw[:19], "distributional_features": raw[19:]})
        return build_qdic_mgf_exploration_event_features(rows)

    def score_event(
        self,
        candidates: Sequence[Any] | np.ndarray | Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[np.ndarray, Any]:
        if isinstance(candidates, (np.ndarray, torch.Tensor)):
            raw = _as_features(
                candidates.detach().cpu().numpy()
                if isinstance(candidates, Tensor)
                else candidates
            )
            if raw.ndim == 1:
                raw = raw[None, :]
        else:
            raw = self.build_event_features(candidates)
        with torch.inference_mode():
            diagnostics = self.forward(
                torch.from_numpy(raw).to(self.feature_mean.device),
                return_diagnostics=True,
            )
        assert isinstance(diagnostics, dict)
        logits = diagnostics["logit"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)  # type: ignore[union-attr]
        if not np.isfinite(logits).all():
            raise FloatingPointError("exploration MGF logits are non-finite")
        if not return_diagnostics:
            return logits, raw
        details: dict[str, Any] = {}
        for key, value in diagnostics.items():
            if isinstance(value, Tensor):
                details[key] = value.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            else:
                details[key] = value
        return logits, details


__all__ = [
    "EXPLORATION_BETA_COUNT",
    "EXPLORATION_FEATURE_INDEX",
    "EXPLORATION_GATE_FEATURE_INDICES",
    "EXPLORATION_RESIDUAL_INPUT_DIM",
    "QueryAdaptiveMGFCalibrator",
]
