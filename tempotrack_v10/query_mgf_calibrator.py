"""Exact empirical log-MGF structured scorer for the V12 experiment.

This module is deliberately separate from the V11 calibrator.  The V11
33-D model and checkpoint loader remain unchanged; this class owns the frozen
35-D schema and the two pre-registered MGF structured modes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .qdic_features import (
    QDIC_MGF_BETA,
    QDIC_MGF_FEATURE_NAMES,
    QDIC_MGF_RAW_DIM,
    build_qdic_mgf_candidate_features,
    build_qdic_mgf_event_features,
)
from .query_distributional_calibrator import GATE_FEATURE_NAMES


GATE_INPUT_DIM = len(GATE_FEATURE_NAMES)
MGF_FEATURE_INDEX = {name: index for index, name in enumerate(QDIC_MGF_FEATURE_NAMES)}
GATE_FEATURE_INDICES = tuple(MGF_FEATURE_INDEX[name] for name in GATE_FEATURE_NAMES)
RESIDUAL_INPUT_DIM = QDIC_MGF_RAW_DIM + 2


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return float(np.log(np.expm1(value)))


def _as_numpy_features(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] != QDIC_MGF_RAW_DIM:
        raise ValueError(
            f"QDIC MGF features must end in [{QDIC_MGF_RAW_DIM}], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("QDIC MGF features must be finite")
    return array


class QueryMomentGeneratingCalibrator(nn.Module):
    """B0-shaped scorer whose structured evidence is exact empirical log-MGF.

    ``mgf_mode`` is pre-registered before any Val/Test result is observed:

    * ``core``: recent/full log-MGF directly form the two temporal branches;
    * ``fused``: each branch averages the current query prototype cue with
      its corresponding log-MGF evidence.
    """

    feature_names = tuple(QDIC_MGF_FEATURE_NAMES)
    mgf_beta = float(QDIC_MGF_BETA)

    def __init__(
        self,
        input_dim: int = QDIC_MGF_RAW_DIM,
        *,
        mgf_mode: str = "core",
        gate_hidden: tuple[int, int] = (16, 8),
        residual_hidden: tuple[int, int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(input_dim) != QDIC_MGF_RAW_DIM:
            raise ValueError(f"QDIC V12 MGF requires input_dim={QDIC_MGF_RAW_DIM}")
        if mgf_mode not in {"core", "fused"}:
            raise ValueError("mgf_mode must be 'core' or 'fused'")
        if len(gate_hidden) != 2 or len(residual_hidden) != 2:
            raise ValueError("gate_hidden and residual_hidden must contain two widths")
        self.input_dim = QDIC_MGF_RAW_DIM
        self.mgf_mode = str(mgf_mode)
        self.register_buffer("feature_mean", torch.zeros(QDIC_MGF_RAW_DIM))
        self.register_buffer("feature_scale", torch.ones(QDIC_MGF_RAW_DIM))
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
        self.structured_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)
        nn.init.zeros_(self.residual_calibrator[-1].weight)
        nn.init.zeros_(self.residual_calibrator[-1].bias)

    def set_normalization(self, mean: Any, scale: Any) -> None:
        mean_array = _as_numpy_features(mean).reshape(-1)
        scale_array = _as_numpy_features(scale).reshape(-1)
        if mean_array.shape != (QDIC_MGF_RAW_DIM,) or scale_array.shape != (QDIC_MGF_RAW_DIM,):
            raise ValueError("QDIC MGF normalization statistics must be [35]")
        if np.any(scale_array <= 0.0) or not np.isfinite(scale_array).all():
            raise ValueError("QDIC MGF feature scale must be finite and positive")
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
        if values.ndim < 1 or values.shape[-1] != QDIC_MGF_RAW_DIM:
            raise ValueError(
                f"QDIC MGF features must end in [{QDIC_MGF_RAW_DIM}], got {tuple(values.shape)}"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("QDIC MGF features must be finite")
        leading = values.shape[:-1]
        flat = values.reshape(-1, QDIC_MGF_RAW_DIM)
        normalized = (flat - self.feature_mean) / self.feature_scale
        gate_values = flat[:, GATE_FEATURE_INDICES]
        alpha = self.gate(gate_values).squeeze(-1)
        q_fast = flat[:, MGF_FEATURE_INDEX["query_fast_cosine"]]
        q_slow = flat[:, MGF_FEATURE_INDEX["query_slow_cosine"]]
        fast_mean = flat[:, MGF_FEATURE_INDEX["projected_fast_mean"]]
        fast_variance = flat[:, MGF_FEATURE_INDEX["projected_fast_variance"]]
        slow_mean = flat[:, MGF_FEATURE_INDEX["projected_slow_mean"]]
        slow_variance = flat[:, MGF_FEATURE_INDEX["projected_slow_variance"]]
        fast_lmgf = flat[:, MGF_FEATURE_INDEX["projected_fast_log_mgf"]]
        slow_lmgf = flat[:, MGF_FEATURE_INDEX["projected_slow_log_mgf"]]
        if self.mgf_mode == "core":
            fast_branch = fast_lmgf
            slow_branch = slow_lmgf
        else:
            fast_branch = 0.5 * (q_fast + fast_lmgf)
            slow_branch = 0.5 * (q_slow + slow_lmgf)
        structured_score = alpha * fast_branch + (1.0 - alpha) * slow_branch
        residual_input = torch.cat((normalized, alpha[:, None], structured_score[:, None]), dim=-1)
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
            "projected_fast_log_mgf": fast_lmgf.reshape(leading),
            "projected_slow_log_mgf": slow_lmgf.reshape(leading),
            "projected_fast_mean": fast_mean.reshape(leading),
            "projected_fast_variance": fast_variance.reshape(leading),
            "projected_slow_mean": slow_mean.reshape(leading),
            "projected_slow_variance": slow_variance.reshape(leading),
            "mgf_beta": self.mgf_beta,
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
                    "MGF candidate tuples must be "
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
        raise TypeError("unsupported MGF candidate payload")

    def build_event_features(self, candidates: Sequence[Any]) -> np.ndarray:
        if not candidates:
            raise ValueError("QDIC MGF event requires at least one candidate")
        rows = []
        for candidate in candidates:
            item = self._candidate_mapping(candidate)
            raw = build_qdic_mgf_candidate_features(
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
                mgf_beta=float(item.get("mgf_beta", QDIC_MGF_BETA)),
            )
            rows.append(
                {
                    "base_features": raw[:19],
                    "distributional_features": raw[19:],
                }
            )
        return build_qdic_mgf_event_features(rows)

    def score_event(
        self,
        candidates: Sequence[Any] | np.ndarray | Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[np.ndarray, Any]:
        if isinstance(candidates, (np.ndarray, torch.Tensor)):
            raw = _as_numpy_features(
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
            raise FloatingPointError("QDIC MGF logits are non-finite")
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
    "GATE_FEATURE_INDICES",
    "GATE_FEATURE_NAMES",
    "MGF_FEATURE_INDEX",
    "QueryMomentGeneratingCalibrator",
    "RESIDUAL_INPUT_DIM",
]
