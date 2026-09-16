"""Shared implementation for the V11 candidate-aware QDIC experiments.

The parent QDIC is kept as a real submodule.  The new models only add an
auxiliary distribution-only branch and a zero-initialized residual interaction
head, so loading a freshly initialized A1/A2 checkpoint is exactly the same
as loading the parent V11 scorer.  Runtime feature construction is delegated
to the parent model, which preserves the existing causal 33-D feature
contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .distributional_identity import DistributionEvidenceEncoder, DistributionIdentityHead
from .qdic_features import (
    FEATURE_NAMES,
    QDIC_DISTRIBUTIONAL_FEATURE_NAMES,
    QDIC_FEATURE_NAMES,
    QDIC_RAW_DIM,
)


LEGACY_FEATURE_INDICES = tuple(QDIC_FEATURE_NAMES.index(name) for name in FEATURE_NAMES)
DISTRIBUTIONAL_FEATURE_INDICES = tuple(
    QDIC_FEATURE_NAMES.index(name) for name in QDIC_DISTRIBUTIONAL_FEATURE_NAMES
)
if len(LEGACY_FEATURE_INDICES) != 24 or len(DISTRIBUTIONAL_FEATURE_INDICES) != 9:
    raise RuntimeError("the V11 QDIC feature split must remain 24-D + 9-D")


def _parent_dtype_device(parent: nn.Module) -> tuple[torch.dtype, torch.device]:
    for value in parent.parameters():
        return value.dtype, value.device
    for name in ("feature_mean", "feature_scale"):
        value = getattr(parent, name, None)
        if isinstance(value, Tensor):
            return value.dtype, value.device
    return torch.float32, torch.device("cpu")


def _as_batch(
    features: Tensor | np.ndarray,
    parent: nn.Module,
    mask: Tensor | np.ndarray | None,
) -> tuple[Tensor, Tensor, bool]:
    dtype, device = _parent_dtype_device(parent)
    values = torch.as_tensor(features, dtype=dtype, device=device)
    squeezed = values.ndim == 2
    if squeezed:
        values = values.unsqueeze(0)
    if values.ndim != 3 or values.shape[-1] != QDIC_RAW_DIM:
        raise ValueError(
            f"candidate-aware QDIC features must be [B,K,{QDIC_RAW_DIM}] or "
            f"[K,{QDIC_RAW_DIM}], got {tuple(values.shape)}"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("candidate-aware QDIC features must be finite")
    if mask is None:
        valid = torch.ones(values.shape[:2], dtype=torch.bool, device=device)
    else:
        valid = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != values.shape[:2]:
            raise ValueError(
                f"candidate mask must be {tuple(values.shape[:2])}, got {tuple(valid.shape)}"
            )
    return values, valid, squeezed


def _parent_diagnostics(parent: nn.Module, values: Tensor) -> dict[str, Tensor]:
    output = parent(values, return_diagnostics=True)
    if not isinstance(output, Mapping):
        raise TypeError("the parent V11 QDIC must return diagnostics as a mapping")
    required = ("logit", "alpha", "structured_score")
    missing = [name for name in required if name not in output]
    if missing:
        raise ValueError("parent QDIC diagnostics missing: " + ",".join(missing))
    result = {str(key): torch.as_tensor(value) for key, value in output.items()}
    for name in required:
        if result[name].shape != values.shape[:2]:
            raise ValueError(
                f"parent diagnostic {name} must be {tuple(values.shape[:2])}, "
                f"got {tuple(result[name].shape)}"
            )
        if not bool(torch.isfinite(result[name]).all()):
            raise ValueError(f"parent diagnostic {name} must be finite")
    return result


def _normalized_features(parent: nn.Module, values: Tensor) -> Tensor:
    mean = getattr(parent, "feature_mean", None)
    scale = getattr(parent, "feature_scale", None)
    if not isinstance(mean, Tensor) or not isinstance(scale, Tensor):
        raise ValueError("parent V11 QDIC must expose feature normalization buffers")
    if mean.shape != (QDIC_RAW_DIM,) or scale.shape != (QDIC_RAW_DIM,):
        raise ValueError("parent V11 QDIC normalization must be [33]")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
        raise ValueError("parent V11 QDIC normalization must be finite")
    if bool((scale <= 0).any()):
        raise ValueError("parent V11 QDIC feature scale must be positive")
    return (values - mean) / scale


def _distribution_parts(
    parent: nn.Module,
    values: Tensor,
    parent_details: Mapping[str, Tensor],
    dist_encoder: DistributionEvidenceEncoder,
    dist_head: DistributionIdentityHead,
) -> tuple[Tensor, Tensor]:
    normalized = _normalized_features(parent, values)
    distribution_token = dist_encoder(normalized[..., DISTRIBUTIONAL_FEATURE_INDICES])
    distribution_logit = dist_head(
        distribution_token,
        parent_details["alpha"],
        parent_details["structured_score"],
    )
    return distribution_token, distribution_logit


def _restore_leading(value: Tensor, squeezed: bool) -> Tensor:
    return value[0] if squeezed else value


def _mask_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _mask_max(values: Tensor, mask: Tensor) -> Tensor:
    any_valid = mask.any(dim=1)
    masked = values.masked_fill(~mask.unsqueeze(-1), -torch.inf)
    result = masked.max(dim=1).values
    return torch.where(any_valid.unsqueeze(-1), result, torch.zeros_like(result))


def _parent_model(parent: Any) -> nn.Module:
    model = getattr(parent, "model", parent)
    if not isinstance(model, nn.Module):
        raise TypeError("parent QDIC must be a torch module or a loaded QDIC artifact")
    return model


class _ScoreEventMixin:
    """Expose the same online numerical boundary as the V11 parent scorer."""

    parent_qdic: nn.Module

    def build_event_features(self, candidates: Sequence[Any]) -> np.ndarray:
        parent = self.parent_qdic
        builder = getattr(parent, "build_event_features", None)
        if callable(builder):
            raw = builder(candidates)
        else:
            artifact_builder = getattr(getattr(parent, "model", None), "build_event_features", None)
            if not callable(artifact_builder):
                raise TypeError("parent QDIC does not expose build_event_features")
            raw = artifact_builder(candidates)
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != QDIC_RAW_DIM or not np.isfinite(raw).all():
            raise ValueError("parent event feature builder returned invalid V11 features")
        return raw

    def score_event(
        self,
        candidates: Sequence[Any] | np.ndarray | Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[np.ndarray, Any]:
        if isinstance(candidates, (np.ndarray, Tensor)):
            raw = np.asarray(
                candidates.detach().cpu().numpy() if isinstance(candidates, Tensor) else candidates,
                dtype=np.float32,
            )
            if raw.ndim == 1:
                raw = raw[None, :]
        else:
            raw = self.build_event_features(candidates)
        if raw.ndim != 2 or raw.shape[1] != QDIC_RAW_DIM or not np.isfinite(raw).all():
            raise ValueError("candidate-aware score_event expects [N,33] finite features")
        with torch.inference_mode():
            details = self.forward(
                torch.from_numpy(raw).unsqueeze(0),
                mask=torch.ones((1, len(raw)), dtype=torch.bool),
                return_diagnostics=True,
            )
        if not isinstance(details, Mapping):
            raise TypeError("candidate-aware forward diagnostics must be a mapping")
        logits = details["logit"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(logits).all():
            raise FloatingPointError("candidate-aware QDIC logits are non-finite")
        output_details: dict[str, np.ndarray] = {}
        for key, value in details.items():
            if not isinstance(value, Tensor):
                continue
            array = value[0].detach().cpu().numpy().astype(np.float32, copy=False)
            output_details[str(key)] = array
        return logits, output_details if return_diagnostics else raw


class DistributionalAuxiliaryQDIC(_ScoreEventMixin, nn.Module):
    """A0-D: unchanged per-candidate V11 scorer plus DIRL supervision."""

    architecture_name = "A0-D"

    def __init__(self, parent_qdic: nn.Module) -> None:
        super().__init__()
        self.parent_qdic = _parent_model(parent_qdic)
        self.dist_encoder = DistributionEvidenceEncoder()
        self.dist_head = DistributionIdentityHead()

    def forward(
        self,
        features: Tensor | np.ndarray,
        mask: Tensor | np.ndarray | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        values, valid, squeezed = _as_batch(features, self.parent_qdic, mask)
        parent = _parent_diagnostics(self.parent_qdic, values)
        token, dist_logit = _distribution_parts(
            self.parent_qdic, values, parent, self.dist_encoder, self.dist_head
        )
        if not return_diagnostics:
            return _restore_leading(parent["logit"], squeezed)
        result = dict(parent)
        result.update(
            {
                "logit": _restore_leading(parent["logit"], squeezed),
                "distribution_logit": _restore_leading(dist_logit, squeezed),
                "distribution_token": _restore_leading(token, squeezed),
                "valid_mask": _restore_leading(valid, squeezed),
            }
        )
        return result


class CandidateAwareQDICBase(_ScoreEventMixin, nn.Module):
    """Common parent, distribution path, legacy path, and candidate token."""

    architecture_name = "candidate-aware-base"

    def __init__(self, parent_qdic: nn.Module) -> None:
        super().__init__()
        self.parent_qdic = _parent_model(parent_qdic)
        self.dist_encoder = DistributionEvidenceEncoder()
        self.dist_head = DistributionIdentityHead()
        self.legacy_evidence_encoder = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.candidate_fusion = nn.Sequential(
            nn.Linear(67, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def _common(
        self,
        values: Tensor,
        valid: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
        parent = _parent_diagnostics(self.parent_qdic, values)
        token, distribution_logit = _distribution_parts(
            self.parent_qdic, values, parent, self.dist_encoder, self.dist_head
        )
        normalized = _normalized_features(self.parent_qdic, values)
        legacy = self.legacy_evidence_encoder(normalized[..., LEGACY_FEATURE_INDICES])
        fusion_input = torch.cat(
            (
                token,
                legacy,
                parent["alpha"].unsqueeze(-1),
                parent["structured_score"].unsqueeze(-1),
                parent["logit"].unsqueeze(-1),
            ),
            dim=-1,
        )
        candidate = self.candidate_fusion(fusion_input)
        return parent, token, distribution_logit, candidate, legacy

    def _interaction(
        self,
        candidate: Tensor,
        distribution_token: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(
        self,
        features: Tensor | np.ndarray,
        mask: Tensor | np.ndarray | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        values, valid, squeezed = _as_batch(features, self.parent_qdic, mask)
        parent, token, distribution_logit, candidate, legacy = self._common(values, valid)
        delta, interaction = self._interaction(candidate, token, valid)
        final = parent["logit"] + delta
        if not return_diagnostics:
            return _restore_leading(final, squeezed)
        result = dict(parent)
        result.update(
            {
                "logit": _restore_leading(final, squeezed),
                "parent_logit": _restore_leading(parent["logit"], squeezed),
                "distribution_logit": _restore_leading(distribution_logit, squeezed),
                "distribution_token": _restore_leading(token, squeezed),
                "legacy_evidence_token": _restore_leading(legacy, squeezed),
                "candidate_representation": _restore_leading(candidate, squeezed),
                "interaction_representation": _restore_leading(interaction, squeezed),
                "delta_set": _restore_leading(delta, squeezed),
                "valid_mask": _restore_leading(valid, squeezed),
            }
        )
        return result


__all__ = [
    "CandidateAwareQDICBase",
    "DISTRIBUTIONAL_FEATURE_INDICES",
    "DistributionalAuxiliaryQDIC",
    "LEGACY_FEATURE_INDICES",
]
