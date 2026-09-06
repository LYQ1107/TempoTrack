"""S2 successor-state models and the shared state transform.

The main flow predicts a fixed successor state conditioned on the source
history and the real temporal gap.  Candidate appearance is only encoded
after sampling to calculate a support score; it is never part of the flow
condition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _segment_values(value: Any) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    if hasattr(value, "appearance"):
        return value.appearance, value.geometry, value.relative_time, value.valid
    if isinstance(value, Mapping):
        return value["appearance"], value["geometry"], value.get("relative_time", value.get("time_offsets")), value.get("valid")
    raise TypeError("expected SegmentInputs-like value")


@dataclass(frozen=True)
class StateTransformSnapshot:
    schema_version: int
    appearance_mean: list[float]
    appearance_components: list[list[float]]
    appearance_std: list[float]
    state_dim: int
    fit_count: int
    fit_data_hash: str
    mode: str = "pca"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "appearance_mean": self.appearance_mean,
            "appearance_components": self.appearance_components,
            "appearance_std": self.appearance_std,
            "state_dim": self.state_dim,
            "fit_count": self.fit_count,
            "fit_data_hash": self.fit_data_hash,
            "mode": self.mode,
        }

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SuccessorStateTransform:
    """Encode ``[geometry_delta(4), appearance_residual(60)]`` states."""

    schema_version = 1

    def __init__(self, appearance_dim: int, *, appearance_latent_dim: int = 60, mode: str = "pca", snapshot: StateTransformSnapshot | Mapping[str, Any] | None = None):
        if appearance_latent_dim != 60:
            raise ValueError("the S2 v2 state contract reserves 60 appearance dimensions")
        if appearance_dim < 1:
            raise ValueError("appearance_dim must be positive")
        if mode not in {"pca", "random_projection_control"}:
            raise ValueError("mode must be pca or random_projection_control")
        self.appearance_dim = int(appearance_dim)
        self.appearance_latent_dim = int(appearance_latent_dim)
        self.mode = mode
        self.mean = torch.zeros(self.appearance_dim)
        self.components = torch.zeros((self.appearance_latent_dim, self.appearance_dim))
        self.std = torch.ones(self.appearance_latent_dim)
        self._snapshot: StateTransformSnapshot | None = None
        if snapshot is not None:
            self.load_snapshot(snapshot)
        elif mode == "random_projection_control":
            generator = torch.Generator().manual_seed(0)
            matrix = torch.randn((self.appearance_latent_dim, self.appearance_dim), generator=generator)
            self.components = torch.linalg.qr(matrix.t(), mode="reduced").Q.t().contiguous()
            self._snapshot = self._make_snapshot(0, "random_projection_control")

    @property
    def state_dim(self) -> int:
        return 64

    def _make_snapshot(self, count: int, mode: str | None = None, fit_hash: str = "") -> StateTransformSnapshot:
        digest = fit_hash or hashlib.sha256(self.components.detach().cpu().numpy().tobytes()).hexdigest()
        return StateTransformSnapshot(
            self.schema_version,
            self.mean.detach().cpu().tolist(),
            self.components.detach().cpu().tolist(),
            self.std.detach().cpu().tolist(),
            self.state_dim,
            int(count),
            digest,
            mode or self.mode,
        )

    def fit(self, train_samples: Sequence[Any] | Tensor) -> StateTransformSnapshot:
        """Fit PCA/whitening on appearance residuals from train only."""
        if self.mode != "pca":
            raise ValueError("fit is unavailable for random_projection_control")
        if torch.is_tensor(train_samples):
            values = train_samples.detach().float()
        else:
            residuals: list[Tensor] = []
            for sample in train_samples:
                if isinstance(sample, Mapping) and "appearance_residual" in sample:
                    residuals.append(torch.as_tensor(sample["appearance_residual"], dtype=torch.float32).reshape(-1, self.appearance_dim))
                elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    left = torch.as_tensor(sample[0], dtype=torch.float32).reshape(-1, self.appearance_dim)
                    right = torch.as_tensor(sample[1], dtype=torch.float32).reshape(-1, self.appearance_dim)
                    residuals.append(right.mean(0, keepdim=True) - left.mean(0, keepdim=True))
                else:
                    residuals.append(torch.as_tensor(sample, dtype=torch.float32).reshape(-1, self.appearance_dim))
            values = torch.cat(residuals, dim=0) if residuals else torch.empty((0, self.appearance_dim))
        if values.ndim != 2 or values.shape[-1] != self.appearance_dim:
            raise ValueError("S2 fit values must be [N,appearance_dim]")
        if values.shape[0] < 2:
            raise ValueError("S2 PCA requires at least two train residuals")
        values = values.float()
        self.mean = values.mean(0)
        centered = values - self.mean
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        components = vh[: min(self.appearance_latent_dim, vh.shape[0])]
        if components.shape[0] < self.appearance_latent_dim:
            padding = torch.zeros((self.appearance_latent_dim - components.shape[0], self.appearance_dim))
            components = torch.cat((components, padding), dim=0)
        self.components = components.contiguous()
        variance = (singular.square() / max(values.shape[0] - 1, 1))[: self.appearance_latent_dim]
        self.std = variance.sqrt().clamp_min(1e-6)
        fit_hash = hashlib.sha256(values.numpy().tobytes()).hexdigest()
        self._snapshot = self._make_snapshot(len(values), "pca", fit_hash)
        return self._snapshot

    def load_snapshot(self, snapshot: StateTransformSnapshot | Mapping[str, Any]) -> None:
        payload = snapshot.to_dict() if isinstance(snapshot, StateTransformSnapshot) else dict(snapshot)
        if int(payload.get("schema_version", 0)) != self.schema_version:
            raise ValueError("unsupported S2 state transform schema")
        if int(payload.get("state_dim", 64)) != 64:
            raise ValueError("S2 state transform must have state_dim=64")
        mean = torch.as_tensor(payload["appearance_mean"], dtype=torch.float32)
        components = torch.as_tensor(payload["appearance_components"], dtype=torch.float32)
        std = torch.as_tensor(payload["appearance_std"], dtype=torch.float32)
        if mean.shape != (self.appearance_dim,) or components.shape != (60, self.appearance_dim) or std.shape != (60,):
            raise ValueError("S2 transform snapshot dimensions do not match appearance_dim")
        self.mean, self.components, self.std = mean, components, std.clamp_min(1e-6)
        self.mode = str(payload.get("mode", "pca"))
        self._snapshot = StateTransformSnapshot(
            self.schema_version, mean.tolist(), components.tolist(), self.std.tolist(), 64,
            int(payload.get("fit_count", 0)), str(payload.get("fit_data_hash", "")), self.mode,
        )

    def snapshot(self) -> StateTransformSnapshot:
        if self._snapshot is None:
            self._snapshot = self._make_snapshot(0, self.mode)
        return self._snapshot

    def snapshot_hash(self) -> str:
        return self.snapshot().hash()

    @staticmethod
    def _last(values: Tensor, valid: Tensor | None) -> Tensor:
        if values.ndim != 3:
            raise ValueError("segment tensors must be [B,L,D]")
        if valid is None:
            return values[:, -1]
        valid = valid.bool()
        if valid.shape != values.shape[:2] or not bool(valid.any(dim=-1).all()):
            raise ValueError("every S2 segment needs at least one valid token")
        indices = valid.long().sum(-1) - 1
        return values[torch.arange(values.shape[0], device=values.device), indices]

    def _appearance_residual(self, source_app: Tensor, target_app: Tensor) -> Tensor:
        source = source_app.mean(dim=-2) if source_app.ndim == 3 else source_app
        target = target_app.mean(dim=-2) if target_app.ndim == 3 else target_app
        residual = (target - source).float()
        mean = self.mean.to(residual.device)
        components = self.components.to(residual.device)
        std = self.std.to(residual.device)
        return ((residual - mean) @ components.t()) / std

    def encode_target(self, source: Any, target: Any) -> Tensor:
        source_app, source_geo, _, source_valid = _segment_values(source)
        target_app, target_geo, _, target_valid = _segment_values(target)
        source_app = source_app if source_app.ndim == 3 else source_app.unsqueeze(0)
        target_app = target_app if target_app.ndim == 3 else target_app.unsqueeze(0)
        source_geo = source_geo if source_geo.ndim == 3 else source_geo.unsqueeze(0)
        target_geo = target_geo if target_geo.ndim == 3 else target_geo.unsqueeze(0)
        source_box = self._last(source_geo, source_valid)
        target_box = self._last(target_geo, target_valid)
        # Geometry is already normalised by SegmentTensorizer.  The first two
        # entries are centre displacement; the latter two are log size change.
        delta = target_box - source_box
        return torch.cat((delta, self._appearance_residual(source_app, target_app)), dim=-1)

    def encode_candidate(self, source: Any, candidate: Any) -> Tensor:
        return self.encode_target(source, candidate)


class FrozenTrajectoryProjection(nn.Module):
    """Compatibility projection; main S2 uses ``SuccessorStateTransform``."""

    def __init__(self, input_dim: int, latent_dim: int = 64, mean: Tensor | None = None, std: Tensor | None = None, *, mode: str = "random_projection_control"):
        super().__init__()
        if mode not in {"random_projection_control", "pca"}:
            raise ValueError("projection mode must be explicit")
        self.mode = mode
        self.projection = nn.Linear(input_dim, latent_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.register_buffer("mean", torch.zeros(input_dim) if mean is None else mean.detach().clone())
        self.register_buffer("std", torch.ones(input_dim) if std is None else std.detach().clone().clamp_min(1e-6))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def fit_standardization(self, values: Tensor) -> None:
        if self.mode != "pca":
            raise ValueError("random projection control cannot be fit as the main projection")
        if values.ndim != 2 or values.shape[-1] != self.mean.numel() or values.shape[0] < 2:
            raise ValueError("projection fit requires at least two [N,input_dim] train values")
        self.mean.copy_(values.detach().float().mean(dim=0))
        self.std.copy_(values.detach().float().std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, values: Tensor) -> Tensor:
        return self.projection((values - self.mean) / self.std)

    def snapshot_hash(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.projection.weight, self.mean, self.std):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


class _ResidualField(nn.Module):
    def __init__(self, state_dim: int, condition_dim: int, hidden_dim: int = 256, layers: int = 4):
        super().__init__()
        blocks: list[nn.Module] = []
        in_dim = state_dim + condition_dim + 1
        for _ in range(layers):
            blocks.extend((nn.Linear(in_dim, hidden_dim), nn.GELU()))
            in_dim = hidden_dim
        blocks.append(nn.Linear(in_dim, state_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        if time.ndim == state.ndim - 1:
            time = time.unsqueeze(-1)
        while condition.ndim < state.ndim:
            # ``sample_states`` integrates [K,B,D] states against [B,C]
            # conditions, whereas ordinary flow training uses [B,D] or
            # [B,K,D].  Insert the sample axis on the left in the former
            # case and the candidate axis before the feature axis in the
            # latter; an unconditional unsqueeze(-2) swaps K and B.
            if (
                state.ndim == condition.ndim + 1
                and condition.ndim >= 2
                and state.shape[1:-1] == condition.shape[:-1]
                and state.shape[0] != condition.shape[0]
            ):
                condition = condition.unsqueeze(0)
            else:
                condition = condition.unsqueeze(-2)
        condition = condition.expand(*state.shape[:-1], condition.shape[-1])
        time = time.expand(*state.shape[:-1], 1)
        return self.net(torch.cat((state, time, condition), dim=-1))


class ContinuationFlowModel(nn.Module):
    def __init__(self, condition_dim: int | None = None, latent_dim: int = 64, hidden_dim: int = 256, layers: int = 4, *, appearance_dim: int | None = None):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.state_dim = self.latent_dim
        self.hidden_dim = int(hidden_dim)
        if condition_dim is None:
            if appearance_dim is None:
                raise ValueError("ContinuationFlowModel needs condition_dim or appearance_dim")
            condition_dim = hidden_dim + 1
        self.condition_dim = int(condition_dim)
        self.history_encoder: nn.Module | None = None
        if appearance_dim is not None:
            self.history_encoder = nn.Sequential(
                nn.Linear(int(appearance_dim) + 4 + 1, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            )
            self.condition_dim = hidden_dim + 1
        self.field = _ResidualField(self.state_dim, self.condition_dim, hidden_dim, layers)
        self.candidate_matchability_head = nn.Sequential(
            nn.Linear(self.condition_dim, max(16, hidden_dim // 2)), nn.GELU(), nn.Linear(max(16, hidden_dim // 2), 1)
        )

    @property
    def exists_head(self) -> nn.Module:
        return self.candidate_matchability_head

    def encode_condition(self, source: Any, gap: Tensor | None = None) -> Tensor:
        if torch.is_tensor(source):
            condition = source
            if condition.shape[-1] != self.condition_dim:
                raise ValueError("condition tensor does not match model condition_dim")
            return condition
        appearance, geometry, time, valid = _segment_values(source)
        appearance = appearance if appearance.ndim == 3 else appearance.unsqueeze(0)
        geometry = geometry if geometry.ndim == 3 else geometry.unsqueeze(0)
        time = time if time.ndim == 2 else time.unsqueeze(0)
        if self.history_encoder is None:
            raise ValueError("a SegmentInputs condition requires appearance_dim at construction")
        if valid is None:
            valid = torch.ones(time.shape, dtype=torch.bool, device=time.device)
        valid = valid.bool()
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("source condition contains an all-padding segment")
        idx = valid.long().sum(-1) - 1
        batch = torch.arange(appearance.shape[0], device=appearance.device)
        last = torch.cat((appearance[batch, idx], geometry[batch, idx], time[batch, idx].unsqueeze(-1)), dim=-1)
        history = self.history_encoder(last)
        gap_value = torch.zeros((appearance.shape[0], 1), dtype=history.dtype, device=history.device) if gap is None else torch.as_tensor(gap, device=history.device, dtype=history.dtype).reshape(-1, 1)
        if gap_value.shape[0] == 1 and history.shape[0] != 1:
            gap_value = gap_value.expand(history.shape[0], -1)
        return torch.cat((history, gap_value), dim=-1)

    def vector_field(self, state: Tensor, flow_time: Tensor, condition: Tensor) -> Tensor:
        return self.field(state, flow_time, condition)

    def forward(self, state: Tensor, time: Tensor, condition: Tensor) -> dict[str, Tensor]:
        return {"velocity": self.vector_field(state, time, condition), "candidate_matchability_logit": self.candidate_matchability_head(condition).squeeze(-1), "existence_logit": self.candidate_matchability_head(condition).squeeze(-1)}

    def compute_loss(self, *args: Any, generator: torch.Generator | None = None, **kwargs: Any) -> dict[str, Tensor]:
        # New call: compute_loss(inputs, targets).  Keep the old tensor call
        # only as a compatibility adapter; it still executes real FM math.
        if len(args) >= 2 and not torch.is_tensor(args[0]):
            inputs, targets = args[0], args[1]
            source_state = targets["source_state"]
            target_state = targets["target_state"]
            condition = self.encode_condition(inputs["source"], inputs.get("gap"))
            exists = targets.get("matchability")
            exists_known = targets.get("existence_known")
            target_valid = targets.get("target_state_valid")
        else:
            if len(args) < 3:
                raise TypeError("legacy flow loss needs source_state,target_state,condition")
            source_state, target_state, condition = args[:3]
            exists = args[3] if len(args) > 3 else kwargs.get("exists")
            exists_known = kwargs.get("existence_known")
            target_valid = kwargs.get("target_state_valid")
        del source_state  # x0 is Gaussian by the S2 contract
        if target_state.shape[-1] != self.state_dim:
            raise ValueError("target_state dimension does not match state_dim")
        if condition.shape[-1] != self.condition_dim:
            raise ValueError("flow condition dimension mismatch")
        noise = torch.randn(target_state.shape, device=target_state.device, dtype=target_state.dtype, generator=generator)
        t = torch.rand(target_state.shape[:-1], device=target_state.device, dtype=target_state.dtype, generator=generator)
        interpolated = (1.0 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * target_state
        target_velocity = target_state - noise
        predicted = self.vector_field(interpolated, t, condition)
        per_item = (predicted - target_velocity).square().mean(dim=-1)
        if target_valid is not None:
            mask = torch.as_tensor(target_valid, device=per_item.device, dtype=torch.bool)
            if mask.shape != per_item.shape:
                raise ValueError("target_state_valid shape mismatch")
        else:
            mask = torch.ones_like(per_item, dtype=torch.bool)
        if bool(mask.any()):
            flow = per_item[mask].mean()
        else:
            flow = per_item.new_zeros(())
        total = flow
        output: dict[str, Tensor] = {"total": total, "flow": flow.detach(), "valid_state_count": mask.sum().detach()}
        if exists is not None:
            logits = self.candidate_matchability_head(condition).squeeze(-1)
            exists = torch.as_tensor(exists, device=logits.device, dtype=logits.dtype)
            known = torch.ones_like(exists, dtype=torch.bool) if exists_known is None else torch.as_tensor(exists_known, device=logits.device, dtype=torch.bool)
            if known.shape != logits.shape or exists.shape != logits.shape:
                raise ValueError("matchability target shape mismatch")
            if bool(known.any()):
                classification = F.binary_cross_entropy_with_logits(logits[known], exists[known])
                total = total + 0.2 * classification
                output["candidate_matchability"] = classification.detach()
                output["existence"] = classification.detach()
                output["existence_count"] = known.sum().detach()
        output["total"] = total
        return output

    @torch.no_grad()
    def sample_states(self, source: Any, gaps: Tensor, *, num_samples: int = 16, steps: int = 32, generator: torch.Generator | None = None) -> Tensor:
        if num_samples < 1 or steps < 1:
            raise ValueError("S2 requires positive num_samples and steps")
        condition = self.encode_condition(source, gaps).detach()
        base = torch.randn((num_samples, condition.shape[0], self.state_dim), device=condition.device, dtype=condition.dtype, generator=generator)
        outputs: list[Tensor] = []
        for sample in base:
            outputs.append(heun_integrate(lambda value, time: self.vector_field(value, time, condition), sample, steps))
        return torch.stack(outputs, dim=1)  # [B,K,state_dim]

    @torch.no_grad()
    def score_candidates(self, inputs: Mapping[str, Any], transform: SuccessorStateTransform, generator: torch.Generator | None = None, *, num_samples: int = 16, steps: int = 32, bandwidth: float = 0.5) -> Tensor:
        source = inputs["source"]
        candidates = inputs["candidates"]
        gaps = torch.as_tensor(inputs["gaps"], dtype=torch.float32, device=next(self.parameters()).device).reshape(-1)
        samples = self.sample_states(source, gaps, num_samples=num_samples, steps=steps, generator=generator)
        candidate_states = torch.stack([transform.encode_candidate(source, item) for item in candidates], dim=0)
        if candidate_states.ndim == 2:
            candidate_states = candidate_states.unsqueeze(0)
        if samples.shape[0] != candidate_states.shape[0]:
            raise ValueError("candidate and source batch sizes differ")
        return normalized_log_mean_kernel_support(samples, candidate_states, bandwidth)


class GaussianSuccessorControl(nn.Module):
    """Same condition/state contract, deterministic Gaussian baseline."""

    def __init__(self, condition_dim: int, state_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, state_dim * 2))
        self.state_dim = int(state_dim)

    def forward(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self.net(condition).chunk(2, dim=-1)
        return mean, log_std.clamp(-7.0, 5.0)


class MDNSuccessorControl(nn.Module):
    """Finite-mixture state baseline with explicit component axis."""

    def __init__(self, condition_dim: int, state_dim: int = 64, components: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.components, self.state_dim = int(components), int(state_dim)
        self.net = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, components * (2 * state_dim + 1)))

    def forward(self, condition: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        values = self.net(condition).reshape(condition.shape[0], self.components, 2 * self.state_dim + 1)
        return values[..., : self.state_dim], values[..., self.state_dim : 2 * self.state_dim], values[..., -1]


class CVAESuccessorControl(nn.Module):
    """Conditional VAE control kept separate from the FM implementation."""

    def __init__(self, condition_dim: int, state_dim: int = 64, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(condition_dim + state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * latent_dim))
        self.decoder = nn.Sequential(nn.Linear(condition_dim + latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, state_dim))
        self.latent_dim, self.state_dim = int(latent_dim), int(state_dim)


@torch.no_grad()
def heun_integrate(field: Callable[[Tensor, Tensor], Tensor], initial: Tensor, steps: int = 32) -> Tensor:
    if steps < 1:
        raise ValueError("Heun integration requires at least one step")
    value = initial
    dt = 1.0 / steps
    for index in range(steps):
        t = torch.full(value.shape[:-1], index / steps, device=value.device, dtype=value.dtype)
        t_next = torch.full_like(t, (index + 1) / steps)
        first = field(value, t)
        predictor = value + dt * first
        second = field(predictor, t_next)
        value = value + 0.5 * dt * (first + second)
    return value


def normalized_log_mean_kernel_support(samples: Tensor, candidate: Tensor, bandwidth: float = 0.5) -> Tensor:
    if bandwidth <= 0:
        raise ValueError("kernel bandwidth must be positive")
    if samples.ndim < 3 or candidate.ndim < 2 or samples.shape[-1] != candidate.shape[-1]:
        raise ValueError("samples/candidate dimensions do not agree")
    # samples [B,K,D], candidate [B,D] or [B,Kc,D].
    if candidate.ndim == 2:
        candidate = candidate.unsqueeze(-2)
    distances = (samples.unsqueeze(-2) - candidate.unsqueeze(-3)).square().sum(dim=-1)
    dimension = samples.shape[-1]
    sigma = torch.as_tensor(float(bandwidth), device=samples.device, dtype=samples.dtype)
    log_kernel = -distances / (2.0 * sigma.square()) - dimension * torch.log(sigma) - 0.5 * dimension * torch.log(torch.as_tensor(2.0 * torch.pi, device=samples.device, dtype=samples.dtype))
    return torch.logsumexp(log_kernel, dim=-2) - torch.log(torch.as_tensor(samples.shape[-2], device=samples.device, dtype=samples.dtype))
