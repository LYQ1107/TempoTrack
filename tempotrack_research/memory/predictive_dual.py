"""Trainable/deployable predictive dual-speed memory."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .state import MemoryState, initialize_state, safe_normalize
from ..models.memory_controller import MemoryController


@dataclass(frozen=True)
class UtilityExample:
    """Counterfactual labels generated from one identical pre-action state."""

    branch_names: tuple[str, ...]
    utility_target: Tensor
    state_hash: str
    policy_snapshot_hash: str
    training_label_hash: str


class UtilityLabelBuilder:
    """Build the runnable M1 memory-action ablation labels.

    The builder is deliberately outside the deployable memory forward path:
    GT identity labels are used only to score the same future rows under the
    three branches.  Every branch starts from the exact supplied state and
    uses the fixed policy snapshot's rates, so changing that snapshot or the
    training-label set changes the recorded label hashes.
    """

    BRANCHES = ("hold", "recent_only", "recent_and_anchor")

    def __init__(self, memory: "PredictiveDualMemory"):
        self.memory = memory

    @staticmethod
    def _hash(value: Any) -> str:
        if isinstance(value, Tensor):
            value = {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.detach().cpu().contiguous().numpy().tobytes().hex()}
        elif isinstance(value, MemoryState):
            value = {"fast": value.fast, "slow": value.slow, "last_seen": value.last_seen, "birth_time": value.birth_time, "write_count": value.write_count, "last_bbox": value.last_bbox}
        elif isinstance(value, dict):
            value = {str(key): UtilityLabelBuilder._hash(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
        elif isinstance(value, (list, tuple)):
            value = [UtilityLabelBuilder._hash(item) for item in value]
        elif hasattr(value, "item") and callable(value.item):
            value = value.item()
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()

    def build(
        self,
        state_snapshot: MemoryState,
        observation: Tensor,
        future_rows: Tensor,
        fixed_policy_snapshot: Mapping[str, Any],
        training_labels: Mapping[str, Any],
    ) -> UtilityExample:
        if not isinstance(state_snapshot, MemoryState):
            raise TypeError("state_snapshot must be the pre-action MemoryState")
        if observation.ndim != 2 or observation.shape[-1] != self.memory.observation_dim:
            raise ValueError("utility observation must be [B,observation_dim]")
        if future_rows.ndim != 3 or future_rows.shape[0] != observation.shape[0] or future_rows.shape[-1] != self.memory.observation_dim:
            raise ValueError("future_rows must be [B,K,observation_dim]")
        known = torch.as_tensor(training_labels.get("candidate_known"), device=observation.device, dtype=torch.bool)
        positive = torch.as_tensor(training_labels.get("positive_mask"), device=observation.device, dtype=torch.bool)
        if known.shape != positive.shape or known.shape != future_rows.shape[:2]:
            raise ValueError("utility training labels must contain [B,K] known/positive masks")
        valid_rows = known & positive
        if not bool(valid_rows.any(dim=-1).all()):
            raise ValueError("utility labels require at least one known positive future row per batch")
        policy = dict(fixed_policy_snapshot)
        if "alpha_fast" not in policy or "alpha_slow" not in policy:
            raise ValueError("fixed policy snapshot must contain alpha_fast and alpha_slow")
        alpha_fast = torch.as_tensor(policy["alpha_fast"], device=observation.device, dtype=observation.dtype).reshape(-1)
        alpha_slow = torch.as_tensor(policy["alpha_slow"], device=observation.device, dtype=observation.dtype).reshape(-1)
        if alpha_fast.numel() == 1:
            alpha_fast = alpha_fast.expand(observation.shape[0])
        if alpha_slow.numel() == 1:
            alpha_slow = alpha_slow.expand(observation.shape[0])
        if alpha_fast.shape != (observation.shape[0],) or alpha_slow.shape != alpha_fast.shape:
            raise ValueError("fixed policy rates must broadcast to batch size")
        if not bool(((alpha_slow >= 0) & (alpha_slow <= alpha_fast) & (alpha_fast < 1)).all()):
            raise ValueError("fixed policy snapshot violates M1 rate constraints")
        branches = self.memory.counterfactual_branches(state_snapshot, observation, alpha_fast, alpha_slow)
        future = torch.nn.functional.normalize(future_rows, dim=-1)
        scores = []
        for name in self.BRANCHES:
            branch = branches[name]
            fast = torch.nn.functional.normalize(branch.fast, dim=-1)
            slow = torch.nn.functional.normalize(branch.slow, dim=-1)
            value = 0.5 * (torch.einsum("bd,bkd->bk", fast, future) + torch.einsum("bd,bkd->bk", slow, future))
            value = value.masked_fill(~known, 0.0)
            value = (value * valid_rows.to(value.dtype)).sum(-1) / valid_rows.sum(-1).clamp_min(1).to(value.dtype)
            scores.append(value)
        utility_target = torch.stack(scores, dim=-1) - scores[0].unsqueeze(-1)
        return UtilityExample(
            self.BRANCHES,
            utility_target.detach(),
            self._hash(state_snapshot),
            self._hash(policy),
            self._hash(training_labels),
        )


def _vector(value: Tensor | float, *, device: torch.device, dtype: torch.dtype, batch: int | None = None) -> Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if batch is not None and result.ndim == 0:
        result = result.expand(batch)
    return result


def build_causal_evidence(
    state: MemoryState,
    observation: Tensor,
    time_since_last_seen: Tensor | float,
    accepted_match_margin: Tensor | float | None,
    geometry_delta: Tensor,
    time_since_birth: Tensor | float,
    *,
    time_scale: float = 1.0,
    missing_margin: bool = False,
) -> Tensor:
    """Build the exact causal eight-dimensional evidence vector."""

    del observation  # the current appearance enters controller separately
    device, dtype = state.fast.device, state.fast.dtype
    consistency = (state.fast * state.slow).sum(dim=-1)
    batch_shape = consistency.shape
    def scalar(value: Tensor | float, default: float = 0.0) -> Tensor:
        if value is None:
            value = default
        result = torch.as_tensor(value, device=device, dtype=dtype)
        if result.ndim == 0:
            result = result.expand(batch_shape)
        return result.reshape(batch_shape)
    geo = torch.as_tensor(geometry_delta, device=device, dtype=dtype)
    if geo.shape[-1] != 4:
        raise ValueError("geometry_delta must have four normalized fields")
    geo = geo.expand(*batch_shape, 4) if geo.ndim == 1 and len(batch_shape) else geo
    margin = torch.zeros_like(consistency) if missing_margin else scalar(accepted_match_margin)
    gap = torch.log1p(scalar(time_since_last_seen).clamp_min(0.0) / max(float(time_scale), 1e-6))
    age = torch.log1p(scalar(time_since_birth).clamp_min(0.0) / max(float(time_scale), 1e-6))
    return torch.cat((consistency.unsqueeze(-1), gap.unsqueeze(-1), margin.unsqueeze(-1), geo.reshape(*batch_shape, 4), age.unsqueeze(-1)), dim=-1)


class PredictiveDualMemory(nn.Module):
    def __init__(self, history_dim: int, observation_dim: int, evidence_dim: int = 8, hidden_dim: int = 128, alpha_min: float = 1e-4, alpha_max: float = 0.99, controller: nn.Module | None = None, *, initial_alpha_fast: float = 0.7, initial_alpha_slow: float = 0.02):
        super().__init__()
        if evidence_dim != 8:
            raise ValueError("M1 v2 evidence schema requires evidence_dim=8")
        self.history_dim = int(history_dim)
        self.observation_dim = int(observation_dim)
        self.evidence_dim = int(evidence_dim)
        self.hidden_dim = int(hidden_dim)
        self.controller = controller or MemoryController(history_dim, observation_dim, evidence_dim, hidden_dim, alpha_min, alpha_max, initial_alpha_fast=initial_alpha_fast, initial_alpha_slow=initial_alpha_slow)
        self.utility_head = nn.Sequential(nn.Linear(history_dim + observation_dim + evidence_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def initialize(self, prototype: Tensor, frame: Tensor | int = 0) -> MemoryState:
        return initialize_state(prototype, frame)

    def update(self, state: MemoryState, observation: Tensor, history_state: Tensor, causal_evidence: Tensor, frame: Tensor | int | None = None, bbox: Tensor | None = None) -> tuple[MemoryState, dict[str, Tensor]]:
        if history_state.shape[-1] != self.history_dim or causal_evidence.shape[-1] != self.evidence_dim:
            raise ValueError("M1 update received an incompatible history/evidence tensor")
        z = safe_normalize(observation, state.fast)
        rates = self.controller(history_state, z, causal_evidence)
        fast = safe_normalize((1 - rates["alpha_fast"].unsqueeze(-1)) * state.fast + rates["alpha_fast"].unsqueeze(-1) * z, state.fast)
        slow = safe_normalize((1 - rates["alpha_slow"].unsqueeze(-1)) * state.slow + rates["alpha_slow"].unsqueeze(-1) * z, state.slow)
        last_seen = state.last_seen if frame is None else torch.as_tensor(frame, device=z.device)
        birth = state.birth_time if state.birth_time is not None else state.last_seen
        old_count = state.write_count if state.write_count is not None else torch.zeros_like(rates["q"])
        new_state = MemoryState(
            fast=fast,
            slow=slow,
            last_seen=last_seen,
            write_count=old_count + rates["q"],
            birth_time=birth,
            last_bbox=bbox if bbox is not None else state.last_bbox,
            diagnostics=dict(state.diagnostics),
        )
        diagnostics = dict(rates)
        diagnostics.update({"prototype_norm_fast": fast.norm(dim=-1), "prototype_norm_slow": slow.norm(dim=-1), "write_rate": (rates["q"] > 0.5).to(z.dtype)})
        return new_state, diagnostics

    def update_from_match(self, state: MemoryState, observation: Tensor, history_state: Tensor, gap: Tensor, accepted_match_margin: Tensor, geometry_delta: Tensor, age: Tensor, frame: Tensor | int | None = None, *, bbox: Tensor | None = None) -> tuple[MemoryState, dict[str, Tensor]]:
        evidence = build_causal_evidence(state, observation, gap, accepted_match_margin, geometry_delta, age)
        return self.update(state, observation, history_state, evidence, frame, bbox=bbox)

    def predict_utility(self, history_state: Tensor, accepted_observation: Tensor, causal_evidence: Tensor) -> Tensor:
        return self.utility_head(torch.cat((history_state, accepted_observation, causal_evidence), dim=-1))

    def counterfactual_branches(self, state: MemoryState, observation: Tensor, alpha_fast: Tensor, alpha_slow: Tensor) -> dict[str, MemoryState]:
        z = safe_normalize(observation, state.fast)
        fast = safe_normalize((1 - alpha_fast.unsqueeze(-1)) * state.fast + alpha_fast.unsqueeze(-1) * z, state.fast)
        slow = safe_normalize((1 - alpha_slow.unsqueeze(-1)) * state.slow + alpha_slow.unsqueeze(-1) * z, state.slow)
        return {
            "hold": state,
            "recent_only": MemoryState(
                fast=fast,
                slow=state.slow,
                last_seen=state.last_seen,
                write_count=state.write_count,
                birth_time=state.birth_time,
                last_bbox=state.last_bbox,
                diagnostics=dict(state.diagnostics),
            ),
            "recent_and_anchor": MemoryState(
                fast=fast,
                slow=slow,
                last_seen=state.last_seen,
                write_count=state.write_count,
                birth_time=state.birth_time,
                last_bbox=state.last_bbox,
                diagnostics=dict(state.diagnostics),
            ),
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path, expected_schema: Mapping[str, Any] | None = None, device: str | torch.device = "cpu") -> "PredictiveDualMemory":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        config = dict(metadata.get("model_config", {}))
        state_dict = payload.get("model") or payload.get("state_dict")
        if state_dict is None:
            state_dict = payload.get("components", {}).get("memory")
        if state_dict is None:
            raise ValueError(f"checkpoint has no predictive memory state: {path}")
        inferred = {"history_dim": config.get("history_dim"), "observation_dim": config.get("observation_dim"), "evidence_dim": config.get("evidence_dim", 8), "hidden_dim": config.get("memory_hidden_dim", config.get("hidden_dim", 128))}
        if inferred["history_dim"] is None or inferred["observation_dim"] is None:
            key = next((key for key in state_dict if key.endswith("controller.net.0.weight") or key.endswith("net.0.weight")), None)
            if key is None:
                raise ValueError("checkpoint lacks model_config for predictive memory")
            input_dim = int(state_dict[key].shape[1])
            inferred["observation_dim"] = (input_dim - int(inferred["evidence_dim"])) // 3
            inferred["history_dim"] = 2 * inferred["observation_dim"]
        memory = cls(**{key: int(value) for key, value in inferred.items()})
        expected_schema = dict(expected_schema or {})
        for key, value in expected_schema.items():
            if key in inferred and int(value) != int(inferred[key]):
                raise ValueError(f"M1 checkpoint schema mismatch for {key}: {inferred[key]} != {value}")
        # Standalone state keys may be prefixed by ``module``/``memory``;
        # only an exact, explicitly declared component prefix is accepted.
        if all(key.startswith("module.") for key in state_dict):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        if any(key.startswith("memory.") for key in state_dict):
            state_dict = {key[len("memory."):]: value for key, value in state_dict.items()}
        memory.load_state_dict(state_dict, strict=True)
        memory.to(device).eval()
        for parameter in memory.parameters():
            parameter.requires_grad_(False)
        return memory


def constrained_rate_penalty(rates: dict[str, Tensor]) -> Tensor:
    """Constraints are structural; this function is a diagnostic assertion."""

    fast, slow = rates["alpha_fast"], rates["alpha_slow"]
    if not torch.isfinite(fast).all() or not torch.isfinite(slow).all() or not bool((slow <= fast + 1e-7).all()) or not bool(((fast >= 0) & (fast < 1)).all()):
        raise ValueError("M1 rate parameterization violated 0 <= slow <= fast < 1")
    return fast.new_zeros(())
