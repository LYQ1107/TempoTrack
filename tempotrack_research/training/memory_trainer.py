"""True multi-step M1 training task and loss wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..losses.predictive import counterfactual_utility_loss, predictive_memory_loss
from ..memory.predictive_dual import PredictiveDualMemory, UtilityExample, UtilityLabelBuilder, build_causal_evidence


@dataclass
class MemoryInputs:
    initial_feature: Tensor
    initial_time: Tensor
    initial_geometry: Tensor
    observations: Tensor
    times: Tensor
    geometry: Tensor
    competition_margin: Tensor
    margin_known: Tensor
    observation_scores: Tensor
    valid: Tensor


@dataclass
class MemoryTargets:
    future_embedding: Tensor
    positive_mask: Tensor
    candidate_known: Tensor
    candidate_valid: Tensor | None = None
    reliability: Tensor | None = None
    reliability_known: Tensor | None = None
    valid: Tensor | None = None


def normalize_memory_targets(
    targets: MemoryTargets,
    *,
    batch_size: int,
    event_count: int,
    candidate_count: int,
) -> MemoryTargets:
    """Normalize the shared-candidate M1 contract before any broadcasting.

    Candidate labels commonly arrive as ``[B,K]`` while the controller emits
    ``T`` event states.  Expansion is allowed only from axis length one; a
    shorter non-singleton target is a data-contract error, never silently
    padded with a negative label.
    """
    if event_count < 1 or candidate_count < 1:
        raise ValueError("M1 normalization requires positive event/candidate axes")

    future = targets.future_embedding
    if future.ndim == 3:
        future = future.unsqueeze(1)
    if future.ndim != 4 or future.shape[0] != batch_size or future.shape[2] != candidate_count:
        raise ValueError("future_embedding must be [B,T,K,D] after candidate construction")
    if future.shape[1] == 1 and event_count > 1:
        future = future.expand(-1, event_count, -1, -1)
    elif future.shape[1] >= event_count:
        future = future[:, :event_count]
    else:
        raise ValueError("future_embedding event axis is shorter than the real unroll")

    def candidate_axis(value: Tensor, name: str) -> Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(1)
        if value.ndim != 3 or value.shape[0] != batch_size or value.shape[2] != candidate_count:
            raise ValueError(f"{name} must be [B,T,K] or [B,K]")
        if value.shape[1] == 1 and event_count > 1:
            return value.expand(-1, event_count, -1)
        if value.shape[1] >= event_count:
            return value[:, :event_count]
        raise ValueError(f"{name} event axis is shorter than the real unroll")

    positive = candidate_axis(targets.positive_mask.bool(), "positive_mask")
    known = candidate_axis(targets.candidate_known.bool(), "candidate_known")
    candidate_valid = targets.candidate_valid
    if candidate_valid is None:
        candidate_valid = torch.ones_like(known, dtype=torch.bool)
    else:
        candidate_valid = candidate_axis(candidate_valid.bool(), "candidate_valid")
    known = known & candidate_valid
    if bool((positive & ~known).any()):
        raise ValueError("M1 positive labels cannot be unknown")
    reliability = targets.reliability
    reliability_known = targets.reliability_known
    if reliability is not None:
        if reliability.ndim == 1:
            reliability = reliability.unsqueeze(1)
        if reliability.shape[0] != batch_size or reliability.shape[1] < event_count:
            raise ValueError("reliability must cover the real event axis")
        reliability = reliability[:, :event_count]
    if reliability_known is not None:
        if reliability_known.ndim == 1:
            reliability_known = reliability_known.unsqueeze(1)
        if reliability_known.shape[0] != batch_size or reliability_known.shape[1] < event_count:
            raise ValueError("reliability_known must cover the real event axis")
        reliability_known = reliability_known[:, :event_count].bool()
    valid = targets.valid
    if valid is not None:
        if valid.ndim == 1:
            valid = valid.unsqueeze(1)
        if valid.shape[0] != batch_size or valid.shape[1] < event_count:
            raise ValueError("M1 valid mask must cover the real event axis")
        valid = valid[:, :event_count].bool()
    return MemoryTargets(future, positive, known, candidate_valid, reliability, reliability_known, valid)


class MemoryTrainingTask(nn.Module):
    """Unroll a controller over an event chunk without per-event detach."""

    def __init__(self, memory: PredictiveDualMemory, *, unroll: int = 16, loss_weights: Mapping[str, float] | None = None):
        super().__init__()
        self.memory = memory
        self.unroll = int(unroll)
        if self.unroll < 1:
            raise ValueError("M1 unroll must be positive")
        self.loss_weights = dict(loss_weights or {})

    def forward(self, inputs: MemoryInputs | Mapping[str, Tensor], targets: MemoryTargets | Mapping[str, Tensor]) -> dict[str, Tensor]:
        if isinstance(inputs, Mapping):
            inputs = MemoryInputs(
                initial_feature=inputs["initial_feature"], initial_time=inputs["initial_time"], initial_geometry=inputs["initial_geometry"],
                observations=inputs["observations"], times=inputs["times"], geometry=inputs["geometry"],
                competition_margin=inputs["competition_margin"], margin_known=inputs["margin_known"],
                observation_scores=inputs["observation_scores"], valid=inputs["valid"],
            )
        if isinstance(targets, Mapping):
            targets = MemoryTargets(
                future_embedding=targets["future_embedding"], positive_mask=targets["positive_mask"],
                candidate_known=targets["candidate_known"], candidate_valid=targets.get("candidate_valid"), reliability=targets.get("reliability"),
                reliability_known=targets.get("reliability_known"), valid=targets.get("valid"),
            )
        if inputs.observations.ndim != 3 or inputs.times.shape != inputs.observations.shape[:2] or inputs.geometry.shape[:2] != inputs.observations.shape[:2]:
            raise ValueError("M1 inputs must contain observations [B,T,D], times [B,T], geometry [B,T,4]")
        if inputs.geometry.shape[-1] != 4 or inputs.competition_margin.shape != inputs.times.shape or inputs.margin_known.shape != inputs.times.shape or inputs.observation_scores.shape != inputs.times.shape or inputs.valid.shape != inputs.times.shape:
            raise ValueError("M1 event fields must all have shape [B,T]")
        if inputs.initial_feature.ndim != 2 or inputs.initial_geometry.shape != (inputs.initial_feature.shape[0], 4) or inputs.initial_time.shape != (inputs.initial_feature.shape[0],):
            raise ValueError("M1 initial state fields have incompatible shapes")
        batch, steps, _ = inputs.observations.shape
        state = self.memory.initialize(inputs.initial_feature, inputs.initial_time)
        previous_geometry = inputs.initial_geometry
        previous_time = inputs.initial_time
        fast_values: list[Tensor] = []
        slow_values: list[Tensor] = []
        rate_values: dict[str, list[Tensor]] = {"q": [], "alpha_fast": [], "alpha_slow": [], "reliability_logit": []}
        limit = min(steps, self.unroll)
        for index in range(limit):
            valid_step = inputs.valid[:, index].bool()
            frame = inputs.times[:, index]
            bbox = inputs.geometry[:, index]
            # History/evidence are derived from the state produced by the
            # preceding event.  No prefix or cached history tensor is read.
            history = torch.cat((state.fast, state.slow), dim=-1)
            gap = (inputs.times[:, index] - previous_time).clamp_min(0)
            geometry_delta = inputs.geometry[:, index] - previous_geometry
            age = (inputs.times[:, index] - inputs.initial_time).clamp_min(0)
            evidence = build_causal_evidence(state, inputs.observations[:, index], gap, inputs.competition_margin[:, index], geometry_delta, age, missing_margin=~inputs.margin_known[:, index].bool())
            observation = torch.where(valid_step.unsqueeze(-1), inputs.observations[:, index], state.fast)
            old_state = state
            updated_state, rates = self.memory.update(
                old_state,
                observation,
                history,
                evidence,
                frame,
                bbox=bbox,
            )
            # Invalid right-padding does not advance time, bbox or write
            # count.  The computation above remains finite but its state is
            # explicitly discarded.
            state = type(updated_state)(
                fast=torch.where(valid_step.unsqueeze(-1), updated_state.fast, old_state.fast),
                slow=torch.where(valid_step.unsqueeze(-1), updated_state.slow, old_state.slow),
                last_seen=torch.where(valid_step, updated_state.last_seen, old_state.last_seen),
                write_count=torch.where(valid_step, updated_state.write_count, old_state.write_count),
                birth_time=updated_state.birth_time,
                last_bbox=torch.where(valid_step.unsqueeze(-1), updated_state.last_bbox if updated_state.last_bbox is not None else bbox, old_state.last_bbox if old_state.last_bbox is not None else previous_geometry),
                diagnostics=dict(updated_state.diagnostics),
            )
            previous_geometry = torch.where(valid_step.unsqueeze(-1), inputs.geometry[:, index], previous_geometry)
            previous_time = torch.where(valid_step, inputs.times[:, index], previous_time)
            fast_values.append(state.fast)
            slow_values.append(state.slow)
            for name in rate_values:
                rate_values[name].append(rates[name])
        if not fast_values:
            raise ValueError("M1 received an empty unroll")
        fast = torch.stack(fast_values, dim=1)
        slow = torch.stack(slow_values, dim=1)
        rates = {name: torch.stack(values, dim=1) for name, values in rate_values.items()}
        target_steps = int(fast.shape[1])
        candidate_count = int(targets.future_embedding.shape[-2]) if targets.future_embedding.ndim >= 3 else 1
        targets = normalize_memory_targets(targets, batch_size=batch, event_count=target_steps, candidate_count=candidate_count)
        future = targets.future_embedding
        positive = targets.positive_mask
        known = targets.candidate_known
        valid_steps = torch.ones((batch, target_steps), dtype=torch.bool, device=fast.device) if targets.valid is None else targets.valid
        if valid_steps.shape != fast.shape[:2]:
            raise ValueError("M1 valid event mask must be [B,T]")
        # Flatten time into independent retrieval queries, while the state is
        # still produced by a single differentiable unroll.
        event_mask = valid_steps.reshape(-1)
        future_flat = future.reshape(-1, future.shape[-2], future.shape[-1])[event_mask]
        positive_flat = positive.reshape(-1, positive.shape[-1])[event_mask]
        known_flat = known.reshape(-1, known.shape[-1])[event_mask]
        rate_flat = {name: value.reshape(-1)[event_mask] for name, value in rates.items()}
        reliability_flat = None if targets.reliability is None else targets.reliability[:, :fast.shape[1]].reshape(-1)[event_mask]
        reliability_known_flat = None if targets.reliability_known is None else targets.reliability_known[:, :fast.shape[1]].reshape(-1)[event_mask]
        retrieval = predictive_memory_loss(
            fast.reshape(-1, fast.shape[-1])[event_mask], slow.reshape(-1, slow.shape[-1])[event_mask],
            future_flat, positive_flat, rate_flat, reliability_flat, reliability_known_flat, known_flat, self.loss_weights,
        )
        # Masking inactive events is done at query construction time in the
        # data task; keeping this count explicit prevents an empty mask being
        # mistaken for a successful optimizer step.
        retrieval["event_count"] = valid_steps.sum().detach()
        retrieval["write_rate"] = torch.stack([rates["q"], rates["alpha_fast"], rates["alpha_slow"]]).mean().detach()
        return retrieval

    def build_utility_labels(
        self,
        state_snapshot: Any,
        observation: Tensor,
        future_rows: Tensor,
        fixed_policy_snapshot: Mapping[str, Any],
        training_labels: Mapping[str, Any],
    ) -> UtilityExample:
        """Expose the complete counterfactual-label branch for ablations."""
        return UtilityLabelBuilder(self.memory).build(
            state_snapshot, observation, future_rows,
            fixed_policy_snapshot, training_labels,
        )

    def utility_objective(self, predicted_utility: Tensor, example: UtilityExample) -> dict[str, Tensor]:
        target = example.utility_target.to(predicted_utility.device, predicted_utility.dtype)
        if predicted_utility.shape != target.shape:
            raise ValueError("utility head output does not match counterfactual label shape")
        loss = counterfactual_utility_loss(predicted_utility, target)
        return {"total": loss, "utility": loss.detach()}


def memory_loss(memory_output: Mapping[str, Tensor], future_embedding: Tensor, same_identity: Tensor, reliability: Tensor | None = None, *, candidate_known: Tensor | None = None, reliability_known: Tensor | None = None) -> dict[str, Tensor]:
    return predictive_memory_loss(
        memory_output["fast"], memory_output["slow"], future_embedding, same_identity,
        memory_output.get("rates"),
        memory_output.get("reliability_logit"), reliability,
        reliability_known, candidate_known,
    )


def utility_loss(predicted_utility: Tensor, branch_states: Mapping[str, Tensor], future_embedding: Tensor) -> dict[str, Tensor]:
    """Train utility scores from the same state and future target."""
    future = future_embedding / future_embedding.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scores = torch.stack([(value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8) * future).sum(-1) for value in branch_states.values()], dim=-1)
    target = scores - scores[..., :1]
    loss = counterfactual_utility_loss(predicted_utility, target)
    return {"total": loss, "utility": loss.detach(), "utility_target": target.detach()}
