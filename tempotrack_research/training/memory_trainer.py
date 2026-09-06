"""True multi-step M1 training task and loss wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..losses.predictive import counterfactual_utility_loss, predictive_memory_loss
from ..memory.predictive_dual import PredictiveDualMemory, UtilityExample, UtilityLabelBuilder


@dataclass
class MemoryInputs:
    prototype: Tensor
    observations: Tensor
    history_states: Tensor
    causal_evidence: Tensor
    frames: Tensor | None = None
    bboxes: Tensor | None = None


@dataclass
class MemoryTargets:
    future_embedding: Tensor
    positive_mask: Tensor
    candidate_known: Tensor
    reliability: Tensor | None = None
    reliability_known: Tensor | None = None
    valid: Tensor | None = None


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
                prototype=inputs["prototype"], observations=inputs["observations"],
                history_states=inputs["history_states"], causal_evidence=inputs["causal_evidence"],
                frames=inputs.get("frames"), bboxes=inputs.get("bboxes"),
            )
        if isinstance(targets, Mapping):
            targets = MemoryTargets(
                future_embedding=targets["future_embedding"], positive_mask=targets["positive_mask"],
                candidate_known=targets["candidate_known"], reliability=targets.get("reliability"),
                reliability_known=targets.get("reliability_known"), valid=targets.get("valid"),
            )
        if inputs.observations.ndim != 3 or inputs.history_states.shape[:2] != inputs.observations.shape[:2] or inputs.causal_evidence.shape[:2] != inputs.observations.shape[:2]:
            raise ValueError("M1 inputs must be [B,T,D], [B,T,2D], [B,T,8]")
        if inputs.causal_evidence.shape[-1] != 8:
            raise ValueError("M1 causal evidence must be eight-dimensional")
        batch, steps, _ = inputs.observations.shape
        state = self.memory.initialize(inputs.prototype)
        fast_values: list[Tensor] = []
        slow_values: list[Tensor] = []
        rate_values: dict[str, list[Tensor]] = {"q": [], "alpha_fast": [], "alpha_slow": [], "reliability_logit": []}
        for index in range(min(steps, self.unroll)):
            frame = None if inputs.frames is None else inputs.frames[:, index]
            bbox = None if inputs.bboxes is None else inputs.bboxes[:, index]
            state, rates = self.memory.update(
                state,
                inputs.observations[:, index],
                inputs.history_states[:, index],
                inputs.causal_evidence[:, index],
                frame,
                bbox=bbox,
            )
            fast_values.append(state.fast)
            slow_values.append(state.slow)
            for name in rate_values:
                rate_values[name].append(rates[name])
        if not fast_values:
            raise ValueError("M1 received an empty unroll")
        fast = torch.stack(fast_values, dim=1)
        slow = torch.stack(slow_values, dim=1)
        rates = {name: torch.stack(values, dim=1) for name, values in rate_values.items()}
        future = targets.future_embedding
        positive = targets.positive_mask.bool()
        known = targets.candidate_known.bool()
        if future.ndim == 3:
            future = future.unsqueeze(1)
        if future.ndim != 4:
            raise ValueError("future_embedding must be [B,T,K,D] or [B,K,D]")
        if positive.ndim == 2:
            positive = positive.unsqueeze(1)
        if known.ndim == 2:
            known = known.unsqueeze(1)
        if positive.shape[:2] != fast.shape[:2] or known.shape != positive.shape or future.shape[:3] != positive.shape:
            raise ValueError("M1 future candidate tensors have inconsistent shapes")
        valid_steps = torch.ones((batch, fast.shape[1]), dtype=torch.bool, device=fast.device) if targets.valid is None else targets.valid.bool()
        if valid_steps.shape != fast.shape[:2]:
            raise ValueError("M1 valid event mask must be [B,T]")
        # Flatten time into independent retrieval queries, while the state is
        # still produced by a single differentiable unroll.
        retrieval = predictive_memory_loss(
            fast.reshape(-1, fast.shape[-1]),
            slow.reshape(-1, slow.shape[-1]),
            future.reshape(-1, future.shape[-2], future.shape[-1]),
            positive.reshape(-1, positive.shape[-1]),
            {name: value.reshape(-1) for name, value in rates.items()},
            targets.reliability.reshape(-1) if targets.reliability is not None else None,
            targets.reliability_known.reshape(-1) if targets.reliability_known is not None else None,
            known.reshape(-1, known.shape[-1]),
            self.loss_weights,
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
