"""S1 causal cross-break prediction and its matched ordinary metric control."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .trajectory_encoder import TrajectoryEncoder
from ..losses.regularization import vicreg_regularization
from ..schemas import LinkEvidence, PairInputs


def _masked_mean(value: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        return value.mean()
    if mask.shape != value.shape[: mask.ndim]:
        raise ValueError(f"mask shape {mask.shape} is incompatible with {value.shape}")
    weights = mask.to(value.dtype)
    while weights.ndim < value.ndim:
        weights = weights.unsqueeze(-1)
    denominator = weights.sum().clamp_min(1.0)
    return (value * weights).sum() / denominator


def _multi_positive_ce(scores: Tensor, positive: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
    if scores.ndim != 2 or positive.shape != scores.shape or known.shape != scores.shape:
        raise ValueError("candidate scores and masks must all have shape [B,K]")
    valid = positive.any(dim=-1) & known.any(dim=-1)
    if not bool(valid.any()):
        return scores.new_zeros(()), scores.new_zeros((), dtype=torch.long)
    floor = torch.finfo(scores.dtype).min
    denominator = torch.logsumexp(scores.masked_fill(~known, floor), dim=-1)
    numerator = torch.logsumexp(scores.masked_fill(~positive, floor), dim=-1)
    return (-(numerator - denominator)[valid]).mean(), valid.sum()


def masked_link_objectives(score: Tensor, positive: Tensor, known: Tensor, candidate_valid: Tensor) -> dict[str, Tensor]:
    """Compute the shared S1/ordinary candidate objectives with real masks."""
    if score.ndim != 2:
        raise ValueError("link scores must have shape [B,K]")
    for name, value in (("positive", positive), ("known", known), ("candidate_valid", candidate_valid)):
        if value.shape != score.shape:
            raise ValueError(f"{name} must have shape {tuple(score.shape)}")
    effective = known.bool() & candidate_valid.bool()
    positive = positive.bool() & effective
    zero = score.sum() * 0.0
    if bool(effective.any()):
        bce = F.binary_cross_entropy_with_logits(score[effective], positive.to(score.dtype)[effective])
    else:
        bce = zero
    ce, ce_rows = _multi_positive_ce(score, positive, effective)
    total = bce + ce
    return {
        "total": total,
        "bce": bce,
        "ce": ce,
        "known_count": effective.sum().to(score.dtype),
        "positive_count": positive.sum().to(score.dtype),
        "positive_rows": ce_rows.to(score.dtype),
    }


def compute_link_evidence(
    context: Mapping[str, Tensor],
    predicted: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    inputs: Mapping[str, Tensor],
    *,
    temperature: float = 0.07,
    dynamic_weight: float = 1.0,
    anchor_weight: float = 0.1,
) -> LinkEvidence:
    """Single score formula used by formal loss and deployment."""
    if temperature <= 0 or dynamic_weight < 0 or anchor_weight < 0:
        raise ValueError("link score weights must be non-negative and temperature positive")
    prediction_identity = F.normalize(predicted["identity"], dim=-1)
    target_identity = F.normalize(target["identity"].detach(), dim=-1)
    if prediction_identity.shape != target_identity.shape:
        raise ValueError("predicted and target identity tensors disagree")
    candidate_axis = prediction_identity.ndim == 3
    if candidate_axis:
        anchor = F.normalize(context["identity"], dim=-1).unsqueeze(1)
    else:
        anchor = F.normalize(context["identity"], dim=-1)
    frozen_anchor = (anchor * target_identity).sum(-1)
    prediction_identity_cosine = (prediction_identity * target_identity).sum(-1)
    target_valid = inputs.get("target_valid")
    query_valid = inputs.get("query_valid", target_valid)
    if target_valid is None:
        target_valid = torch.ones(predicted["dynamic"].shape[:-1], dtype=torch.bool, device=predicted["dynamic"].device)
    if query_valid is None:
        query_valid = target_valid
    token_valid = target_valid.bool() & query_valid.bool()
    dynamic_difference = F.smooth_l1_loss(predicted["dynamic"], target["dynamic"].detach(), reduction="none")
    while token_valid.ndim < dynamic_difference.ndim:
        token_valid = token_valid.unsqueeze(-1)
    denominator = token_valid.to(dynamic_difference.dtype).sum(dim=tuple(range(-2, 0))).clamp_min(1.0)
    dynamic_error = (dynamic_difference * token_valid.to(dynamic_difference.dtype)).sum(dim=tuple(range(-2, 0))) / denominator
    valid = token_valid.squeeze(-1) if token_valid.ndim == predicted["dynamic"].ndim else token_valid
    valid = valid.any(dim=-1) if valid.ndim == prediction_identity.ndim else valid
    score = prediction_identity_cosine / float(temperature) - float(dynamic_weight) * dynamic_error + float(anchor_weight) * frozen_anchor
    return LinkEvidence(score, prediction_identity, dynamic_error, frozen_anchor, valid.bool())


class JEPAIdentityLinker(nn.Module):
    """Student/EMA-teacher segment model.

    ``predict`` receives only the encoded source and query times.  Candidate
    appearance is used by ``compute_loss``/``score_candidates`` in the
    independent target branch and can therefore not leak into the predictor.
    """

    def __init__(
        self,
        appearance_dim: int = 256,
        hidden_dim: int = 256,
        layers: int = 4,
        heads: int = 8,
        ff_dim: int = 1024,
        dynamic_dim: int = 64,
        target_momentum_start: float = 0.99,
        target_momentum_end: float = 0.9999,
        loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.context_encoder = TrajectoryEncoder(
            appearance_dim,
            hidden_dim,
            layers,
            heads,
            ff_dim,
            dynamic_dim=dynamic_dim,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.source_dynamic = nn.Linear(dynamic_dim, hidden_dim)
        self.query_time = nn.Sequential(
            nn.Linear(1, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.dynamic_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dynamic_dim)
        )
        self.identity_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.target_momentum_start = float(target_momentum_start)
        self.target_momentum_end = float(target_momentum_end)
        weights = dict(loss_weights or {})
        self.loss_weights = {
            "prediction": float(weights.get("prediction", 1.0)),
            "identity": float(weights.get("identity", 1.0)),
            "regularization": float(weights.get("regularization", 0.01)),
        }
        if any(value < 0 for value in self.loss_weights.values()) or not any(self.loss_weights.values()):
            raise ValueError("S1 loss weights must be non-negative and not all zero")
        if not (0.0 <= self.target_momentum_start < 1.0 and self.target_momentum_start <= self.target_momentum_end < 1.0):
            raise ValueError("invalid S1 EMA momentum range")
        self.register_buffer("ema_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_schedule_steps", torch.tensor(100000, dtype=torch.long))

    def train(self, mode: bool = True) -> "JEPAIdentityLinker":
        super().train(mode)
        # ``Module.train`` recursively changes target_encoder back to train.
        self.target_encoder.eval()
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        return self

    def encode_context(self, appearance: Any, geometry: Tensor | None = None, time_offsets: Tensor | None = None, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        if geometry is None and hasattr(appearance, "appearance"):
            return self.context_encoder(appearance)
        return self.context_encoder(appearance, geometry, time_offsets, valid_mask)

    @torch.no_grad()
    def encode_target(self, appearance: Any, geometry: Tensor | None = None, time_offsets: Tensor | None = None, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        self.target_encoder.eval()
        if geometry is None and hasattr(appearance, "appearance"):
            return self.target_encoder(appearance)
        return self.target_encoder(appearance, geometry, time_offsets, valid_mask)

    @torch.no_grad()
    def score_pair_inputs(self, inputs: PairInputs) -> LinkEvidence:
        """Score the exact tensor contract used by training."""
        source = inputs.source
        candidates = inputs.candidates
        source_app = source.appearance.unsqueeze(0) if source.appearance.ndim == 2 else source.appearance
        source_geo = source.geometry.unsqueeze(0) if source.geometry.ndim == 2 else source.geometry
        source_time = source.local_time.unsqueeze(0) if source.local_time.ndim == 1 else source.local_time
        source_valid = source.valid.unsqueeze(0) if source.valid.ndim == 1 else source.valid
        candidate_app = candidates.appearance
        candidate_geo = candidates.geometry
        candidate_time = candidates.local_time
        candidate_valid_tokens = candidates.valid
        if candidate_app.ndim == 3:
            candidate_app = candidate_app.unsqueeze(0)
            candidate_geo = candidate_geo.unsqueeze(0)
            candidate_time = candidate_time.unsqueeze(0)
            candidate_valid_tokens = candidate_valid_tokens.unsqueeze(0)
        if candidate_app.ndim != 4:
            raise ValueError("PairInputs candidates must be [B,K,L,D]")
        batch, candidate_count, length, dim = candidate_app.shape
        if source_app.shape[0] != batch:
            if source_app.shape[0] == 1:
                source_app = source_app.expand(batch, -1, -1)
                source_geo = source_geo.expand(batch, -1, -1)
                source_time = source_time.expand(batch, -1)
                source_valid = source_valid.expand(batch, -1)
            else:
                raise ValueError("PairInputs source/candidate batch mismatch")
        context = self.encode_context(source_app, source_geo, source_time, source_valid)
        flat_target = self.encode_target(
            candidate_app.reshape(batch * candidate_count, length, dim),
            candidate_geo.reshape(batch * candidate_count, length, candidate_geo.shape[-1]),
            candidate_time.reshape(batch * candidate_count, length),
            candidate_valid_tokens.reshape(batch * candidate_count, length),
        )
        target = {key: value.reshape(batch, candidate_count, *value.shape[1:]) for key, value in flat_target.items() if torch.is_tensor(value)}
        query = inputs.query.relative_times if inputs.query is not None else candidate_time
        query_valid = inputs.query.valid if inputs.query is not None else candidate_valid_tokens
        if query.ndim == 2:
            query = query.unsqueeze(0)
        if query_valid is not None and query_valid.ndim == 2:
            query_valid = query_valid.unsqueeze(0)
        predicted = self.predict(context, query, query_valid=query_valid)
        evidence = compute_link_evidence(context, predicted, target, {"target_valid": candidate_valid_tokens, "query_valid": query_valid})
        candidate_valid = inputs.candidate_valid
        if candidate_valid is None:
            candidate_valid = torch.ones((batch, candidate_count), dtype=torch.bool, device=source_app.device)
        else:
            candidate_valid = torch.as_tensor(candidate_valid, dtype=torch.bool, device=source_app.device)
            if candidate_valid.ndim == 1:
                candidate_valid = candidate_valid.unsqueeze(0).expand(batch, -1)
        evidence.valid = evidence.valid & candidate_valid
        return evidence

    @staticmethod
    def _query_tensor(query: Any) -> Tensor:
        if hasattr(query, "relative_times"):
            query = query.relative_times
        if not torch.is_tensor(query):
            query = torch.as_tensor(query, dtype=torch.float32)
        return query

    def _predict_flat(self, context: Mapping[str, Tensor], query_times: Tensor, query_valid: Tensor | None = None) -> dict[str, Tensor]:
        source_dynamic = context["dynamic"]
        source_valid = context["valid"].bool()
        source_summary = context["identity_raw"]
        if source_dynamic.ndim != 3 or query_times.ndim != 2:
            raise ValueError("flat S1 prediction expects source [B,L,*] and query [B,Q]")
        keys = self.source_dynamic(source_dynamic)
        queries = self.query_time(query_times.to(keys.dtype).unsqueeze(-1))
        # No source group is fully empty because the encoder filtered it; an
        # all-empty input still yields a finite zero prediction.
        safe_valid = source_valid.any(dim=-1)
        safe_keys = keys.masked_fill(~source_valid.unsqueeze(-1), 0.0)
        attended, _ = self.cross_attention(
            queries,
            safe_keys,
            safe_keys,
            key_padding_mask=~source_valid,
            need_weights=False,
        )
        attended = torch.where(safe_valid[:, None, None], attended, torch.zeros_like(attended))
        source = source_summary.unsqueeze(1).expand(-1, query_times.shape[1], -1)
        dynamic = self.dynamic_predictor(torch.cat((attended, source, queries), dim=-1))
        if query_valid is None:
            query_valid = torch.ones(query_times.shape, dtype=torch.bool, device=query_times.device)
        else:
            query_valid = query_valid.bool().to(query_times.device)
            if query_valid.shape != query_times.shape:
                raise ValueError("S1 query_valid must match query_times")
        qweights = query_valid.to(attended.dtype).unsqueeze(-1)
        qsummary = (attended * qweights).sum(dim=1) / qweights.sum(dim=1).clamp_min(1.0)
        identity_raw = self.identity_predictor(torch.cat((qsummary, source_summary), dim=-1))
        identity = F.normalize(identity_raw, dim=-1)
        valid = query_valid
        return {
            "dynamic": dynamic,
            "identity_raw": identity_raw,
            "identity": identity,
            "summary": identity_raw,
            "valid": valid,
            "query_times": query_times,
        }

    def predict(self, context: Mapping[str, Tensor], target_relative_times: Any, query_valid: Tensor | None = None) -> dict[str, Tensor]:
        times = self._query_tensor(target_relative_times).to(context["summary"].device)
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(context["summary"].shape[0], -1)
            if query_valid is not None:
                query_valid = torch.as_tensor(query_valid, device=times.device, dtype=torch.bool)
                if query_valid.ndim == 1:
                    query_valid = query_valid.unsqueeze(0).expand(context["summary"].shape[0], -1)
        if times.ndim == 2:
            valid = None if query_valid is None else torch.as_tensor(query_valid, device=times.device, dtype=torch.bool)
            return self._predict_flat(context, times, valid)
        if times.ndim == 3:
            batch, candidates, length = times.shape
            if context["summary"].shape[0] != batch:
                raise ValueError("query batch does not match context batch")
            repeated = {
                key: value.unsqueeze(1).expand(-1, candidates, *value.shape[1:]).reshape(batch * candidates, *value.shape[1:])
                for key, value in context.items()
                if torch.is_tensor(value) and key not in {"query_times"}
            }
            flat_valid = None if query_valid is None else torch.as_tensor(query_valid, device=times.device, dtype=torch.bool).reshape(batch * candidates, length)
            flat = self._predict_flat(repeated, times.reshape(batch * candidates, length), flat_valid)
            return {
                key: value.reshape(batch, candidates, *value.shape[1:])
                for key, value in flat.items()
            }
        raise ValueError("target_relative_times must have rank 1, 2, or 3")

    @torch.no_grad()
    def update_target(
        self,
        successful_optimizer_step: bool = True,
        *,
        optimizer_step: int | None = None,
        schedule_steps: int | None = None,
    ) -> float:
        if not successful_optimizer_step:
            return self.current_momentum(schedule_steps)
        schedule = int(schedule_steps or self.ema_schedule_steps.item())
        schedule = max(schedule, 1)
        step = int(self.ema_steps.item() if optimizer_step is None else optimizer_step)
        progress = min(1.0, max(0.0, step / schedule))
        momentum = self.target_momentum_start + progress * (
            self.target_momentum_end - self.target_momentum_start
        )
        for target, source in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.data.mul_(momentum).add_(source.data, alpha=1.0 - momentum)
        for target, source in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            if target.dtype.is_floating_point:
                target.data.copy_(source.data)
            else:
                target.data.copy_(source.data)
        self.ema_steps.copy_(torch.as_tensor(step + 1, device=self.ema_steps.device))
        return float(momentum)

    def current_momentum(self, schedule_steps: int | None = None) -> float:
        schedule = max(int(schedule_steps or self.ema_schedule_steps.item()), 1)
        progress = min(1.0, float(self.ema_steps.item()) / schedule)
        return self.target_momentum_start + progress * (
            self.target_momentum_end - self.target_momentum_start
        )

    def forward(self, appearance: Any, geometry: Tensor | None = None, time_offsets: Tensor | None = None, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        return self.encode_context(appearance, geometry, time_offsets, valid_mask)

    def _representation_regularization(self, encoded: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        valid = encoded["valid"]
        values: list[Tensor] = []
        if encoded["identity_raw"].ndim == 2:
            values.append(encoded["identity_raw"])
        values.append(encoded["pre_tokens"][valid])
        values.append(encoded["dynamic"][valid])
        values = [value.float() for value in values if value.numel()]
        if not values:
            zero = encoded["summary"].new_zeros(())
            return zero, {"regularization_count": zero}
        # Keep separate dimensions independent; concatenating different head
        # widths would turn the regularizer into a hidden padding operation.
        losses = [vicreg_regularization(value) for value in values if value.shape[0] >= 2]
        if not losses:
            zero = encoded["summary"].new_zeros(())
            return zero, {"regularization_count": encoded["summary"].new_tensor(sum(v.shape[0] for v in values))}
        total = torch.stack([item["total"] for item in losses]).mean()
        return total, {
            "regularization_count": encoded["summary"].new_tensor(sum(v.shape[0] for v in values)),
            "regularization": total.detach(),
        }

    def compute_loss(self, episode: Mapping[str, Tensor], labels: Mapping[str, Tensor] | None = None) -> dict[str, Tensor]:
        labels = dict(labels or {})
        context = self.encode_context(
            episode["context_appearance"], episode["context_geometry"],
            episode["context_time"], episode.get("context_valid")
        )
        target_appearance = episode["target_appearance"]
        target_geometry = episode["target_geometry"]
        target_time = episode["target_time"]
        target_valid_input = episode.get("target_valid")
        candidate_axis = target_appearance.ndim == 4
        if candidate_axis:
            batch_size, candidate_count, target_length, appearance_dim = target_appearance.shape
            flat = {
                "appearance": target_appearance.reshape(batch_size * candidate_count, target_length, appearance_dim),
                "geometry": target_geometry.reshape(batch_size * candidate_count, target_length, target_geometry.shape[-1]),
                "time": target_time.reshape(batch_size * candidate_count, target_length),
                "valid": None if target_valid_input is None else target_valid_input.reshape(batch_size * candidate_count, target_length),
            }
        else:
            flat = {"appearance": target_appearance, "geometry": target_geometry, "time": target_time, "valid": target_valid_input}
        with torch.no_grad():
            target_flat = self.encode_target(flat["appearance"], flat["geometry"], flat["time"], flat["valid"])
        if candidate_axis:
            target = {key: value.reshape(batch_size, candidate_count, *value.shape[1:]) if torch.is_tensor(value) and value.shape[0] == batch_size * candidate_count else value for key, value in target_flat.items()}
        else:
            target = target_flat
        query = episode.get("query_times", episode["target_time"])
        query_valid = episode.get("target_valid")
        predicted = self.predict(context, query, query_valid=query_valid)
        target_dynamic = target["dynamic"]
        if candidate_axis:
            target_token_valid = episode.get("target_valid", torch.ones_like(predicted["valid"])).bool()
            query_token_valid = episode.get("query_valid", target_token_valid).bool()
            positive_mask = episode.get("positive_mask", episode.get("positive")).bool()
            known_mask = episode.get("candidate_known", torch.ones_like(positive_mask)).bool()
            candidate_valid = episode.get("candidate_valid", torch.ones_like(positive_mask)).bool()
            if positive_mask.shape != known_mask.shape or positive_mask.shape != predicted["identity"].shape[:2]:
                raise ValueError("S1 candidate masks do not match predicted candidate axis")
            evidence = compute_link_evidence(context, predicted, target, {"target_valid": target_token_valid, "query_valid": query_token_valid})
            dynamic_valid = target_token_valid & query_token_valid & positive_mask.unsqueeze(-1)
        else:
            target_token_valid = episode.get("target_valid", target["valid"]).bool()
            positive_mask = labels.get("positive", episode.get("positive", torch.ones(target_token_valid.shape[0], dtype=torch.bool, device=target_token_valid.device))).bool()
            while positive_mask.ndim < target_token_valid.ndim:
                positive_mask = positive_mask.unsqueeze(-1)
            evidence = compute_link_evidence(context, predicted, target, {"target_valid": target_token_valid, "query_valid": target_token_valid})
            dynamic_valid = target_token_valid & positive_mask
        if predicted["dynamic"].shape != target_dynamic.shape:
            raise ValueError("S1 prediction and target dynamic shapes disagree")
        if bool(dynamic_valid.any()):
            prediction_loss = F.smooth_l1_loss(
                predicted["dynamic"][dynamic_valid], target_dynamic.detach()[dynamic_valid], reduction="mean"
            )
        else:
            prediction_loss = context["summary"].new_zeros(())

        if candidate_axis:
            objectives = masked_link_objectives(evidence.score, positive_mask, known_mask, candidate_valid)
            identity_loss = objectives["total"]
            identity_count = objectives["known_count"]
        else:
            same = episode.get("same_identity", episode.get("positive", torch.ones_like(evidence.score))).bool().reshape(-1, 1)
            known = episode.get("candidate_known", torch.ones_like(same)).bool().reshape(-1, 1)
            objectives = masked_link_objectives(evidence.score.reshape(-1, 1), same, known, torch.ones_like(known))
            identity_loss = objectives["total"]
            identity_count = objectives["known_count"]
        regularization, reg_metrics = self._representation_regularization(context)
        total = (
            self.loss_weights["prediction"] * prediction_loss
            + self.loss_weights["identity"] * identity_loss
            + self.loss_weights["regularization"] * regularization
        )
        output: dict[str, Tensor] = {
            "total": total,
            "prediction": prediction_loss.detach(),
            "identity": identity_loss.detach(),
            "identity_queries": identity_count.detach(),
            "link_bce": objectives["bce"].detach(),
            "link_ce": objectives["ce"].detach(),
            "link_known_count": objectives["known_count"].detach(),
            **reg_metrics,
            "dynamic_valid_count": dynamic_valid.sum().detach(),
            "prediction_weight": context["summary"].new_tensor(self.loss_weights["prediction"]),
            "identity_weight": context["summary"].new_tensor(self.loss_weights["identity"]),
            "regularization_weight": context["summary"].new_tensor(self.loss_weights["regularization"]),
        }
        return output

    @staticmethod
    def _tracklet_inputs(tracklet: Mapping[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        appearance = torch.as_tensor(tracklet["appearance"], dtype=torch.float32, device=device)
        if appearance.ndim == 2:
            appearance = appearance.unsqueeze(0)
        if appearance.ndim != 3:
            raise ValueError("tracklet appearance must be [L,D]")
        if "bboxes" not in tracklet and "bbox" not in tracklet:
            raise ValueError("tracklet is missing real bboxes")
        box_values = tracklet.get("bboxes")
        if box_values is None:
            box_values = tracklet.get("bbox")
        if box_values is None:
            raise ValueError("tracklet is missing real bboxes")
        boxes = torch.as_tensor(box_values, dtype=torch.float32, device=device)
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        if boxes.ndim != 2 or boxes.shape[-1] < 4 or boxes.shape[0] != appearance.shape[1]:
            raise ValueError("tracklet bboxes must have one real box per appearance token")
        boxes = boxes[:, :4]
        width_values = tracklet.get("image_widths")
        height_values = tracklet.get("image_heights")
        if width_values is None or height_values is None:
            raise ValueError("tracklet is missing real image dimensions for geometry normalization")
        widths = torch.as_tensor(width_values, dtype=torch.float32, device=device).reshape(-1)
        heights = torch.as_tensor(height_values, dtype=torch.float32, device=device).reshape(-1)
        if widths.shape[0] != appearance.shape[1] or heights.shape[0] != appearance.shape[1]:
            raise ValueError("tracklet image dimensions must align with appearance tokens")
        widths = widths.clamp_min(1e-6)
        heights = heights.clamp_min(1e-6)
        x1, y1, x2, y2 = boxes.unbind(-1)
        box_width = (x2 - x1).clamp_min(1e-6)
        box_height = (y2 - y1).clamp_min(1e-6)
        geometry = torch.stack(((x1 + x2) / (2.0 * widths), (y1 + y2) / (2.0 * heights), torch.log(box_width / widths), torch.log(box_height / heights)), dim=-1).unsqueeze(0)
        times = torch.as_tensor(tracklet.get("time_offsets", tracklet.get("frames")), dtype=torch.float32, device=device)
        if times.ndim != 1 or times.shape[0] != appearance.shape[1]:
            raise ValueError("tracklet must contain one real time per appearance token")
        times = times.unsqueeze(0)
        return appearance, geometry, times

    @torch.no_grad()
    def score_link(self, left: Mapping[str, Any], right: Mapping[str, Any], mode: str = "forward_only") -> dict[str, Tensor]:
        if mode not in {"forward_only", "bidirectional_inpainting"}:
            raise ValueError("mode must be forward_only or bidirectional_inpainting")
        device = next(self.parameters()).device
        la, lg, lt = self._tracklet_inputs(left, device)
        ra, rg, rt = self._tracklet_inputs(right, device)
        lc = self.encode_context(la, lg, lt)
        target = self.encode_target(ra, rg, rt)
        left_absolute = torch.as_tensor(left.get("absolute_times", left.get("frames")), dtype=torch.float64, device=device).reshape(-1)
        right_absolute = torch.as_tensor(right.get("absolute_times", right.get("frames")), dtype=torch.float64, device=device).reshape(-1)
        query = (right_absolute - left_absolute[-1]).to(torch.float32).unsqueeze(0)
        predicted = self.predict(lc, query)
        evidence = compute_link_evidence(lc, predicted, target, {"target_valid": torch.ones_like(query, dtype=torch.bool), "query_valid": torch.ones_like(query, dtype=torch.bool)})
        error = evidence.dynamic_error.mean()
        if mode == "bidirectional_inpainting":
            rc = self.encode_context(ra, rg, rt)
            back_target = self.encode_target(la, lg, lt)
            reverse_query = (left_absolute - right_absolute[-1]).to(torch.float32).unsqueeze(0)
            reverse = self.predict(rc, reverse_query)
            reverse_evidence = compute_link_evidence(rc, reverse, back_target, {"target_valid": torch.ones_like(reverse_query, dtype=torch.bool), "query_valid": torch.ones_like(reverse_query, dtype=torch.bool)})
            reverse_error = reverse_evidence.dynamic_error.mean()
            error = 0.5 * (error + reverse_error)
        return {
            "edge_score": evidence.score.reshape(-1).mean(),
            "prediction_error": error,
            "anchor_similarity": evidence.frozen_anchor.reshape(-1).mean(),
            "predicted_identity": evidence.prediction_identity,
        }

    @torch.no_grad()
    def score_candidates(self, tracklets: Sequence[Mapping[str, Any]], candidate_graph: Any, generator: torch.Generator | None = None, mode: str = "forward_only", *, ledger: Any | None = None, tensorizer: Any | None = None, edge_batch_size: int = 64) -> Tensor:
        del generator
        if mode not in {"forward_only", "bidirectional_inpainting"}:
            raise ValueError("mode must be forward_only or bidirectional_inpainting")
        edge_index = candidate_graph.edge_index if hasattr(candidate_graph, "edge_index") else candidate_graph["edge_index"]
        edge_index = edge_index.detach().cpu() if torch.is_tensor(edge_index) else torch.as_tensor(edge_index)
        if edge_index.numel() == 0:
            return torch.empty(0, device=next(self.parameters()).device)
        values: list[Tensor] = []
        if ledger is not None and tensorizer is not None:
            # This is the production path: candidate tensorization and query
            # construction both come from the shared absolute-clock factory.
            for start in range(0, edge_index.shape[1], max(1, int(edge_batch_size))):
                batch_inputs: list[PairInputs] = []
                for source, target in edge_index[:, start : start + max(1, int(edge_batch_size))].t().tolist():
                    batch_inputs.append(tensorizer.build_pair((ledger, tracklets[int(source)]["rows"]), [(ledger, tracklets[int(target)]["rows"]) ]))
                for pair in batch_inputs:
                    values.append(self.score_pair_inputs(pair).score.reshape(-1)[0])
            return torch.stack(values).to(next(self.parameters()).device)
        # Compatibility path for callers that only have routed tracklet views.
        # It still calls the real predictor and computes each pair's absolute
        # query independently; no pair-gap output is cached across edges.
        for source, target in edge_index.t().tolist():
            values.append(self.score_link(tracklets[int(source)], tracklets[int(target)], mode=mode)["edge_score"].reshape(-1).mean())
        return torch.stack(values)

    @torch.no_grad()
    def score_chain(self, context_segments: Sequence[Mapping[str, Any]], heldout_segment: Mapping[str, Any], mode: str = "forward_only") -> dict[str, Any]:
        if len(context_segments) < 2:
            raise ValueError("chain inpainting needs at least two context segments")
        device = next(self.parameters()).device
        context_parts = [self._tracklet_inputs(item, device) for item in context_segments]
        held_app, held_geo, held_time = self._tracklet_inputs(heldout_segment, device)
        appearance = torch.cat([part[0] for part in context_parts], dim=1)
        geometry = torch.cat([part[1] for part in context_parts], dim=1)
        # Keep the original segment time origin; query times are never reset
        # independently for the held-out segment.
        times = torch.cat([part[2] for part in context_parts], dim=1)
        context_absolute = torch.cat([
            torch.as_tensor(item.get("absolute_times", item.get("frames")), dtype=torch.float64, device=device).reshape(1, -1)
            for item in context_segments
        ], dim=1)
        held_absolute = torch.as_tensor(heldout_segment.get("absolute_times", heldout_segment.get("frames")), dtype=torch.float64, device=device).reshape(1, -1)
        context = self.encode_context(appearance, geometry, times)
        target = self.encode_target(held_app, held_geo, held_time)
        predicted = self.predict(context, (held_absolute - context_absolute[:, -1:]).to(torch.float32))
        error = F.smooth_l1_loss(predicted["dynamic"], target["dynamic"], reduction="mean")
        return {"heldout": heldout_segment, "mode": mode, "error": error, "edge_score": -error, "context_count": len(context_segments)}

    @torch.no_grad()
    def chain_leave_one_segment_out(self, tracklets: list[Mapping[str, Any]], path: list[int], rounds: int = 2, mode: str = "forward_only", compatibility_threshold: float | None = None, calibration_hash: str = "") -> dict[str, Any]:
        if len(path) < 3:
            return {"path": list(path), "refined_path": list(path), "refined": False, "reason": "need at least three segments", "decisions": []}
        current = list(path)
        decisions: list[dict[str, Any]] = []
        for round_index in range(max(1, int(rounds))):
            candidates = []
            for position in range(1, len(current) - 1):
                evidence = self.score_chain(
                    [tracklets[current[position - 1]], tracklets[current[position + 1]]],
                    tracklets[current[position]], mode,
                )
                candidates.append((float(evidence["error"].cpu()), position, evidence))
            if not candidates:
                break
            error, position, evidence = max(candidates, key=lambda item: item[0])
            # A high-error held-out segment can invalidate its adjacent links,
            # but the B node and every B observation remain in the result.
            compatibility = -float(error)
            removed = [
                [current[position - 1], current[position]],
                [current[position], current[position + 1]],
            ] if compatibility_threshold is not None and compatibility < float(compatibility_threshold) else []
            decisions.append({
                "round": round_index,
                "masked_segment": current[position],
                "context": [current[position - 1], current[position + 1]],
                "error": error,
                "compatibility": compatibility,
                "compatibility_threshold": compatibility_threshold,
                "calibration_hash": calibration_hash,
                "removed_edges": removed,
                "node_retained": True,
                "evidence": {key: float(value.cpu()) if torch.is_tensor(value) and value.ndim == 0 else value for key, value in evidence.items() if key in {"edge_score", "context_count"}},
            })
            # No mutation of current path: projection must re-solve edges while
            # preserving the held-out node.  The old implementation deleted B.
            break
        return {"path": list(path), "refined_path": list(current), "refined": bool(decisions), "decisions": decisions, "rounds": int(rounds), "mode": mode}


class PairMetricLinker(nn.Module):
    """Ordinary metric with the same encoder capacity and candidate masks."""

    def __init__(self, *encoder_args: Any, **encoder_kwargs: Any) -> None:
        super().__init__()
        self.encoder = TrajectoryEncoder(*encoder_args, **encoder_kwargs)
        self.temperature_log = nn.Parameter(torch.tensor(-2.5))

    @property
    def temperature(self) -> Tensor:
        return F.softplus(self.temperature_log) + 1e-3

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        return self.encoder(*args, **kwargs)

    def score_pair(self, left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> Tensor:
        return (left["identity"] * right["identity"]).sum(-1) / self.temperature

    def compute_loss(self, episode: Mapping[str, Tensor], labels: Mapping[str, Tensor]) -> dict[str, Tensor]:
        left = self.encoder(episode["left_appearance"], episode["left_geometry"], episode["left_time"], episode.get("left_valid"))
        if episode.get("target_appearance") is not None and episode["target_appearance"].ndim == 4:
            batch, candidates, length, dim = episode["target_appearance"].shape
            right = self.encoder(
                episode["target_appearance"].reshape(batch * candidates, length, dim),
                episode["target_geometry"].reshape(batch * candidates, length, episode["target_geometry"].shape[-1]),
                episode["target_time"].reshape(batch * candidates, length),
                episode.get("target_valid").reshape(batch * candidates, length),
            )
            right_identity = right["identity"].reshape(batch, candidates, -1)
            logits = (left["identity"].unsqueeze(1) * right_identity).sum(-1) / self.temperature
            target = labels.get("positive", episode.get("positive_mask")).to(logits.dtype)
            known = labels.get("candidate_known", episode.get("candidate_known", torch.ones_like(target, dtype=torch.bool))).bool()
            if target.ndim == 1:
                target = target.unsqueeze(-1)
            if known.ndim == 1:
                known = known.unsqueeze(-1)
            loss, count = _multi_positive_ce(logits, target.bool(), known)
            if int(count.item()) == 0:
                raise ValueError("ordinary metric candidate batch has no known positive row")
            return {"total": loss, "metric_multi_positive": loss.detach(), "known_pairs": known.sum().detach()}
        right = self.encoder(episode["right_appearance"], episode["right_geometry"], episode["right_time"], episode.get("right_valid"))
        logits = self.score_pair(left, right)
        target = labels.get("same_identity", episode.get("same_identity")).to(logits.dtype)
        known = labels.get("candidate_known", episode.get("candidate_known", torch.ones_like(target, dtype=torch.bool))).bool()
        if not bool(known.any()):
            raise ValueError("ordinary metric batch has no known candidate labels")
        loss = F.binary_cross_entropy_with_logits(logits[known], target[known])
        return {"total": loss, "metric_bce": loss.detach(), "known_pairs": known.sum().detach()}
