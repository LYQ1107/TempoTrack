"""S1 causal cross-break prediction and its matched ordinary metric control."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .trajectory_encoder import TrajectoryEncoder
from ..losses.regularization import vicreg_regularization


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

    @staticmethod
    def _query_tensor(query: Any) -> Tensor:
        if hasattr(query, "relative_times"):
            query = query.relative_times
        if not torch.is_tensor(query):
            query = torch.as_tensor(query, dtype=torch.float32)
        return query

    def _predict_flat(self, context: Mapping[str, Tensor], query_times: Tensor) -> dict[str, Tensor]:
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
        identity_raw = self.identity_predictor(torch.cat((attended.mean(dim=1), source_summary), dim=-1))
        identity = F.normalize(identity_raw, dim=-1)
        valid = torch.ones(query_times.shape, dtype=torch.bool, device=query_times.device)
        return {
            "dynamic": dynamic,
            "identity_raw": identity_raw,
            "identity": identity,
            "summary": identity_raw,
            "valid": valid,
            "query_times": query_times,
        }

    def predict(self, context: Mapping[str, Tensor], target_relative_times: Any) -> dict[str, Tensor]:
        times = self._query_tensor(target_relative_times).to(context["summary"].device)
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(context["summary"].shape[0], -1)
        if times.ndim == 2:
            return self._predict_flat(context, times)
        if times.ndim == 3:
            batch, candidates, length = times.shape
            if context["summary"].shape[0] != batch:
                raise ValueError("query batch does not match context batch")
            repeated = {
                key: value.unsqueeze(1).expand(-1, candidates, *value.shape[1:]).reshape(batch * candidates, *value.shape[1:])
                for key, value in context.items()
                if torch.is_tensor(value) and key not in {"query_times"}
            }
            flat = self._predict_flat(repeated, times.reshape(batch * candidates, length))
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
        with torch.no_grad():
            target = self.encode_target(
                episode["target_appearance"], episode["target_geometry"],
                episode["target_time"], episode.get("target_valid")
            )
        query = episode.get("query_times", episode["target_time"])
        predicted = self.predict(context, query)
        target_dynamic = target["dynamic"]
        if predicted["dynamic"].shape != target_dynamic.shape:
            raise ValueError("S1 prediction and target dynamic shapes disagree")
        valid = episode.get("target_valid", target["valid"]).bool()
        positive = labels.get("positive", episode.get("positive", torch.ones(valid.shape[0], dtype=torch.bool, device=valid.device))).bool()
        while positive.ndim < valid.ndim:
            positive = positive.unsqueeze(-1)
        dynamic_valid = valid & positive
        if bool(dynamic_valid.any()):
            prediction_loss = F.smooth_l1_loss(
                predicted["dynamic"][dynamic_valid], target_dynamic.detach()[dynamic_valid], reduction="mean"
            )
        else:
            prediction_loss = context["summary"].new_zeros(())

        # A single target is represented as K=1; candidate episodes may pass
        # precomputed candidate identity tensors for a genuine multi-positive
        # objective.  The target branch never feeds the predictor.
        candidate_identity = episode.get("candidate_identity")
        candidate_known = episode.get("candidate_known")
        positive_mask = episode.get("positive_mask")
        if candidate_identity is not None:
            if candidate_identity.ndim != 3:
                raise ValueError("candidate_identity must be [B,K,H]")
            source_identity = context["identity"].unsqueeze(1)
            scores = (source_identity * F.normalize(candidate_identity, dim=-1)).sum(-1) / 0.07
            known = torch.ones_like(scores, dtype=torch.bool) if candidate_known is None else candidate_known.bool()
            positives = (torch.ones_like(scores, dtype=torch.bool) if positive_mask is None else positive_mask.bool())
            identity_loss, identity_count = _multi_positive_ce(scores, positives, known)
        else:
            target_identity = target["identity"].detach()
            similarity = (context["identity"] * target_identity).sum(-1) / 0.07
            same = episode.get("same_identity", positive.squeeze(-1)).to(similarity.dtype)
            known = episode.get("candidate_known", torch.ones_like(same, dtype=torch.bool)).bool()
            if bool(known.any()):
                identity_loss = F.binary_cross_entropy_with_logits(similarity[known], same[known])
                identity_count = known.sum()
            else:
                identity_loss = similarity.new_zeros(())
                identity_count = similarity.new_zeros((), dtype=torch.long)
        regularization, reg_metrics = self._representation_regularization(context)
        total = prediction_loss + identity_loss + 0.01 * regularization
        output: dict[str, Tensor] = {
            "total": total,
            "prediction": prediction_loss.detach(),
            "identity": identity_loss.detach(),
            "identity_queries": identity_count.detach(),
            **reg_metrics,
            "dynamic_valid_count": dynamic_valid.sum().detach(),
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
        query = (rt - lt[:, -1:])
        predicted = self.predict(lc, query)
        forward_error = F.smooth_l1_loss(predicted["dynamic"], target["dynamic"], reduction="mean")
        anchor_similarity = (lc["identity"] * target["identity"]).sum(-1).mean()
        error = forward_error
        if mode == "bidirectional_inpainting":
            rc = self.encode_context(ra, rg, rt)
            back_target = self.encode_target(la, lg, lt)
            reverse = self.predict(rc, (lt - rt[:, -1:]))
            reverse_error = F.smooth_l1_loss(reverse["dynamic"], back_target["dynamic"], reduction="mean")
            error = 0.5 * (forward_error + reverse_error)
        return {
            "edge_score": -error + 0.1 * anchor_similarity,
            "prediction_error": error,
            "anchor_similarity": anchor_similarity,
        }

    @torch.no_grad()
    def score_candidates(self, tracklets: Sequence[Mapping[str, Any]], candidate_graph: Any, generator: torch.Generator | None = None, mode: str = "forward_only") -> Tensor:
        del generator
        edge_index = candidate_graph.edge_index if hasattr(candidate_graph, "edge_index") else candidate_graph["edge_index"]
        edge_index = edge_index.detach().cpu() if torch.is_tensor(edge_index) else torch.as_tensor(edge_index)
        if edge_index.numel() == 0:
            return torch.empty(0, device=next(self.parameters()).device)
        # Encode every unique segment once.  Scoring remains deterministic and
        # does not launch a separate model path for every edge.
        cache: dict[int, tuple[dict[str, Tensor], dict[str, Tensor], Tensor, Tensor, Tensor]] = {}
        for node in sorted(set(int(value) for value in edge_index.flatten().tolist())):
            appearance, geometry, times = self._tracklet_inputs(tracklets[node], next(self.parameters()).device)
            cache[node] = (
                self.encode_context(appearance, geometry, times),
                self.encode_target(appearance, geometry, times),
                appearance,
                geometry,
                times,
            )
        values = []
        for source, target in edge_index.t().tolist():
            left_context, _, _, _, left_times = cache[int(source)]
            right_context, right_target, _, _, right_times = cache[int(target)]
            predicted = self.predict(left_context, right_times - left_times[:, -1:])
            forward_error = F.smooth_l1_loss(predicted["dynamic"], right_target["dynamic"], reduction="mean")
            anchor_similarity = (left_context["identity"] * right_target["identity"]).sum(-1).mean()
            error = forward_error
            if mode == "bidirectional_inpainting":
                _, left_target, _, _, _ = cache[int(source)]
                reverse = self.predict(right_context, left_times - right_times[:, -1:])
                reverse_error = F.smooth_l1_loss(reverse["dynamic"], left_target["dynamic"], reduction="mean")
                error = 0.5 * (forward_error + reverse_error)
            values.append(-error + 0.1 * anchor_similarity)
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
        context = self.encode_context(appearance, geometry, times)
        target = self.encode_target(held_app, held_geo, held_time)
        predicted = self.predict(context, held_time - times[:, -1:])
        error = F.smooth_l1_loss(predicted["dynamic"], target["dynamic"], reduction="mean")
        return {"heldout": heldout_segment, "mode": mode, "error": error, "edge_score": -error, "context_count": len(context_segments)}

    @torch.no_grad()
    def chain_leave_one_segment_out(self, tracklets: list[Mapping[str, Any]], path: list[int], rounds: int = 2, mode: str = "forward_only") -> dict[str, Any]:
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
            removed = [
                [current[position - 1], current[position]],
                [current[position], current[position + 1]],
            ] if error > 0 else []
            decisions.append({
                "round": round_index,
                "masked_segment": current[position],
                "context": [current[position - 1], current[position + 1]],
                "error": error,
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
        right = self.encoder(episode["right_appearance"], episode["right_geometry"], episode["right_time"], episode.get("right_valid"))
        logits = self.score_pair(left, right)
        target = labels.get("same_identity", episode.get("same_identity")).to(logits.dtype)
        known = labels.get("candidate_known", episode.get("candidate_known", torch.ones_like(target, dtype=torch.bool))).bool()
        if not bool(known.any()):
            raise ValueError("ordinary metric batch has no known candidate labels")
        loss = F.binary_cross_entropy_with_logits(logits[known], target[known])
        return {"total": loss, "metric_bce": loss.detach(), "known_pairs": known.sum().detach()}
