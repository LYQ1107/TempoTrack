"""S1 cross-break identity prediction and same-capacity metric control."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .trajectory_encoder import TrajectoryEncoder


def _masked_mean(value: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        return value.mean()
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


class JEPAIdentityLinker(nn.Module):
    """Context encoder + EMA target encoder + predictor.

    The target segment is only accepted by ``encode_target``/``compute_loss``;
    ``predict`` itself receives context and relative query times only.
    """

    def __init__(self, appearance_dim: int = 256, hidden_dim: int = 256, layers: int = 4, heads: int = 8, ff_dim: int = 1024, dynamic_dim: int = 64, target_momentum_start: float = 0.99, target_momentum_end: float = 0.9999):
        super().__init__()
        self.context_encoder = TrajectoryEncoder(appearance_dim, hidden_dim, layers, heads, ff_dim, dynamic_dim=dynamic_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.target_momentum_start = float(target_momentum_start)
        self.target_momentum_end = float(target_momentum_end)
        self.register_buffer("ema_steps", torch.zeros((), dtype=torch.long))

    def encode_context(self, appearance: Tensor, geometry: Tensor, time_offsets: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        return self.context_encoder(appearance, geometry, time_offsets, valid_mask)

    @torch.no_grad()
    def encode_target(self, appearance: Tensor, geometry: Tensor, time_offsets: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        return self.target_encoder(appearance, geometry, time_offsets, valid_mask)

    def predict(self, context: dict[str, Tensor], target_relative_times: Tensor) -> Tensor:
        summary = context["summary"]
        times = target_relative_times.to(summary.dtype)
        if summary.ndim == 2:
            if times.ndim == 1:
                times = times.view(1, -1).expand(summary.shape[0], -1)
            if times.ndim != 2 or times.shape[0] != summary.shape[0]:
                raise ValueError("target_relative_times must be [L] or [B,L] for [B,H] context")
            base = summary.unsqueeze(1).expand(-1, times.shape[1], -1)
            return self.predictor(torch.cat((base, times.unsqueeze(-1)), dim=-1))
        if summary.ndim == 3:
            if times.ndim == 2:
                times = times.unsqueeze(1).expand(-1, summary.shape[1], -1)
            if times.ndim != 3 or times.shape[:2] != summary.shape[:2]:
                raise ValueError("target_relative_times must be [B,N,L] for [B,N,H] context")
            base = summary.unsqueeze(-2).expand(-1, -1, times.shape[-1], -1)
            return self.predictor(torch.cat((base, times.unsqueeze(-1)), dim=-1))
        raise ValueError("context summary must have rank 2 or 3")

    @torch.no_grad()
    def update_target(self, successful_optimizer_step: bool = True) -> float:
        if not successful_optimizer_step:
            return self.current_momentum()
        progress = min(1.0, float(self.ema_steps.item()) / 100000.0)
        momentum = self.target_momentum_start + progress * (self.target_momentum_end - self.target_momentum_start)
        for target, source in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.data.mul_(momentum).add_(source.data, alpha=1.0 - momentum)
        self.ema_steps.add_(1)
        return momentum

    def current_momentum(self) -> float:
        progress = min(1.0, float(self.ema_steps.item()) / 100000.0)
        return self.target_momentum_start + progress * (self.target_momentum_end - self.target_momentum_start)

    def forward(self, appearance: Tensor, geometry: Tensor, time_offsets: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        return self.encode_context(appearance, geometry, time_offsets, valid_mask)

    def compute_loss(self, episode: Mapping[str, Tensor], labels: Mapping[str, Tensor] | None = None) -> dict[str, Tensor]:
        context = self.encode_context(episode["context_appearance"], episode["context_geometry"], episode["context_time"], episode.get("context_valid"))
        with torch.no_grad():
            target = self.encode_target(episode["target_appearance"], episode["target_geometry"], episode["target_time"], episode.get("target_valid"))
        predicted = self.predict(context, episode["target_time"])
        target_summary = target["summary"]
        if predicted.ndim == 3 and target_summary.ndim == 2:
            target_summary = target_summary.unsqueeze(1).expand_as(predicted)
        if predicted.shape != target_summary.shape:
            target_summary = target_summary.reshape_as(predicted)
        valid = episode.get("target_valid")
        l_predict = _masked_mean(F.smooth_l1_loss(predicted, target_summary.detach(), reduction="none").mean(dim=-1), valid)
        identity_loss = predicted.new_zeros(())
        if labels and "positive" in labels:
            positive = labels["positive"].bool()
            if positive.any():
                identity_loss = (1.0 - F.cosine_similarity(predicted.reshape(-1, predicted.shape[-1]), target_summary.detach().reshape(-1, target_summary.shape[-1])))[positive.reshape(-1)].mean()
        flat_predicted = predicted.reshape(-1, predicted.shape[-1])
        variance = torch.sqrt(flat_predicted.var(dim=0, unbiased=False) + 1e-4)
        l_var = F.relu(1.0 - variance).mean()
        centered = flat_predicted - flat_predicted.mean(dim=0, keepdim=True)
        cov = centered.transpose(0, 1) @ centered / max(centered.shape[0] - 1, 1)
        l_cov = (cov.fill_diagonal_(0).square().mean()) if cov.ndim == 2 else predicted.new_zeros(())
        l_reg = l_var + l_cov
        total = l_predict + identity_loss + 0.01 * l_reg
        return {"total": total, "prediction": l_predict.detach(), "identity": identity_loss.detach(), "regularization": l_reg.detach(), "representation_std": variance.detach().mean()}

    @staticmethod
    def _tracklet_inputs(tracklet: Mapping[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        appearance = torch.as_tensor(tracklet["appearance"], dtype=torch.float32, device=device)
        if appearance.ndim == 2:
            appearance = appearance.unsqueeze(0)
        boxes = torch.as_tensor(tracklet.get("bboxes", tracklet.get("bbox")), dtype=torch.float32, device=device)
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        if boxes.shape[-1] >= 5:
            boxes = boxes[..., :4]
        if boxes.shape[0] != appearance.shape[1]:
            boxes = boxes[:1].expand(appearance.shape[1], -1)
        times = torch.as_tensor(tracklet.get("time_offsets", tracklet.get("frames", list(range(appearance.shape[1])))), dtype=torch.float32, device=device)
        times = times.reshape(1, -1)
        geometry = boxes.unsqueeze(0)
        return appearance, geometry, times

    @torch.no_grad()
    def score_link(self, left: Mapping[str, Any], right: Mapping[str, Any], mode: str = "forward_only") -> dict[str, Tensor]:
        """Score one legal A→B edge; candidate features are only used here."""
        if mode not in {"forward_only", "bidirectional_inpainting"}:
            raise ValueError("mode must be forward_only or bidirectional_inpainting")
        device = next(self.parameters()).device
        la, lg, lt = self._tracklet_inputs(left, device)
        ra, rg, rt = self._tracklet_inputs(right, device)
        lc = self.encode_context(la, lg, lt)
        rc = self.encode_context(ra, rg, rt)
        lpred = self.predict(lc, (rt - lt[:, -1:]).clamp_min(0.0))
        rtarget = self.encode_target(ra, rg, rt)
        right_summary = rtarget["summary"].unsqueeze(1).expand_as(lpred)
        forward_error = (lpred - right_summary).square().mean()
        anchor_similarity = (lc["identity"] * rc["identity"]).sum(-1).mean()
        error = forward_error
        if mode == "bidirectional_inpainting":
            rpred = self.predict(rc, (lt - rt[:, -1:]).abs())
            ltarget = self.encode_target(la, lg, lt)
            left_summary = ltarget["summary"].unsqueeze(1).expand_as(rpred)
            error = 0.5 * (forward_error + (rpred - left_summary).square().mean())
        return {"edge_score": -error + 0.1 * anchor_similarity, "prediction_error": error, "anchor_similarity": anchor_similarity}

    @torch.no_grad()
    def score_candidates(self, tracklets: list[Mapping[str, Any]], candidate_graph: Any, generator: torch.Generator | None = None, mode: str = "forward_only") -> Tensor:
        scores = []
        edge_index = candidate_graph.edge_index if hasattr(candidate_graph, "edge_index") else candidate_graph["edge_index"]
        for source, target in edge_index.t().tolist():
            scores.append(self.score_link(tracklets[source], tracklets[target], mode)["edge_score"])
        return torch.stack(scores) if scores else torch.empty(0, device=next(self.parameters()).device)

    @torch.no_grad()
    def chain_leave_one_segment_out(self, tracklets: list[Mapping[str, Any]], path: list[int], rounds: int = 2, mode: str = "forward_only") -> dict[str, Any]:
        if len(path) < 3:
            return {"path": path, "refined": False, "reason": "need at least three segments"}
        decisions = []
        current = list(path)
        for round_index in range(rounds):
            middle = len(current) // 2
            left, right = tracklets[current[middle - 1]], tracklets[current[middle + 1]]
            score = self.score_link(left, right, mode)["edge_score"]
            decisions.append({"round": round_index, "masked_segment": current[middle], "context": [current[middle - 1], current[middle + 1]], "score": float(score.cpu())})
            if float(score) < 0:
                current.pop(middle)
        return {"path": path, "refined_path": current, "refined": current != path, "decisions": decisions, "rounds": rounds, "mode": mode}


class PairMetricLinker(nn.Module):
    """Ordinary learned metric with the exact S1 encoder capacity."""

    def __init__(self, *encoder_args: Any, **encoder_kwargs: Any):
        super().__init__()
        self.encoder = TrajectoryEncoder(*encoder_args, **encoder_kwargs)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        return self.encoder(*args, **kwargs)

    def score_pair(self, left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> Tensor:
        return F.cosine_similarity(left["identity"], right["identity"], dim=-1) / self.temperature.clamp_min(1e-3)

    def compute_loss(self, episode: Mapping[str, Tensor], labels: Mapping[str, Tensor]) -> dict[str, Tensor]:
        left = self.encoder(episode["left_appearance"], episode["left_geometry"], episode["left_time"], episode.get("left_valid"))
        right = self.encoder(episode["right_appearance"], episode["right_geometry"], episode["right_time"], episode.get("right_valid"))
        logits = self.score_pair(left, right)
        target = labels["same_identity"].to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return {"total": loss, "metric_bce": loss.detach()}
