"""Behaviorally explicit dual-timescale MASA tracker.

This class owns only the frontend prototype update and assignment choice.  The
detector observations, filtering, lifecycle, and replay entry point remain the
ones implemented by :class:`MasaTaoTracker`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmdet.registry import MODELS

from .association_types import AssociationTrace
from .masa_tao_tracker import MasaTaoTracker


@MODELS.register_module()
class MasaDualTimescaleTracker(MasaTaoTracker):
    def __init__(
        self,
        *args: Any,
        alpha_fast: float = 0.70,
        alpha_slow: float = 0.15,
        fast_accept_threshold: float = 0.60,
        dual_logit_scale: float = 12.0,
        assignment_mode: str = "official_greedy",
        **kwargs: Any,
    ) -> None:
        if not 0.0 < alpha_slow < alpha_fast < 1.0:
            raise ValueError("require 0 < alpha_slow < alpha_fast < 1")
        if assignment_mode not in {"official_greedy", "hungarian_legacy"}:
            raise ValueError(f"unknown assignment_mode={assignment_mode}")
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow = float(alpha_slow)
        self.fast_accept_threshold = float(fast_accept_threshold)
        self.dual_logit_scale = float(dual_logit_scale)
        self.assignment_mode = assignment_mode
        self._last_fast_score: Tensor | None = None
        self._last_slow_score: Tensor | None = None
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._last_fast_score = None
        self._last_slow_score = None

    def update(
        self,
        ids: Tensor,
        bboxes: Tensor,
        embeds: Tensor,
        labels: Tensor,
        scores: Tensor,
        frame_id: int,
    ) -> None:
        tracklet_inds = ids > -1
        if embeds.ndim == 2 and embeds.shape[1] > 0:
            self._embedding_dim = int(embeds.shape[1])
            self._last_device = embeds.device
        for track_id, bbox, embed, label, score in zip(
            ids[tracklet_inds],
            bboxes[tracklet_inds],
            embeds[tracklet_inds],
            labels[tracklet_inds],
            scores[tracklet_inds],
        ):
            track_id = int(track_id)
            z = F.normalize(embed, p=2, dim=0)
            if track_id in self.tracks:
                state = self.tracks[track_id]
                fast = F.normalize(
                    (1.0 - self.alpha_fast) * state["embed_fast"] + self.alpha_fast * z,
                    p=2,
                    dim=0,
                )
                slow = F.normalize(
                    (1.0 - self.alpha_slow) * state["embed_slow"] + self.alpha_slow * z,
                    p=2,
                    dim=0,
                )
                state["embed_fast"] = fast
                state["embed_slow"] = slow
                state["embed"] = fast
                state["bbox"] = bbox
                state["last_frame"] = int(frame_id)
                state["label"] = label
                state["score"] = score
            else:
                self.tracks[track_id] = {
                    "bbox": bbox,
                    "embed_fast": z.clone(),
                    "embed_slow": z.clone(),
                    "embed": z.clone(),
                    "label": label,
                    "score": score,
                    "last_frame": int(frame_id),
                }
        invalid = [
            key for key, value in self.tracks.items()
            if int(frame_id) - int(value["last_frame"]) >= self.memo_tracklet_frames
        ]
        for key in invalid:
            self.tracks.pop(key, None)

    def _ordered_memory(self) -> dict[str, Tensor]:
        device = self._last_device
        dim = self._embedding_dim
        if not self.tracks:
            return {
                "bboxes": torch.empty((0, 4), device=device),
                "labels": torch.empty((0,), dtype=torch.long, device=device),
                "embeds": torch.empty((0, dim), device=device),
                "embeds_fast": torch.empty((0, dim), device=device),
                "embeds_slow": torch.empty((0, dim), device=device),
                "ids": torch.empty((0,), dtype=torch.long, device=device),
                "frame_ids": torch.empty((0,), dtype=torch.long, device=device),
            }
        rows = list(self.tracks.items())
        return {
            "bboxes": torch.cat([value["bbox"].reshape(1, -1)[:, :4] for _, value in rows]),
            "labels": torch.stack([value["label"].reshape(()) for _, value in rows]).long(),
            "embeds": torch.cat([value["embed"].reshape(1, -1) for _, value in rows]),
            "embeds_fast": torch.cat([value["embed_fast"].reshape(1, -1) for _, value in rows]),
            "embeds_slow": torch.cat([value["embed_slow"].reshape(1, -1) for _, value in rows]),
            "ids": torch.as_tensor([key for key, _ in rows], dtype=torch.long, device=device),
            "frame_ids": torch.as_tensor([value["last_frame"] for _, value in rows], dtype=torch.long, device=device),
        }

    def _prototype_score(self, det_embed: Tensor, proto_embed: Tensor) -> Tensor:
        det_n = F.normalize(det_embed, dim=1)
        proto_n = F.normalize(proto_embed, dim=1)
        cosine = det_n @ proto_n.t()
        logits = self.dual_logit_scale * cosine
        bisoft = 0.5 * (logits.softmax(dim=1) + logits.softmax(dim=0))
        return 0.5 * (bisoft + cosine)

    def _compute_match_scores(
        self,
        embeds: Tensor,
        memory: dict[str, Tensor],
        bboxes: Tensor,
        frame_id: int,
    ) -> Tensor:
        if memory["ids"].numel() == 0:
            self._last_fast_score = embeds.new_empty((embeds.shape[0], 0))
            self._last_slow_score = embeds.new_empty((embeds.shape[0], 0))
            return embeds.new_empty((embeds.shape[0], 0))
        fast = self._prototype_score(embeds, memory["embeds_fast"])
        slow = self._prototype_score(embeds, memory["embeds_slow"])
        scores = torch.where(fast >= self.fast_accept_threshold, fast, torch.maximum(fast, slow))
        if self.max_distance != -1:
            current = torch.full((bboxes.shape[0],), int(frame_id), dtype=torch.long, device=bboxes.device)
            scores = scores * self.compute_distance_mask(bboxes, memory["bboxes"], current, memory["frame_ids"])
        self._last_fast_score = fast
        self._last_slow_score = slow
        return scores

    def _diagnostic_scores(self) -> tuple[Tensor | None, Tensor | None]:
        return self._last_fast_score, self._last_slow_score

    def _assign_matches(self, match_scores: Tensor, memo_ids: Tensor, det_scores: Tensor):
        if self.assignment_mode == "official_greedy":
            return self._assign_official_greedy(match_scores, memo_ids, det_scores)
        return self._assign_hungarian_legacy(match_scores, memo_ids, det_scores)

    def _assign_hungarian_legacy(
        self, match_scores: Tensor, memo_ids: Tensor, det_scores: Tensor
    ) -> tuple[Tensor, AssociationTrace]:
        count, memo_count = match_scores.shape
        ids = torch.full((count,), -1, dtype=torch.long, device=det_scores.device)
        accepted = torch.zeros((count,), dtype=torch.bool, device=det_scores.device)
        accepted_score = torch.full((count,), float("nan"), dtype=match_scores.dtype, device=det_scores.device)
        margin = torch.full((count,), float("nan"), dtype=match_scores.dtype, device=det_scores.device)
        assigned = torch.full((count,), -1, dtype=torch.long, device=det_scores.device)
        if count and memo_count:
            import numpy as np

            cost = match_scores.detach().cpu().numpy().astype(np.float64)
            invalid = cost < float(self.match_score_thr)
            cost[invalid] = 1e6
            rows, cols = linear_sum_assignment(-cost)
            for row, col in zip(rows.tolist(), cols.tolist()):
                value = float(match_scores[row, col].detach().cpu())
                if value >= self.match_score_thr and float(det_scores[row].detach().cpu()) > self.obj_score_thr:
                    ids[row] = memo_ids[col]
                    accepted[row] = True
                    accepted_score[row] = match_scores[row, col]
                    assigned[row] = int(col)
                    if memo_count > 1:
                        values = torch.cat((match_scores[row, :col], match_scores[row, col + 1:]))
                        margin[row] = match_scores[row, col] - values.max()
        return ids, AssociationTrace(
            frame_id=-1,
            ids=ids,
            accepted=accepted,
            accepted_score=accepted_score,
            detection_margin=margin,
            assigned_memo_index=assigned,
            score_matrix=match_scores.detach().clone() if self.debug_association_trace else None,
        )

