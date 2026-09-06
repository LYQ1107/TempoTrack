"""Pure-feature causal replay shared by M0, M1 and backend inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from ..schemas import ObservationBatch
from ..data.tracklet_store import TrackletRecord, TrackletStore
from .fixed_dual import FixedDualMemory
from .predictive_dual import PredictiveDualMemory, build_causal_evidence
from .state import MemoryState


@dataclass
class FrameAssignment:
    frame_index: int
    observation_rows: np.ndarray
    local_ids: np.ndarray
    accepted_mask: np.ndarray
    match_scores: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass
class _ActiveTrack:
    local_id: int
    video_id: int
    rows: list[int]
    first_frame: int
    last_frame: int
    last_bbox: torch.Tensor
    label: int
    state: MemoryState


class FrozenObservationTracker:
    """Causal tracker which never reads images, GT, or backend scores."""

    def __init__(self, config: Mapping[str, Any], memory: FixedDualMemory | PredictiveDualMemory):
        self.config = dict(config)
        self.memory = memory
        self.max_gap = int(self.config.get("max_gap", 90))
        self.match_threshold = float(self.config.get("match_score_thr", self.config.get("match_threshold", 0.5)))
        self.with_cats = bool(self.config.get("with_cats", False))
        self._tracks: dict[int, _ActiveTrack] = {}
        self._finished: dict[int, _ActiveTrack] = {}
        self._next_id = 0
        self._video_id = -1
        self._assignments: list[FrameAssignment] = []

    def reset(self, video: Mapping[str, Any] | int) -> None:
        self._tracks.clear()
        self._finished.clear()
        self._next_id = 0
        self._assignments.clear()
        self._video_id = int(video.get("video_id", -1) if isinstance(video, Mapping) else video)

    @staticmethod
    def _cosine(query: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        return F.normalize(query, dim=-1) @ F.normalize(prototype, dim=-1).t()

    def _score(self, embeddings: torch.Tensor, tracks: list[_ActiveTrack], labels: torch.Tensor) -> torch.Tensor:
        if not tracks:
            return embeddings.new_empty((len(embeddings), 0))
        fast = torch.stack([track.state.fast for track in tracks])
        slow = torch.stack([track.state.slow for track in tracks])
        if isinstance(self.memory, FixedDualMemory) and self.memory.mode == "single_ema":
            score = self._cosine(embeddings, fast)
        else:
            fast_cos = self._cosine(embeddings, fast)
            slow_cos = self._cosine(embeddings, slow)
            scale = float(getattr(self.memory, "logit_scale", self.config.get("logit_scale", 10.0)))
            fast_logits = scale * fast_cos
            slow_logits = scale * slow_cos
            fast_bi = (fast_logits.softmax(dim=1) + fast_logits.softmax(dim=0)) / 2
            slow_bi = (slow_logits.softmax(dim=1) + slow_logits.softmax(dim=0)) / 2
            score = torch.where(fast_bi >= self.match_threshold, fast_bi, torch.maximum(fast_bi, slow_bi))
            score = (score + torch.maximum(fast_cos, slow_cos)) / 2
        if self.with_cats:
            track_labels = torch.as_tensor([track.label for track in tracks], device=labels.device)
            score = score.masked_fill(labels[:, None] != track_labels[None, :], -torch.inf)
        return score

    def step(self, frame: ObservationBatch) -> FrameAssignment:
        if len(frame.keys) and self._video_id not in {-1, int(frame.video_ids[0]) if frame.video_ids is not None else self._video_id}:
            raise ValueError("FrozenObservationTracker received a different video without reset")
        if frame.video_ids is not None and len(frame.video_ids):
            self._video_id = int(frame.video_ids[0])
        frame_index = int(frame.frame_indices[0]) if len(frame.frame_indices) else (int(frame.frame_index) if getattr(frame, "frame_index", None) is not None else (self._assignments[-1].frame_index + 1 if self._assignments else 0))
        rows = np.asarray(getattr(frame, "rows", np.arange(len(frame.keys))), dtype=np.int64)
        embeddings = torch.as_tensor(np.asarray(frame.appearance), dtype=torch.float32)
        boxes = torch.as_tensor(np.asarray(frame.bboxes_xyxy), dtype=torch.float32)
        labels = torch.as_tensor(np.asarray(frame.category_ids), dtype=torch.long)
        local_ids = np.full(len(frame.keys), -1, dtype=np.int64)
        accepted = np.zeros(len(frame.keys), dtype=bool)
        score_matrix = self._score(embeddings, list(self._tracks.values()), labels)
        tracks = list(self._tracks.values())
        track_position = {track.local_id: position for position, track in enumerate(tracks)}
        if len(tracks) and len(embeddings):
            from scipy.optimize import linear_sum_assignment

            score_cpu = score_matrix.detach().cpu().numpy()
            cost = np.where(np.isfinite(score_cpu), -score_cpu, 1e9)
            det_indices, track_indices = linear_sum_assignment(cost)
            for det_index, track_index in zip(det_indices.tolist(), track_indices.tolist()):
                if not np.isfinite(score_cpu[det_index, track_index]) or score_cpu[det_index, track_index] < self.match_threshold:
                    continue
                track = tracks[track_index]
                if frame_index - track.last_frame > self.max_gap:
                    continue
                local_ids[det_index] = track.local_id
                accepted[det_index] = True
        # Births are applied only after the whole frame's matching decision.
        for index in np.flatnonzero(local_ids < 0).tolist():
            local_ids[index] = self._next_id
            self._next_id += 1
        # Update states after assignment.  Unmatched tracks are retained until
        # max_gap so zero-detection frames advance their lifecycle.
        by_id = {track.local_id: track for track in tracks}
        for index, local_id in enumerate(local_ids.tolist()):
            z = F.normalize(embeddings[index], dim=-1)
            if local_id in by_id:
                track = by_id[local_id]
                old_bbox = track.last_bbox
                if isinstance(self.memory, PredictiveDualMemory):
                    old_state = track.state
                    fast = old_state.fast
                    slow = old_state.slow
                    position = track_position.get(track.local_id, -1)
                    raw_margin = float(score_matrix[index, position].item()) if position >= 0 and score_matrix.shape[1] else 0.0
                    image_width = max(float(frame.image_widths[index]) if frame.image_widths is not None else 1.0, 1.0)
                    image_height = max(float(frame.image_heights[index]) if frame.image_heights is not None else 1.0, 1.0)
                    old_width = max(float(old_bbox[2] - old_bbox[0]), 1e-6)
                    old_height = max(float(old_bbox[3] - old_bbox[1]), 1e-6)
                    current_width = max(float(boxes[index, 2] - boxes[index, 0]), 1e-6)
                    current_height = max(float(boxes[index, 3] - boxes[index, 1]), 1e-6)
                    geometry_delta = torch.stack(((boxes[index, 0] + boxes[index, 2]) / (2 * image_width) - (old_bbox[0] + old_bbox[2]) / (2 * image_width), (boxes[index, 1] + boxes[index, 3]) / (2 * image_height) - (old_bbox[1] + old_bbox[3]) / (2 * image_height), torch.log(torch.as_tensor(current_width / old_width, dtype=z.dtype)), torch.log(torch.as_tensor(current_height / old_height, dtype=z.dtype))))
                    evidence = build_causal_evidence(old_state, z, torch.as_tensor(float(frame_index - track.last_frame), dtype=z.dtype), torch.as_tensor(raw_margin, dtype=z.dtype), geometry_delta, torch.as_tensor(float(frame_index - track.first_frame), dtype=z.dtype), missing_margin=not bool(accepted[index]))
                    history = torch.cat((old_state.fast, old_state.slow), dim=-1)
                    track.state, diagnostics = self.memory.update(old_state, z, history, evidence, frame_index, bbox=boxes[index])
                else:
                    track.state, diagnostics = self.memory.update(track.state, z, torch.as_tensor(float(accepted[index]), dtype=z.dtype), frame_index)
                track.rows.append(int(rows[index]) if len(rows) > index else int(index))
                track.last_frame = frame_index
                track.last_bbox = boxes[index].detach().clone()
                track.label = int(labels[index])
            else:
                if isinstance(self.memory, PredictiveDualMemory):
                    state = self.memory.initialize(z, frame_index)
                else:
                    state = self.memory.initialize(z, frame_index)
                track = _ActiveTrack(local_id, self._video_id, [int(rows[index]) if len(rows) > index else int(index)], frame_index, frame_index, boxes[index].detach().clone(), int(labels[index]), state)
                self._tracks[local_id] = track
        expired = [local_id for local_id, track in self._tracks.items() if frame_index - track.last_frame > self.max_gap]
        for local_id in expired:
            track = self._tracks.pop(local_id)
            track.state = track.state.detach()
            self._finished[local_id] = track
        assignment = FrameAssignment(frame_index, rows, local_ids, accepted, score_matrix.detach().cpu().numpy() if score_matrix.numel() else np.empty((len(rows), 0), dtype=np.float32), {"active_tracks": len(self._tracks), "births": int((~accepted).sum()), "zero_detection_frame": len(rows) == 0})
        self._assignments.append(assignment)
        return assignment

    def finalize(self) -> TrackletStore:
        records = []
        all_tracks = {**self._finished, **self._tracks}
        for local_id, track in sorted(all_tracks.items()):
            records.append(TrackletRecord(local_id, track.video_id, list(track.rows), track.first_frame, track.last_frame, True, None, None))
        store = TrackletStore(records)
        return store

    @property
    def assignments(self) -> list[FrameAssignment]:
        return list(self._assignments)
