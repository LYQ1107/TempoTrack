"""Replay the native cache through the exact official association core."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from mmengine.structures import InstanceData

from masa.models.tracker.masa_tao_tracker import MasaTaoTracker
from ..data.native_observation_recorder import NativeFrameObservation


class OfficialMasaReplay:
    def __init__(self, tracker_cfg: Mapping[str, Any] | None = None, *, device: str = "cpu") -> None:
        config = dict(tracker_cfg or {})
        config.pop("type", None)
        self.device = torch.device(device)
        self.tracker = MasaTaoTracker(**config)

    def reset(self) -> None:
        self.tracker.reset()

    @staticmethod
    def _instance(data: NativeFrameObservation, output: InstanceData) -> InstanceData:
        return output

    @property
    def last_trace(self):
        return self.tracker.last_association_trace

    def step(self, frame: NativeFrameObservation) -> InstanceData:
        bboxes = torch.from_numpy(frame.boxes_xyxy).to(self.device, dtype=torch.float32)
        labels = torch.from_numpy(frame.labels).to(self.device, dtype=torch.long)
        scores = torch.from_numpy(frame.scores).to(self.device, dtype=torch.float32)
        embeds = torch.from_numpy(frame.embeddings_raw).to(self.device, dtype=torch.float32)
        return self.tracker.associate_precomputed(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
            embeds=embeds,
            frame_id=int(frame.frame_id),
        )

