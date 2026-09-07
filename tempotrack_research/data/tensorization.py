"""Single train/deploy trajectory tensor contract for V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from ..config import object_hash
from ..schemas import PairInputs, PredictionQuery, SegmentClock, SegmentInputs
from .observation_store import ObservationLedger


@dataclass(frozen=True)
class TransformSpec:
    schema_version: int = 1
    time_unit: str = "frame"
    time_scale: float = 1.0
    geometry_fields: tuple[str, ...] = ("cx_norm", "cy_norm", "log_w_norm", "log_h_norm")
    max_source_tokens: int = 16
    max_target_tokens: int = 8

    def __post_init__(self) -> None:
        if self.time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if self.time_unit not in {"frame", "second"}:
            raise ValueError("time_unit must be frame or second")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "time_unit": self.time_unit,
            "time_scale": float(self.time_scale),
            "geometry_fields": list(self.geometry_fields),
            "max_source_tokens": int(self.max_source_tokens),
            "max_target_tokens": int(self.max_target_tokens),
        }

    def content_hash(self) -> str:
        return object_hash(self.to_dict())


def _valid_indices(valid: Tensor) -> Tensor:
    if valid.ndim != 2:
        raise ValueError("valid must be [B,L]")
    result: list[Tensor] = []
    for row in valid.bool():
        indices = torch.nonzero(row, as_tuple=False).flatten()
        if indices.numel() == 0:
            result.append(torch.tensor(-1, device=valid.device))
        else:
            result.append(indices[-1])
    return torch.stack(result).long()


class TrajectoryTensorizer:
    """Encode segments with relative local time and normalized geometry."""

    def __init__(self, spec: TransformSpec | Mapping[str, Any] | None = None):
        if isinstance(spec, TransformSpec):
            self.spec = spec
        else:
            self.spec = TransformSpec(**dict(spec or {}))

    @property
    def tensor_contract_hash(self) -> str:
        return self.spec.content_hash()

    def select_rows(self, ledger: ObservationLedger, rows: Sequence[int] | np.ndarray, *, role: str = "source") -> np.ndarray:
        """Select a causal prefix without changing the immutable ledger.

        ``role`` is part of the V4 tensor contract.  Source/context segments
        keep the most recent observations while target/event segments keep
        their causal prefix.  The selected rows remain in absolute-time order
        and every caller uses the returned rows for both features and clocks.
        """
        if role not in {"source", "target", "event", "context", "heldout"}:
            raise ValueError(f"unknown tensorization role: {role}")
        rows_array = np.asarray(rows, dtype=np.int64).reshape(-1)
        if rows_array.size == 0:
            raise ValueError("segment must contain at least one observation")
        if np.any(rows_array < 0) or np.any(rows_array >= ledger.row_count):
            raise IndexError("segment row out of bounds")
        limit = self.spec.max_source_tokens if role in {"source", "context"} else self.spec.max_target_tokens
        if limit > 0 and rows_array.size > limit:
            rows_array = rows_array[-limit:] if role in {"source", "context"} else rows_array[:limit]
        return rows_array

    def encode_segment(self, ledger: ObservationLedger, rows: Sequence[int] | np.ndarray, *, role: str = "source") -> tuple[SegmentInputs, SegmentClock]:
        rows_array = self.select_rows(ledger, rows, role=role)
        # Caller controls a segment's causal order, but the contract refuses
        # non-monotonic input instead of silently constructing a different one.
        times_np = np.asarray(ledger.arrays["frame_times"][rows_array], dtype=np.float64)
        if np.any(np.diff(times_np) < 0):
            raise ValueError("segment rows must be sorted by observation time")
        ledger_unit = str(ledger.metadata.get("time_unit", self.spec.time_unit))
        if ledger_unit != self.spec.time_unit:
            raise ValueError(f"ledger time unit {ledger_unit!r} disagrees with tensor contract {self.spec.time_unit!r}")
        batch = ledger.model_batch(rows_array)
        app = torch.as_tensor(np.asarray(batch.appearance, dtype=np.float32), dtype=torch.float32).clone()
        boxes = torch.as_tensor(np.asarray(batch.bboxes_xyxy, dtype=np.float32), dtype=torch.float32).clone()
        widths = torch.as_tensor(np.asarray(ledger.arrays["image_widths"][rows_array], dtype=np.float32)).clamp_min(1e-6)
        heights = torch.as_tensor(np.asarray(ledger.arrays["image_heights"][rows_array], dtype=np.float32)).clamp_min(1e-6)
        x1, y1, x2, y2 = boxes.unbind(-1)
        bw = (x2 - x1).clamp_min(1e-6)
        bh = (y2 - y1).clamp_min(1e-6)
        geometry = torch.stack(((x1 + x2) / (2 * widths), (y1 + y2) / (2 * heights), torch.log(bw / widths), torch.log(bh / heights)), dim=-1)
        # Do time subtraction in float64 before exposing a float32 model
        # feature.  Absolute timestamps are routing metadata only.
        raw_times64 = torch.as_tensor(times_np, dtype=torch.float64)
        local = ((raw_times64 - raw_times64[-1]) / float(self.spec.time_scale)).to(torch.float32)
        valid = torch.ones((len(rows_array),), dtype=torch.bool)
        return SegmentInputs(app, geometry, local, valid), SegmentClock(raw_times64, raw_times64[0], raw_times64[-1], self.spec.time_unit, float(self.spec.time_scale))

    def build_prediction_query(self, source_clock: SegmentClock, target_clocks: Sequence[SegmentClock], target_valid: Tensor | None = None, *, mode: str = "forward_only") -> PredictionQuery:
        if not target_clocks:
            raise ValueError("prediction query needs at least one target clock")
        if mode not in {"forward_only", "bidirectional_inpainting"}:
            raise ValueError(f"unsupported prediction mode: {mode}")
        values = []
        for clock in target_clocks:
            if clock.time_unit != source_clock.time_unit or float(clock.scale) != float(source_clock.scale):
                raise ValueError("source and target clocks use incompatible time contracts")
            times = torch.as_tensor(clock.observation_times, dtype=torch.float64)
            values.append(((times - torch.as_tensor(source_clock.last_time, dtype=torch.float64)) / float(source_clock.scale)).to(torch.float32))
        length = max(int(item.numel()) for item in values)
        result = torch.zeros((len(values), length), dtype=torch.float32)
        valid = torch.zeros((len(values), length), dtype=torch.bool)
        for index, value in enumerate(values):
            result[index, : value.numel()] = value
            valid[index, : value.numel()] = True
        if target_valid is not None:
            valid = valid & target_valid.bool()
        return PredictionQuery(result, valid, mode)

    def build_pair(self, source_ref: tuple[ObservationLedger, Sequence[int]], candidate_refs: Sequence[tuple[ObservationLedger, Sequence[int]]]) -> PairInputs:
        if not candidate_refs:
            raise ValueError("pair requires at least one candidate")
        source, source_clock = self.encode_segment(*source_ref, role="source")
        candidates: list[SegmentInputs] = []
        clocks: list[SegmentClock] = []
        for ledger, rows in candidate_refs:
            segment, clock = self.encode_segment(ledger, rows, role="target")
            candidates.append(segment)
            clocks.append(clock)
        max_len = max(int(item.appearance.shape[0]) for item in candidates)
        dim = int(source.appearance.shape[-1])
        app = source.appearance.new_zeros((len(candidates), max_len, dim))
        geo = source.geometry.new_zeros((len(candidates), max_len, 4))
        times = source.local_time.new_zeros((len(candidates), max_len))
        valid = torch.zeros((len(candidates), max_len), dtype=torch.bool)
        for index, item in enumerate(candidates):
            length = item.appearance.shape[0]
            app[index, :length] = item.appearance
            geo[index, :length] = item.geometry
            times[index, :length] = item.local_time
            valid[index, :length] = item.valid
        candidate_batch = SegmentInputs(app, geo, times, valid)
        query = self.build_prediction_query(source_clock, clocks)
        return PairInputs(source, candidate_batch, query, torch.ones((len(candidates),), dtype=torch.bool))

    def build_inpainting_query(self, context_refs: Sequence[tuple[ObservationLedger, Sequence[int]]], heldout_ref: tuple[ObservationLedger, Sequence[int]]) -> PairInputs:
        """Build the explicit bidirectional held-out query from absolute clocks."""
        if len(context_refs) < 2:
            raise ValueError("inpainting requires at least two context segments")
        context_segments = [self.encode_segment(ledger, rows, role="context") for ledger, rows in context_refs]
        heldout, held_clock = self.encode_segment(heldout_ref[0], heldout_ref[1], role="heldout")
        appearance = torch.cat([item[0].appearance for item in context_segments], dim=0).unsqueeze(0)
        geometry = torch.cat([item[0].geometry for item in context_segments], dim=0).unsqueeze(0)
        local = torch.cat([item[0].local_time for item in context_segments], dim=0).unsqueeze(0)
        valid = torch.cat([item[0].valid for item in context_segments], dim=0).unsqueeze(0)
        source = SegmentInputs(appearance, geometry, local, valid)
        target = SegmentInputs(heldout.appearance.unsqueeze(0), heldout.geometry.unsqueeze(0), heldout.local_time.unsqueeze(0), heldout.valid.unsqueeze(0))
        source_clock = SegmentClock(
            torch.cat([item[1].observation_times for item in context_segments]),
            context_segments[0][1].first_time,
            context_segments[-1][1].last_time,
            context_segments[0][1].time_unit,
            context_segments[0][1].scale,
        )
        query = self.build_prediction_query(source_clock, [held_clock], heldout.valid.unsqueeze(0), mode="bidirectional_inpainting")
        return PairInputs(source, target, query, torch.ones((1,), dtype=torch.bool))

    def encode_graph_node(self, segment: SegmentInputs, clock: SegmentClock, graph_origin: float, transform_spec: TransformSpec | None = None) -> Tensor:
        spec = transform_spec or self.spec
        valid = segment.valid.bool()
        if not bool(valid.any()):
            raise ValueError("graph node cannot be all padding")
        app = segment.appearance[valid]
        geo = segment.geometry[valid]
        times = torch.as_tensor(clock.observation_times, dtype=segment.appearance.dtype, device=segment.appearance.device)
        return torch.cat((app.mean(0), geo[-1], ((times[-1] - float(graph_origin)) / float(spec.time_scale)).reshape(1)))


def padded_segment(segments: Sequence[SegmentInputs]) -> SegmentInputs:
    if not segments:
        raise ValueError("cannot pad zero segments")
    length = max(int(item.appearance.shape[-2]) for item in segments)
    dim = int(segments[0].appearance.shape[-1])
    app = segments[0].appearance.new_zeros((len(segments), length, dim))
    geo = segments[0].geometry.new_zeros((len(segments), length, 4))
    time = segments[0].local_time.new_zeros((len(segments), length))
    valid = torch.zeros((len(segments), length), dtype=torch.bool, device=app.device)
    for index, segment in enumerate(segments):
        n = int(segment.appearance.shape[-2])
        app[index, :n] = segment.appearance
        geo[index, :n] = segment.geometry
        time[index, :n] = segment.local_time
        valid[index, :n] = segment.valid
    return SegmentInputs(app, geo, time, valid)


__all__ = ["TransformSpec", "TrajectoryTensorizer", "padded_segment"]
