"""Mutable tracklet grouping that references ledger rows, not copied images."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np

from ..schemas import Tracklet


@dataclass
class TrackletRecord:
    local_id: int
    video_id: int
    observation_rows: list[int]
    first_frame: int
    last_frame: int
    active: bool = True
    recent_prototype_row: int | None = None
    anchor_prototype_row: int | None = None


class TrackletStore:
    def __init__(self, records: Iterable[TrackletRecord] = ()):
        self.records = list(records)

    def validate(self, frame_indices: np.ndarray) -> None:
        seen: set[int] = set()
        for tracklet in self.records:
            rows = list(tracklet.observation_rows)
            if rows != sorted(rows, key=lambda row: int(frame_indices[row])):
                raise ValueError(f"tracklet {tracklet.local_id} is not time ordered")
            for row in rows:
                if row in seen:
                    raise ValueError(f"observation row {row} assigned more than once")
                seen.add(row)

    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]

    @classmethod
    def from_assignments(cls, video_ids: np.ndarray, frame_indices: np.ndarray, assignments: np.ndarray) -> "TrackletStore":
        groups: dict[tuple[int, int], list[int]] = {}
        for row, assignment in enumerate(assignments.tolist()):
            if int(assignment) < 0:
                continue
            groups.setdefault((int(video_ids[row]), int(assignment)), []).append(row)
        records = []
        for (video_id, local_id), rows in sorted(groups.items()):
            rows.sort(key=lambda row: int(frame_indices[row]))
            records.append(TrackletRecord(local_id, video_id, rows, int(frame_indices[rows[0]]), int(frame_indices[rows[-1]])))
        return cls(records)
