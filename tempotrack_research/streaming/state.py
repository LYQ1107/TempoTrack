from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..memory.identity_history import IdentityHistory, MemoryObservation


class IdentityState(Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    ARCHIVED = "archived"


@dataclass
class DormantIdentity:
    identity_id: int
    video_id: int
    history: IdentityHistory
    last_frame: int
    last_bbox: np.ndarray
    slow_prototype: np.ndarray


@dataclass
class TentativeTrack:
    frontend_track_id: int
    video_id: int
    rows: list[int] = field(default_factory=list)
    observation_uids: list[str] = field(default_factory=list)
    observations: list[MemoryObservation] = field(default_factory=list)
    first_frame: int = -1
    last_frame: int = -1
    decision_frame: int | None = None
