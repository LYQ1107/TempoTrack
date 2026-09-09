"""Typed diagnostics shared by native and replayed association.

The tracker deliberately keeps the diagnostic tensors on the device only for
the current frame.  ``score_matrix`` is optional because retaining every
pairwise matrix for a full TAO run is unnecessarily expensive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class AssociationTrace:
    frame_id: int
    ids: Tensor
    accepted: Tensor
    accepted_score: Tensor
    detection_margin: Tensor
    assigned_memo_index: Tensor
    score_matrix: Optional[Tensor] = None
    fast_score: Optional[Tensor] = None
    slow_score: Optional[Tensor] = None

