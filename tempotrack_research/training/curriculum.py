"""Gap-bucket curriculum with deterministic epoch state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GapCurriculum:
    boundaries: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

    def active_max_bucket(self, progress: float) -> int:
        progress = min(max(float(progress), 0.0), 1.0)
        return min(len(self.boundaries) - 1, sum(progress >= boundary for boundary in self.boundaries))
