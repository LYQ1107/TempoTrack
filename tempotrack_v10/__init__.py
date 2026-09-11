"""Shared V10.3 TempoTrack association core.

Frontend integrations must import this package and keep their own detector,
association embedding, native affinity, and ID bookkeeping unchanged.
"""

from .contract import (
    FrameCollisionError,
    PreAssociationSnapshot,
    SnapshotContractError,
)
from .overlay import (
    OverlayProposal,
    TempoTrackConfig,
    TempoTrackOverlay,
)
from .adapters import COVTrackAssociationDecision, COVTrackTempoAdapter

__all__ = [
    "FrameCollisionError",
    "OverlayProposal",
    "PreAssociationSnapshot",
    "SnapshotContractError",
    "TempoTrackConfig",
    "TempoTrackOverlay",
    "COVTrackAssociationDecision",
    "COVTrackTempoAdapter",
]
