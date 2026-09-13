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
from .qdic_features import QDIC_FEATURE_NAMES, QDIC_RAW_DIM, QDIC_RECENT_K
from .qdic_loader import QDIC_STATUS, QDICV11Artifact, load_qdic_checkpoint
from .query_distributional_calibrator import QueryDistributionalCalibrator
from .adapters import (
    COVTrackAssociationDecision,
    COVTrackTempoAdapter,
    MasaPreAssociationDecision,
    MasaPreAssociationTrace,
    MasaTaoPreAssociationAdapter,
    OVTrackTempoAdapter,
    OVTrackTempoConfig,
    install_masa_overlay,
    load_ovtrack_tempo_config,
    CORE_SHA_V10_FULL,
    OVTrackPlusDecision,
    OVTrackPlusTempoAdapter,
)

__all__ = [
    "FrameCollisionError",
    "OverlayProposal",
    "PreAssociationSnapshot",
    "SnapshotContractError",
    "TempoTrackConfig",
    "TempoTrackOverlay",
    "QDIC_FEATURE_NAMES",
    "QDIC_RAW_DIM",
    "QDIC_RECENT_K",
    "QDIC_STATUS",
    "QDICV11Artifact",
    "QueryDistributionalCalibrator",
    "load_qdic_checkpoint",
    "COVTrackAssociationDecision",
    "COVTrackTempoAdapter",
    "MasaPreAssociationDecision",
    "MasaPreAssociationTrace",
    "MasaTaoPreAssociationAdapter",
    "OVTrackTempoAdapter",
    "OVTrackTempoConfig",
    "install_masa_overlay",
    "load_ovtrack_tempo_config",
    "CORE_SHA_V10_FULL",
    "OVTrackPlusDecision",
    "OVTrackPlusTempoAdapter",
]
