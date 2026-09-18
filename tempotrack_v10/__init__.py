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
from .qdic_mgf_loader import QDIC_MGF_STATUS, QDICMGFArtifact, load_qdic_mgf_checkpoint
from .query_distributional_calibrator import QueryDistributionalCalibrator
from .query_mgf_calibrator import QueryMomentGeneratingCalibrator
from .candidate_aware_qdic import DistributionalAuxiliaryQDIC
from .deepset_qdic import DeepSetQDIC
from .dgsa_qdic import DistributionGuidedSetAttentionQDIC
from .candidate_aware_qdic_loader import (
    CANDIDATE_AWARE_STATUS,
    CandidateAwareQDICArtifact,
    load_candidate_aware_checkpoint,
)
from .distributional_identity import DistributionEvidenceEncoder, DistributionIdentityHead
from .distributional_losses import (
    distributional_ranking_loss,
    hard_negative_margin_loss,
    listwise_group_loss,
)
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
    "QDIC_MGF_STATUS",
    "QDICMGFArtifact",
    "QueryDistributionalCalibrator",
    "QueryMomentGeneratingCalibrator",
    "DistributionEvidenceEncoder",
    "DistributionIdentityHead",
    "DistributionalAuxiliaryQDIC",
    "DeepSetQDIC",
    "DistributionGuidedSetAttentionQDIC",
    "CANDIDATE_AWARE_STATUS",
    "CandidateAwareQDICArtifact",
    "load_candidate_aware_checkpoint",
    "distributional_ranking_loss",
    "hard_negative_margin_loss",
    "listwise_group_loss",
    "load_qdic_checkpoint",
    "load_qdic_mgf_checkpoint",
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
