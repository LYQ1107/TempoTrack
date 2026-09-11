"""Thin frontend adapters for the shared V10 association overlay."""

from .covtrack import COVTrackAssociationDecision, COVTrackTempoAdapter

from .masa import (
    MasaPreAssociationDecision,
    MasaPreAssociationTrace,
    MasaTaoPreAssociationAdapter,
    install_masa_overlay,
)
from .ovtrack_plus import (
    CORE_SHA_V10_FULL,
    OVTrackPlusDecision,
    OVTrackPlusTempoAdapter,
)

from .ovtrack import (
    OVTrackTempoAdapter,
    OVTrackTempoConfig,
    load_ovtrack_tempo_config,
)

__all__ = [
    "MasaPreAssociationDecision",
    "MasaPreAssociationTrace",
    "MasaTaoPreAssociationAdapter",
    "install_masa_overlay",
    "COVTrackAssociationDecision",
    "COVTrackTempoAdapter",
    "OVTrackTempoAdapter",
    "OVTrackTempoConfig",
    "load_ovtrack_tempo_config",
    "CORE_SHA_V10_FULL",
    "OVTrackPlusDecision",
    "OVTrackPlusTempoAdapter",
]
