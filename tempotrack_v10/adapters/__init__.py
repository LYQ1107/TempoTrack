"""Thin frontend adapters for the shared V10 association overlay."""

from .ovtrack import (
    OVTrackTempoAdapter,
    OVTrackTempoConfig,
    load_ovtrack_tempo_config,
)

__all__ = [
    "OVTrackTempoAdapter",
    "OVTrackTempoConfig",
    "load_ovtrack_tempo_config",
]
