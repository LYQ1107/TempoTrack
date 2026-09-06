"""Safe observation and episode storage for TempoTrack research runs.

The inventory/build commands are intentionally usable in a bare Python
environment.  Episode construction imports the torch-backed association
stack lazily instead of making every data utility import CUDA dependencies.
"""

from .manifests import collect_environment_inventory, write_repository_audit
from .observation_store import ObservationLedger

__all__ = [
    "ObservationLedger",
    "EpisodeManifest",
    "build_episode_manifests",
    "collect_environment_inventory",
    "write_repository_audit",
]


def __getattr__(name: str):
    if name in {"EpisodeManifest", "build_episode_manifests"}:
        from .episodes import EpisodeManifest, build_episode_manifests
        return {"EpisodeManifest": EpisodeManifest, "build_episode_manifests": build_episode_manifests}[name]
    raise AttributeError(name)
