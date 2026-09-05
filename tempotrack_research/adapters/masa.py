"""Optional MASA adapter with deliberately local imports."""

from __future__ import annotations

import importlib.util
from typing import Any, Mapping


def legacy_runtime_available() -> dict[str, Any]:
    modules = {name: importlib.util.find_spec(name) is not None for name in ("torch", "mmcv", "mmengine", "mmdet")}
    return {"available": all(modules.values()), "modules": modules}


def build_legacy_tracker_config(frontend: str, tracker_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "MasaOVMOTTracker", "frontend": frontend, **dict(tracker_kwargs or {})}


def run_legacy(*args: Any, **kwargs: Any) -> Any:
    """Import the legacy stack only when a caller explicitly runs it."""

    try:
        from masa.models.mot.masa import MASA  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on external stack
        raise RuntimeError("MASA legacy runtime is unavailable; activate the verified masa environment") from exc
    raise NotImplementedError("Use the repository's existing MASA inference entrypoint for detector execution")
