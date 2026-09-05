"""Two-axis implementation/training/evaluation progress registry."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..config import object_hash
from ..registry import METHODS, RESEARCH_SCHEMES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def initial_scheme(name: str) -> dict[str, Any]:
    base = name.split("_", 1)[1] if name.startswith("m0_") or name.startswith("m1_") else name
    if base == "no_offline":
        base = "single_ema"
    if base == "s5_bc":
        base = "s5_rl_edit"
    return {"scheme": name, "implementation": "NOT_STARTED", "training": "NOT_RUN", "evaluation": "NOT_RUN", "run_signature": "", "code_hash": "", "data_hash": None, "checkpoint": None, "metrics": None, "blocking_evidence": None, "next_command": None, "updated_at": _now(), "source_review": "pending", "implemented_files": [], "config": METHODS.get(base).config if base in METHODS else None, "build_status": "NOT_RUN", "train_entry": METHODS.get(base).train_entry if base in METHODS else None, "infer_entry": METHODS.get(base).infer_entry if base in METHODS else None, "trial_status": "NOT_RUN", "full_status": "NOT_RUN", "eval_status": "NOT_RUN", "limitations": []}


def ensure_progress(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"schema_version": 1, "project": "tempotrack_iclr", "updated_at": _now(), "schemes": {}}
    schemes = payload.setdefault("schemes", {})
    for name in RESEARCH_SCHEMES:
        schemes.setdefault(name, initial_scheme(name))
    payload["updated_at"] = _now()
    _atomic_write(payload, path)
    return payload


def load_progress(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return ensure_progress(path)


def update_scheme(path: str | Path, scheme: str, **fields: Any) -> dict[str, Any]:
    payload = ensure_progress(path)
    if scheme not in payload["schemes"]:
        raise ValueError(f"Unknown scheme: {scheme}")
    payload["schemes"][scheme].update(fields)
    payload["schemes"][scheme]["updated_at"] = _now()
    payload["updated_at"] = _now()
    _atomic_write(payload, Path(path))
    return payload


def progress_hash(payload: Mapping[str, Any]) -> str:
    return object_hash(payload)
