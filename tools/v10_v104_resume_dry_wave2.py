#!/usr/bin/env python3
"""Resume the primary supervisor after a failed Wave2 contract dry run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time


STATE_PATH = Path("/data2/usr_for_deadline/tempotrack_v10_unified/v104_persistent_supervisor/state.json")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
    finally:
        try:
            os.unlink(raw)
        except FileNotFoundError:
            pass


def main() -> int:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get("status") != "BLOCKED" or state.get("current_stage") != "DRY_WAVE2":
        raise RuntimeError(
            f"unexpected primary state: {state.get('status')} {state.get('current_stage')}"
        )
    stage = state.setdefault("stages", {}).setdefault("DRY_WAVE2", {})
    if stage.get("status") != "BLOCKED":
        raise RuntimeError(f"unexpected dry stage: {stage.get('status')}")
    old_root = Path(str(stage.get("output_root", "")))
    if not str(old_root):
        raise RuntimeError("failed dry stage has no output root")
    retry_root = old_root.with_name(f"{old_root.name}__retry{int(stage.get('attempts', 1)):02d}")
    while retry_root.exists():
        retry_root = retry_root.with_name(f"{retry_root.name}x")
    stage.update(
        {
            "status": "PENDING_RETRY",
            "retry_of": str(old_root),
            "output_root": str(retry_root),
            "repair_reason": "search validator now reads nested hardened contract gate schema",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    state["status"] = "RUNNING"
    state["current_stage"] = "DRY_WAVE2"
    state["resume_reason"] = "resume dry Wave2 after nested contract-gate schema repair"
    state["resume_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state.pop("stopped_at", None)
    _atomic_json(STATE_PATH, state)
    print(json.dumps({"old_root": str(old_root), "retry_root": str(retry_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
