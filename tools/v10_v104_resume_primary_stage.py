#!/usr/bin/env python3
"""Resume the V10.4 primary supervisor after a repaired stage gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


V10_ROOT = Path("/data2/usr_for_deadline/tempotrack_v10_unified")
STATE_ROOT = V10_ROOT / "v104_persistent_supervisor"
STATE_PATH = STATE_ROOT / "state.json"
GATE = V10_ROOT / "search" / "covtrack_q1_hardened_contract_smoke_FINAL_20260913" / "contract_gate.json"
RECEIPT = STATE_ROOT / "final_validate_repair.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
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
    stage = state.get("stages", {}).get("FINAL_VALIDATE", {})
    if state.get("status") != "BLOCKED" or state.get("current_stage") != "FINAL_VALIDATE":
        raise RuntimeError(f"unexpected primary state: {state.get('status')} {state.get('current_stage')}")
    if not isinstance(stage, dict) or stage.get("status") != "BLOCKED":
        raise RuntimeError(f"unexpected final-validate stage: {stage}")
    if not GATE.is_file():
        raise FileNotFoundError(GATE)
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError(f"final gate is not PASS: {gate.get('status')}")

    stage = dict(stage)
    stage.update(
        {
            "status": "PENDING_RETRY",
            "repair_reason": "validator module-path bootstrap repaired and gate regenerated",
            "gate_sha256": sha256(GATE),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    state.setdefault("stages", {})["FINAL_VALIDATE"] = stage
    state["status"] = "RUNNING"
    state["current_stage"] = "FINAL_VALIDATE"
    state.pop("stopped_at", None)
    state["resume_reason"] = "resume after FINAL_VALIDATE validator repair"
    state["resume_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(STATE_PATH, state)
    atomic_json(
        RECEIPT,
        {
            "schema_version": 1,
            "status": "PASS",
            "repair": "FINAL_VALIDATE_REPO_PATH_BOOTSTRAP",
            "gate": str(GATE),
            "gate_sha256": sha256(GATE),
            "validator": "/data2/usr_for_deadline/tempotrack_v104_search_hardening/tools/v10_validate_q1_contract_smoke.py",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print(json.dumps({"state": str(STATE_PATH), "gate": str(GATE), "receipt": str(RECEIPT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
