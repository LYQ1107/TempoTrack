#!/usr/bin/env python3
"""Prepare a resumable retry after the recorded COV native-control setup failure.

This changes only the durable V10.4 supervisor state.  The failed attempt is
kept intact and the retry is assigned a new immutable output root.
"""

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
OLD_ROOT = V10_ROOT / "search" / "covtrack_native_control_final_20260913"
RETRY_ROOT = V10_ROOT / "search" / "covtrack_native_control_final_20260913__retry01"
RECEIPT = STATE_ROOT / "native_control_setup_repair.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(raw)
        except FileNotFoundError:
            pass


def main() -> int:
    if RETRY_ROOT.exists():
        raise RuntimeError(f"refusing to reuse retry root: {RETRY_ROOT}")
    state = read_json(STATE_PATH)
    native = state.get("stages", {}).get("NATIVE_CONTROL", {})
    if state.get("status") != "BLOCKED" or state.get("current_stage") != "NATIVE_CONTROL":
        raise RuntimeError(f"unexpected primary state: {state.get('status')} {state.get('current_stage')}")
    if not isinstance(native, dict) or native.get("status") != "BLOCKED":
        raise RuntimeError(f"native-control stage is not the recorded failure: {native}")
    failed_receipt = OLD_ROOT / "native_control" / "receipt.json"
    if not failed_receipt.is_file():
        raise FileNotFoundError(failed_receipt)
    failed = read_json(failed_receipt)
    if failed.get("status") != "FAILED":
        raise RuntimeError(f"unexpected failed receipt status: {failed.get('status')}")

    source = Path("/data2/usr_for_deadline/COVTrack_9b0ced_final_clean")
    setup = {
        "data": str(source / "data"),
        "saved_models": str(source / "saved_models"),
        "data_target": os.readlink(source / "data") if (source / "data").is_symlink() else None,
        "saved_models_target": os.readlink(source / "saved_models") if (source / "saved_models").is_symlink() else None,
    }
    class_file = source / "data" / "lvis" / "annotations" / "lvis_classes_v1.txt"
    prompt_file = source / "saved_models" / "pretrained_models" / "detpro_prompt.pt"
    if not class_file.is_file() or not prompt_file.is_file():
        raise FileNotFoundError(f"COV runtime assets missing: {class_file} {prompt_file}")

    native = dict(native)
    native.update(
        {
            "status": "PENDING_RETRY",
            "output_root": str(RETRY_ROOT),
            "retry_of": str(OLD_ROOT),
            "repair_reason": "COV clean checkout lacked untracked runtime data/saved_models links; failed attempt preserved",
            "repair_setup": setup,
            "failed_receipt_sha256": sha256(failed_receipt),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    state.setdefault("stages", {})["NATIVE_CONTROL"] = native
    state["status"] = "RUNNING"
    state["current_stage"] = "NATIVE_CONTROL"
    state.pop("stopped_at", None)
    state["resume_reason"] = "resume after environment-only runtime asset repair"
    state["resume_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(STATE_PATH, state)

    atomic_json(
        RECEIPT,
        {
            "schema_version": 1,
            "status": "PASS",
            "repair": "COV_NATIVE_CONTROL_RUNTIME_ASSET_LINKS",
            "failed_attempt": str(failed_receipt),
            "failed_attempt_sha256": sha256(failed_receipt),
            "retry_output_root": str(RETRY_ROOT),
            "source_checkout": str(source),
            "source_git_commit": "9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b",
            "runtime_assets": {
                "class_file": str(class_file),
                "class_file_sha256": sha256(class_file),
                "prompt_file": str(prompt_file),
                "prompt_file_sha256": sha256(prompt_file),
                **setup,
            },
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print(json.dumps({"state": str(STATE_PATH), "receipt": str(RECEIPT), "retry_root": str(RETRY_ROOT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
