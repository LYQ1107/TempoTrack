#!/usr/bin/env python3
"""Mark pre-feature-contract COV trials as diagnostic-only without deleting data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def mark(root: Path, reason: str) -> dict:
    root = root.resolve()
    receipt_path = root / "receipt.json"
    receipt = {}
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            receipt = {"receipt_parse_error": f"{type(exc).__name__}: {exc}"}
    marker = {
        "schema_version": 1,
        "artifact": "covtrack_trial_selection_status",
        "status": "INVALID_PRE_FEATURE_CONTRACT_FIX",
        "usage": "DIAGNOSTIC_ONLY",
        "reason": reason,
        "trial_id": receipt.get("trial_id", root.name),
        "trial_root": str(root),
        "original_receipt": str(receipt_path),
        "original_receipt_sha256": _sha256(receipt_path),
        "original_status": receipt.get("status"),
        "marked_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(root / "selection_status.json", marker)
    return marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument(
        "--reason",
        default=(
            "pre-fix online reranker feature contract was not checkpoint-bound; "
            "retain outputs as diagnostic evidence only"
        ),
    )
    args = parser.parse_args()
    for root in args.root:
        print(json.dumps(mark(root, args.reason), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
