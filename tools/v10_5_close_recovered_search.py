#!/usr/bin/env python3
"""Close a bounded V10.4 search after immutable shard recovery.

This command is deliberately separate from the live controller.  It refuses
to alter the state while the controller PID is alive, then replaces only the
logical full-result index with PASS rows whose prediction/summary artifacts
have matching hashes.  Original partial results and receipts remain intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import time
from typing import Any, Mapping


DEFAULT_ROOT = Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_v104_best20h_20260914")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def valid_full(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "PASS":
        return False
    for key in ("prediction", "summary"):
        path = Path(str(row.get(key, "")))
        if not path.is_file() or row.get(key + "_sha256") != sha256(path):
            return False
    return isinstance(row.get("metrics"), Mapping)


def candidate_id(row: Mapping[str, Any]) -> str:
    spec = row.get("spec") or row.get("candidate", {}).get("spec", {})
    return str(spec.get("trial_id", row.get("candidate", {}).get("spec", {}).get("trial_id", "")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--controller-pid", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    state_path = root / "20h_search_state.json"
    state = read_json(state_path)
    pid = args.controller_pid or int(state.get("pid") or 0)
    if pid > 0 and Path(f"/proc/{pid}").exists():
        raise RuntimeError(f"CONTROLLER_STILL_LIVE:{pid}")

    indexed: dict[str, dict[str, Any]] = {}
    full_path = root / "full_results.json"
    if full_path.is_file():
        old = read_json(full_path)
        for row in [old.get("anchor"), *old.get("new_results", [])]:
            if isinstance(row, Mapping) and valid_full(row):
                indexed[candidate_id(row)] = dict(row)
    recovery_rows = []
    for path in sorted((root / "full").glob("*__recovery*/full_result.json")):
        row = read_json(path)
        if valid_full(row):
            recovery_rows.append(dict(row))
            indexed[candidate_id(row)] = dict(row)

    requested = []
    for spec in state.get("new_full_candidates", []):
        if isinstance(spec, Mapping) and str(spec.get("trial_id", "")):
            requested.append(str(spec["trial_id"]))
    missing = [item for item in requested if item not in indexed]
    if missing:
        raise RuntimeError("RECOVERY_RESULTS_MISSING:" + ",".join(missing))

    anchor = None
    if full_path.is_file():
        old = read_json(full_path)
        if isinstance(old.get("anchor"), Mapping) and valid_full(old["anchor"]):
            anchor = old["anchor"]
    new_results = [indexed[item] for item in requested]
    atomic_json(full_path, {
        "anchor": anchor,
        "new_results": new_results,
        "recovery_rows": recovery_rows,
        "completion_mode": "IMMUTABLE_SHARD_RECOVERY",
        "created_at": time.time(),
    })
    state.update({
        "status": "COMPLETED",
        "completion_mode": "IMMUTABLE_SHARD_RECOVERY",
        "recovery_results": [str(root / "full" / (candidate_id(row) + "__recovery")) for row in recovery_rows],
        "incomplete_jobs": [],
        "incomplete_full": [],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_json(state_path, state)
    print(json.dumps({"status": "COMPLETED", "full_results": str(full_path), "candidates": requested, "recovered": [candidate_id(row) for row in recovery_rows]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
