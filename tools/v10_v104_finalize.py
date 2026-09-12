#!/usr/bin/env python3
"""Finalize V10.4 only after every downstream receipt is terminal and valid."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


REPO = Path("/data2/usr_for_deadline/tempotrack_v104_search_hardening")
V10_ROOT = Path("/data2/usr_for_deadline/tempotrack_v10_unified")
DOWNSTREAM_STATE = V10_ROOT / "v104_downstream_supervisor" / "state.json"
MASA_STATE = V10_ROOT / "v104_masa_downstream_supervisor" / "state.json"
STATE_ROOT = V10_ROOT / "v104_finalize_supervisor"


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def atomic_json(path: Path, value):
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
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = STATE_ROOT / "state.json"
    state = read_json(state_path) or {"schema_version": 1, "status": "RUNNING", "created_at": time.time()}
    state["pid"] = os.getpid()
    while True:
        downstream = read_json(DOWNSTREAM_STATE) or {}
        masa = read_json(MASA_STATE) or {}
        downstream_done = downstream.get("status") == "COMPLETED"
        masa_done = masa.get("status") == "COMPLETED"
        state.update({
            "status": "RUNNING",
            "current_stage": "WAIT_RESULTS",
            "next_action": f"wait downstream/MASA: {downstream_done}/{masa_done}",
            "downstream_state": str(DOWNSTREAM_STATE),
            "masa_state": str(MASA_STATE),
            "heartbeat": time.time(),
        })
        atomic_json(state_path, state)
        if downstream_done and masa_done:
            break
        time.sleep(30)

    report = REPO / "reports/tempotrack_v10/FINAL_V10_4_FULL_SEARCH.md"
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "LD_PRELOAD": env.get("LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0"),
    })
    command = [
        "/home/lwr/anaconda3/envs/masaenv/bin/python",
        str(REPO / "tools/v10_v104_generate_report.py"),
        "--output", str(report),
    ]
    result = subprocess.run(command, cwd=str(REPO), env=env, capture_output=True, text=True)
    state["report_command"] = command
    state["report_stdout"] = result.stdout[-2000:]
    state["report_stderr"] = result.stderr[-2000:]
    if result.returncode != 0 or not report.is_file():
        state.update({"status": "BLOCKED", "current_stage": "REPORT", "next_action": "inspect report generator"})
        atomic_json(state_path, state)
        return 2

    check = subprocess.run(["git", "-C", str(REPO), "diff", "--check"], capture_output=True, text=True)
    state["diff_check"] = {"returncode": check.returncode, "stderr": check.stderr[-2000:]}
    if check.returncode != 0:
        state.update({"status": "BLOCKED", "current_stage": "DIFF_CHECK", "next_action": "inspect whitespace errors"})
        atomic_json(state_path, state)
        return 3

    add = subprocess.run(["git", "-C", str(REPO), "add", "--", "reports/tempotrack_v10/FINAL_V10_4_FULL_SEARCH.md"], capture_output=True, text=True)
    commit = subprocess.run(["git", "-C", str(REPO), "commit", "-m", "Record V10.4 full downstream results"], capture_output=True, text=True)
    state["git_commit_stdout"] = commit.stdout[-2000:]
    state["git_commit_stderr"] = commit.stderr[-2000:]
    if add.returncode != 0 or commit.returncode != 0:
        state.update({"status": "BLOCKED", "current_stage": "GIT_COMMIT", "next_action": "inspect report commit"})
        atomic_json(state_path, state)
        return 4
    push = subprocess.run(["git", "-C", str(REPO), "push", "origin", "HEAD:codex/v104-search-hardening"], capture_output=True, text=True)
    state["git_push_stdout"] = push.stdout[-2000:]
    state["git_push_stderr"] = push.stderr[-2000:]
    if push.returncode != 0:
        state.update({"status": "BLOCKED", "current_stage": "GIT_PUSH", "next_action": "inspect push output"})
        atomic_json(state_path, state)
        return 5
    state.update({"status": "COMPLETED", "current_stage": "COMPLETED", "report": str(report), "report_sha256": __import__('hashlib').sha256(report.read_bytes()).hexdigest(), "completed_at": time.time()})
    atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
