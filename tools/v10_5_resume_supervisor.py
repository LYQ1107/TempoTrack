#!/usr/bin/env python3
"""Resume the bounded V10.5 search only after its own controller is terminal.

This supervisor is deliberately conservative: it never signals a live process,
never restarts a RUNNING state whose PID disappeared, and launches at most one
explicit extended-stage resume for a terminal partial/deadline state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def live(pid):
    try:
        return int(pid) > 0 and Path(f"/proc/{int(pid)}").exists()
    except (TypeError, ValueError):
        return False


def log(handle, message):
    handle.write(time.strftime("[%Y-%m-%dT%H:%M:%SZ] ", time.gmtime()) + message + "\n")
    handle.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v104_search_hardening"))
    parser.add_argument("--root", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_v104_best20h_20260914"))
    parser.add_argument("--extend-stage-hours", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.root = args.root.resolve()
    state_path = args.root / "20h_search_state.json"
    marker = args.root / "extended_resume_started.json"
    log_path = args.root / "resume_supervisor.log"
    args.root.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        while True:
            state = read_json(state_path)
            if not isinstance(state, dict):
                time.sleep(args.poll_seconds)
                continue
            status = state.get("status")
            pid = state.get("pid")
            if status == "RUNNING":
                if live(pid):
                    time.sleep(args.poll_seconds)
                    continue
                log(handle, "RUNNING state with missing controller PID; refusing automatic restart")
                return 2
            if status == "COMPLETED":
                log(handle, "controller completed; no resume needed")
                return 0
            if status not in ("PARTIAL_FAILURE_OR_DEADLINE", "FAILED"):
                time.sleep(args.poll_seconds)
                continue
            if marker.is_file():
                log(handle, "resume marker already exists; exiting")
                return 0
            if live(pid):
                log(handle, f"terminal status but controller PID {pid} still exists; waiting")
                time.sleep(args.poll_seconds)
                continue
            command = [
                "/home/lwr/anaconda3/envs/masaenv/bin/python",
                str(args.repo / "tools/v10_5_best_search.py"),
                "--repo", str(args.repo),
                "--root", str(args.root),
                "--gpus", "0,1,2,3,4,5,6,7,8,9",
                "--hours", "20",
                "--stage-a-hours", "4",
                "--final-reserve-hours", "1",
                "--full-shards", "10",
                "--poll-seconds", "20",
                "--min-ram-gib", "24",
                "--extend-stage-hours", str(args.extend_stage_hours),
                "--resume",
            ]
            marker.write_text(json.dumps({"status": "STARTING", "command": command, "started_at": time.time()}, indent=2) + "\n", encoding="utf-8")
            env = os.environ.copy()
            env.setdefault("LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0")
            log(handle, "launching extended resume: " + " ".join(command))
            with log_path.open("a", encoding="utf-8") as child_log:
                child = subprocess.Popen(command, cwd=str(args.repo), env=env, stdout=child_log, stderr=subprocess.STDOUT, start_new_session=True)
            finalizer_log = (args.root / "finalizer_resume.log").open("a", encoding="utf-8")
            subprocess.Popen([
                "/home/lwr/anaconda3/envs/masaenv/bin/python",
                str(args.repo / "tools/v10_5_finalize_report.py"),
                "--root", str(args.root), "--repo", str(args.repo), "--poll-seconds", "60",
            ], cwd=str(args.repo), env=env, stdout=finalizer_log, stderr=subprocess.STDOUT, start_new_session=True)
            marker.write_text(json.dumps({"status": "STARTED", "pid": child.pid, "command": command, "started_at": time.time()}, indent=2) + "\n", encoding="utf-8")
            log(handle, f"extended resume started with PID {child.pid}")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
