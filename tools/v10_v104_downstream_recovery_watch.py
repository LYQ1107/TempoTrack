#!/usr/bin/env python3
"""Wait for a stale downstream attempt, then adopt validated OV artifacts.

This watcher never signals or replaces a live worker.  It starts one fresh
downstream supervisor only after the older supervisor has exited and the
independent OV takeover has completed both official evaluations.  The fresh
supervisor therefore enters ``adopt_ov_early`` instead of re-running any
detector or shard.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


PYTHON = "/home/lwr/anaconda3/envs/masaenv/bin/python"
REPO = Path("/data2/usr_for_deadline/tempotrack_v104_search_hardening")
V10_ROOT = Path("/data2/usr_for_deadline/tempotrack_v10_unified")
DOWNSTREAM_STATE = V10_ROOT / "v104_downstream_supervisor" / "state.json"
OV_EARLY_STATE = V10_ROOT / "v104_downstream" / "ov_early_parallel_20260913" / "state.json"
DOWNSTREAM_SCRIPT = REPO / "tools/v10_v104_downstream_supervisor.py"
STATE_ROOT = V10_ROOT / "v104_downstream_supervisor"
LOG = STATE_ROOT / "recovery_watch.log"
OLD_PID = 15823


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def early_complete() -> bool:
    state = read_json(OV_EARLY_STATE)
    if state.get("status") != "COMPLETED":
        return False
    lanes = state.get("lanes", {})
    return all(
        isinstance(lanes.get(split), dict)
        and lanes[split].get("status") == "PASS"
        and Path(str(lanes[split].get("prediction", ""))).is_file()
        and Path(str(lanes[split].get("summary", ""))).is_file()
        for split in ("test", "val")
    )


def main() -> int:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"watcher_pid={os.getpid()} old_pid={OLD_PID}\n")
        while True:
            downstream = read_json(DOWNSTREAM_STATE)
            if downstream.get("status") == "COMPLETED":
                log.write("downstream already COMPLETED; exiting\n")
                return 0
            if live(OLD_PID):
                log.write("old supervisor still live; no action\n")
                time.sleep(30)
                continue
            if not early_complete():
                log.write("old supervisor exited but early OV is not complete; waiting\n")
                time.sleep(30)
                continue
            log.write("old supervisor exited and early OV PASS; launching adoption supervisor\n")
            env = os.environ.copy()
            env.setdefault("LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0")
            env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
            result = subprocess.run(
                [PYTHON, str(DOWNSTREAM_SCRIPT), "--state-root", str(STATE_ROOT)],
                cwd=str(REPO),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log.write(f"adoption supervisor returncode={result.returncode}\n")
            return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
