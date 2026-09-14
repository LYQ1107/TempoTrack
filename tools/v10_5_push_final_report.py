#!/usr/bin/env python3
"""Push exactly the verified V10.4 final report when it becomes available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v104_search_hardening"))
    parser.add_argument("--report", type=Path, default=Path("reports/tempotrack_v10/V10_4_20H_BEST_SEARCH.md"))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = (repo / args.report).resolve() if not args.report.is_absolute() else args.report.resolve()
    while True:
        if report.is_file():
            text = report.read_text(encoding="utf-8")
            if (
                "controller_status: COMPLETED" in text
                and "status | Base TETA" in text
                and "## Original full COV native baseline" in text
                and "## Full-Test deltas vs original full COV native baseline" in text
            ):
                break
        time.sleep(args.poll_seconds)
    staged = run(["git", "add", "--", str(report.relative_to(repo))], repo)
    if staged.returncode != 0:
        raise RuntimeError(staged.stderr[-4000:])
    commit = run(["git", "diff", "--cached", "--quiet", "--", str(report.relative_to(repo))], repo)
    if commit.returncode == 0:
        print(json.dumps({"status": "ALREADY_PUSHED", "report": str(report)}))
        return 0
    if commit.returncode != 1:
        raise RuntimeError(commit.stderr[-4000:])
    created = run(["git", "commit", "-m", "Add verified V10.4 final report", "--", str(report.relative_to(repo))], repo)
    if created.returncode != 0:
        raise RuntimeError(created.stderr[-4000:])
    pushed = run(["git", "push", "origin", "HEAD:codex/v104-search-hardening"], repo)
    if pushed.returncode != 0:
        raise RuntimeError(pushed.stderr[-4000:])
    print(json.dumps({"status": "PUSHED", "report": str(report), "commit": run(["git", "rev-parse", "HEAD"], repo).stdout.strip()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
