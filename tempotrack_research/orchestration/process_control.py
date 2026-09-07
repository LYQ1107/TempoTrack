"""Read-only process inspection and narrowly-scoped quiescing for V3.

The V3 runner must never discover a Python process by name and terminate it.
This module records enough identity (uid, cwd, argv, process start ticks and
run root) to make a stop decision auditable.  Quiescing is limited to a
validated process group belonging to the current user and the old research
root.
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _proc_start_ticks(pid: int) -> int | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # comm may contain spaces and parentheses.  The fields after the
        # final ')' are stable; starttime is field 22, i.e. index 19 here.
        tail = value.rsplit(")", 1)[-1].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def _argv(pid: int) -> tuple[str, ...]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return tuple(item.decode("utf-8", errors="replace") for item in raw.split(b"\0") if item)
    except OSError:
        return ()


def _cwd(pid: int) -> Path | None:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
    except OSError:
        return None


def _uid(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/status").read_text(encoding="utf-8").split("Uid:", 1)[1].split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _inside(path: Path | None, root: Path) -> bool:
    if path is None:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _run_root(argv: Sequence[str], repo: Path, old_run_root: Path) -> Path | None:
    candidates: list[Path] = []
    for index, value in enumerate(argv):
        if value == "--run-root" and index + 1 < len(argv):
            candidates.append(Path(argv[index + 1]))
        if "outputs/research_v2" in value or "outputs/research_v3" in value:
            candidates.append(Path(value))
    for candidate in candidates:
        candidate = candidate if candidate.is_absolute() else (repo / candidate)
        candidate = candidate.resolve()
        if _inside(candidate, old_run_root) or _inside(old_run_root, candidate):
            return candidate
    return old_run_root if any("tempotrack_research" in value for value in argv) else None


def _last_checkpoint(run_dir: Path | None, old_run_root: Path) -> Path | None:
    if run_dir is None:
        return None
    root = run_dir if run_dir.is_dir() else old_run_root
    try:
        candidates = sorted(root.glob("**/last.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    except OSError:
        return None
    return candidates[0] if candidates else None


@dataclass(frozen=True)
class OwnedJob:
    pid: int
    process_start_ticks: int
    uid: int
    cwd: Path
    argv: tuple[str, ...]
    run_dir: Path | None
    last_checkpoint: Path | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cwd"] = str(self.cwd)
        value["argv"] = list(self.argv)
        value["run_dir"] = str(self.run_dir) if self.run_dir is not None else None
        value["last_checkpoint"] = str(self.last_checkpoint) if self.last_checkpoint is not None else None
        return value


def inspect_owned_jobs(repo: Path, old_run_root: Path) -> list[OwnedJob]:
    """Return only current-project TempoTrack research processes.

    The function is intentionally conservative: a process must have a
    resolvable cwd inside the repository and an argv containing the research
    module (or the repository path).  Unrelated user jobs are not returned and
    therefore cannot be selected by the quiesce operation.
    """

    repo = repo.resolve()
    old_run_root = old_run_root.resolve()
    jobs: list[OwnedJob] = []
    for entry in sorted(Path("/proc").glob("[0-9]*"), key=lambda item: int(item.name)):
        try:
            pid = int(entry.name)
        except ValueError:
            continue
        start = _proc_start_ticks(pid)
        uid = _uid(pid)
        cwd = _cwd(pid)
        argv = _argv(pid)
        if start is None or uid is None or cwd is None or not _inside(cwd, repo):
            continue
        command = " ".join(argv)
        if "tempotrack_research" not in command and str(repo) not in command:
            continue
        run_dir = _run_root(argv, repo, old_run_root)
        jobs.append(OwnedJob(pid, start, uid, cwd, argv, run_dir, _last_checkpoint(run_dir, old_run_root), "RUNNING"))
    return jobs


def quiesce_owned_jobs(jobs: Sequence[OwnedJob], *, policy: str, evidence_dir: Path, timeout_seconds: int = 20) -> dict[str, Any]:
    """TERM only validated old-root process groups and record the result."""

    evidence_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"policy": policy, "requested": [], "stopped": [], "blocked": [], "checked_at": time.time()}
    if policy not in {"owned", "quiesce-owned"}:
        result["blocked"].append({"reason": f"unsupported policy: {policy}"})
        (evidence_dir / "quiesce.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    current_uid = os.getuid()
    current_pgid = os.getpgrp()
    for job in jobs:
        item = {"pid": job.pid, "process_start_ticks": job.process_start_ticks, "run_dir": str(job.run_dir) if job.run_dir else None, "last_checkpoint": str(job.last_checkpoint) if job.last_checkpoint else None}
        if job.uid != current_uid or job.run_dir is None or "research_v2" not in str(job.run_dir):
            result["blocked"].append({**item, "reason": "not-current-user-or-not-old-research-root"})
            continue
        try:
            pgid = os.getpgid(job.pid)
        except ProcessLookupError:
            result["stopped"].append({**item, "status": "already-exited"})
            continue
        if pgid == current_pgid or pgid <= 1:
            result["blocked"].append({**item, "reason": "unsafe-process-group"})
            continue
        # Revalidate every member we can observe before signalling the group.
        members: list[int] = []
        unsafe = False
        for candidate in Path("/proc").glob("[0-9]*"):
            try:
                candidate_pid = int(candidate.name)
                if os.getpgid(candidate_pid) != pgid:
                    continue
            except (OSError, ValueError):
                continue
            members.append(candidate_pid)
            candidate_cwd = _cwd(candidate_pid)
            candidate_uid = _uid(candidate_pid)
            if candidate_uid != current_uid or not _inside(candidate_cwd, job.cwd):
                unsafe = True
        if unsafe or job.pid not in members:
            result["blocked"].append({**item, "reason": "process-group-member-failed-revalidation", "members": members})
            continue
        result["requested"].append({**item, "pgid": pgid, "members": members})
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError as exc:
            result["blocked"].append({**item, "reason": f"SIGTERM failed: {exc}", "pgid": pgid})
            continue
        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while time.monotonic() < deadline:
            if not Path(f"/proc/{job.pid}").exists():
                break
            time.sleep(0.2)
        result["stopped"].append({**item, "pgid": pgid, "alive_after_timeout": Path(f"/proc/{job.pid}").exists()})
    (evidence_dir / "quiesce.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return result


__all__ = ["OwnedJob", "inspect_owned_jobs", "quiesce_owned_jobs"]
