"""Generate a non-destructive data2 cleanup inventory for V10 Agent A.

The script never removes or renames anything. It intentionally reports only
obvious TempoTrack intermediate candidates and marks every row as requiring
main-agent approval before deletion.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import time


DATA_ROOT = Path("/data2/usr_for_deadline")
REPORT_ROOTS = [
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9/reports"),
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified/reports"),
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/v10_core_detector/reports"),
]
KEYWORDS = ("failed", "retry", "partial", "stale", "pending", "tmp")


def _active_refs() -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = entry.name
        try:
            cwd = os.path.realpath(entry / "cwd")
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, OSError):
            continue
        text = f"{cwd} {cmd}"
        if "tempotrack" in text.lower() or "v9" in text.lower() or "v10" in text.lower():
            refs.setdefault(cwd, []).append(f"{pid}:{cmd.strip()[:160]}")
    return refs


def _size(path: Path) -> int:
    try:
        result = subprocess.run(["du", "-sb", "--", str(path)], check=True, capture_output=True, text=True)
        return int(result.stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return path.stat().st_size if path.is_file() else 0


def _report_references(path: Path) -> str:
    hits: list[str] = []
    needle = str(path)
    for root in REPORT_ROOTS:
        if not root.exists():
            continue
        try:
            result = subprocess.run(
                ["rg", "-l", "--fixed-strings", needle, str(root)],
                check=False,
                capture_output=True,
                text=True,
            )
            hits.extend(line for line in result.stdout.splitlines() if line)
        except OSError:
            pass
    return ";".join(sorted(set(hits)))


def main() -> int:
    output = Path(os.environ.get("V10_DATA2_DRYRUN", "/data2/usr_for_deadline/DATA2_CLEANUP_DRYRUN.tsv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    active = _active_refs()
    rows: list[dict[str, str | int]] = []
    if DATA_ROOT.exists():
        for child in sorted(DATA_ROOT.iterdir()):
            name = child.name.lower()
            if not any(token in name for token in KEYWORDS):
                continue
            resolved = str(child.resolve())
            active_pid = ";".join(active.get(resolved, []))
            rows.append(
                {
                    "path": resolved,
                    "size_bytes": _size(child),
                    "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(child.stat().st_mtime)),
                    "active_pid": active_pid,
                    "referenced_by_final_report": _report_references(child),
                    "replaced_by_complete_artifact": "UNKNOWN_REQUIRES_REVIEW",
                    "safe_to_delete": "NO_UNAUTHORIZED_DELETE",
                    "reason": "name matches intermediate keyword; dry-run only; main-agent review required",
                }
            )
    fields = [
        "path",
        "size_bytes",
        "mtime",
        "active_pid",
        "referenced_by_final_report",
        "replaced_by_complete_artifact",
        "safe_to_delete",
        "reason",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
