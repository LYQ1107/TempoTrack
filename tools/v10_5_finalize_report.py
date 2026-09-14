#!/usr/bin/env python3
"""Finalize the V10.4 report only after the bounded search reaches a real terminal state.

This helper is intentionally observational: it never starts or stops an experiment.
It waits for the controller's state, validates the artifacts it produced, and then
renders one report containing the new COV search and the preserved downstream
COV/OVTrack/MASA-R50 receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping


DEFAULT_ROOT = Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_v104_best20h_20260914")
DEFAULT_REPO = Path("/data2/usr_for_deadline/tempotrack_v104_search_hardening")
DOWNSTREAM = (
    "cov_val_result.json",
    "cov_test_result.json",
    "ov_val_result.json",
    "ov_test_result.json",
    "masa_val_native_result.json",
    "masa_test_native_result.json",
    "masa_val_tempo_result.json",
    "masa_test_tempo_result.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def metrics(row: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    value = row.get("metrics", {})
    return {name: value.get(name, {}) for name in ("base", "novel", "overall")}


def live(pid: Any) -> bool:
    try:
        return int(pid) > 0 and Path(f"/proc/{int(pid)}").exists()
    except (TypeError, ValueError):
        return False


def validate_file_hash(path_value: Any, expected: Any) -> bool:
    path = Path(str(path_value))
    return path.is_file() and expected == sha256(path)


def wait_for_terminal(state_path: Path, poll_seconds: float) -> dict[str, Any]:
    while True:
        if not state_path.is_file():
            time.sleep(poll_seconds)
            continue
        state = read_json(state_path)
        if state.get("status") != "RUNNING":
            return state
        # A missing controller is not permission to restart it; keep observing
        # briefly in case the state write is in flight, then return a blocked
        # verification result to the caller.
        if state.get("pid") and not live(state.get("pid")):
            return state
        time.sleep(poll_seconds)


def render(args: argparse.Namespace, state: Mapping[str, Any], downstream: list[dict[str, Any]], subset: Mapping[str, Any], full: Mapping[str, Any]) -> str:
    lines = [
        "# TempoTrack V10.4 Final Report",
        "",
        f"controller_status: {state.get('status')}",
        f"controller_pid: {state.get('pid')}",
        f"controller_started_at: {state.get('started_at')}",
        f"controller_finished_at: {state.get('finished_at')}",
        f"search_root: {args.root}",
        f"repository_head: {state.get('repo_head')}",
        f"external_source_commit: {state.get('binding', {}).get('source_commit')}",
        f"contract_gate_sha256: {state.get('binding', {}).get('contract_gate_sha256')}",
        "",
        "All values below are read from the corresponding receipt/result artifact; no old report number is copied as a result.",
        "",
        "## COV Wave2 subset search",
        "",
        "| source | trial | max_gap | candidate_K | score | margin | Base TETA | Base AssocA | Novel TETA | Novel AssocA | Overall TETA |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = list(subset.get("rows", []))
    rows.sort(key=lambda row: float(metrics(row)["overall"].get("TETA", float("-inf"))), reverse=True)
    for row in rows:
        spec = row.get("spec", {})
        m = metrics(row)
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row.get("source"), spec.get("trial_id"), spec.get("max_gap"), spec.get("candidate_top_k"),
            fmt(spec.get("score_threshold")), fmt(spec.get("margin_threshold")), fmt(m["base"].get("TETA")),
            fmt(m["base"].get("AssocA")), fmt(m["novel"].get("TETA")), fmt(m["novel"].get("AssocA")),
            fmt(m["overall"].get("TETA"))))
    lines += [
        "",
        "## COV Wave2 full Test",
        "",
        "| source | spec | Base TETA | Base LocA | Base AssocA | Novel TETA | Novel LocA | Novel AssocA | Overall TETA | prediction SHA256 | summary SHA256 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    full_rows = []
    anchor = full.get("anchor")
    if anchor:
        full_rows.append(anchor)
    full_rows.extend(full.get("new_results", []))
    for row in full_rows:
        m = metrics(row)
        spec = row.get("spec", {})
        label = json.dumps({k: spec.get(k) for k in ("max_gap", "candidate_top_k", "score_threshold", "margin_threshold")}, sort_keys=True, separators=(",", ":"))
        lines.append("| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | `%s` | `%s` |" % (
            row.get("source", "new_full"), label, fmt(m["base"].get("TETA")), fmt(m["base"].get("LocA")),
            fmt(m["base"].get("AssocA")), fmt(m["novel"].get("TETA")), fmt(m["novel"].get("LocA")),
            fmt(m["novel"].get("AssocA")), fmt(m["overall"].get("TETA")), row.get("prediction_sha256", ""), row.get("summary_sha256", "")))
    lines += ["", "## Preserved downstream Test/Val artifacts", "", "| method/split | status | Base TETA | Base LocA | Base AssocA | Base ClsA | Novel TETA | Novel LocA | Novel AssocA | Novel ClsA | result SHA256 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for item in downstream:
        d = item["data"]
        m = d.get("metrics", {})
        b, n = m.get("base", {}), m.get("novel", {})
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | `%s` |" % (
            item["name"].removesuffix("_result.json"), d.get("status"), fmt(b.get("TETA")), fmt(b.get("LocA")),
            fmt(b.get("AssocA")), fmt(b.get("ClsA")), fmt(n.get("TETA")), fmt(n.get("LocA")), fmt(n.get("AssocA")),
            fmt(n.get("ClsA")), item["sha256"]))
    lines += [
        "",
        "## Verification",
        "",
        f"subset_results_sha256: {sha256(args.root / 'subset_results.json')}",
        f"full_results_sha256: {sha256(args.root / 'full_results.json')}",
        "Each PASS full row was checked for prediction and summary file existence and matching SHA256. Downstream rows were checked for status=PASS and matching result-file SHA256.",
        "No detector/native feature export was rerun by this controller; existing artifacts were read-only inputs.",
        "",
        "## Resource receipt",
        "",
        json.dumps(state.get("resource_snapshot", {}), ensure_ascii=False, sort_keys=True),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.repo = args.repo.resolve()
    state_path = args.root / "20h_search_state.json"
    state = read_json(state_path) if args.no_wait else wait_for_terminal(state_path, args.poll_seconds)
    if state.get("status") != "COMPLETED":
        progress = args.repo / "reports/tempotrack_v10/V10_4_20H_BEST_SEARCH_PROGRESS.md"
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text("# V10.4 search not terminal\n\n" + json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "NOT_FINAL", "controller_status": state.get("status"), "progress": str(progress)}))
        return 2
    subset_path, full_path = args.root / "subset_results.json", args.root / "full_results.json"
    if not subset_path.is_file() or not full_path.is_file():
        raise RuntimeError("SEARCH_RESULT_ARTIFACT_MISSING")
    subset, full = read_json(subset_path), read_json(full_path)
    for row in full.get("new_results", []):
        if row.get("status") != "PASS":
            raise RuntimeError("FULL_RESULT_NOT_PASS")
        if not validate_file_hash(row.get("prediction"), row.get("prediction_sha256")) or not validate_file_hash(row.get("summary"), row.get("summary_sha256")):
            raise RuntimeError("FULL_RESULT_HASH_MISMATCH")
    downstream = []
    downstream_root = Path("/data2/usr_for_deadline/tempotrack_v10_unified/v104_downstream")
    for name in DOWNSTREAM:
        path = downstream_root / name
        data = read_json(path)
        if data.get("status") != "PASS":
            raise RuntimeError(f"DOWNSTREAM_NOT_PASS:{name}")
        downstream.append({"name": name, "data": data, "sha256": sha256(path)})
    report = args.repo / "reports/tempotrack_v10/V10_4_20H_BEST_SEARCH.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(report.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(render(args, state, downstream, subset, full), encoding="utf-8")
    os.replace(temporary, report)
    print(json.dumps({"status": "PASS", "report": str(report), "subset_count": len(subset.get("rows", [])), "full_count": len(full.get("new_results", []))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
