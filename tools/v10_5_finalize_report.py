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
# This is the original, complete COV native Test evaluation.  The bounded
# search also has a small subset baseline, but that artifact is diagnostic
# only and must never be used for paper deltas.
FULL_COV_BASELINE_RECEIPT = Path("/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_20h_20260914/full_results.json")
FULL_COV_BASELINE_SUMMARY = Path("/data2/usr_for_deadline/tempotrack_v10_unified/tempo_full/covtrack/test_v10_runtime_gate_sharded/merged/evaluation/COVTrack_V10_Tempo_Test/teta_summary_results.pth")
FULL_COV_BASELINE_MANIFEST = Path("/data2/usr_for_deadline/tempotrack_v10_unified/tempo_full/covtrack/test_v10_runtime_gate_sharded/merged/merge_manifest.json")
FULL_METRICS = ("TETA", "LocA", "AssocA", "ClsA", "LocRe", "LocPr", "AssocRe", "AssocPr", "ClsRe", "ClsPr")
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


def load_recovery_full_results(root: Path) -> list[dict[str, Any]]:
    """Collect independently recovered full-Test candidates without trusting names.

    Recovery attempts intentionally live outside the controller's ``full_results``
    file so they cannot overwrite the original attempt.  Once a recovery has a
    PASS receipt, it is a valid full-Test result and must appear in the final
    report.  The receipt remains the source of truth; both output hashes are
    rechecked here and duplicate predictions are suppressed.
    """
    recovered: list[dict[str, Any]] = []
    seen_predictions: set[str] = set()
    for receipt_path in sorted((root / "full").glob("*__recovery*/recovery_result.json")):
        try:
            row = read_json(receipt_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict) or row.get("status") != "PASS":
            continue
        prediction_sha = row.get("prediction_sha256")
        if prediction_sha and prediction_sha in seen_predictions:
            continue
        if not validate_file_hash(row.get("prediction"), prediction_sha) or not validate_file_hash(row.get("summary"), row.get("summary_sha256")):
            raise RuntimeError(f"RECOVERY_FULL_RESULT_HASH_MISMATCH:{receipt_path}")
        seen_predictions.add(str(prediction_sha))
        item = dict(row)
        item["source"] = "recovered_full"
        item["recovery_receipt"] = str(receipt_path)
        item["recovery_receipt_sha256"] = sha256(receipt_path)
        recovered.append(item)
    return recovered


def load_candidate_full_results(root: Path) -> list[dict[str, Any]]:
    """Recover PASS full-Test rows from immutable per-candidate receipts.

    A resume controller may rewrite its aggregate ``full_results.json`` while
    leaving completed candidate directories untouched.  The per-candidate
    receipt is therefore also an authoritative source, provided its hashes are
    verified again.  PARTIAL/FAILED candidate receipts are deliberately
    excluded rather than represented as metric rows.
    """
    rows: list[dict[str, Any]] = []
    for result_path in sorted((root / "full").glob("*/full_result.json")):
        try:
            row = read_json(result_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict) or row.get("status") != "PASS":
            continue
        if not validate_file_hash(row.get("prediction"), row.get("prediction_sha256")) or not validate_file_hash(row.get("summary"), row.get("summary_sha256")):
            raise RuntimeError(f"CANDIDATE_FULL_RESULT_HASH_MISMATCH:{result_path}")
        item = dict(row)
        item.setdefault("source", "candidate_full")
        item["candidate_receipt"] = str(result_path)
        item["candidate_receipt_sha256"] = sha256(result_path)
        rows.append(item)
    return rows


def load_original_full_cov_baseline() -> dict[str, Any]:
    """Load and validate the complete original COV native Test baseline.

    The baseline is read from a structured PASS receipt and its original
    evaluator summary, rather than copied from a previous markdown report.
    This makes it impossible for the finalizer to silently fall back to the
    11,500-frame diagnostic subset.
    """
    receipt = read_json(FULL_COV_BASELINE_RECEIPT)
    matches = [row for row in receipt.get("baselines", []) if row.get("name") == "COV native baseline"]
    if len(matches) != 1:
        raise RuntimeError("ORIGINAL_FULL_COV_BASELINE_RECEIPT_MISSING_OR_AMBIGUOUS")
    row = matches[0]
    if row.get("status") != "PASS" or row.get("scope") != "FULL_TEST":
        raise RuntimeError("ORIGINAL_FULL_COV_BASELINE_NOT_FULL_PASS")
    if Path(str(row.get("path"))).resolve() != FULL_COV_BASELINE_SUMMARY.resolve():
        raise RuntimeError("ORIGINAL_FULL_COV_BASELINE_PATH_MISMATCH")
    if not FULL_COV_BASELINE_SUMMARY.is_file() or not FULL_COV_BASELINE_MANIFEST.is_file():
        raise RuntimeError("ORIGINAL_FULL_COV_BASELINE_ARTIFACT_MISSING")
    manifest = read_json(FULL_COV_BASELINE_MANIFEST)
    if manifest.get("status") != "PASS" or int(manifest.get("images", 0)) != 52155:
        raise RuntimeError("ORIGINAL_FULL_COV_BASELINE_MANIFEST_NOT_COMPLETE")
    value = metrics(row)
    for split in ("base", "novel", "overall"):
        if any(name not in value[split] for name in FULL_METRICS):
            raise RuntimeError(f"ORIGINAL_FULL_COV_BASELINE_METRICS_INCOMPLETE:{split}")
    return {
        "row": row,
        "metrics": value,
        "receipt_path": str(FULL_COV_BASELINE_RECEIPT),
        "receipt_sha256": sha256(FULL_COV_BASELINE_RECEIPT),
        "summary_path": str(FULL_COV_BASELINE_SUMMARY),
        "summary_sha256": sha256(FULL_COV_BASELINE_SUMMARY),
        "manifest_path": str(FULL_COV_BASELINE_MANIFEST),
        "manifest_sha256": sha256(FULL_COV_BASELINE_MANIFEST),
        "annotation": manifest.get("annotation"),
        "annotation_sha256": manifest.get("annotation_sha256"),
        "images": manifest.get("images"),
        "rows": manifest.get("rows"),
    }


def metric_deltas(candidate: Mapping[str, Any], baseline: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for split in ("overall", "base", "novel"):
        result[split] = {}
        for name in FULL_METRICS:
            left, right = candidate.get(split, {}).get(name), baseline.get(split, {}).get(name)
            if left is None or right is None:
                continue
            result[split][name] = float(left) - float(right)
    return result


def fmt_delta(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.6f}"


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


def render(args: argparse.Namespace, state: Mapping[str, Any], downstream: list[dict[str, Any]], subset: Mapping[str, Any], full: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
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
        "## Original full COV native baseline",
        "",
        "This is the only baseline used for subsequent full-Test metric deltas. The 11,500-frame subset baseline is diagnostic-only.",
        "",
        f"scope: {baseline['row'].get('scope')}",
        f"images: {baseline['images']}",
        f"rows: {baseline['rows']}",
        f"annotation: {baseline['annotation']}",
        f"annotation_sha256: {baseline['annotation_sha256']}",
        f"summary: {baseline['summary_path']}",
        f"summary_sha256: {baseline['summary_sha256']}",
        f"merge_manifest: {baseline['manifest_path']}",
        f"merge_manifest_sha256: {baseline['manifest_sha256']}",
        f"baseline_receipt: {baseline['receipt_path']}",
        f"baseline_receipt_sha256: {baseline['receipt_sha256']}",
        "",
        "| split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("overall", "base", "novel"):
        m = baseline["metrics"][split]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            split, *(fmt(m.get(name)) for name in FULL_METRICS)))
    lines += [
        "",
        "## COV Wave2 subset search (DIAGNOSTIC_ONLY)",
        "",
        "These rows are not used for the original-baseline delta and are not full-Test claims.",
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
    lines += [
        "",
        "## Full-Test deltas vs original full COV native baseline",
        "",
        "Positive/negative deltas below are candidate minus the complete original baseline above; no subset value is involved.",
        "",
        "| source/spec | split | TETA | LocA | AssocA | ClsA | LocRe | LocPr | AssocRe | AssocPr | ClsRe | ClsPr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in full_rows:
        m = metrics(row)
        spec = row.get("spec", {})
        label = json.dumps({k: spec.get(k) for k in ("max_gap", "candidate_top_k", "score_threshold", "margin_threshold")}, sort_keys=True, separators=(",", ":"))
        delta = metric_deltas(m, baseline["metrics"])
        for split in ("overall", "base", "novel"):
            lines.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                label, split, *(fmt_delta(delta[split].get(name)) for name in FULL_METRICS)))
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
    if state.get("status") != "COMPLETED":
        lines += [
            "",
            "## Incomplete work (transparent status)",
            "",
            "This report does not claim that every planned search shard completed. The controller ended in a terminal partial/deadline state; incomplete and failed attempts remain recorded in the state receipt and were not converted into metric rows.",
            f"controller_terminal_status: {state.get('status')}",
            f"incomplete_jobs: {json.dumps(state.get('incomplete_jobs', []), ensure_ascii=False, sort_keys=True)}",
            f"incomplete_full: {json.dumps(state.get('incomplete_full', []), ensure_ascii=False, sort_keys=True)}",
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
    if state.get("status") not in {"COMPLETED", "PARTIAL_FAILURE_OR_DEADLINE"}:
        progress = args.repo / "reports/tempotrack_v10/V10_4_20H_BEST_SEARCH_PROGRESS.md"
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text("# V10.4 search not terminal\n\n" + json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "NOT_FINAL", "controller_status": state.get("status"), "progress": str(progress)}))
        return 2
    subset_path, full_path = args.root / "subset_results.json", args.root / "full_results.json"
    if not subset_path.is_file() or not full_path.is_file():
        raise RuntimeError("SEARCH_RESULT_ARTIFACT_MISSING")
    subset, full = read_json(subset_path), read_json(full_path)
    baseline = load_original_full_cov_baseline()
    for row in full.get("new_results", []):
        if row.get("status") != "PASS":
            raise RuntimeError("FULL_RESULT_NOT_PASS")
        if not validate_file_hash(row.get("prediction"), row.get("prediction_sha256")) or not validate_file_hash(row.get("summary"), row.get("summary_sha256")):
            raise RuntimeError("FULL_RESULT_HASH_MISMATCH")
    candidate_results = load_candidate_full_results(args.root)
    recovery_results = load_recovery_full_results(args.root)
    existing_prediction_hashes = {
        str(row.get("prediction_sha256"))
        for row in full.get("new_results", [])
        if row.get("prediction_sha256")
    }
    discovered_results = []
    for row in candidate_results + recovery_results:
        prediction_sha = row.get("prediction_sha256")
        if prediction_sha and str(prediction_sha) in existing_prediction_hashes:
            continue
        if prediction_sha:
            existing_prediction_hashes.add(str(prediction_sha))
        discovered_results.append(row)
    if discovered_results:
        full = dict(full)
        full["new_results"] = list(full.get("new_results", [])) + discovered_results
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
    temporary.write_text(render(args, state, downstream, subset, full, baseline), encoding="utf-8")
    os.replace(temporary, report)
    report_status = "PASS" if state.get("status") == "COMPLETED" else "PARTIAL_REPORT"
    print(json.dumps({"status": report_status, "controller_status": state.get("status"), "report": str(report), "subset_count": len(subset.get("rows", [])), "full_count": len(full.get("new_results", []))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
