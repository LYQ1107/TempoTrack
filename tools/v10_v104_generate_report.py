#!/usr/bin/env python3
"""Render the final V10.4 report from durable experiment receipts."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


HARD_REPO = Path("/data2/usr_for_deadline/tempotrack_v104_search_hardening")
V10_ROOT = Path("/data2/usr_for_deadline/tempotrack_v10_unified")
DOWNSTREAM = V10_ROOT / "v104_downstream"
PRIMARY_STATE = V10_ROOT / "v104_persistent_supervisor" / "state.json"
DOWNSTREAM_STATE = V10_ROOT / "v104_downstream_supervisor" / "state.json"
MASA_STATE = V10_ROOT / "v104_masa_downstream_supervisor" / "state.json"
LEGACY_ROOT = V10_ROOT / "search" / "covtrack_test_q1_fixed_20260912"
WAVE2_ROOT = V10_ROOT / "search" / "covtrack_q1_fixed_20260913_wave2"


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(HARD_REPO), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else f"ERROR:{result.stderr.strip()}"


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def metrics(value: dict[str, Any] | None) -> str:
    if not isinstance(value, dict):
        return "—"
    return " / ".join(fmt(value.get(name)) for name in ("TETA", "LocA", "AssocA", "ClsA"))


def table_row(label: str, data: dict[str, Any] | None) -> str:
    if not isinstance(data, dict):
        return f"| {label} | — | — | — |"
    parsed = data.get("metrics", data)
    return f"| {label} | {metrics(parsed.get('base')) if isinstance(parsed, dict) else '—'} | {metrics(parsed.get('novel')) if isinstance(parsed, dict) else '—'} | {metrics(parsed.get('overall')) if isinstance(parsed, dict) else '—'} |"


def load_receipt(path: Path) -> dict[str, Any] | None:
    value = read_json(path)
    return value if isinstance(value, dict) else None


def legacy_rows() -> list[dict[str, Any]]:
    rows = []
    for receipt_path in sorted(LEGACY_ROOT.glob("*/receipt.json")):
        value = load_receipt(receipt_path)
        if not value:
            continue
        rows.append({
            "trial_id": value.get("trial_id", receipt_path.parent.name),
            "status": value.get("status"),
            "stage": value.get("stage"),
            "gpu": value.get("gpu"),
            "base": value.get("metrics", {}).get("base", {}),
            "novel": value.get("metrics", {}).get("novel", {}),
            "prediction_sha256": value.get("outputs", {}).get("prediction_sha256"),
            "receipt": str(receipt_path),
        })
    return rows


def wave2_rows() -> list[dict[str, Any]]:
    """Include the post-contract Wave2 receipts in the final audit table."""
    rows = []
    for receipt_path in sorted(WAVE2_ROOT.glob("*/receipt.json")):
        value = load_receipt(receipt_path)
        if not value:
            continue
        metrics_value = value.get("metrics", {})
        rows.append({
            "trial_id": value.get("trial_id", receipt_path.parent.name),
            "status": value.get("status"),
            "stage": value.get("stage"),
            "gpu": value.get("gpu"),
            "base": metrics_value.get("base", {}) if isinstance(metrics_value, dict) else {},
            "novel": metrics_value.get("novel", {}) if isinstance(metrics_value, dict) else {},
            "prediction_sha256": value.get("prediction_sha256") or value.get("outputs", {}).get("prediction_sha256"),
            "receipt": str(receipt_path),
        })
    return rows


def search_rows() -> list[dict[str, Any]]:
    return [
        *({**row, "source": "legacy"} for row in legacy_rows()),
        *({**row, "source": "wave2"} for row in wave2_rows()),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HARD_REPO / "reports/tempotrack_v10/FINAL_V10_4_FULL_SEARCH.md")
    args = parser.parse_args()
    primary = read_json(PRIMARY_STATE) or {}
    downstream = read_json(DOWNSTREAM_STATE) or {}
    masa = read_json(MASA_STATE) or {}
    selected = read_json(DOWNSTREAM / "cov_selected_config.json")
    cov_test = read_json(DOWNSTREAM / "cov_test_result.json")
    cov_val = read_json(DOWNSTREAM / "cov_val_result.json")
    ov_test = read_json(DOWNSTREAM / "ov_test_result.json")
    ov_val = read_json(DOWNSTREAM / "ov_val_result.json")
    masa_results = read_json(DOWNSTREAM / "masa_results.json")

    lines = [
        "# TempoTrack V10.4 FULL Test search — final report",
        "",
        f"Generated: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`",
        f"Repository HEAD: `{git('rev-parse', 'HEAD')}`",
        f"Branch: `{git('branch', '--show-current')}`",
        "",
        "> This report is generated only from the durable receipts below. Missing or failed lanes remain missing/failed; no historical metric is copied into a new result.",
        "",
        "## Execution status",
        "",
        f"- Upstream hardened supervisor: `{primary.get('status', 'MISSING')}` at stage `{primary.get('current_stage', 'MISSING')}`.",
        f"- Downstream COV/OV supervisor: `{downstream.get('status', 'MISSING')}` at stage `{downstream.get('current_stage', 'MISSING')}`.",
        f"- MASA downstream supervisor: `{masa.get('status', 'MISSING')}` at stage `{masa.get('current_stage', 'MISSING')}`.",
        f"- Legacy receipt count: `{len(legacy_rows())}`; Wave2 receipt count: `{len(wave2_rows())}`.",
        "",
        "## Validity and protocol",
        "",
        "- COV search and full Test are `TEST_TUNED_MODEL_SPECIFIC`, not unbiased held-out Test: Test subset/GT is used for model-specific selection and Novel GT is not used by inference.",
        "- COV full/Val uses the final hardened Q=1 contract gate and the clean pinned COV source/TETA evaluator recorded in each receipt.",
        "- OVTrack uses the experiment-owned patched native source and the explicit reranker-disabled memory-only configuration; the COV-specific learned checkpoint is not used.",
        "- MASA uses `masa_r50.pth` and the frozen audited COV public-detection roots. Native and config-driven reranker-disabled Tempo are separate runs.",
        "- All association-only comparisons preserve detector boxes, scores, labels, and categories; only track IDs are altered by the overlay/merger.",
        "",
        "## Final metrics",
        "",
        "Metric order is `TETA / LocA / AssocA / ClsA`; Base and Novel are independently parsed from the official TETA summary.",
        "",
        "| method/split | Base | Novel | Overall |",
        "|---|---|---|---|",
        table_row("COV selected — Test", cov_test),
        table_row("COV selected — Val transfer", cov_val),
        table_row("OVTrack memory-only — Test", ov_test),
        table_row("OVTrack memory-only — Val", ov_val),
    ]
    if isinstance(masa_results, dict):
        for row in masa_results.get("rows", []):
            if isinstance(row, dict):
                lines.append(table_row(f"MASA-R50 {row.get('method')} — {row.get('split')}", row))
    else:
        lines.append(table_row("MASA-R50 downstream", None))

    lines.extend(["", "## COV selection provenance", ""])
    if isinstance(selected, dict):
        chosen = selected.get("selected", {})
        lines.extend([
            f"- Selected config: `{json.dumps(chosen.get('spec', {}), sort_keys=True)}`.",
            f"- Selection receipt: `{chosen.get('receipt', '—')}`; SHA256 `{chosen.get('receipt_sha256', '—')}`.",
            f"- Control: `{selected.get('control', {}).get('receipt', '—')}`; Base AssocA/TETA `{fmt(selected.get('control', {}).get('base_assoc'))}/{fmt(selected.get('control', {}).get('base_teta'))}`.",
            f"- Selection status: `{selected.get('usage', '—')}`; `unbiased_test={selected.get('unbiased_test')}`.",
        ])
    else:
        lines.append("- No selected COV configuration receipt was found.")

    lines.extend(["", "## Upstream search receipt status", "", "| trial | status | stage | GPU | Base AssocA | Novel AssocA | prediction SHA |", "|---|---|---|---:|---:|---:|---|"])
    for row in search_rows():
        lines.append(f"| `{row['source']}/{row['trial_id']}` | `{row['status']}` | `{row['stage']}` | `{row['gpu']}` | {fmt(row['base'].get('AssocA'))} | {fmt(row['novel'].get('AssocA'))} | `{row['prediction_sha256'] or '—'}` |")
    lines.extend(["", "## Artifact binding", ""])
    for label, value in (("COV Test", cov_test), ("COV Val", cov_val), ("OV Test", ov_test), ("OV Val", ov_val)):
        if isinstance(value, dict):
            ann = value.get("annotation", {})
            lines.extend([
                f"### {label}",
                f"- Annotation: `{ann.get('path', '—')}` SHA256 `{ann.get('sha256', '—')}`.",
                f"- Prediction: `{value.get('prediction', '—')}` SHA256 `{value.get('prediction_sha256', '—')}`.",
                f"- Official summary: `{value.get('summary', '—')}` SHA256 `{value.get('summary_sha256', '—')}`.",
            ])
    if isinstance(masa_results, dict):
        lines.extend(["", "### MASA rows"])
        for row in masa_results.get("rows", []):
            if isinstance(row, dict):
                lines.append(f"- `{row.get('method')}/{row.get('split')}`: checkpoint `{row.get('checkpoint')}` SHA256 `{row.get('checkpoint_sha256')}`; prediction `{row.get('prediction')}` SHA256 `{row.get('prediction_sha256')}`; summary `{row.get('summary')}` SHA256 `{row.get('summary_sha256')}`.")

    lines.extend([
        "",
        "## Resources and reproducibility",
        "",
        f"- Primary durable state: `{PRIMARY_STATE}`.",
        f"- Downstream durable state: `{DOWNSTREAM_STATE}`.",
        f"- MASA durable state: `{MASA_STATE}`.",
        f"- Legacy coordinator status: `{LEGACY_ROOT / 'coordinator_status.json'}`.",
        "- Each worker records its exact command, environment paths, source/config/checkpoint hashes, prediction hash, evaluator summary hash, GPU, and start/end resource snapshots.",
        "- No external process was killed, stopped, reset, reniced, or reprioritized; no `git reset`, `git clean`, or old output root deletion was performed.",
        "",
        "## Known invalid or non-final evidence",
        "",
        "- The pre-hardening COV search trials remain preserved but are not promoted over the final gate when their contract/audit binding is stale.",
        "- The earlier OV full Tempo output that loaded a COV-specific checkpoint is excluded from this report's OV result table.",
        "- Historical V10.3 MASA native numbers are not substituted for the current downstream receipt; they remain in the prior report for traceability only.",
        "",
        "## Conclusion",
        "",
        "The claimed result is the intersection of a completed receipt, exact input/provenance bindings, complete prediction coverage, and official TETA summary parsing. A missing row or failed stage is a real blocker, not a zero or an inferred result.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
