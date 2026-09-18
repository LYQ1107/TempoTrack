#!/usr/bin/env python3
"""Close the DSSL Val gate after the calibrated D-wave.

This is a post-processing supervisor only.  It never launches tracking or
evaluation.  It waits for the existing calibrated D-wave supervisor, reads
the B0 and D-card receipts, applies the frozen Base-only gate, and writes a
receipt that either fail-closes the experiment or authorizes a later C/H/Test
extension.  The Current Test is never read by this tool.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


DEFAULT_D_CARDS = ("D1_LS010", "D2_LS025", "D3_LS050", "D4_LS100")
DEFAULT_TRAIN_CARDS = ("B0_OFFICIAL_V11", "B1_DSSL_ARCH_ONLY", *DEFAULT_D_CARDS)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _selected_row(report: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if report.get("status") != "PASS":
        raise ValueError(f"{label} report is not PASS")
    if report.get("test_status") not in (None, "UNTOUCHED"):
        raise ValueError(f"{label} report has unexpected Test status")
    if report.get("novel_used_for_selection") not in (None, False):
        raise ValueError(f"{label} report used Novel for selection")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} report has no metric rows")
    selection = report.get("selection")
    selected_id = selection.get("selected_trial_id") if isinstance(selection, dict) else None
    candidates = [row for row in rows if isinstance(row, dict)]
    if selected_id is not None:
        candidates = [row for row in candidates if row.get("trial_id") == selected_id]
    if len(candidates) != 1:
        raise ValueError(f"{label} report does not identify one selected row")
    row = dict(candidates[0])
    for split in ("overall", "base", "novel"):
        if not isinstance(row.get(split), Mapping):
            raise ValueError(f"{label} selected row has no {split} metrics")
    return row


def _internal_receipt(train_root: Path, card_id: str) -> dict[str, Any]:
    path = train_root / card_id / "training.json"
    value = read_json(path)
    if value.get("card_id") not in (None, card_id):
        raise ValueError(f"training receipt card mismatch: {path}")
    selected = value.get("selected_checkpoint")
    if not isinstance(selected, Mapping):
        selected = value.get("best_checkpoint")
    if not isinstance(selected, Mapping):
        raise ValueError(f"training receipt has no selected checkpoint: {path}")
    required = ("final_mrr", "net_correction")
    if any(key not in selected for key in required):
        raise ValueError(f"training receipt lacks gate metrics: {path}")
    return {
        "card_id": card_id,
        "receipt": str(path.resolve()),
        "best_epoch": value.get("best_epoch"),
        "final_mrr": float(selected["final_mrr"]),
        "final_top1": float(selected.get("final_top1", float("nan"))),
        "net_correction": float(selected["net_correction"]),
        "final_listwise_loss": float(selected.get("final_listwise_loss", float("nan"))),
    }


def _card_report_paths(runtime: Mapping[str, Any], cards: Sequence[str]) -> dict[str, Path]:
    records = runtime.get("completed_cards")
    if not isinstance(records, list):
        raise ValueError("D-wave runtime has no completed_cards")
    result: dict[str, Path] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        card = str(record.get("card_id", ""))
        report = record.get("report")
        if card in cards and report:
            result[card] = Path(str(report)).resolve()
    missing = [card for card in cards if card not in result]
    if missing:
        raise ValueError(f"D-wave runtime is missing card reports: {missing}")
    return result


def build_gate(
    *,
    b0_report: Mapping[str, Any],
    d_reports: Mapping[str, Mapping[str, Any]],
    train_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    b0_row = _selected_row(b0_report, label="B0")
    b0_internal = train_receipts["B0_OFFICIAL_V11"]
    d_rows: list[dict[str, Any]] = []
    for card_id, report in d_reports.items():
        row = _selected_row(report, label=card_id)
        row["card_id"] = card_id
        row["internal"] = dict(train_receipts[card_id])
        d_rows.append(row)
    ranked = sorted(
        d_rows,
        key=lambda row: (
            -float(row["base"]["AssocA"]),
            -float(row["base"]["TETA"]),
            -float(row["base"]["AssocPr"]),
            -float(row["internal"]["final_mrr"]),
            str(row["card_id"]),
        ),
    )
    selected = ranked[0]
    b1 = train_receipts["B1_DSSL_ARCH_ONLY"]
    internal_candidates = [b1] + [row["internal"] for row in d_rows]
    gate = {
        "val_base_assocA_gt_B0": float(selected["base"]["AssocA"]) > float(b0_row["base"]["AssocA"]),
        "net_correction_gt_B0": float(selected["internal"]["net_correction"]) > float(b0_internal["net_correction"]),
        "repeated_internal_final_mrr_improvement": any(
            float(row["final_mrr"]) > float(b0_internal["final_mrr"])
            for row in internal_candidates
        ),
        "overall_teta_not_obvious_collapse": float(selected["overall"]["TETA"])
        >= float(b0_row["overall"]["TETA"]) - 1.0,
    }
    passed = all(gate.values())
    return {
        "status": "DSSL_GATE_PASS" if passed else "DSSL_NEGATIVE_GATE",
        "artifact": "v11_post_dssl_val_gate",
        "selection_metric": "Official Val Base AssocA, then Base TETA, then Base AssocPr, then internal holdout final_mrr",
        "novel_used_for_selection": False,
        "overall_used_for_selection": False,
        "control_B0": {"row": b0_row, "internal": dict(b0_internal)},
        "cards": ranked,
        "lambda_struct_star": selected["internal"].get("lambda_struct"),
        "selected_card": selected["card_id"],
        "gate": gate,
        "gate_rule_note": "Overall TETA collapse is operationalized as selected Overall TETA >= B0 Overall TETA - 1.0.",
        "extension": {
            "allowed": passed,
            "cards": ["C1_LC005", "C2_LC010", "C3_LC025", "H1_TEMPORAL_CONFLICT"],
            "current_test": "one confirmation only after frozen Val selection" if passed else "NOT_RUN_GATE_FAILED",
        },
    }


def write_gate_reports(gate: Mapping[str, Any], output_root: Path, report_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "dssl_gate.json", gate)
    atomic_write_json(report_root / "dssl_gate.json", gate)
    lines = [
        "# Official-Train DSSL Val Gate",
        "",
        f"Status: **{gate['status']}**",
        "",
        "Selection uses Official Val Base only; Novel and Overall are diagnostic.",
        "Current Test is not read or launched by this gate.",
        "",
        "## Gate conditions",
        "",
        "| condition | result |",
        "|---|---|",
    ]
    for key, value in gate["gate"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    lines += [
        "",
        f"Selected card: `{gate['selected_card']}`",
        f"Extension allowed: `{gate['extension']['allowed']}`",
    ]
    if gate["status"] == "DSSL_NEGATIVE_GATE":
        lines += [
            "",
            "The DSSL gate failed; C1–C3, H1, and Current Test must not be launched.",
            "This is a negative result under the frozen protocol, not an invitation to continue threshold search.",
        ]
    (report_root / "DSSL_NEGATIVE_RESULT_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def wait_for_d_wave(path: Path, poll_seconds: float) -> dict[str, Any]:
    while True:
        if path.is_file():
            try:
                value = read_json(path)
                if value.get("status") in {"D_WAVE_COMPLETE", "DSSL_GATE_PASS", "DSSL_NEGATIVE_GATE"}:
                    return value
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        time.sleep(max(1.0, float(poll_seconds)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-wave-runtime", type=Path, required=True)
    parser.add_argument("--b0-report", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    runtime = read_json(args.d_wave_runtime.resolve()) if args.once else wait_for_d_wave(args.d_wave_runtime.resolve(), args.poll_seconds)
    if runtime.get("status") != "D_WAVE_COMPLETE":
        return 0
    cards = list(DEFAULT_D_CARDS)
    report_paths = _card_report_paths(runtime, cards)
    reports = {card: read_json(path) for card, path in report_paths.items()}
    train_receipts = {
        card: _internal_receipt(args.train_root.resolve(), card)
        for card in DEFAULT_TRAIN_CARDS
    }
    gate = build_gate(
        b0_report=read_json(args.b0_report.resolve()),
        d_reports=reports,
        train_receipts=train_receipts,
    )
    write_gate_reports(gate, args.output_root.resolve(), args.report_root.resolve())
    runtime = dict(runtime)
    runtime.update({"status": gate["status"], "gate": gate, "test_started": False, "gate_ended_at_unix": time.time()})
    atomic_write_json(args.d_wave_runtime.resolve(), runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
