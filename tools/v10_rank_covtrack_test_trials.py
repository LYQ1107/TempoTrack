#!/usr/bin/env python3
"""Rank completed COV V10 Test-tuned trials without accepting failed receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(receipt: dict[str, Any], split: str, name: str) -> float | None:
    value = receipt.get("metrics", {}).get(split)
    if not isinstance(value, dict) or name not in value:
        return None
    if value[name] is None:
        return None
    try:
        number = float(value[name])
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _receipts(roots: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    for root in roots:
        for path in sorted(root.glob("**/receipt.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") != "COMPLETED":
                continue
            protocol = value.get("protocol", {})
            if protocol.get("test_tuned_model_specific") is not True or protocol.get("unbiased_test") is not False:
                continue
            if value.get("outputs", {}).get("prediction_sha256") is None:
                continue
            found.append((path, value))
    return found


def rank(args: argparse.Namespace) -> int:
    roots = [Path(value).resolve() for value in args.root]
    control_path = Path(args.control_receipt).resolve()
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control_base_assoc = _metric(control, "base", "AssocA")
    control_base_teta = _metric(control, "base", "TETA")
    if control_base_assoc is None or control_base_teta is None:
        raise ValueError("control receipt lacks finite Base AssocA/TETA")
    rows: list[dict[str, Any]] = []
    for path, receipt in _receipts(roots):
        base_assoc = _metric(receipt, "base", "AssocA")
        base_teta = _metric(receipt, "base", "TETA")
        novel_assoc = _metric(receipt, "novel", "AssocA")
        # A missing Novel metric is an evaluator/provenance failure, never a
        # ranking advantage. Keep the receipt on disk but fail closed here.
        if base_assoc is None or base_teta is None or novel_assoc is None:
            continue
        row = {
            "trial_id": receipt.get("trial_id"),
            "receipt": str(path),
            "receipt_sha256": _sha256(path),
            "stage": receipt.get("stage"),
            "spec": receipt.get("spec"),
            "prediction_sha256": receipt.get("outputs", {}).get("prediction_sha256"),
            "base": receipt.get("metrics", {}).get("base"),
            "novel": receipt.get("metrics", {}).get("novel"),
            "novel_assoc_for_ranking": novel_assoc,
            "base_assoc_delta_vs_control": base_assoc - control_base_assoc,
            "base_teta_delta_vs_control": base_teta - control_base_teta,
            "eligible": base_assoc >= control_base_assoc - 1.0 and base_teta >= control_base_teta - 1.0,
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            bool(row["eligible"]),
            -float(row["novel_assoc_for_ranking"]),
            -float(row["base"].get("AssocA", float("-inf"))),
            -float(row["base"].get("TETA", float("-inf"))),
            str(row.get("trial_id")),
        )
    )
    selected = [row for row in rows if row["eligible"]][: args.top_k]
    output = {
        "schema_version": 1,
        "artifact": "tempotrack_v10_covtrack_test_tuned_rank",
        "selection_protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "selection_note": "Test subset is used for model-specific diagnostics; this is not a paper-unbiased Test result.",
        "subset_provenance": {
            "novel_gt_used_for_subset_selection": True,
            "test_gt_used_for_hyperparameter_selection": True,
            "novel_gt_used_for_inference": False,
        },
        "control_receipt": str(control_path),
        "control_receipt_sha256": _sha256(control_path),
        "control_base": control.get("metrics", {}).get("base"),
        "all_completed": rows,
        "selected_top_k": selected,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# COVTrack V10 Test-tuned subset ranking",
        "",
        "> This is `TEST_TUNED_MODEL_SPECIFIC`; it is not an unbiased Test result.",
        "",
        f"Control Base TETA/AssocA: `{control_base_teta:.6f}/{control_base_assoc:.6f}`",
        "",
        "| rank | trial | eligible | Base TETA | Base AssocA | Novel AssocA | ΔBase AssocA | prediction hash |",
        "|---:|---|:---:|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(selected, start=1):
        novel = row.get("novel") or {}
        markdown.append(
            f"| {index} | `{row['trial_id']}` | {row['eligible']} | "
            f"{row['base']['TETA']:.6f} | {row['base']['AssocA']:.6f} | "
            f"{novel.get('AssocA', float('nan')):.6f} | {row['base_assoc_delta_vs_control']:+.6f} | "
            f"`{row['prediction_sha256']}` |"
        )
    Path(args.markdown).resolve().write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"completed": len(rows), "selected": [row["trial_id"] for row in selected], "output": str(output_path)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--control-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    return rank(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
