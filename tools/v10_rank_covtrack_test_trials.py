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


def _audit_for_trial(
    receipt_path: Path,
    receipt: dict[str, Any],
    global_audit: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if global_audit is not None:
        trials = global_audit.get("trials", {})
        if isinstance(trials, dict):
            receipt_trial_id = str(receipt.get("trial_id", ""))
            requested = str(receipt.get("requested_trial_id", ""))
            for key in (requested, receipt_trial_id):
                value = trials.get(key)
                if isinstance(value, dict):
                    return value
            for value in trials.values():
                if not isinstance(value, dict):
                    continue
                if str(value.get("effective_trial_id", "")) == receipt_trial_id:
                    return value
                audit_receipt = value.get("receipt")
                if audit_receipt and Path(audit_receipt).resolve() == receipt_path.resolve():
                    return value
        return None
    sidecar = receipt_path.parent / "selection_audit.json"
    if not sidecar.is_file():
        return None
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _receipts(
    roots: list[Path],
    *,
    global_audit: dict[str, Any] | None = None,
) -> tuple[list[tuple[Path, dict[str, Any]]], list[dict[str, Any]]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(root.glob("**/receipt.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                rejected.append({"trial_id": path.parent.name, "reason": [f"RECEIPT_INVALID:{type(exc).__name__}"]})
                continue
            trial_id = str(value.get("trial_id", path.parent.name))
            reasons: list[str] = []
            if value.get("status") != "COMPLETED":
                rejected.append({"trial_id": trial_id, "reason": ["RECEIPT_NOT_COMPLETED"]})
                continue
            protocol = value.get("protocol", {})
            if protocol.get("test_tuned_model_specific") is not True or protocol.get("unbiased_test") is not False:
                rejected.append({"trial_id": trial_id, "reason": ["PROTOCOL_NOT_TEST_TUNED_MODEL_SPECIFIC"]})
                continue
            audit = _audit_for_trial(path, value, global_audit)
            if not isinstance(audit, dict):
                reasons.append("SELECTION_AUDIT_MISSING")
            else:
                if audit.get("status") != "PASS":
                    reasons.append("SELECTION_AUDIT_NOT_PASS")
                if audit.get("usage") != "SEARCH_SELECTION":
                    reasons.append("SELECTION_AUDIT_NOT_SELECTION")
            outputs = value.get("outputs", {})
            for key, label in (("prediction", "PREDICTION"), ("diagnostics", "DIAGNOSTICS"), ("summary", "SUMMARY")):
                output_path = outputs.get(key)
                expected_hash = outputs.get(f"{key}_sha256")
                if not output_path or not Path(output_path).is_file():
                    reasons.append(f"{label}_MISSING")
                elif not expected_hash or _sha256(Path(output_path)) != expected_hash:
                    reasons.append(f"{label}_HASH_MISMATCH")
            if reasons:
                rejected.append({"trial_id": trial_id, "reason": sorted(set(reasons))})
                continue
            found.append((path, value))
    return found, rejected


def _input_binding(receipt: dict[str, Any]) -> dict[str, Any]:
    inputs = receipt.get("inputs", {})
    source = receipt.get("external_source", {})
    return {
        "annotation_sha256": inputs.get("annotation_sha256"),
        "external_source_commit": source.get("commit"),
        "external_checkpoint_sha256": inputs.get("external_checkpoint_sha256"),
        "external_config_sha256": inputs.get("external_config_sha256"),
    }


def _binding_reasons(
    control: dict[str, Any],
    trial: dict[str, Any],
    *,
    expected_annotation_sha256: str | None,
) -> list[str]:
    control_binding = _input_binding(control)
    trial_binding = _input_binding(trial)
    reasons: list[str] = []
    if any(value is None for value in control_binding.values()):
        reasons.append("CONTROL_INPUT_BINDING_MISSING")
    if any(value is None for value in trial_binding.values()):
        reasons.append("TRIAL_INPUT_BINDING_MISSING")
    for key in control_binding:
        if control_binding[key] is not None and trial_binding[key] is not None and control_binding[key] != trial_binding[key]:
            reasons.append(f"INPUT_BINDING_MISMATCH:{key}")
    if expected_annotation_sha256 is not None:
        if control_binding.get("annotation_sha256") != expected_annotation_sha256:
            reasons.append("CONTROL_ANNOTATION_MISMATCH")
        if trial_binding.get("annotation_sha256") != expected_annotation_sha256:
            reasons.append("TRIAL_ANNOTATION_MISMATCH")
    return reasons


def rank(args: argparse.Namespace) -> int:
    roots = [Path(value).resolve() for value in args.root]
    control_path = Path(args.control_receipt).resolve()
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control_base_assoc = _metric(control, "base", "AssocA")
    control_base_teta = _metric(control, "base", "TETA")
    if control_base_assoc is None or control_base_teta is None:
        raise ValueError("control receipt lacks finite Base AssocA/TETA")
    global_audit = None
    search_audit_arg = getattr(args, "search_audit", None)
    if search_audit_arg:
        global_audit = json.loads(Path(search_audit_arg).resolve().read_text(encoding="utf-8"))
        if (
            global_audit.get("status") not in {"PASS", "PARTIAL_PASS"}
            or global_audit.get("usage") != "SEARCH_SELECTION"
        ):
            raise ValueError("search audit is not selectable")
    expected_annotation_arg = getattr(args, "expected_annotation", None)
    expected_annotation_sha256 = (
        _sha256(Path(expected_annotation_arg).resolve()) if expected_annotation_arg else None
    )
    rows: list[dict[str, Any]] = []
    receipts, rejected = _receipts(roots, global_audit=global_audit)
    control_binding = _input_binding(control)
    control_binding_reasons = _binding_reasons(
        control,
        control,
        expected_annotation_sha256=expected_annotation_sha256,
    )
    for path, receipt in receipts:
        base_assoc = _metric(receipt, "base", "AssocA")
        base_teta = _metric(receipt, "base", "TETA")
        novel_assoc = _metric(receipt, "novel", "AssocA")
        # A missing Novel metric is an evaluator/provenance failure, never a
        # ranking advantage. Keep the receipt on disk but fail closed here.
        reasons: list[str] = []
        if base_assoc is None:
            reasons.append("BASE_ASSOCA_MISSING")
        if base_teta is None:
            reasons.append("BASE_TETA_MISSING")
        if novel_assoc is None:
            reasons.append("NOVEL_ASSOCA_MISSING")
        reasons.extend(
            _binding_reasons(
                control,
                receipt,
                expected_annotation_sha256=expected_annotation_sha256,
            )
        )
        if control_binding_reasons:
            reasons.extend(control_binding_reasons)
        if reasons:
            rejected.append({"trial_id": receipt.get("trial_id", path.parent.name), "reason": sorted(set(reasons))})
            continue
        novel_teta = _metric(receipt, "novel", "TETA")
        if novel_teta is None:
            rejected.append({"trial_id": receipt.get("trial_id", path.parent.name), "reason": ["NOVEL_TETA_MISSING"]})
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
            "novel_teta_for_ranking": novel_teta,
            "base_assoc_delta_vs_control": base_assoc - control_base_assoc,
            "base_teta_delta_vs_control": base_teta - control_base_teta,
            "eligible": base_assoc >= control_base_assoc - 1.0 and base_teta >= control_base_teta - 1.0,
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            not bool(row["eligible"]),
            -float(row["novel_assoc_for_ranking"]),
            -float(row["novel_teta_for_ranking"]),
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
        "control_binding": control_binding,
        "control_binding_rejected": control_binding_reasons,
        "search_audit": search_audit_arg,
        "all_completed": rows,
        "rejected_trials": rejected,
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
        f"Rejected trials: `{len(rejected)}`",
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
    parser.add_argument("--search-audit")
    parser.add_argument("--expected-annotation")
    return rank(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
