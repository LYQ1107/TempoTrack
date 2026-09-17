#!/usr/bin/env python3
"""Write the auditable final QDIC-MO comparison report.

The report is deliberately assembled only after the architecture comparison,
winner-only loss search, structural diagnostics, and the selected candidate's
Full-Test result exist.  Every metric delta is ``final - original COV native
Full-Test baseline``; subset, shard, and tuned-reference results are retained
as context only and are never silently substituted for that baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping


SPLITS = ("overall", "base", "novel")
METRICS = (
    "TETA",
    "LocA",
    "AssocA",
    "ClsA",
    "LocRe",
    "LocPr",
    "AssocRe",
    "AssocPr",
    "ClsRe",
    "ClsPr",
)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _require_metrics(record: Mapping[str, Any], label: str) -> dict[str, dict[str, float]]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"{label} does not contain a metrics mapping")
    result: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        values = metrics.get(split)
        if not isinstance(values, Mapping):
            raise RuntimeError(f"{label} is missing metrics split {split!r}")
        row: dict[str, float] = {}
        for name in METRICS:
            value = _number(values.get(name))
            if value is None:
                raise RuntimeError(f"{label} is missing finite {split}.{name}")
            row[name] = value
        result[split] = row
    return result


def _find_named_baseline(results: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    rows = results.get("baselines", [])
    if not isinstance(rows, list):
        raise RuntimeError("baseline results does not contain a baselines list")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("name")) == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one baseline named {name!r}, got {len(matches)}")
    return matches[0]


def _load_original_baseline(path: Path) -> dict[str, Any]:
    results = _read_json(path)
    if not isinstance(results, Mapping):
        raise RuntimeError("baseline results must be a JSON object")
    row = _find_named_baseline(results, "COV native baseline")
    if row.get("status") != "PASS":
        raise RuntimeError("COV native baseline is not PASS")
    if row.get("scope") not in {"FULL_TEST", "FULL_TEST_ORIGINAL_BASELINE"}:
        raise RuntimeError(
            "COV native baseline is not an original Full-Test result: "
            f"{row.get('scope')!r}"
        )
    if row.get("comparison_eligible") is False:
        raise RuntimeError("COV native baseline is marked comparison-ineligible")
    return {
        "name": row.get("name"),
        "status": row.get("status"),
        "scope": row.get("scope"),
        "path": row.get("path"),
        "metrics": _require_metrics(row, "COV native baseline"),
        "source_results": str(path),
        "source_results_sha256": _sha256(path),
        "summary_hash": row.get("summary_hash"),
        "summary_path": row.get("summary_path"),
        "provenance": row.get("provenance"),
    }


def _load_q1_references(path: Path) -> dict[str, dict[str, Any]]:
    results = _read_json(path)
    if not isinstance(results, Mapping):
        return {}
    references: dict[str, dict[str, Any]] = {}
    for name in ("Q1 OP00", "Q1 tuned champion"):
        try:
            row = _find_named_baseline(results, name)
        except RuntimeError:
            continue
        if row.get("status") != "PASS" or not isinstance(row.get("metrics"), Mapping):
            continue
        try:
            metrics = _require_metrics(row, name)
        except RuntimeError:
            continue
        references[name] = {
            "name": name,
            "status": row.get("status"),
            "scope": row.get("scope"),
            "path": row.get("path"),
            "metrics": metrics,
        }
    return references


def _load_completed(path: Path, label: str, allowed_statuses: set[str]) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    status = str(value.get("status"))
    if status not in allowed_statuses:
        raise RuntimeError(f"{label} is not complete: {status}")
    return dict(value)


def _load_optional_artifact(path: Path | None, label: str) -> dict[str, Any]:
    if path is None:
        return {"status": "NOT_PROVIDED", "payload": None}
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return {
        "status": str(value.get("status", "UNKNOWN")),
        "path": str(path),
        "sha256": _sha256(path),
        "artifact": value.get("artifact"),
        "payload": dict(value),
    }


def _margin_search_analysis(
    comparison_path: Path | None,
    distribution_path: Path | None,
    extension_path: Path | None,
) -> dict[str, Any]:
    """Load the prior margin-champion audit without conflating it with QDIC."""
    comparison = _load_optional_artifact(comparison_path, "margin search comparison")
    distributions = _load_optional_artifact(distribution_path, "margin score distributions")
    extension = _load_optional_artifact(extension_path, "margin extension plan")
    payload = comparison.get("payload") or {}
    distribution_payload = distributions.get("payload") or {}
    margin_rows = distribution_payload.get("margins", {})
    compact_distributions: dict[str, Any] = {}
    if isinstance(margin_rows, Mapping):
        for name, row in margin_rows.items():
            if not isinstance(row, Mapping):
                continue
            values = row.get("distribution", {})
            if not isinstance(values, Mapping):
                values = {}
            compact_distributions[str(name)] = {
                "trial_id": row.get("trial_id"),
                "score_threshold": row.get("score_threshold"),
                "margin_threshold": row.get("margin_threshold"),
                "winner_score_min": values.get("winner_score_min"),
                "p01": values.get("p01"),
                "p03": values.get("p03"),
                "p05": values.get("p05"),
                "p10": values.get("p10"),
                "p25": values.get("p25"),
                "p50": values.get("p50"),
                "p95": values.get("p95"),
            }
    return {
        "status": "PASS" if comparison.get("status") == "PASS" and distributions.get("status") == "PASS" else "INCOMPLETE",
        "comparison": {
            "path": comparison.get("path"),
            "sha256": comparison.get("sha256"),
            "status": comparison.get("status"),
        },
        "overall_margin_champion": payload.get("margin_champions"),
        "ov_margin_champion": payload.get("ov_margin_champion"),
        "program_champion": payload.get("program_champion"),
        "absolute_overall_teta_leader": payload.get("absolute_overall_teta_leader"),
        "score_distribution_artifact": {
            "path": distributions.get("path"),
            "sha256": distributions.get("sha256"),
            "status": distributions.get("status"),
            "margins": compact_distributions,
        },
        "extension_plan": {
            "path": extension.get("path"),
            "sha256": extension.get("sha256"),
            "status": extension.get("status"),
            "payload": extension.get("payload"),
        },
        "interpretation": "Margin champion selection and score search are bounded sequential evidence; they do not establish a global score-by-margin optimum.",
    }


def _load_final_result(path: Path) -> dict[str, Any]:
    result = _load_completed(path, "final candidate Full-Test result", {"PASS", "COMPLETED"})
    if not isinstance(result.get("metrics"), Mapping):
        raise RuntimeError("final candidate Full-Test result has no metrics")
    result["validated_metrics"] = _require_metrics(result, "final candidate Full-Test result")
    if result.get("unbiased_test") is True:
        raise RuntimeError("final result unexpectedly claims to be an unbiased Test")
    return result


def _candidate_training_receipt(final: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = final.get("checkpoint")
    if not checkpoint:
        return {}
    training_path = Path(str(checkpoint)).resolve().parent / "training.json"
    if not training_path.is_file():
        return {
            "status": "MISSING",
            "path": str(training_path),
        }
    value = _read_json(training_path)
    if not isinstance(value, Mapping):
        raise RuntimeError("candidate training receipt is not a JSON object")
    return {
        "status": "PASS",
        "path": str(training_path),
        "sha256": _sha256(training_path),
        "architecture_name": value.get("architecture_name"),
        "training_split": value.get("training_split"),
        "paper_status": value.get("paper_status"),
        "paper_valid": value.get("paper_valid"),
        "base_only_supervision": value.get("base_only_supervision"),
        "novel_gt_used": value.get("novel_gt_used"),
        "test_gt_used_for_optimizer": value.get("test_gt_used_for_optimizer"),
        "test_weights_used": value.get("test_weights_used"),
        "parent_v11_checkpoint": value.get("parent_v11_checkpoint"),
        "parent_v11_checkpoint_hash": value.get("parent_v11_checkpoint_hash"),
        "features": value.get("features"),
        "features_hash": value.get("features_hash"),
        "parent_frozen": value.get("parent_frozen"),
        "train_video_count": len(value.get("train_video_ids", [])),
        "holdout_video_count": len(value.get("holdout_video_ids", [])),
        "architecture_config": value.get("architecture_config"),
    }


def _canonical_architecture(value: Any) -> str:
    name = str(value).strip().upper()
    if name == "A0":
        return "A0"
    if name in {"A0-D", "A0_D", "A0DIST", "A0-DISTLOSS"}:
        return "A0-D"
    if name in {"A1", "A1-DS-QDIC", "DS-QDIC"}:
        return "A1-DS-QDIC"
    if name in {"A2", "A2-DGSA-QDIC", "DGSA-QDIC"}:
        return "A2-DGSA-QDIC"
    return name


def _validate_selection(
    final: Mapping[str, Any],
    architecture: Mapping[str, Any],
    loss_search: Mapping[str, Any],
) -> dict[str, Any]:
    if architecture.get("status") != "COMPLETED":
        raise RuntimeError("architecture comparison is not COMPLETED")
    if loss_search.get("status") != "COMPLETED":
        raise RuntimeError("winner-only loss search is not COMPLETED")
    selected = loss_search.get("selected")
    if not isinstance(selected, Mapping):
        raise RuntimeError("loss search has no selected checkpoint")
    final_arch = _canonical_architecture(final.get("architecture"))
    selected_arch = _canonical_architecture(selected.get("architecture_name"))
    if final_arch != selected_arch:
        raise RuntimeError(
            f"final architecture does not match loss-search winner: {final_arch!r} != {selected_arch!r}"
        )
    for key in ("lambda_dist", "lambda_hard"):
        final_value = _number(final.get(key))
        selected_value = _number(selected.get(key))
        if final_value is None or selected_value is None or abs(final_value - selected_value) > 1e-12:
            raise RuntimeError(
                f"final {key} does not match loss-search winner: {final_value!r} != {selected_value!r}"
            )
    final_hash = final.get("checkpoint_sha256")
    selected_hash = selected.get("checkpoint_hash")
    if final_hash and selected_hash and str(final_hash) != str(selected_hash):
        raise RuntimeError("final checkpoint hash does not match loss-search selected checkpoint")
    winner_arch = loss_search.get("winner_architecture")
    if winner_arch is not None and _canonical_architecture(winner_arch) != selected_arch:
        raise RuntimeError("loss-search winner_architecture disagrees with selected checkpoint")
    comparison_rows = architecture.get("architectures", [])
    if not isinstance(comparison_rows, list):
        raise RuntimeError("architecture comparison has no architecture rows")
    matching = [
        row
        for row in comparison_rows
        if isinstance(row, Mapping)
        and _canonical_architecture(row.get("architecture_name")) == selected_arch
    ]
    if not matching:
        raise RuntimeError("selected loss-search architecture is absent from architecture comparison")
    return {
        "selected_architecture": selected_arch,
        "lambda_dist": _number(selected.get("lambda_dist")),
        "lambda_hard": _number(selected.get("lambda_hard")),
        "checkpoint": selected.get("checkpoint"),
        "checkpoint_hash": selected.get("checkpoint_hash"),
        "architecture_comparison_row": matching[0],
    }


def _deltas(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {
        split: {
            name: float(left[split][name]) - float(right[split][name])
            for name in METRICS
        }
        for split in SPLITS
    }


def _reference_analysis(
    final_metrics: Mapping[str, Any], references: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, reference in references.items():
        metrics = reference["metrics"]
        delta = _deltas(final_metrics, metrics)
        result[name] = {
            "scope": reference.get("scope"),
            "metrics": metrics,
            "delta_final_minus_reference": delta,
            "overall_questions": {
                "association_exceeds_reference": delta["overall"]["AssocA"] > 0.0,
                "association_delta": delta["overall"]["AssocA"],
                "classification_accuracy_delta": delta["overall"]["ClsA"],
                "classification_recall_delta": delta["overall"]["ClsRe"],
                "localization_accuracy_delta": delta["overall"]["LocA"],
                "primary_negative_component": min(
                    ("LocA", delta["overall"]["LocA"]),
                    ("ClsA", delta["overall"]["ClsA"]),
                    key=lambda item: item[1],
                )[0],
            },
        }
    preferred = "Q1 tuned champion" if "Q1 tuned champion" in result else "Q1 OP00"
    return {
        "preferred_reference": preferred if preferred in result else None,
        "references": result,
    }


def _walk_metric_records(value: Any, path: str = "root") -> list[dict[str, Any]]:
    """Collect nested candidate records from old/new score-search receipts."""
    found: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("metrics"), Mapping):
            identity = (
                value.get("trial_id")
                or value.get("name")
                or value.get("label")
                or value.get("candidate_id")
                or path
            )
            try:
                metrics = _require_metrics(value, f"score-search record {identity}")
            except RuntimeError:
                metrics = None
            if metrics is not None:
                found.append(
                    {
                        "id": str(identity),
                        "score_threshold": value.get("score_threshold"),
                        "margin_threshold": value.get("margin_threshold"),
                        "metrics": metrics,
                        "source_path": path,
                    }
                )
        for key, child in value.items():
            found.extend(_walk_metric_records(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_metric_records(child, f"{path}[{index}]"))
    return found


def _score_analysis(
    path: Path | None,
    final_metrics: Mapping[str, Any],
    q1: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "NOT_PROVIDED",
            "records": [],
            "classification_recovery": "NOT_EVALUABLE_SCORE_SEARCH_EVIDENCE_MISSING",
        }
    value = _read_json(path)
    records = _walk_metric_records(value)
    # Keep one record per identity, preserving the first occurrence in a
    # nested aggregate that may repeat the same candidate under summaries.
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(str(record["id"]), record)
    rows = list(unique.values())
    if q1 is None:
        return {
            "status": "PASS_NO_Q1_REFERENCE",
            "source": str(path),
            "source_sha256": _sha256(path),
            "records": rows,
            "classification_recovery": "NOT_EVALUABLE_Q1_REFERENCE_MISSING",
        }
    q1_metrics = q1["metrics"]
    for row in rows:
        row["delta_final_minus_q1"] = _deltas(row["metrics"], q1_metrics)
    named: dict[str, list[dict[str, Any]]] = {key: [] for key in ("SCORE_OFF", "P03", "P05")}
    for row in rows:
        identity = str(row["id"]).upper().replace("-", "_")
        for key in named:
            if key in identity:
                named[key].append(row)
    selected_rows = {key: (values[0] if values else None) for key, values in named.items()}
    recoverable = []
    for key, row in selected_rows.items():
        if row is None:
            continue
        metrics = row["metrics"]["overall"]
        recoverable.append(
            {
                "label": key,
                "id": row["id"],
                "cls_a": metrics["ClsA"],
                "cls_re": metrics["ClsRe"],
                "assoc_a": metrics["AssocA"],
                "delta_cls_a_vs_q1": metrics["ClsA"] - q1_metrics["overall"]["ClsA"],
                "delta_cls_re_vs_q1": metrics["ClsRe"] - q1_metrics["overall"]["ClsRe"],
                "delta_assoc_a_vs_q1": metrics["AssocA"] - q1_metrics["overall"]["AssocA"],
            }
        )
    expected = tuple(named)
    complete_named = all(selected_rows[key] is not None for key in expected)
    classification_recovery = (
        "RECOVERED"
        if any(
            row["cls_a"] >= q1_metrics["overall"]["ClsA"] - 0.5
            and row["cls_re"] >= q1_metrics["overall"]["ClsRe"] - 0.5
            for row in recoverable
        )
        else ("NOT_RECOVERED" if complete_named else "INCOMPLETE_SCORE_SEARCH_EVIDENCE")
    )
    near_limit = None
    if complete_named:
        all_assoc_high = all(
            row["assoc_a"] >= q1_metrics["overall"]["AssocA"] + 0.5
            for row in recoverable
        )
        all_cls_low = all(
            row["cls_a"] <= q1_metrics["overall"]["ClsA"] - 0.5
            and row["cls_re"] <= q1_metrics["overall"]["ClsRe"] - 0.5
            for row in recoverable
        )
        if all_assoc_high and all_cls_low:
            near_limit = "THRESHOLD_CALIBRATION_NEAR_LIMIT"
    return {
        "status": "PASS",
        "source": str(path),
        "source_sha256": _sha256(path),
        "reference": q1.get("name"),
        "records": rows,
        "named_records": selected_rows,
        "classification_recovery": classification_recovery,
        "threshold_stop_condition": near_limit,
        "required_labels_present": {key: selected_rows[key] is not None for key in expected},
        "recovery_rows": recoverable,
        "note": "SCORE_OFF/P03/P05 are interpreted only from their own recorded candidate metrics; no cross-margin score quantile is substituted.",
    }


def _fmt(value: Any) -> str:
    number = _number(value)
    return "NA" if number is None else f"{number:.3f}"


def _metric_table(
    baseline: Mapping[str, Any], final: Mapping[str, Any], delta: Mapping[str, Any]
) -> str:
    lines = [
        "| Metric | COV native original | Final QDIC-MO | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name in METRICS:
        lines.append(
            f"| {name} | {_fmt(baseline[name])} | {_fmt(final[name])} | {_fmt(delta[name])} |"
        )
    return "\n".join(lines)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    baseline = report["baseline"]
    final = report["final"]
    lines = [
        "# TempoTrack V11 / QDIC-MO final experimental report",
        "",
        "> `TEST_TUNED_MODEL_SPECIFIC`; `NOT_UNBIASED_TEST`.",
        "",
        "All reported deltas below are **final result minus the original COV native Full-Test baseline**. Subset, shard, Q1-tuned, and structural-search results are not used as the baseline.",
        "",
        "## Final selection",
        "",
        f"- Architecture: `{final.get('architecture')}`",
        f"- lambda_dist: `{final.get('lambda_dist')}`; lambda_hard: `{final.get('lambda_hard')}`",
        f"- Final result: `{final.get('source')}`",
        f"- Final result SHA256: `{final.get('source_sha256')}`",
        f"- Checkpoint: `{final.get('checkpoint')}`",
        f"- Checkpoint SHA256: `{final.get('checkpoint_sha256')}`",
        "",
        "## Final versus original COV native baseline",
        "",
        f"- Baseline source: `{baseline.get('source_results')}`",
        f"- Baseline source SHA256: `{baseline.get('source_results_sha256')}`",
        f"- Official baseline summary: `{baseline.get('summary_path')}`",
        "",
    ]
    for split in SPLITS:
        lines.extend(
            [
                f"### {split.title()}",
                "",
                _metric_table(
                    baseline["metrics"][split],
                    final["metrics"][split],
                    report["deltas"][split],
                ),
                "",
            ]
        )
    q1 = report["q1_analysis"]
    lines.extend(["## Q1 diagnosis", ""])
    preferred = q1.get("preferred_reference")
    if preferred is None:
        lines.append("Q1 reference metrics are unavailable; no Q1-based causal interpretation is claimed.")
    else:
        lines.append(f"Preferred Q1 reference for diagnosis: `{preferred}`. It is not the final baseline.")
        row = q1["references"][preferred]
        questions = row["overall_questions"]
        lines.extend(
            [
                "",
                f"1. Association exceeds Q1: `{questions['association_exceeds_reference']}`; Delta AssocA=`{_fmt(questions['association_delta'])}`.",
                f"2. The larger negative Overall component between LocA and ClsA is `{questions['primary_negative_component']}`; Delta LocA=`{_fmt(questions['localization_accuracy_delta'])}`, Delta ClsA=`{_fmt(questions['classification_accuracy_delta'])}`.",
                f"3. Delta ClsRe versus Q1=`{_fmt(questions['classification_recall_delta'])}`.",
                "",
                "| Q1 reference | Overall TETA Δ | Overall AssocA Δ | Overall ClsA Δ | Overall ClsRe Δ | Base AssocA Δ | Novel AssocA Δ |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, ref in q1["references"].items():
            d = ref["delta_final_minus_reference"]
            lines.append(
                f"| {name} | {_fmt(d['overall']['TETA'])} | {_fmt(d['overall']['AssocA'])} | {_fmt(d['overall']['ClsA'])} | {_fmt(d['overall']['ClsRe'])} | {_fmt(d['base']['AssocA'])} | {_fmt(d['novel']['AssocA'])} |"
            )
    score = report["score_analysis"]
    lines.extend(
        [
            "",
            "## SCORE_OFF / P03 / P05 classification check",
            "",
            f"- Evidence status: `{score.get('status')}`",
            f"- Classification recovery: `{score.get('classification_recovery')}`",
            f"- Threshold stop condition: `{score.get('threshold_stop_condition')}`",
            "",
            "The score search is a bounded sequential search. A single margin's score distribution is not a global score×margin optimum, and each margin's score quantiles must remain causal to that margin.",
        ]
    )
    if score.get("recovery_rows"):
        lines.extend(
            [
                "",
                "| Candidate | ClsA | ClsRe | AssocA | ΔClsA vs Q1 | ΔClsRe vs Q1 | ΔAssocA vs Q1 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in score["recovery_rows"]:
            lines.append(
                f"| {row['label']} | {_fmt(row['cls_a'])} | {_fmt(row['cls_re'])} | {_fmt(row['assoc_a'])} | {_fmt(row['delta_cls_a_vs_q1'])} | {_fmt(row['delta_cls_re_vs_q1'])} | {_fmt(row['delta_assoc_a_vs_q1'])} |"
            )
    margin = report.get("margin_search", {})
    lines.extend(
        [
            "",
            "## Margin champion and score-distribution audit",
            "",
            f"- Audit status: `{margin.get('status')}`",
            f"- `OVERALL_MARGIN_CHAMPION`: `{(margin.get('overall_margin_champion') or {}).get('trial_id')}`; margin=`{(margin.get('overall_margin_champion') or {}).get('margin_threshold')}`.",
            f"- `OV_MARGIN_CHAMPION`: `{(margin.get('ov_margin_champion') or {}).get('trial_id')}`; margin=`{(margin.get('ov_margin_champion') or {}).get('margin_threshold')}`.",
            "- These are separate selection definitions. Wave-S score search is a bounded sequential search around one selected margin and is not a global score×margin optimum.",
            "",
            "| Margin trial | Margin | winner min | p01 | p03 | p05 | p10 | p25 | p50 | p95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in (margin.get("score_distribution_artifact", {}).get("margins", {}) or {}).items():
        lines.append(
            f"| {name} | {_fmt(row.get('margin_threshold'))} | {_fmt(row.get('winner_score_min'))} | {_fmt(row.get('p01'))} | {_fmt(row.get('p03'))} | {_fmt(row.get('p05'))} | {_fmt(row.get('p10'))} | {_fmt(row.get('p25'))} | {_fmt(row.get('p50'))} | {_fmt(row.get('p95'))} |"
        )
    extension_payload = (margin.get("extension_plan") or {}).get("payload") or {}
    lines.extend(
        [
            "",
            f"- Extension plan: `{extension_payload.get('status', 'NOT_PROVIDED')}`; `launch={extension_payload.get('launch')}`. It is recorded for review and is not auto-started.",
        ]
    )
    lines.extend(
        [
            "",
            "## Structural and identity diagnostics",
            "",
            f"- Structural analysis: `{report['structure']['analysis_path']}` (SHA256 `{report['structure']['analysis_sha256']}`)",
            f"- Structural behavior sanity: `{report['structure']['sanity_path']}` (SHA256 `{report['structure']['sanity_sha256']}`), status=`{report['structure']['sanity_status']}`",
            "- Strict identity correctness excludes local-track IDs with mixed GT identity mapping; those rows are reported as `ambiguous_identity_mapping`/`accepted_ambiguous`.",
            "- G720 is `LONG_HORIZON_EXTRAPOLATION`: runtime legal horizon=720 while learned feature normalization max_gap=360.",
            f"- Bottleneck report: `{report['structure'].get('bottleneck_report')}`",
            "",
            "## Training and provenance",
            "",
            f"- Architecture comparison: `{report['architecture_comparison']['path']}` (SHA256 `{report['architecture_comparison']['sha256']}`)",
            f"- Winner-only loss search: `{report['loss_search']['path']}` (SHA256 `{report['loss_search']['sha256']}`)",
            f"- Parent V11 checkpoint: `{final.get('parent_v11_checkpoint')}` (SHA256 `{final.get('parent_v11_checkpoint_hash')}`)",
            f"- Candidate training receipt: `{final.get('training_receipt', {}).get('path')}` (status `{final.get('training_receipt', {}).get('status')}`)",
            f"- Final preflight: `{final.get('preflight')}` (SHA256 `{final.get('preflight_sha256')}`)",
            f"- Runtime environment source: `{final.get('runtime_environment', {}).get('source', 'embedded in final preflight')}`",
            "- Training contract: Base-only supervision, video-disjoint train/holdout, Novel GT unused, Test GT unused for optimizer, parent V11 frozen.",
            "",
            "## Reproducibility index",
            "",
            "The structured JSON beside this report contains the complete metric matrices, deltas, selection checks, score-search evidence, structural hashes, candidate diagnostics, and final-result provenance.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline_results).resolve()
    final_path = Path(args.final_result).resolve()
    architecture_path = Path(args.architecture_comparison).resolve()
    loss_path = Path(args.loss_search).resolve()
    structure_path = Path(args.structural_analysis).resolve()
    sanity_path = Path(args.structural_sanity).resolve()
    baseline = _load_original_baseline(baseline_path)
    final_raw = _load_final_result(final_path)
    architecture = _load_completed(architecture_path, "architecture comparison", {"COMPLETED"})
    loss_search = _load_completed(loss_path, "winner-only loss search", {"COMPLETED"})
    selection = _validate_selection(final_raw, architecture, loss_search)
    structural = _load_completed(structure_path, "structural analysis", {"COMPLETED"})
    sanity = _load_json_sanity = _read_json(sanity_path)
    if _load_json_sanity.get("status") != "PASS":
        raise RuntimeError("structural behavior sanity is not PASS")
    if structural.get("structure_gate", {}).get("sanity_sha256") not in {None, _sha256(sanity_path)}:
        raise RuntimeError("structural analysis sanity hash does not match supplied sanity artifact")
    q1_references = _load_q1_references(baseline_path)
    final_metrics = final_raw["validated_metrics"]
    training_receipt = _candidate_training_receipt(final_raw)
    score_path = None if args.score_search is None else Path(args.score_search).resolve()
    margin_comparison_path = (
        None if args.margin_search is None else Path(args.margin_search).resolve()
    )
    margin_distribution_path = (
        None
        if args.margin_score_distributions is None
        else Path(args.margin_score_distributions).resolve()
    )
    margin_extension_path = (
        None if args.margin_extension_plan is None else Path(args.margin_extension_plan).resolve()
    )
    preferred_q1_name = "Q1 tuned champion" if "Q1 tuned champion" in q1_references else "Q1 OP00"
    q1 = q1_references.get(preferred_q1_name)
    score_analysis = _score_analysis(score_path, final_metrics, q1)
    margin_search = _margin_search_analysis(
        margin_comparison_path,
        margin_distribution_path,
        margin_extension_path,
    )
    structure_gate = structural.get("structure_gate", {})
    report = {
        "schema_version": 1,
        "artifact": "tempotrack_v11_qdic_mo_final_experimental_report",
        "status": "COMPLETED",
        "labels": ["TEST_TUNED_MODEL_SPECIFIC", "NOT_UNBIASED_TEST"],
        "baseline": baseline,
        "final": {
            "source": str(final_path),
            "source_sha256": _sha256(final_path),
            "status": final_raw.get("status"),
            "artifact": final_raw.get("artifact"),
            "architecture": final_raw.get("architecture"),
            "lambda_dist": final_raw.get("lambda_dist"),
            "lambda_hard": final_raw.get("lambda_hard"),
            "checkpoint": final_raw.get("checkpoint"),
            "checkpoint_sha256": final_raw.get("checkpoint_sha256"),
            "preflight": final_raw.get("preflight"),
            "preflight_sha256": final_raw.get("preflight_sha256"),
            "plan": final_raw.get("plan"),
            "plan_sha256": final_raw.get("plan_sha256"),
            "parent_v11_checkpoint": training_receipt.get("parent_v11_checkpoint"),
            "parent_v11_checkpoint_hash": training_receipt.get("parent_v11_checkpoint_hash"),
            "runtime_environment": final_raw.get("runtime_environment"),
            "training_receipt": training_receipt,
            "metrics": final_metrics,
            "diagnostics": final_raw.get("diagnostics"),
            "structure_gate": final_raw.get("structure_gate"),
        },
        "deltas": _deltas(final_metrics, baseline["metrics"]),
        "q1_analysis": _reference_analysis(final_metrics, q1_references),
        "score_analysis": score_analysis,
        "margin_search": margin_search,
        "selection": selection,
        "architecture_comparison": {
            "path": str(architecture_path),
            "sha256": _sha256(architecture_path),
            "status": architecture.get("status"),
            "paper_status": architecture.get("paper_status"),
            "paper_valid": architecture.get("paper_valid"),
            "contract": architecture.get("contract"),
        },
        "loss_search": {
            "path": str(loss_path),
            "sha256": _sha256(loss_path),
            "status": loss_search.get("status"),
            "winner_architecture": loss_search.get("winner_architecture"),
            "search_contract": loss_search.get("search_contract"),
            "selected": loss_search.get("selected"),
        },
        "structure": {
            "analysis_path": str(structure_path),
            "analysis_sha256": _sha256(structure_path),
            "analysis_status": structural.get("status"),
            "sanity_path": str(sanity_path),
            "sanity_sha256": _sha256(sanity_path),
            "sanity_status": sanity.get("status"),
            "gate": structure_gate,
            "bottleneck_report": structural.get("outputs", {}).get("diagnosis"),
            "horizon_interpretation": structural.get("horizon_interpretation"),
        },
        "provenance": {
            "baseline_results": str(baseline_path),
            "baseline_results_sha256": _sha256(baseline_path),
            "final_result": str(final_path),
            "final_result_sha256": _sha256(final_path),
            "architecture_comparison": str(architecture_path),
            "architecture_comparison_sha256": _sha256(architecture_path),
            "loss_search": str(loss_path),
            "loss_search_sha256": _sha256(loss_path),
            "structural_analysis": str(structure_path),
            "structural_analysis_sha256": _sha256(structure_path),
            "structural_sanity": str(sanity_path),
            "structural_sanity_sha256": _sha256(sanity_path),
            "final_embedded_preflight": final_raw.get("preflight"),
            "final_embedded_runtime_environment": final_raw.get("runtime_environment"),
            "candidate_training_receipt": training_receipt,
        },
        "generated_at_unix": time.time(),
    }
    output_root = Path(args.output_root).resolve()
    _write_json(output_root / "final_report.json", report)
    _write_markdown(output_root / "final_report.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--final-result", required=True)
    parser.add_argument("--architecture-comparison", required=True)
    parser.add_argument("--loss-search", required=True)
    parser.add_argument("--structural-analysis", required=True)
    parser.add_argument("--structural-sanity", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--score-search")
    parser.add_argument("--margin-search")
    parser.add_argument("--margin-score-distributions")
    parser.add_argument("--margin-extension-plan")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
