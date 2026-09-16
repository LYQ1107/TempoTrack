#!/usr/bin/env python3
"""Aggregate the structural Full-Test evidence into QDIC bottleneck reports.

This is a post-hoc, read-only analysis of completed structural candidates.  It
does not read Test GT during inference and it never uses a structural metric
to alter a running search plan.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tempotrack_v10.qdic_trainer import sha256  # noqa: E402


TRIAL_IDS = (
    "ST1_K16_G360",
    "ST2_K32_G360",
    "ST3_K64_G360",
    "ST4_K32_G180",
    "ST5_K32_G720",
    "ST6_K64_G720",
)
EXPECTED_SPECS = {
    "ST1_K16_G360": (16, 360),
    "ST2_K32_G360": (32, 360),
    "ST3_K64_G360": (64, 360),
    "ST4_K32_G180": (32, 180),
    "ST5_K32_G720": (32, 720),
    "ST6_K64_G720": (64, 720),
}
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
GAP_BINS = ("gap_le_180", "gap_181_360", "gap_361_720", "gap_gt_720")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _structure_gate(root: Path) -> dict[str, Any]:
    state_path = root / "search_state.json"
    sanity_path = root / "structural_behavior_sanity.json"
    if not state_path.is_file() or not sanity_path.is_file():
        raise RuntimeError("structural search gate artifacts are missing")
    state = _read_json(state_path)
    sanity = _read_json(sanity_path)
    trials = {
        str(item.get("trial_id")): item
        for item in state.get("trials", [])
        if isinstance(item, Mapping)
    }
    incomplete = [
        trial_id
        for trial_id in TRIAL_IDS
        if trial_id not in trials
        or trials[trial_id].get("status") != "COMPLETED"
        or not trials[trial_id].get("full_test_end_unix")
    ]
    if incomplete:
        raise RuntimeError(f"structural search is incomplete: {incomplete}")
    if sanity.get("status") != "PASS" or state.get("structural_behavior_sanity_status") != "PASS":
        raise RuntimeError("structural behavior sanity is not PASS")
    preflight = root / "preflight.json"
    if not preflight.is_file() or sanity.get("preflight_sha256") != sha256(preflight):
        raise RuntimeError("structural preflight provenance hash mismatch")
    return {
        "state_status": state.get("status"),
        "state_sha256": sha256(state_path),
        "sanity_sha256": sha256(sanity_path),
        "preflight_sha256": sha256(preflight),
        "trial_ids": list(TRIAL_IDS),
    }


def _load_candidates(root: Path) -> list[dict[str, Any]]:
    result_path = root / "structure_search_results.json"
    candidates: list[dict[str, Any]] = []
    if result_path.is_file():
        value = _read_json(result_path)
        candidates = [dict(item) for item in value.get("candidates", [])]
    if not candidates:
        state = _read_json(root / "search_state.json")
        candidates = [dict(item) for item in state.get("trials", [])]
    by_id = {str(item.get("trial_id")): item for item in candidates}
    loaded: list[dict[str, Any]] = []
    for trial_id in TRIAL_IDS:
        if trial_id not in by_id:
            raise RuntimeError(f"structural candidate missing from results: {trial_id}")
        item = dict(by_id[trial_id])
        expected_k, expected_gap = EXPECTED_SPECS[trial_id]
        if int(item.get("candidate_top_k", -1)) != expected_k or int(item.get("max_gap", -1)) != expected_gap:
            raise RuntimeError(
                f"structural candidate spec mismatch for {trial_id}: "
                f"got K={item.get('candidate_top_k')} gap={item.get('max_gap')}"
            )
        diagnostic = item.get("diagnostics")
        if not isinstance(diagnostic, Mapping):
            path = item.get("diagnostics_path")
            if path and Path(str(path)).is_file():
                diagnostic = _read_json(Path(str(path)))
        if not isinstance(diagnostic, Mapping):
            raise RuntimeError(f"diagnostics missing for {trial_id}")
        item["diagnostics"] = dict(diagnostic)
        item["horizon_label"] = (
            "LONG_HORIZON_EXTRAPOLATION"
            if int(item.get("max_gap", 360)) > 360
            else "WITHIN_CHECKPOINT_FEATURE_HORIZON"
        )
        loaded.append(item)
    return loaded


def _metric_row(item: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trial_id": item.get("trial_id"),
        "candidate_top_k": item.get("candidate_top_k"),
        "max_gap": item.get("max_gap"),
        "horizon_label": item.get("horizon_label"),
        "status": item.get("status"),
    }
    metrics = item.get("metrics", {})
    for split in ("overall", "base", "novel"):
        values = metrics.get(split, {}) if isinstance(metrics, Mapping) else {}
        for name in METRICS:
            row[f"{split}_{name}"] = values.get(name) if isinstance(values, Mapping) else None
    return row


def _group_rows(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostic = item["diagnostics"]
    groups = diagnostic.get("groups", {})
    rows: list[dict[str, Any]] = []
    for group_name in ("overall", "base", "novel"):
        group = groups.get(group_name, {}) if isinstance(groups, Mapping) else {}
        rows.append(
            {
                "trial_id": item.get("trial_id"),
                "candidate_top_k": item.get("candidate_top_k"),
                "max_gap": item.get("max_gap"),
                "horizon_label": item.get("horizon_label"),
                "group": group_name,
                "association_events": group.get("association_events"),
                "positive_events": group.get("positive_events"),
                "candidate_recall": group.get("candidate_recall"),
                "prefilter_recall_at_64": (group.get("prefilter_recall_at") or {}).get("64"),
                "qdic_recall_at_64": (group.get("qdic_recall_at") or {}).get("64"),
                "strict_association_recall": group.get("association_recall"),
                "relaxed_association_recall": group.get("association_recall_relaxed"),
                "strict_association_precision": group.get("association_precision"),
                "relaxed_association_precision": group.get("association_precision_relaxed"),
                "accepted_total": group.get("accepted_total"),
                "accepted_correct_strict": group.get("accepted_correct"),
                "accepted_correct_relaxed": group.get("accepted_correct_relaxed"),
                "ambiguous_identity_mapping": group.get("ambiguous_identity_mapping"),
                "accepted_ambiguous": group.get("accepted_ambiguous"),
                "false_merge": group.get("false_merge"),
                "accepted_unresolved": group.get("accepted_unresolved"),
                "rejected_true_association": group.get("rejected_true_association"),
            }
        )
    return rows


def _rank_rows(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostic = item["diagnostics"]
    groups = diagnostic.get("groups", {})
    rows: list[dict[str, Any]] = []
    for group_name in ("overall", "base", "novel"):
        group = groups.get(group_name, {}) if isinstance(groups, Mapping) else {}
        pre = group.get("prefilter_recall_at", {}) or {}
        post = group.get("qdic_recall_at", {}) or {}
        for rank in (1, 8, 16, 32, 64):
            rows.append(
                {
                    "trial_id": item.get("trial_id"),
                    "candidate_top_k": item.get("candidate_top_k"),
                    "max_gap": item.get("max_gap"),
                    "horizon_label": item.get("horizon_label"),
                    "group": group_name,
                    "rank": rank,
                    "prefilter_recall": pre.get(str(rank)),
                    "qdic_recall": post.get(str(rank)),
                    "qdic_minus_prefilter": (
                        None
                        if _number(post.get(str(rank))) is None or _number(pre.get(str(rank))) is None
                        else _number(post.get(str(rank))) - _number(pre.get(str(rank)))
                    ),
                    "prefilter_rank_p50": (group.get("prefilter_rank_quantiles") or {}).get("p50"),
                    "qdic_rank_p50": (group.get("qdic_rank_quantiles") or {}).get("p50"),
                    "prefilter_rank_p95": (group.get("prefilter_rank_quantiles") or {}).get("p95"),
                    "qdic_rank_p95": (group.get("qdic_rank_quantiles") or {}).get("p95"),
                }
            )
    return rows


def _temporal_rows(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    temporal = item["diagnostics"].get("temporal_gap_bins", {})
    rows: list[dict[str, Any]] = []
    for gap_bin in GAP_BINS:
        group = temporal.get(gap_bin, {}) if isinstance(temporal, Mapping) else {}
        rows.append(
            {
                "trial_id": item.get("trial_id"),
                "candidate_top_k": item.get("candidate_top_k"),
                "max_gap": item.get("max_gap"),
                "horizon_label": item.get("horizon_label"),
                "gap_bin": gap_bin,
                "association_events": group.get("association_events"),
                "positive_events": group.get("positive_events"),
                "candidate_recall": group.get("candidate_recall"),
                "prefilter_recall_at_64": (group.get("prefilter_recall_at") or {}).get("64"),
                "qdic_recall_at_64": (group.get("qdic_recall_at") or {}).get("64"),
                "strict_association_recall": group.get("association_recall"),
                "relaxed_association_recall": group.get("association_recall_relaxed"),
                "ambiguous_identity_mapping": group.get("ambiguous_identity_mapping"),
                "accepted_ambiguous": group.get("accepted_ambiguous"),
                "false_merge": group.get("false_merge"),
                "accepted_unresolved": group.get("accepted_unresolved"),
            }
        )
    return rows


def _diagnosis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = next(
        row for row in rows if row.get("group") == "overall"
    )
    supply = _number(overall.get("candidate_recall"))
    pre = _number(overall.get("prefilter_recall_at_64"))
    post = _number(overall.get("qdic_recall_at_64"))
    accepted = _number(overall.get("accepted_total")) or 0.0
    unresolved = _number(overall.get("accepted_unresolved")) or 0.0
    ambiguous = _number(overall.get("accepted_ambiguous")) or 0.0
    rejected = _number(overall.get("rejected_true_association")) or 0.0
    components: list[str] = []
    if supply is not None and supply < 0.70:
        components.append("CANDIDATE_SUPPLY_LIMITED")
    if pre is not None and post is not None and pre - post > 0.08:
        components.append("QDIC_RANKING_LIMITED")
    if accepted > 0 and unresolved / accepted > 0.25:
        components.append("STALE_MEMORY_LIMITED")
    if accepted > 0 and ambiguous / accepted > 0.10:
        components.append("AMBIGUOUS_IDENTITY_MAPPING_PRESENT")
    return {
        "label": components[0] if len(components) == 1 else "MIXED",
        "components": components,
        "evidence": {
            "candidate_recall": supply,
            "prefilter_recall_at_64": pre,
            "qdic_recall_at_64": post,
            "prefilter_minus_qdic_recall_at_64": (
                None if pre is None or post is None else pre - post
            ),
            "accepted_total": accepted,
            "accepted_unresolved": unresolved,
            "accepted_ambiguous": ambiguous,
            "rejected_true_association": rejected,
            "strict_statistics_exclude_mixed_identity_mappings": True,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    structure_root = Path(args.structure_search_root).resolve()
    output_root = Path(args.output_root).resolve()
    gate = _structure_gate(structure_root)
    candidates = _load_candidates(structure_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metrics = [_metric_row(item) for item in candidates]
    groups: list[dict[str, Any]] = []
    ranks: list[dict[str, Any]] = []
    temporal: list[dict[str, Any]] = []
    diagnoses: list[dict[str, Any]] = []
    for item in candidates:
        group_rows = _group_rows(item)
        groups.extend(group_rows)
        ranks.extend(_rank_rows(item))
        temporal.extend(_temporal_rows(item))
        diagnoses.append(
            {
                "trial_id": item.get("trial_id"),
                "candidate_top_k": item.get("candidate_top_k"),
                "max_gap": item.get("max_gap"),
                "horizon_label": item.get("horizon_label"),
                "diagnosis": _diagnosis(group_rows),
            }
        )

    overall_metric_candidates = [
        item for item in candidates if isinstance(item.get("metrics", {}).get("overall"), Mapping)
    ]
    best = max(
        overall_metric_candidates,
        key=lambda item: float(item["metrics"]["overall"].get("TETA", float("-inf"))),
        default=None,
    )
    final = {
        "schema_version": 1,
        "artifact": "tempotrack_v11_qdic_structural_search_final_analysis",
        "status": "COMPLETED",
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "structure_gate": gate,
        "source_structure_root": str(structure_root),
        "source_structure_results_sha256": (
            sha256(structure_root / "structure_search_results.json")
            if (structure_root / "structure_search_results.json").is_file()
            else None
        ),
        "learned_feature_contract": {
            "decision_candidate_top_k": 8,
            "max_gap": 360,
        },
        "horizon_interpretation": {
            "g720_label": "LONG_HORIZON_EXTRAPOLATION",
            "runtime_legal_horizon": 720,
            "checkpoint_feature_normalization_max_gap": 360,
            "note": "G720 gaps above 360 are inference extrapolation outside the learned feature horizon, not a pure memory-horizon effect.",
        },
        "candidates": metrics,
        "overall_teta_champion": None if best is None else best.get("trial_id"),
        "bottleneck_diagnosis": diagnoses,
        "outputs": {
            "metrics": str(output_root / "structural_search_final_metrics.csv"),
            "groups": str(output_root / "candidate_supply_analysis.csv"),
            "ranks": str(output_root / "rank_bucket_analysis.csv"),
            "temporal": str(output_root / "temporal_gap_analysis.csv"),
            "diagnosis": str(output_root / "qdic_bottleneck_report.md"),
        },
        "generated_at_unix": time.time(),
    }
    _write_json(output_root / "structural_search_final.json", final)
    _write_json(output_root / "structural_search_final_metrics.json", final)
    _write_csv(output_root / "structural_search_final_metrics.csv", metrics)
    _write_csv(output_root / "candidate_supply_analysis.csv", groups)
    _write_csv(output_root / "rank_bucket_analysis.csv", ranks)
    _write_csv(output_root / "temporal_gap_analysis.csv", temporal)
    _write_csv(output_root / "correction_regression.csv", groups)
    _write_csv(output_root / "decision_funnel.csv", groups)

    lines = [
        "# QDIC bottleneck report",
        "",
        "> `TEST_TUNED_MODEL_SPECIFIC`; `NOT_UNBIASED_TEST`.",
        "",
        f"- Structural source: `{structure_root}`",
        f"- Overall TETA champion: `{final['overall_teta_champion']}`",
        "- QDIC learned feature contract: decision K=8, feature normalization max_gap=360.",
        "- G720 is `LONG_HORIZON_EXTRAPOLATION`: runtime legal horizon=720, but gaps above 360 are outside the learned feature normalization horizon.",
        "",
        "## Candidate supply / rank / decision summary",
        "",
        "| Trial | K | max_gap | Horizon | Candidate recall | Prefilter R@64 | QDIC R@64 | Strict Assoc recall | Relaxed Assoc recall | Ambiguous mapping | Accepted unresolved |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        if row.get("group") != "overall":
            continue
        def fmt(value: Any) -> str:
            number = _number(value)
            return "NA" if number is None else f"{number:.4f}"
        lines.append(
            f"| {row.get('trial_id')} | {row.get('candidate_top_k')} | {row.get('max_gap')} | {row.get('horizon_label')} | "
            f"{fmt(row.get('candidate_recall'))} | {fmt(row.get('prefilter_recall_at_64'))} | {fmt(row.get('qdic_recall_at_64'))} | "
            f"{fmt(row.get('strict_association_recall'))} | {fmt(row.get('relaxed_association_recall'))} | "
            f"{row.get('ambiguous_identity_mapping', 'NA')} | {row.get('accepted_unresolved', 'NA')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`accepted_correct` and strict association recall/precision require the assigned local track to map uniquely to the target GT identity. Mixed mappings are counted under `ambiguous_identity_mapping` and are excluded from strict correctness; relaxed metrics are reported separately for diagnosis.",
            "",
        ]
    )
    for item in diagnoses:
        diagnosis = item["diagnosis"]
        lines.append(
            f"- `{item['trial_id']}`: `{diagnosis['label']}`; components=`{','.join(diagnosis['components']) or 'none'}`; evidence=`{json.dumps(diagnosis['evidence'], sort_keys=True)}`"
        )
    lines.extend(
        [
            "",
            "## Output index",
            "",
            "- `structural_search_final.json`: gated aggregate and provenance hashes.",
            "- `structural_search_final_metrics.csv`: Overall/Base/Novel ten metrics per structural candidate.",
            "- `candidate_supply_analysis.csv`: candidate supply and strict/relaxed decision funnel.",
            "- `rank_bucket_analysis.csv`: prefilter versus QDIC rank recall at 1/8/16/32/64.",
            "- `temporal_gap_analysis.csv`: causal gap-bin breakdown.",
            "- `correction_regression.csv` and `decision_funnel.csv`: auditable event-count views.",
        ]
    )
    (output_root / "qdic_bottleneck_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-search-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
