#!/usr/bin/env python3
"""Aggregate the completed V12 Test-tuned ranking and Full-Test artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


RANKING_METHODS = (
    "Q1_SUPPORT",
    "MEAN",
    "MAX",
    "MO2",
    "EXACT_LMGF_BETA_M2",
    "EXACT_LMGF_BETA_M1",
    "EXACT_LMGF_BETA_M05",
    "EXACT_LMGF_BETA_P025",
    "EXACT_LMGF_BETA_P05",
    "EXACT_LMGF_BETA_P1",
    "EXACT_LMGF_BETA_P2",
    "EXACT_LMGF_BETA_P4",
    "B0",
) + tuple(f"E{index:02d}" for index in range(1, 19))
EXACT_BETAS = {
    "EXACT_LMGF_BETA_M2": -2.0,
    "EXACT_LMGF_BETA_M1": -1.0,
    "EXACT_LMGF_BETA_M05": -0.5,
    "EXACT_LMGF_BETA_P025": 0.25,
    "EXACT_LMGF_BETA_P05": 0.5,
    "EXACT_LMGF_BETA_P1": 1.0,
    "EXACT_LMGF_BETA_P2": 2.0,
    "EXACT_LMGF_BETA_P4": 4.0,
}
TETA_FIELDS = ("TETA", "LocA", "AssocA", "ClsA", "LocRe", "LocPr", "AssocRe", "AssocPr", "ClsRe", "ClsPr")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ranking_row(ranking: dict[str, Any], method: str) -> dict[str, Any]:
    metrics = ranking.get("metrics", {})
    result: dict[str, Any] = {}
    for split_key, prefix in (("TRAIN_HOLDOUT", "train_holdout"), ("VAL", "val"), ("TEST", "test")):
        values = metrics.get(split_key, {}).get(method, {})
        overall = values.get("Overall", {})
        novel = values.get("Novel", {})
        result[f"{prefix}_top1"] = _number(overall.get("top1"))
        result[f"{prefix}_novel_top1"] = _number(novel.get("top1"))
        result[f"{prefix}_mrr"] = _number(overall.get("mrr"))
        result[f"{prefix}_novel_mrr"] = _number(novel.get("mrr"))
        result[f"{prefix}_corrections"] = overall.get("raw_wrong_to_method_correct")
        result[f"{prefix}_regressions"] = overall.get("raw_correct_to_method_wrong")
        result[f"{prefix}_net_correction"] = overall.get("net_correction")
    return result


def _method_metadata(ranking: dict[str, Any], method: str) -> dict[str, Any]:
    if method in ranking.get("card_provenance", {}):
        provenance = ranking["card_provenance"][method]
        card_type = str(provenance.get("card_type", "fixed"))
        return {
            "beta": _number(provenance.get("mgf_beta")),
            "mode": provenance.get("mgf_mode"),
            "adaptive": card_type == "adaptive",
            "card_type": card_type,
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        }
    if method == "B0":
        return {"beta": None, "mode": "legacy", "adaptive": False, "card_type": "b0"}
    if method in EXACT_BETAS:
        return {"beta": EXACT_BETAS[method], "mode": "analytic", "adaptive": False, "card_type": "analytic"}
    return {"beta": None, "mode": "analytic", "adaptive": False, "card_type": "analytic"}


def _candidate_metrics(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics = read_json(path)
    if not isinstance(metrics, dict) or not all(isinstance(metrics.get(key), dict) for key in ("overall", "base", "novel")):
        raise RuntimeError(f"invalid Test-tuned metrics: {path}")
    merged_manifest = path.parent / "manifest.json"
    if not merged_manifest.is_file():
        raise FileNotFoundError(f"merged manifest missing: {merged_manifest}")
    manifest = read_json(merged_manifest)
    if manifest.get("status") != "PASS" or int(manifest.get("detector_forward_calls", -1)) != 0:
        raise RuntimeError(f"merged replay contract failed: {merged_manifest}")
    if manifest.get("gt_loaded_during_replay") is not False:
        raise RuntimeError(f"GT was loaded during replay: {merged_manifest}")
    evaluation_files = sorted((path.parent / "evaluation").glob("*/evaluation.json"))
    evaluation = read_json(evaluation_files[0]) if evaluation_files else {}
    return metrics, manifest, evaluation


def _flatten_teta(row: dict[str, Any], metrics: dict[str, Any]) -> None:
    for split in ("overall", "base", "novel"):
        values = metrics.get(split, {})
        for field in TETA_FIELDS:
            row[f"test_{split}_{field}"] = _number(values.get(field))
    row["test_teta"] = row.get("test_overall_TETA")
    row["test_assocA"] = row.get("test_overall_AssocA")
    row["test_base_assocA"] = row.get("test_base_AssocA")
    row["test_novel_assocA"] = row.get("test_novel_AssocA")
    row["test_locA"] = row.get("test_overall_LocA")
    row["test_clsA"] = row.get("test_overall_ClsA")


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        float(row.get("test_novel_assocA") if row.get("test_novel_assocA") is not None else float("-inf")),
        float(row.get("test_assocA") if row.get("test_assocA") is not None else float("-inf")),
        float(row.get("test_teta") if row.get("test_teta") is not None else float("-inf")),
        float(row.get("test_base_assocA") if row.get("test_base_assocA") is not None else float("-inf")),
        str(row.get("candidate_id", "")),
    )


def _delta(best: dict[str, Any], baseline: dict[str, Any], metric: str, split: str = "overall") -> float | None:
    left = _number(best.get(f"test_{split}_{metric}"))
    right = _number(baseline.get(f"test_{split}_{metric}"))
    return None if left is None or right is None else left - right


def finalize(*, ranking_path: Path, replay_root: Path, output_root: Path) -> dict[str, Any]:
    ranking = read_json(ranking_path.resolve())
    if ranking.get("status") != "PASS" or ranking.get("paper_status") != "TEST_TUNED_EXPLORATION":
        raise RuntimeError("ranking receipt is not TEST_TUNED_EXPLORATION")
    rows: list[dict[str, Any]] = []
    full_paths = sorted(replay_root.resolve().glob("*/*/merged/test_tuned_metrics.json"))
    for metrics_path in full_paths:
        candidate_id = metrics_path.parts[-4]
        trial_id = metrics_path.parts[-3]
        metrics, manifest, evaluation = _candidate_metrics(metrics_path)
        provenance = manifest.get("mgf_provenance", {})
        role = provenance.get("comparison_role")
        method = "B0" if candidate_id.startswith("B0_") or role == "B0_OFFICIAL_V11_TEST_TUNED_COMPARATOR" else str(provenance.get("card_id") or candidate_id.split("_", 1)[0])
        if method not in RANKING_METHODS and not method.startswith("E"):
            raise RuntimeError(f"full replay method is not registered: {method}")
        row = {
            "rank": None,
            "method": method,
            "candidate_id": candidate_id,
            "trial_id": trial_id,
            "beta": _method_metadata(ranking, method).get("beta"),
            "mode": _method_metadata(ranking, method).get("mode"),
            "adaptive": _method_metadata(ranking, method).get("adaptive"),
            "paper_status": "TEST_TUNED_EXPLORATION",
            "paper_valid": False,
            "full_replay_count": 1,
            "full_replay_manifest": str((metrics_path.parent / "manifest.json").resolve()),
            "full_replay_manifest_sha256": sha256_file(metrics_path.parent / "manifest.json"),
            "evaluation_manifest": evaluation.get("summary") or (str(next(iter((metrics_path.parent / "evaluation").glob("*/evaluation.json")), "")) if evaluation else None),
            "score_threshold": manifest.get("thresholds", {}).get("score_threshold"),
            "margin_threshold": manifest.get("thresholds", {}).get("margin_threshold"),
            "test_tuned": True,
        }
        row.update(_ranking_row(ranking, method))
        _flatten_teta(row, metrics)
        rows.append(row)

    # Preserve all event-ranking methods in the final table, even when a
    # method was not allocated a causal Full-Test replay.
    existing_methods = {str(row["method"]) for row in rows}
    for method in RANKING_METHODS:
        if method in existing_methods:
            continue
        metadata = _method_metadata(ranking, method)
        row = {
            "rank": None,
            "method": method,
            "candidate_id": None,
            "trial_id": None,
            "beta": metadata.get("beta"),
            "mode": metadata.get("mode"),
            "adaptive": metadata.get("adaptive"),
            "paper_status": "TEST_TUNED_EXPLORATION",
            "paper_valid": False,
            "full_replay_count": 0,
            "full_replay_manifest": None,
            "full_replay_manifest_sha256": None,
            "evaluation_manifest": None,
            "score_threshold": None,
            "margin_threshold": None,
            "test_tuned": True,
        }
        row.update(_ranking_row(ranking, method))
        for split in ("overall", "base", "novel"):
            for field in TETA_FIELDS:
                row[f"test_{split}_{field}"] = None
        row.update({"test_teta": None, "test_assocA": None, "test_base_assocA": None, "test_novel_assocA": None, "test_locA": None, "test_clsA": None})
        rows.append(row)

    full_mgf = [row for row in rows if str(row["method"]).startswith("E") and row.get("test_teta") is not None]
    full_b0 = [row for row in rows if row["method"] == "B0" and row.get("test_teta") is not None]
    if not full_mgf:
        raise RuntimeError("no completed MGF Full-Test evaluation is available")
    if not full_b0:
        raise RuntimeError("no completed B0 Full-Test evaluation is available")
    best_mgf = max(full_mgf, key=_selection_key)
    best_b0 = max(full_b0, key=_selection_key)
    best_mgf["selected_best_mgf"] = True
    best_b0["selected_b0_reference"] = True
    for rank, row in enumerate(sorted(full_mgf + full_b0, key=_selection_key, reverse=True), start=1):
        row["rank"] = rank
    deltas = {
        "overall_AssocA": _delta(best_mgf, best_b0, "AssocA", "overall"),
        "novel_AssocA": _delta(best_mgf, best_b0, "AssocA", "novel"),
        "base_AssocA": _delta(best_mgf, best_b0, "AssocA", "base"),
        "overall_TETA": _delta(best_mgf, best_b0, "TETA", "overall"),
        "overall_LocA": _delta(best_mgf, best_b0, "LocA", "overall"),
        "overall_ClsA": _delta(best_mgf, best_b0, "ClsA", "overall"),
    }
    novel_delta = deltas["novel_AssocA"]
    overall_delta = deltas["overall_AssocA"]
    if novel_delta is not None and novel_delta >= 1.0 and (overall_delta is not None and overall_delta >= 0.0):
        label = "MGF_STRONG_TEST_TUNED"
    elif (novel_delta is not None and novel_delta > 0.0) or (overall_delta is not None and overall_delta > 0.0):
        label = "MGF_SMALL_TEST_TUNED_GAIN"
    else:
        label = "MGF_NO_TEST_TUNED_GAIN"
    best_method_metadata = _method_metadata(ranking, str(best_mgf["method"]))
    result = {
        "status": "PASS",
        "artifact": "v12_mgf_final_20h_test_tuned_leaderboard",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "test_used_for_model_selection": True,
        "test_used_for_beta_selection": True,
        "test_used_for_operating_point_selection": True,
        "ranking": str(ranking_path.resolve()),
        "ranking_sha256": sha256_file(ranking_path.resolve()),
        "replay_root": str(replay_root.resolve()),
        "full_test_candidate_count": len(full_mgf) + len(full_b0),
        "best_mgf": best_mgf,
        "b0_reference": best_b0,
        "deltas_best_mgf_vs_b0": deltas,
        "best_beta": best_method_metadata.get("beta"),
        "best_mode": best_method_metadata.get("mode"),
        "adaptive_won": bool(best_method_metadata.get("adaptive")),
        "final_label": label,
        "selection_order": ["Test Novel AssocA", "Test Overall AssocA", "Test TETA", "Test Base AssocA"],
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "final_20h_test_tuned_leaderboard.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_root / "final_20h_test_tuned_leaderboard.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    markdown = output_root / "final_20h_test_tuned_report.md"
    b0 = best_b0
    mgf = best_mgf
    lines = [
        f"BEST MGF vs B0: Test Overall AssocA: B0 = {b0.get('test_assocA')}; MGF = {mgf.get('test_assocA')}; Delta = {deltas['overall_AssocA']}",
        f"Test Novel AssocA: B0 = {b0.get('test_novel_assocA')}; MGF = {mgf.get('test_novel_assocA')}; Delta = {deltas['novel_AssocA']}",
        f"Test Base AssocA: B0 = {b0.get('test_base_assocA')}; MGF = {mgf.get('test_base_assocA')}; Delta = {deltas['base_AssocA']}",
        f"Test TETA: B0 = {b0.get('test_teta')}; MGF = {mgf.get('test_teta')}; Delta = {deltas['overall_TETA']}",
        "",
        "# 20-Hour Test-Tuned MGF Exploration",
        "",
        "This experiment intentionally uses Test for model and operating-point exploration to estimate the upper potential of Moment-Generating Temporal Matching. The resulting metrics are model-specific and test-tuned, not unbiased Test estimates.",
        "",
        f"- best beta = {result['best_beta']}",
        f"- best mode = {result['best_mode']}",
        f"- adaptive won = {'YES' if result['adaptive_won'] else 'NO'}",
        f"- final label = **{label}**",
        "",
        "## Best MGF vs B0 metric comparison",
        "",
        "| Split | Metric | B0 | Best MGF | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for split in ("overall", "base", "novel"):
        for field in TETA_FIELDS:
            delta = _delta(mgf, b0, field, split)
            lines.append(f"| {split} | {field} | {b0.get(f'test_{split}_{field}')} | {mgf.get(f'test_{split}_{field}')} | {delta} |")
    lines.extend(["", "## Full-Test rows", "", "| Rank | Method | Candidate | beta | mode | Novel AssocA | Overall AssocA | TETA | Base AssocA | margin |", "|---:|---|---|---:|---|---:|---:|---:|---:|---:|"])
    for row in sorted((item for item in rows if item.get("test_teta") is not None), key=lambda item: (item.get("rank") is None, item.get("rank") or 10**9)):
        lines.append(f"| {row.get('rank') or '-'} | {row['method']} | {row.get('candidate_id') or '-'} | {row.get('beta') or '-'} | {row.get('mode') or '-'} | {row.get('test_novel_assocA')} | {row.get('test_assocA')} | {row.get('test_teta')} | {row.get('test_base_assocA')} | {row.get('margin_threshold')} |")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result["outputs"] = {"csv": str(csv_path), "json": str(json_path), "markdown": str(markdown)}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "best_mgf": best_mgf.get("candidate_id"), "b0": best_b0.get("candidate_id"), "label": label, "outputs": result["outputs"]}, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-json", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    finalize(ranking_path=args.ranking_json, replay_root=args.replay_root, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
