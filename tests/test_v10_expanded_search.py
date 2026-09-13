from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "v10_v104_expanded_search.py"
SPEC = importlib.util.spec_from_file_location("v10_v104_expanded_search", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["v10_v104_expanded_search"] = MODULE
SPEC.loader.exec_module(MODULE)


QUANTILES = {
    "score_p05": 1.367088943719864,
    "score_p25": 2.371457517147064,
    "score_p50": 3.5821245908737183,
    "margin_p05": 0.1311543583869934,
    "margin_p25": 0.6632569432258606,
    "margin_p50": 1.566097915172577,
}


def make_plan():
    return {
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "contract_mode": "hardened",
        "contract_gate": "/tmp/gate.json",
        "contract_gate_sha256": "gate",
        "threshold_source": "smoke",
        "threshold_quantiles": QUANTILES,
        "expected_inputs": {"subset_annotation_sha256": "subset", "full_test_annotation_sha256": "full"},
        "trials": [],
    }


def metric_row(spec, *, overall=1.0, novel=1.0, assoc=1.0):
    metrics = {name: overall for name in MODULE.METRIC_NAMES}
    novel_metrics = {name: novel for name in MODULE.METRIC_NAMES}
    novel_metrics["AssocA"] = assoc
    return {"spec": spec, "base": dict(metrics), "novel": novel_metrics, "overall": dict(metrics)}


def test_threshold_grid_is_exactly_81_and_values_are_from_quantiles():
    specs = MODULE.stage1_specs(make_plan())
    assert len(specs) == 81
    scores = MODULE.threshold_grid(QUANTILES, "score")
    margins = MODULE.threshold_grid(QUANTILES, "margin")
    assert len(scores) == 9 and len(margins) == 9
    assert scores[0] == 0.0 and margins[0] == 0.0
    assert scores[-1] == 1.5 * QUANTILES["score_p50"]
    assert margins[-1] == 1.5 * QUANTILES["margin_p50"]


def test_threshold_grid_exact_values_without_float_literal_typo():
    scores = MODULE.threshold_grid(QUANTILES, "score")
    margins = MODULE.threshold_grid(QUANTILES, "margin")
    assert scores[0:7] == [0.0, QUANTILES["score_p05"] * 0.5, QUANTILES["score_p05"], (QUANTILES["score_p05"] + QUANTILES["score_p25"]) * 0.5, QUANTILES["score_p25"], (QUANTILES["score_p25"] + QUANTILES["score_p50"]) * 0.5, QUANTILES["score_p50"]]
    assert margins[0:7] == [0.0, QUANTILES["margin_p05"] * 0.5, QUANTILES["margin_p05"], (QUANTILES["margin_p05"] + QUANTILES["margin_p25"]) * 0.5, QUANTILES["margin_p25"], (QUANTILES["margin_p25"] + QUANTILES["margin_p50"]) * 0.5, QUANTILES["margin_p50"]]
    assert scores[-2:] == [1.25 * QUANTILES["score_p50"], 1.5 * QUANTILES["score_p50"]]
    assert margins[-2:] == [1.25 * QUANTILES["margin_p50"], 1.5 * QUANTILES["margin_p50"]]


def test_structure_grid_is_5_by_5():
    rows = [metric_row({"max_gap": 360, "candidate_top_k": 8, "score_threshold": 1.0, "margin_threshold": 1.0}, overall=i) for i in range(25)]
    specs = MODULE._stage2_specs(rows)
    assert len(specs) == 25
    assert {x["max_gap"] for x in specs} == set(MODULE.STRUCTURE_GAPS)
    assert {x["candidate_top_k"] for x in specs} == set(MODULE.STRUCTURE_TOPKS)


def test_exact_spec_deduplication():
    specs = [
        {"trial_id": "a", "max_gap": 60, "candidate_top_k": 8, "score_threshold": 1, "margin_threshold": 2},
        {"trial_id": "b", "max_gap": 60, "candidate_top_k": 8, "score_threshold": 1.0, "margin_threshold": 2.0},
    ]
    assert len(MODULE.dedupe_specs(specs)) == 1


def test_old_full_receipt_requires_exact_provenance(tmp_path):
    base = {"external_checkpoint_sha256": "ck", "external_config_sha256": "cfg", "base_config_sha256": "base", "teta_init_sha256": "teta", "external_cov_commit": "cov"}
    receipt = {
        "status": "COMPLETED", "stage": "full", "spec": {"max_gap": 60, "candidate_top_k": 8, "score_threshold": 1, "margin_threshold": 2},
        "protocol": {"test_tuned_model_specific": True, "unbiased_test": False},
        "runtime_contract": {"status": "PASS"},
        "inputs": {"annotation_sha256": "full", "external_checkpoint_sha256": "ck", "external_config_sha256": "cfg", "base_config_sha256": "base", "teta_init_sha256": "teta", "overlay_sha256": "ov", "runtime_sha256": "rt"},
        "external_source": {"commit": "cov"}, "search_plan": {"contract_gate_sha256": "gate"},
        "metrics": {split: {name: 1.0 for name in MODULE.METRIC_NAMES} for split in ("base", "novel", "overall")},
        "outputs": {},
    }
    assert not MODULE.valid_old_full_receipt(receipt, full_annotation_sha="full", expected=base, contract_gate_sha="gate", overlay_sha="ov", runtime_sha="rt")


def test_subset_receipt_cannot_be_full():
    receipt = {"status": "COMPLETED", "stage": "subset"}
    assert not MODULE.valid_old_full_receipt(receipt, full_annotation_sha="full", expected={}, contract_gate_sha="gate", overlay_sha="ov", runtime_sha="rt")


def test_completed_exact_receipt_can_be_reused(tmp_path):
    prediction = tmp_path / "prediction.json"
    summary = tmp_path / "summary.pth"
    prediction.write_text("prediction", encoding="utf-8")
    summary.write_text("summary", encoding="utf-8")
    base = {"external_checkpoint_sha256": "ck", "external_config_sha256": "cfg", "base_config_sha256": "base", "teta_init_sha256": "teta", "external_cov_commit": "cov"}
    receipt = {
        "status": "COMPLETED", "stage": "full", "spec": {"max_gap": 60, "candidate_top_k": 8, "score_threshold": 1, "margin_threshold": 2},
        "protocol": {"test_tuned_model_specific": True, "unbiased_test": False},
        "runtime_contract": {"status": "PASS"},
        "inputs": {"annotation_sha256": "full", "external_checkpoint_sha256": "ck", "external_config_sha256": "cfg", "base_config_sha256": "base", "teta_init_sha256": "teta", "overlay_sha256": "ov", "runtime_sha256": "rt"},
        "external_source": {"commit": "cov"}, "search_plan": {"contract_gate_sha256": "gate"},
        "metrics": {split: {name: 1.0 for name in MODULE.METRIC_NAMES} for split in ("base", "novel", "overall")},
        "outputs": {"prediction": str(prediction), "prediction_sha256": MODULE.sha256(prediction), "summary": str(summary), "summary_sha256": MODULE.sha256(summary)},
    }
    assert MODULE.valid_old_full_receipt(receipt, full_annotation_sha="full", expected=base, contract_gate_sha="gate", overlay_sha="ov", runtime_sha="rt")


def test_local_refine_has_no_negative_or_oversized_topk():
    spec = {"max_gap": 30, "candidate_top_k": 4, "score_threshold": 0.0, "margin_threshold": 0.0}
    row = metric_row(spec)
    values = MODULE._stage_full_refine_specs([row], factors=(0.90, 1, 1.10), prefix="x")
    assert all(x["score_threshold"] >= 0 and x["margin_threshold"] >= 0 and x["candidate_top_k"] <= 64 for x in values)


def test_immutable_plan_is_not_overwritten(tmp_path):
    plan = {"protocol": "TEST_TUNED_MODEL_SPECIFIC", "unbiased_test": False, "trials": []}
    path, digest = MODULE.write_immutable_plan(tmp_path, plan)
    assert MODULE.sha256(path) == digest
    with pytest.raises(RuntimeError, match="IMMUTABLE"):
        MODULE.write_immutable_plan(tmp_path, {**plan, "trials": [{"max_gap": 1}]})


def test_official_metric_rankings_and_champions():
    rows = [
        metric_row({"max_gap": 60, "candidate_top_k": 8, "score_threshold": 1, "margin_threshold": 1}, overall=2, novel=1, assoc=3),
        metric_row({"max_gap": 120, "candidate_top_k": 16, "score_threshold": 2, "margin_threshold": 2}, overall=3, novel=2, assoc=1),
    ]
    assert MODULE.ranked(rows, "overall", "TETA")[0]["spec"]["max_gap"] == 120
    assert MODULE.ranked(rows, "novel", "TETA")[0]["spec"]["max_gap"] == 120
    assert MODULE.ranked(rows, "novel", "AssocA")[0]["spec"]["max_gap"] == 60


def test_failed_trial_is_not_a_ranking_observation():
    failed = {"status": "FAILED", "metrics": {"overall": {"TETA": 999}}}
    assert not MODULE._metrics_from_receipt(failed)


def test_stage2_uses_at_most_four_threshold_pairs():
    rows = []
    for i in range(8):
        spec = {"max_gap": 360, "candidate_top_k": 8, "score_threshold": float(i), "margin_threshold": float(i) + 0.25}
        rows.append(metric_row(spec, overall=8 - i, novel=8 - i, assoc=8 - i))
    specs = MODULE._stage2_specs(rows)
    assert len(specs) <= 4 * 25
    assert len(specs) == 3 * 25


def test_structure_neighbors_are_local_and_ordered():
    assert MODULE.structure_neighbors(120, MODULE.STRUCTURE_GAPS) == [60, 120, 240]
    assert MODULE.structure_neighbors(8, MODULE.STRUCTURE_TOPKS) == [4, 8, 16]


def test_stage3_refine_respects_local_coarse_neighbors():
    spec = {"max_gap": 120, "candidate_top_k": 16, "score_threshold": QUANTILES["score_p25"], "margin_threshold": QUANTILES["margin_p25"]}
    rows = MODULE._stage3_specs([metric_row(spec)], MODULE.threshold_grid(QUANTILES, "score"), MODULE.threshold_grid(QUANTILES, "margin"))
    assert rows
    assert all(x["max_gap"] in MODULE.STRUCTURE_GAPS and x["candidate_top_k"] in MODULE.STRUCTURE_TOPKS for x in rows)
    assert all(x["score_threshold"] >= 0 and x["margin_threshold"] >= 0 for x in rows)


def test_build_stage_plan_inherits_hardened_contract():
    plan = MODULE.build_stage_plan(make_plan(), "subset", MODULE.stage1_specs(make_plan())[:1])
    assert plan["contract_mode"] == "hardened"
    assert plan["expected_inputs"] == make_plan()["expected_inputs"]
    assert plan["search_fields"] == list(MODULE.SEARCH_FIELDS)


def test_current_receipt_requires_plan_and_annotation_binding():
    receipt = {"status": "COMPLETED", "stage": "subset", "spec": {"max_gap": 1, "candidate_top_k": 4, "score_threshold": 0, "margin_threshold": 0}}
    assert not MODULE.valid_current_receipt(receipt, stage="subset", plan_sha="p", annotation_sha="a", expected={})


def test_local_multiplier_refine_is_nonnegative_and_bounded():
    values = MODULE.local_threshold_values(0.0, [], multipliers=(0.95, 1.0, 1.05))
    assert values == [0.0]
    spec = {"max_gap": 360, "candidate_top_k": 64, "score_threshold": 1.0, "margin_threshold": 1.0}
    rows = MODULE._stage_full_refine_specs([metric_row(spec)], factors=(0.95, 1.0, 1.05), prefix="micro")
    assert all(row["candidate_top_k"] <= 64 for row in rows)
