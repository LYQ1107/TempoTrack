from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _plan_module():
    return _load("search_hardening_harness", ROOT / "tools" / "v10_search_covtrack_full_test.py")


def _scheduler_module():
    return _load("search_hardening_scheduler", ROOT / "tools" / "v10_run_covtrack_search.py")


def _valid_gate():
    return {
        "status": "PASS",
        "expected_q": 1,
        "actual_q": 1,
        "context_candidate_top_k": 64,
        "reranker_missing_evidence": 0,
    }


def _write_plan(tmp_path: Path, gate: dict, *, mutate_gate_after=False) -> tuple[Path, Path]:
    gate_path = tmp_path / "contract_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    gate_hash = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "protocol": "TEST_TUNED_MODEL_SPECIFIC",
                "unbiased_test": False,
                "contract_gate": str(gate_path),
                "contract_gate_sha256": gate_hash,
                "threshold_source": "smoke_initialization_points",
                "threshold_quantiles": {"score_p25": 2.0, "margin_p25": 0.5},
                "trials": [
                    {
                        "trial_id": "anchor",
                        "max_gap": 360,
                        "candidate_top_k": 8,
                        "score_threshold": 0.0,
                        "margin_threshold": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if mutate_gate_after:
        gate_path.write_text(json.dumps({**gate, "actual_q": 4}), encoding="utf-8")
    return plan_path, gate_path


def test_search_plan_preserves_contract_gate_metadata():
    module = _plan_module()
    plan_path = ROOT / "configs/research/v10/covtrack_q1_fixed_test_search_specs.json"
    plan = module._load_search_plan(plan_path)
    assert plan.protocol == "TEST_TUNED_MODEL_SPECIFIC"
    assert plan.unbiased_test is False
    assert plan.contract_gate
    assert plan.contract_gate_sha256
    assert plan.threshold_source == "post_fix_q1_2video_smoke"
    assert plan.threshold_quantiles["score_p25"] == pytest.approx(2.371457517147064)
    assert len(plan.trials) == 12


def test_search_plan_bad_gate_hash_fails(tmp_path):
    module = _plan_module()
    plan_path, gate_path = _write_plan(tmp_path, _valid_gate(), mutate_gate_after=True)
    plan = module._load_search_plan(plan_path)
    with pytest.raises(RuntimeError, match="SEARCH_CONTRACT_GATE_HASH_MISMATCH"):
        module._validate_contract_gate(plan)


def test_search_plan_nonpass_gate_fails(tmp_path):
    module = _plan_module()
    plan_path, _ = _write_plan(tmp_path, {**_valid_gate(), "status": "FAIL"})
    plan = module._load_search_plan(plan_path)
    with pytest.raises(RuntimeError, match="SEARCH_CONTRACT_GATE_NOT_PASS"):
        module._validate_contract_gate(plan)


def _diagnostics(**updates):
    value = {
        "frames": 2,
        "reranker_expected_query_observations": 1,
        "reranker_actual_query_observations": 1,
        "reranker_context_candidate_top_k": 64,
        "reranker_decision_candidate_top_k": 8,
        "reranker_context_contract_mismatch": False,
        "reranker_missing_evidence": 0,
        "reranker_native_memo_bootstrap_count": 0,
        "full_capability_status_counts": {"FULL_Q1_RERANKER_RUNTIME_ACTIVE": 2},
    }
    value.update(updates)
    return value


def test_trial_runtime_contract_missing_evidence_fails(tmp_path):
    module = _plan_module()
    path = tmp_path / "diagnostics.json"
    path.write_text(json.dumps(_diagnostics(reranker_missing_evidence=1)), encoding="utf-8")
    result = module._validate_runtime_contract(
        diagnostics_path=path,
        spec={"candidate_top_k": 8},
    )
    assert result["status"] == "FAIL"
    assert "missing_evidence" in result["failures"]


def test_trial_runtime_contract_wrong_q_fails(tmp_path):
    module = _plan_module()
    path = tmp_path / "diagnostics.json"
    path.write_text(json.dumps(_diagnostics(reranker_actual_query_observations=4)), encoding="utf-8")
    result = module._validate_runtime_contract(diagnostics_path=path, spec={"candidate_top_k": 8})
    assert result["status"] == "FAIL"
    assert "actual_q" in result["failures"]


def test_trial_runtime_contract_context_k_mismatch_fails(tmp_path):
    module = _plan_module()
    path = tmp_path / "diagnostics.json"
    path.write_text(json.dumps(_diagnostics(reranker_context_candidate_top_k=32)), encoding="utf-8")
    result = module._validate_runtime_contract(diagnostics_path=path, spec={"candidate_top_k": 8})
    assert result["status"] == "FAIL"
    assert "context_k" in result["failures"]


def test_ram_guard_blocks_new_launch_but_not_running_workers(monkeypatch):
    module = _scheduler_module()
    args = type("Args", (), {"min_available_ram_gb": 24.0, "launch_reserve_ram_gb": 4.0})()
    monkeypatch.setattr(module, "_mem_available_gib", lambda: 20.0)
    allowed, available, required = module._ram_launch_gate(args)
    assert allowed is False
    assert available == 20.0
    assert required == 28.0
    # The helper is pure: it cannot signal or alter already-running workers.


def test_current_launch_head_is_not_rejected_only_because_branch_advanced(tmp_path):
    audit = _load("postcontract_audit", ROOT / "tools" / "v10_audit_covtrack_postcontract_search.py")
    receipt = {
        "status": "COMPLETED",
        "repo": {"head": "e24b2db4e2297c51ad6718e70b3e1fd520ee6ff8"},
        "spec": {"trial_id": "anchor", "max_gap": 360, "candidate_top_k": 8, "score_threshold": 0.0, "margin_threshold": 0.0},
        "inputs": {
            "annotation_sha256": "a",
            "external_checkpoint_sha256": "b",
            "external_config_sha256": "c",
            "external_checkpoint": str(tmp_path / "checkpoint"),
            "external_config": str(tmp_path / "config"),
            "base_config": str(tmp_path / "base"),
            "base_config_sha256": "d",
            "tempo_config": str(tmp_path / "tempo"),
        },
        "external_source": {"commit": "9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b"},
    }
    for key in ("checkpoint", "config", "base", "tempo"):
        (tmp_path / key).write_text(key, encoding="utf-8")
    for key in ("external_checkpoint", "external_config", "base_config", "tempo_config"):
        path = Path(receipt["inputs"][key])
        receipt["inputs"][f"{key}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    result = audit._audit_trial(
        receipt_path=receipt_path,
        expected_spec=receipt["spec"],
        root=tmp_path,
        annotation_sha256="a",
        launch_head="e24b2db4e2297c51ad6718e70b3e1fd520ee6ff8",
        external_commit=receipt["external_source"]["commit"],
        checkpoint_sha256="b",
        external_config_sha256="c",
    )
    assert "LAUNCH_HEAD_MISMATCH" not in result["failure_reasons"]


def _rank_fixture(tmp_path: Path, *, control_annotation: str = "ann", trial_annotation: str = "ann", bad_prediction_hash: bool = False):
    module = _load("rank_hardening", ROOT / "tools" / "v10_rank_covtrack_test_trials.py")
    root = tmp_path / "trials"
    trial = root / "anchor"
    trial.mkdir(parents=True)
    prediction = trial / "prediction.json"
    diagnostics = trial / "diagnostics.json"
    summary = trial / "summary.pth"
    for path, value in ((prediction, "prediction"), (diagnostics, "diagnostics"), (summary, "summary")):
        path.write_text(value, encoding="utf-8")
    def binding(annotation):
        return {
            "annotation_sha256": annotation,
            "external_source_commit": "source",
            "external_checkpoint_sha256": "checkpoint",
            "external_config_sha256": "config",
        }
    receipt = {
        "status": "COMPLETED",
        "trial_id": "anchor",
        "protocol": {"test_tuned_model_specific": True, "unbiased_test": False},
        "inputs": binding(trial_annotation),
        "external_source": {"commit": "source"},
        "outputs": {
            "prediction": str(prediction),
            "prediction_sha256": "wrong" if bad_prediction_hash else hashlib.sha256(prediction.read_bytes()).hexdigest(),
            "diagnostics": str(diagnostics),
            "diagnostics_sha256": hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
            "summary": str(summary),
            "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        },
        "metrics": {
            "base": {"AssocA": 10.0, "TETA": 10.0},
            "novel": {"AssocA": 9.0, "TETA": 9.0},
        },
    }
    (trial / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps({
            "inputs": binding(control_annotation),
            "external_source": {"commit": "source"},
            "metrics": {"base": {"AssocA": 10.0, "TETA": 10.0}},
        }),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "root": [str(root)],
            "control_receipt": str(control),
            "output": str(tmp_path / "rank.json"),
            "markdown": str(tmp_path / "rank.md"),
            "top_k": 4,
            "search_audit": None,
            "expected_annotation": None,
        },
    )()
    return module, args


def test_rank_rejects_trial_without_selection_audit(tmp_path):
    module, args = _rank_fixture(tmp_path)
    assert module.rank(args) == 0
    output = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert any("SELECTION_AUDIT_MISSING" in row["reason"] for row in output["rejected_trials"])


def test_rank_rejects_control_annotation_mismatch(tmp_path):
    module, args = _rank_fixture(tmp_path, control_annotation="different")
    trial_dir = Path(args.root[0]) / "anchor"
    (trial_dir / "selection_audit.json").write_text(
        json.dumps({"status": "PASS", "usage": "SEARCH_SELECTION"}), encoding="utf-8"
    )
    assert module.rank(args) == 0
    output = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert any("INPUT_BINDING_MISMATCH:annotation_sha256" in row["reason"] for row in output["rejected_trials"])


def test_rank_rejects_prediction_hash_mismatch(tmp_path):
    module, args = _rank_fixture(tmp_path, bad_prediction_hash=True)
    assert module.rank(args) == 0
    output = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert any("PREDICTION_HASH_MISMATCH" in row["reason"] for row in output["rejected_trials"])
