from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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


def test_v2_search_plan_preserves_expected_input_binding():
    module = _plan_module()
    plan = module._load_search_plan(
        ROOT / "configs/research/v10/covtrack_q1_fixed_test_search_specs_v2.json"
    )
    assert plan.expected_inputs["subset_annotation_sha256"] == (
        "6a1245c5bcc2e9838caf3c5f545256a216ff52003eafd9c47f64c95de3118938"
    )
    assert plan.expected_inputs["external_cov_commit"] == (
        "9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b"
    )
    assert plan.contract_mode == "hardened"
    assert plan.expected_inputs["teta_source_root"].endswith("/tet_a62a9c0_clean/teta")
    assert plan.expected_inputs["teta_require_tracked_clean"] is True


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


def test_hardened_gate_requires_bootstrap_counter(tmp_path):
    module = _plan_module()
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(_valid_gate()), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "protocol": "TEST_TUNED_MODEL_SPECIFIC",
                "unbiased_test": False,
                "contract_mode": "hardened",
                "contract_gate": str(gate_path),
                "contract_gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
                "trials": [],
            }
        ),
        encoding="utf-8",
    )
    plan = module._load_search_plan(plan_path)
    with pytest.raises(RuntimeError, match="SEARCH_CONTRACT_BOOTSTRAP_COUNTER_MISSING"):
        module._validate_contract_gate(plan)


def test_hardened_gate_requires_teta_provenance_after_bootstrap(tmp_path):
    module = _plan_module()
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps({**_valid_gate(), "reranker_native_memo_bootstrap_count": 0}),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "protocol": "TEST_TUNED_MODEL_SPECIFIC",
                "unbiased_test": False,
                "contract_mode": "hardened",
                "contract_gate": str(gate_path),
                "contract_gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
                "trials": [],
            }
        ),
        encoding="utf-8",
    )
    plan = module._load_search_plan(plan_path)
    with pytest.raises(RuntimeError, match="SEARCH_CONTRACT_TETA_PROVENANCE_MISSING"):
        module._validate_contract_gate(plan)


def test_validator_reads_top_level_runtime_contract_sha(tmp_path):
    module = _load(
        "q1_contract_validator_config_binding",
        ROOT / "tools" / "v10_validate_q1_contract_smoke.py",
    )
    config = tmp_path / "hardened.yaml"
    config.write_text(
        "runtime_contract_sha: top-level-revision\ntempo:\n  max_gap: 360\n",
        encoding="utf-8",
    )
    assert module._runtime_contract_sha_from_config(config) == "top-level-revision"


def test_search_gate_runtime_revision_must_match_current_base_config(tmp_path):
    scheduler = _scheduler_module()
    repo = tmp_path / "repo"
    (repo / "tempotrack_v10").mkdir(parents=True)
    overlay = repo / "tempotrack_v10" / "overlay.py"
    runtime = repo / "tempotrack_v10" / "covtrack_runtime.py"
    overlay.write_text("overlay\n", encoding="utf-8")
    runtime.write_text("runtime\n", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text(
        "runtime_contract_sha: new-runtime\n"
        "tempo:\n"
        "  reranker_checkpoint: /tmp/x.pt\n",
        encoding="utf-8",
    )
    gate = {
        "runtime_contract_sha": "old-runtime",
        "overlay_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
    }
    with pytest.raises(RuntimeError, match="SEARCH_GATE_RUNTIME_REVISION_MISMATCH"):
        scheduler._validate_gate_runtime_binding(
            contract_gate=gate,
            base_config=base,
            repo=repo,
        )


def test_search_gate_overlay_hash_must_match_current_repo(tmp_path):
    scheduler = _scheduler_module()
    repo = tmp_path / "repo"
    (repo / "tempotrack_v10").mkdir(parents=True)
    overlay = repo / "tempotrack_v10" / "overlay.py"
    runtime = repo / "tempotrack_v10" / "covtrack_runtime.py"
    overlay.write_text("overlay\n", encoding="utf-8")
    runtime.write_text("runtime\n", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text("runtime_contract_sha: current\n", encoding="utf-8")
    gate = {
        "runtime_contract_sha": "current",
        "overlay_sha256": "wrong-overlay",
        "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
    }
    with pytest.raises(RuntimeError, match="SEARCH_GATE_CURRENT_OVERLAY_HASH_MISMATCH"):
        scheduler._validate_gate_runtime_binding(
            contract_gate=gate,
            base_config=base,
            repo=repo,
        )


def test_search_gate_runtime_hash_must_match_current_repo(tmp_path):
    scheduler = _scheduler_module()
    repo = tmp_path / "repo"
    (repo / "tempotrack_v10").mkdir(parents=True)
    overlay = repo / "tempotrack_v10" / "overlay.py"
    runtime = repo / "tempotrack_v10" / "covtrack_runtime.py"
    overlay.write_text("overlay\n", encoding="utf-8")
    runtime.write_text("runtime\n", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text("runtime_contract_sha: current\n", encoding="utf-8")
    gate = {
        "runtime_contract_sha": "current",
        "overlay_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
        "runtime_sha256": "wrong-runtime",
    }
    with pytest.raises(RuntimeError, match="SEARCH_GATE_CURRENT_RUNTIME_HASH_MISMATCH"):
        scheduler._validate_gate_runtime_binding(
            contract_gate=gate,
            base_config=base,
            repo=repo,
        )


def test_teta_tracked_dirty_fails_expected_input_binding(tmp_path):
    module = _scheduler_module()
    teta_root = tmp_path / "teta_source"
    package = teta_root / "teta"
    package.mkdir(parents=True)
    init_file = package / "__init__.py"
    init_file.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(teta_root)], check=True)
    subprocess.run(["git", "-C", str(teta_root), "add", "teta/__init__.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(teta_root),
            "-c",
            "user.name=TempoTrack test",
            "-c",
            "user.email=tempotrack-test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )
    init_file.write_text("VALUE = 2\n", encoding="utf-8")
    dependency = module._teta_dependency(teta_root)
    assert dependency["tracked_source_clean"] is False
    annotation = tmp_path / "annotation.json"
    annotation.write_text("{}\n", encoding="utf-8")
    plan = SimpleNamespace(
        expected_inputs={
            "teta_source_root": str(teta_root.resolve()),
            "teta_init_sha256": hashlib.sha256(init_file.read_bytes()).hexdigest(),
            "teta_git_commit": dependency["git_commit"],
            "teta_require_tracked_clean": True,
        }
    )
    args = SimpleNamespace(teta_source_root=str(teta_root), annotation=str(annotation))
    with pytest.raises(RuntimeError, match="SEARCH_EXPECTED_INPUT_TETA_SOURCE_TRACKED_DIRTY"):
        module._validate_expected_inputs(plan, args, reranker_checkpoint_sha256=None)


def test_teta_import_preflight_records_exact_import_paths(tmp_path):
    module = _plan_module()
    teta_root = tmp_path / "teta_root"
    (teta_root / "teta").mkdir(parents=True)
    (teta_root / "teta" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    source = tmp_path / "cov_source"
    (source / "ovtrack" / "datasets").mkdir(parents=True)
    (source / "ovtrack" / "__init__.py").write_text("\n", encoding="utf-8")
    cov_init = source / "ovtrack" / "datasets" / "__init__.py"
    cov_init.write_text("\n", encoding="utf-8")
    result = module._run_teta_import_preflight(
        stream_python=sys.executable,
        source=source,
        teta_source_root=teta_root,
    )
    assert result["status"] == "PASS"
    assert result["actual_imports"]["teta_file"] == result["expected_teta_init"]
    assert result["actual_imports"]["cov_dataset_file"] == result["expected_cov_dataset_init"]

    wrong_init = tmp_path / "wrong_teta_init.py"
    wrong_init.write_text("\n", encoding="utf-8")
    original_resolver = module._resolve_teta_source_root
    module._resolve_teta_source_root = lambda value: (teta_root.resolve(), wrong_init.resolve())
    try:
        with pytest.raises(RuntimeError, match="TETA_IMPORT_PATH_MISMATCH"):
            module._run_teta_import_preflight(
                stream_python=sys.executable,
                source=source,
                teta_source_root=teta_root,
            )
    finally:
        module._resolve_teta_source_root = original_resolver


def test_teta_source_root_requires_real_importable_package(tmp_path):
    module = _plan_module()
    with pytest.raises(RuntimeError, match="TETA_SOURCE_ROOT_INVALID"):
        module._resolve_teta_source_root(tmp_path)
    root = tmp_path / "teta"
    package = root / "teta"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    resolved = module._resolve_teta_source_root(root)
    assert resolved == (root.resolve(), (package / "__init__.py").resolve())


def test_runtime_env_places_teta_import_parent_before_ambient_path(tmp_path):
    module = _plan_module()
    root = tmp_path / "teta"
    package = root / "teta"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    args = SimpleNamespace(
        gpu="0",
        img_prefix=str(tmp_path / "frames"),
        teta_source_root=str(root),
    )
    env = module._runtime_env(
        args,
        tmp_path / "repo",
        tmp_path / "source",
        tmp_path / "trial",
        tmp_path / "tempo.yaml",
    )
    assert env["PYTHONPATH"].split(":")[0] == str(root.resolve())


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
        "reranker_status": {
            "status": "EXACT_V9_MODEL_CODE_AND_WEIGHTS",
            "checkpoint_sha256": "checkpoint",
            "base_only_supervision": True,
            "novel_gt_used": False,
            "test_weights_used": False,
        },
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


def test_trial_runtime_contract_missing_bootstrap_counter_fails(tmp_path):
    module = _plan_module()
    path = tmp_path / "diagnostics.json"
    diagnostics = _diagnostics()
    diagnostics.pop("reranker_native_memo_bootstrap_count")
    path.write_text(json.dumps(diagnostics), encoding="utf-8")
    result = module._validate_runtime_contract(
        diagnostics_path=path,
        spec={"candidate_top_k": 8},
    )
    assert result["status"] == "FAIL"
    assert "native_memo_bootstrap_counter_missing" in result["failures"]


def test_trial_runtime_contract_reranker_provenance_is_fail_closed(tmp_path):
    module = _plan_module()
    path = tmp_path / "diagnostics.json"
    diagnostics = _diagnostics(
        reranker_status={
            "status": "LEGACY",
            "checkpoint_sha256": "checkpoint",
            "base_only_supervision": False,
            "novel_gt_used": True,
            "test_weights_used": True,
        }
    )
    path.write_text(json.dumps(diagnostics), encoding="utf-8")
    result = module._validate_runtime_contract(
        diagnostics_path=path,
        spec={"candidate_top_k": 8},
    )
    assert result["status"] == "FAIL"
    assert {
        "reranker_provenance_status",
        "reranker_not_base_only",
        "reranker_novel_gt_used",
        "reranker_test_weights_used",
    }.issubset(result["failures"])


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
        expected_base_config=tmp_path / "base",
        expected_base_config_sha256=hashlib.sha256((tmp_path / "base").read_bytes()).hexdigest(),
    )
    assert "LAUNCH_HEAD_MISMATCH" not in result["failure_reasons"]


def _rank_fixture(tmp_path: Path, *, control_annotation: str = "ann", trial_annotation: str = "ann", bad_prediction_hash: bool = False):
    module = _load("rank_hardening", ROOT / "tools" / "v10_rank_covtrack_test_trials.py")
    root = tmp_path / "trials"
    trial = root / "anchor"
    trial.mkdir(parents=True)
    external_checkpoint = tmp_path / "external.pth"
    external_config = tmp_path / "external.py"
    external_checkpoint.write_text("external checkpoint", encoding="utf-8")
    external_config.write_text("external config", encoding="utf-8")
    prediction = trial / "prediction.json"
    diagnostics = trial / "diagnostics.json"
    summary = trial / "summary.pth"
    for path, value in ((prediction, "prediction"), (diagnostics, "diagnostics"), (summary, "summary")):
        path.write_text(value, encoding="utf-8")
    def binding(annotation):
        return {
            "annotation_sha256": annotation,
            "external_source_commit": "source",
            "external_checkpoint_sha256": hashlib.sha256(external_checkpoint.read_bytes()).hexdigest(),
            "external_config_sha256": hashlib.sha256(external_config.read_bytes()).hexdigest(),
            "external_checkpoint": str(external_checkpoint),
            "external_config": str(external_config),
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
    control_prediction = tmp_path / "control_prediction.json"
    control_summary = tmp_path / "control_summary.pth"
    control_prediction.write_text("control prediction", encoding="utf-8")
    control_summary.write_text("control summary", encoding="utf-8")
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps({
            "status": "COMPLETED",
            "protocol": {"disabled_overlay_control": True},
            "inputs": binding(control_annotation),
            "external_source": {"commit": "source"},
            "outputs": {
                "prediction": str(control_prediction),
                "prediction_sha256": hashlib.sha256(control_prediction.read_bytes()).hexdigest(),
                "summary": str(control_summary),
                "summary_sha256": hashlib.sha256(control_summary.read_bytes()).hexdigest(),
            },
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
    assert module.rank(args) == 2
    output = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert any("SELECTION_AUDIT_MISSING" in row["reason"] for row in output["rejected_trials"])


def test_rank_rejects_control_annotation_mismatch(tmp_path):
    module, args = _rank_fixture(tmp_path, control_annotation="different")
    trial_dir = Path(args.root[0]) / "anchor"
    (trial_dir / "selection_audit.json").write_text(
        json.dumps({"status": "PASS", "usage": "SEARCH_SELECTION"}), encoding="utf-8"
    )
    assert module.rank(args) == 2
    output = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert any("INPUT_BINDING_MISMATCH:annotation_sha256" in row["reason"] for row in output["rejected_trials"])


def test_rank_rejects_prediction_hash_mismatch(tmp_path):
    module, args = _rank_fixture(tmp_path, bad_prediction_hash=True)
    assert module.rank(args) == 2
    output = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert any("PREDICTION_HASH_MISMATCH" in row["reason"] for row in output["rejected_trials"])


def _audit_receipt_fixture(
    tmp_path: Path,
    audit_module,
    *,
    requested_id: str = "anchor",
    effective_id: str | None = None,
    legacy: bool = True,
    status: str = "COMPLETED",
    checkpoint_sha: str = "reranker-checkpoint",
):
    effective_id = effective_id or requested_id
    root = tmp_path / "search"
    trial = root / effective_id
    trial.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base.yaml"
    base.write_text("tempo: {}\n", encoding="utf-8")
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({"images": [], "categories": []}), encoding="utf-8")
    external_checkpoint = tmp_path / "external.pth"
    external_config = tmp_path / "external.py"
    external_checkpoint.write_text("external checkpoint", encoding="utf-8")
    external_config.write_text("external config", encoding="utf-8")
    spec = {
        "trial_id": effective_id,
        "max_gap": 360,
        "candidate_top_k": 8,
        "score_threshold": 0.0,
        "margin_threshold": 0.0,
    }
    tempo = trial / "tempo.yaml"
    plan_module = _plan_module()
    plan_module._materialize_config(base, tempo, spec, disabled=False)
    diagnostics = trial / "diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "frames": 1,
                "reranker_expected_query_observations": 1,
                "reranker_actual_query_observations": 1,
                "reranker_context_candidate_top_k": 64,
                "reranker_decision_candidate_top_k": 8,
                "reranker_context_contract_mismatch": False,
                "reranker_missing_evidence": 0,
                "reranker_native_memo_bootstrap_count": 0,
                "full_capability_status_counts": {"FULL_Q1_RERANKER_RUNTIME_ACTIVE": 1},
                "reranker_status": {
                    "status": "EXACT_V9_MODEL_CODE_AND_WEIGHTS",
                    "checkpoint_sha256": checkpoint_sha,
                    "base_only_supervision": True,
                    "novel_gt_used": False,
                    "test_weights_used": False,
                },
            }
        ),
        encoding="utf-8",
    )
    prediction = trial / "prediction.json"
    summary = trial / "summary.pth"
    prediction.write_text("prediction", encoding="utf-8")
    summary.write_text("summary", encoding="utf-8")
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "status": status,
        "trial_id": effective_id,
        "requested_trial_id": requested_id,
        "repo": {"head": "launch"},
        "spec": spec,
        "external_source": {"commit": "external-commit"},
        "inputs": {
            "annotation": str(annotation),
            "annotation_sha256": sha(annotation),
            "external_checkpoint": str(external_checkpoint),
            "external_checkpoint_sha256": sha(external_checkpoint),
            "external_config": str(external_config),
            "external_config_sha256": sha(external_config),
            "tempo_config": str(tempo),
            "tempo_config_sha256": sha(tempo),
        },
        "outputs": {
            "prediction": str(prediction),
            "prediction_sha256": sha(prediction),
            "diagnostics": str(diagnostics),
            "diagnostics_sha256": sha(diagnostics),
            "summary": str(summary),
            "summary_sha256": sha(summary),
        },
        "metrics": {
            "base": {"AssocA": 10.0, "TETA": 10.0},
            "novel": {"AssocA": 9.0, "TETA": 9.0},
        },
        "protocol": {"test_tuned_model_specific": True, "unbiased_test": False},
    }
    if not legacy:
        receipt["inputs"]["base_config"] = str(base)
        receipt["inputs"]["base_config_sha256"] = sha(base)
    receipt_path = trial / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return {
        "root": root,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "base": base,
        "base_sha": sha(base),
        "annotation": annotation,
        "external_checkpoint": external_checkpoint,
        "external_config": external_config,
        "expected_spec": {**spec, "trial_id": requested_id},
        "audit_kwargs": {
            "receipt_path": receipt_path,
            "expected_spec": {**spec, "trial_id": requested_id},
            "root": root,
            "annotation_sha256": sha(annotation),
            "launch_head": "launch",
            "external_commit": "external-commit",
            "checkpoint_sha256": checkpoint_sha,
            "external_config_sha256": sha(external_config),
            "external_checkpoint_sha256": sha(external_checkpoint),
            "expected_base_config": base,
            "expected_base_config_sha256": sha(base),
        },
    }


def test_legacy_receipt_without_base_config_reconstructs_materialized_config(tmp_path):
    audit = _load("legacy_audit", ROOT / "tools" / "v10_audit_covtrack_postcontract_search.py")
    fixture = _audit_receipt_fixture(tmp_path, audit)
    receipt = fixture["receipt"]
    diagnostics_path = Path(receipt["outputs"]["diagnostics"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics.pop("reranker_native_memo_bootstrap_count")
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    receipt["outputs"]["diagnostics_sha256"] = hashlib.sha256(
        diagnostics_path.read_bytes()
    ).hexdigest()
    fixture["receipt_path"].write_text(json.dumps(receipt), encoding="utf-8")
    result = audit._audit_trial(**fixture["audit_kwargs"])
    assert result["status"] == "PASS", result
    assert result["contract_classification"] == "SEARCH_SELECTION_LEGACY_CONTRACT"
    assert result["base_config_binding"]["legacy_reconstructed"] is True
    assert result["base_config_binding"]["reconstruction_status"] == "PASS"


def test_legacy_receipt_bad_reconstructed_config_fails(tmp_path):
    audit = _load("legacy_audit_bad_base", ROOT / "tools" / "v10_audit_covtrack_postcontract_search.py")
    fixture = _audit_receipt_fixture(tmp_path, audit)
    fixture["base"].write_text("tempo:\n  changed: true\n", encoding="utf-8")
    fixture["audit_kwargs"]["expected_base_config"] = fixture["base"]
    fixture["audit_kwargs"]["expected_base_config_sha256"] = hashlib.sha256(
        fixture["base"].read_bytes()
    ).hexdigest()
    result = audit._audit_trial(**fixture["audit_kwargs"])
    assert result["status"] == "FAIL"
    assert "LEGACY_BASE_CONFIG_RECONSTRUCTION_MISMATCH" in result["failure_reasons"]


def test_hardened_audit_requires_bootstrap_counter(tmp_path):
    audit = _load("hardened_contract_audit", ROOT / "tools" / "v10_audit_covtrack_postcontract_search.py")
    fixture = _audit_receipt_fixture(tmp_path, audit)
    receipt = fixture["receipt"]
    diagnostics_path = Path(receipt["outputs"]["diagnostics"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics.pop("reranker_native_memo_bootstrap_count")
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    receipt["outputs"]["diagnostics_sha256"] = hashlib.sha256(
        diagnostics_path.read_bytes()
    ).hexdigest()
    fixture["receipt_path"].write_text(json.dumps(receipt), encoding="utf-8")
    result = audit._audit_trial(**fixture["audit_kwargs"], contract_mode="hardened")
    assert result["status"] == "FAIL"
    assert "HARDENED_BOOTSTRAP_COUNTER_MISSING" in result["failure_reasons"]


def test_audit_prefers_completed_retry_over_failed_direct(tmp_path):
    audit = _load("retry_audit", ROOT / "tools" / "v10_audit_covtrack_postcontract_search.py")
    fixture = _audit_receipt_fixture(tmp_path, audit, effective_id="anchor__retry01")
    direct = fixture["root"] / "anchor"
    direct.mkdir()
    (direct / "receipt.json").write_text(json.dumps({"status": "FAILED"}), encoding="utf-8")
    assert audit._find_receipt(fixture["root"], "anchor") == fixture["receipt_path"]


def test_audit_retry_ignores_effective_trial_id_in_spec_comparison(tmp_path):
    audit = _load("retry_spec_audit", ROOT / "tools" / "v10_audit_covtrack_postcontract_search.py")
    fixture = _audit_receipt_fixture(tmp_path, audit, effective_id="anchor__retry01")
    result = audit._audit_trial(**fixture["audit_kwargs"])
    assert "TRIAL_SPEC_MISMATCH" not in result["failure_reasons"]
    assert result["requested_trial_id"] == "anchor"
    assert result["effective_trial_id"] == "anchor__retry01"


def test_audit_multiple_completed_retries_fails_closed(tmp_path):
    audit = _load("multiple_retry_audit", ROOT / "tools" / "v10_audit_covtrack_postcontract_search.py")
    first = _audit_receipt_fixture(tmp_path, audit, effective_id="anchor__retry01")
    second_dir = first["root"] / "anchor__retry02"
    second_dir.mkdir()
    second = json.loads(first["receipt_path"].read_text(encoding="utf-8"))
    second["trial_id"] = "anchor__retry02"
    second["requested_trial_id"] = "anchor"
    (second_dir / "receipt.json").write_text(json.dumps(second), encoding="utf-8")
    with pytest.raises(RuntimeError, match="MULTIPLE_COMPLETED_ATTEMPTS"):
        audit._find_receipt(first["root"], "anchor")


def test_audit_uses_passed_checkpoint_sha_not_global_constant(tmp_path):
    audit = _load("custom_checkpoint_audit", ROOT / "tools" / "v10_audit_covtrack_postcontract_search.py")
    fixture = _audit_receipt_fixture(tmp_path, audit, checkpoint_sha="custom-checkpoint")
    result = audit._audit_trial(**fixture["audit_kwargs"])
    assert "RERANKER_CHECKPOINT_HASH_MISMATCH" not in result["failure_reasons"]


def test_global_audit_partial_pass_remains_selectable(tmp_path):
    audit = _load("partial_global_audit", ROOT / "tools" / "v10_audit_covtrack_postcontract_search.py")
    gate = tmp_path / "gate.json"
    gate_data = _valid_gate()
    gate.write_text(json.dumps(gate_data), encoding="utf-8")
    plan = tmp_path / "plan.json"
    specs = [
        {"trial_id": name, "max_gap": 360, "candidate_top_k": 8, "score_threshold": 0.0, "margin_threshold": 0.0}
        for name in ("a", "b", "c")
    ]
    plan.write_text(
        json.dumps({
            "protocol": "TEST_TUNED_MODEL_SPECIFIC",
            "unbiased_test": False,
            "contract_gate": str(gate),
            "contract_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
            "threshold_source": "test",
            "trials": specs,
        }),
        encoding="utf-8",
    )
    fixture = _audit_receipt_fixture(tmp_path / "a", audit, requested_id="a")
    for name in ("a", "b"):
        receipt = json.loads(json.dumps(fixture["receipt"]))
        receipt["trial_id"] = name
        receipt["requested_trial_id"] = name
        receipt["spec"]["trial_id"] = name
        target = tmp_path / "search" / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    failed = tmp_path / "search" / "c"
    failed.mkdir(parents=True)
    (failed / "receipt.json").write_text(json.dumps({"status": "FAILED", "trial_id": "c"}), encoding="utf-8")
    annotation = fixture["annotation"]
    base = fixture["base"]
    args = SimpleNamespace(
        search_root=str(tmp_path / "search"),
        search_plan=str(plan),
        contract_gate=str(gate),
        annotation=str(annotation),
        base_config=str(base),
        base_config_sha256=None,
        launch_head="launch",
        external_commit="external-commit",
        checkpoint_sha256="reranker-checkpoint",
        external_checkpoint_sha256=hashlib.sha256(
            fixture["external_checkpoint"].read_bytes()
        ).hexdigest(),
        external_config_sha256=None,
        output=str(tmp_path / "audit.json"),
    )
    result_code = audit.audit(args)
    result = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert result_code == 0, result
    assert result["status"] == "PARTIAL_PASS"
    assert result["usage"] == "SEARCH_SELECTION"


def test_rank_accepts_partial_pass_global_but_only_pass_trials(tmp_path):
    module, args = _rank_fixture(tmp_path)
    trial_dir = Path(args.root[0]) / "anchor"
    audit_path = tmp_path / "global_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "PARTIAL_PASS",
                "usage": "SEARCH_SELECTION",
                "trials": {
                    "anchor": {"status": "PASS", "usage": "SEARCH_SELECTION"}
                },
            }
        ),
        encoding="utf-8",
    )
    args.search_audit = str(audit_path)
    assert module.rank(args) == 0
    output = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert [row["trial_id"] for row in output["selected_top_k"]] == ["anchor"]


def test_gate_checkpoint_must_match_base_config_checkpoint(tmp_path):
    scheduler = _scheduler_module()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text(f"tempo:\n  reranker_checkpoint: {checkpoint}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SEARCH_GATE_CHECKPOINT_MISMATCH"):
        scheduler._validate_gate_checkpoint_binding(
            {"checkpoint_sha256": "wrong"},
            base,
        )


def test_disabled_overlay_does_not_require_q1_contract_gate(tmp_path):
    scheduler = _scheduler_module()
    plan_path, _ = _write_plan(tmp_path, _valid_gate())
    plan = scheduler._load_search_plan(plan_path)
    args = SimpleNamespace(
        repo=str(ROOT),
        source=str(ROOT),
        annotation=str(tmp_path / "annotation.json"),
        img_prefix=str(tmp_path),
        external_config=str(tmp_path / "external.py"),
        external_checkpoint=str(tmp_path / "external.pth"),
        base_config=str(tmp_path / "base.yaml"),
        output_root=str(tmp_path / "out"),
        stage="subset",
        stream_python="python",
        evaluator_python="python",
        evaluator_cores=1,
        disabled_overlay=True,
    )
    command = scheduler._build_command(args, dict(plan.trials[0]), "0", plan)
    assert "--disabled-overlay" in command
    assert "--contract-gate" not in command
    worker = _plan_module()
    worker_args = SimpleNamespace(
        disabled_overlay=True,
        search_plan=str(plan_path),
        search_plan_sha256=plan.sha256,
        contract_gate=None,
        contract_gate_sha256=None,
        threshold_source=plan.threshold_source,
    )
    binding = worker._worker_search_plan_binding(worker_args)
    assert binding["contract_gate"] is None


def test_disabled_overlay_stream_success_reaches_evaluator_and_completes(tmp_path, monkeypatch):
    module = _plan_module()
    repo = ROOT
    source = tmp_path / "source"
    source.mkdir()
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps({"images": [{"id": 1}], "categories": [{"id": 1}]}),
        encoding="utf-8",
    )
    external_config = tmp_path / "external.py"
    external_checkpoint = tmp_path / "external.pth"
    external_config.write_text("config", encoding="utf-8")
    external_checkpoint.write_text("checkpoint", encoding="utf-8")
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        "frontend: covtrack\ntempo:\n  enabled: true\n  reranker_weight: 1.0\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "out"
    evaluator_calls: list[str] = []

    def fake_run_logged(command, *, cwd, env, log_path):
        if log_path.name == "stream.log":
            trial_root = log_path.parent
            stream_root = trial_root / "stream"
            stream_root.mkdir(parents=True)
            (stream_root / "stream_manifest.json").write_text(
                json.dumps({"status": "PASS", "frames": 1}),
                encoding="utf-8",
            )
            (stream_root / "tao_track.json").write_text("[]", encoding="utf-8")
        else:
            evaluator_calls.append(log_path.name)
            summary = log_path.parent / "evaluation" / "COV_V10_TEMPO" / "teta_summary_results.pth"
            summary.parent.mkdir(parents=True)
            summary.write_text("summary", encoding="utf-8")
        return 123, 0, 0.0

    monkeypatch.setattr(module, "_run_logged", fake_run_logged)
    monkeypatch.setattr(module, "_resource_snapshot", lambda gpu: {"gpu": gpu})
    monkeypatch.setattr(
        module,
        "_parse_summary_with_evaluator",
        lambda *args, **kwargs: {
            "status": "PARSED",
            "base": {"TETA": 1.0, "AssocA": 1.0},
            "novel": {"TETA": 1.0, "AssocA": 1.0},
        },
    )
    args = SimpleNamespace(
        repo=str(repo),
        source=str(source),
        annotation=str(annotation),
        img_prefix=str(tmp_path),
        external_config=str(external_config),
        external_checkpoint=str(external_checkpoint),
        base_config=str(base_config),
        output_root=str(output_root),
        trial_id="native_control",
        requested_trial_id=None,
        spec_file=None,
        spec_json=json.dumps(
            {
                "trial_id": "native_control",
                "max_gap": 360,
                "candidate_top_k": 8,
                "score_threshold": 0.0,
                "margin_threshold": 0.0,
            }
        ),
        stage="subset",
        gpu="0",
        stream_python="python",
        evaluator_python="python",
        evaluator_name="COV_V10_TEMPO",
        evaluator_cores=1,
        disabled_overlay=True,
        search_plan=None,
        search_plan_sha256=None,
        contract_gate=None,
        contract_gate_sha256=None,
        threshold_source=None,
    )
    assert module.run_trial(args) == 0
    receipt = json.loads((output_root / "native_control" / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "COMPLETED"
    assert receipt["runtime_contract"]["status"] == "NOT_APPLICABLE"
    assert evaluator_calls == ["evaluation.log"]


def _expected_input_plan(tmp_path: Path, *, updates: dict[str, str]):
    scheduler = _scheduler_module()
    annotation = tmp_path / "annotation.json"
    annotation.write_text("annotation", encoding="utf-8")
    checkpoint = tmp_path / "external.pth"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    expected = {
        "subset_annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
        "reranker_checkpoint_sha256": "reranker",
        "external_cov_commit": "source-commit",
        "external_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    plan = SimpleNamespace(expected_inputs=expected)
    args = SimpleNamespace(
        annotation=str(annotation),
        source=str(source),
        external_checkpoint=str(checkpoint),
        external_config=str(tmp_path / "external.py"),
        base_config=str(tmp_path / "base.yaml"),
    )
    Path(args.external_config).write_text("external", encoding="utf-8")
    Path(args.base_config).write_text("base", encoding="utf-8")
    expected["external_config_sha256"] = hashlib.sha256(
        Path(args.external_config).read_bytes()
    ).hexdigest()
    expected["base_config_sha256"] = hashlib.sha256(
        Path(args.base_config).read_bytes()
    ).hexdigest()
    expected.update(updates)
    return scheduler, plan, args


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("subset_annotation_sha256", "wrong", "ANNOTATION"),
        ("external_cov_commit", "wrong", "EXTERNAL_COMMIT"),
        ("external_checkpoint_sha256", "wrong", "EXTERNAL_CHECKPOINT"),
        ("reranker_checkpoint_sha256", "wrong", "RERANKER_CHECKPOINT"),
    ],
)
def test_search_plan_expected_inputs_mismatch_fails(monkeypatch, tmp_path, field, value, error):
    scheduler, plan, args = _expected_input_plan(tmp_path, updates={field: value})
    monkeypatch.setattr(
        scheduler,
        "_git_value",
        lambda _path, *git_args: "" if git_args == ("status", "--porcelain") else "source-commit",
    )
    with pytest.raises(RuntimeError, match=f"SEARCH_EXPECTED_INPUT_{error}_MISMATCH"):
        scheduler._validate_expected_inputs(plan, args, reranker_checkpoint_sha256="reranker")


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("external_config_sha256", "EXTERNAL_CONFIG"),
        ("base_config_sha256", "BASE_CONFIG"),
    ],
)
def test_search_plan_expected_config_hash_mismatch_fails(monkeypatch, tmp_path, field, error):
    scheduler, plan, args = _expected_input_plan(tmp_path, updates={field: "wrong"})
    monkeypatch.setattr(
        scheduler,
        "_git_value",
        lambda _path, *git_args: "" if git_args == ("status", "--porcelain") else "source-commit",
    )
    with pytest.raises(RuntimeError, match=f"SEARCH_EXPECTED_INPUT_{error}_MISMATCH"):
        scheduler._validate_expected_inputs(plan, args, reranker_checkpoint_sha256="reranker")
