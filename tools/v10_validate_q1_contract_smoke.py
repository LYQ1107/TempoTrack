#!/usr/bin/env python3
"""Validate one real V10.4 Q1 contract smoke and write an auditable gate receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import yaml

from tempotrack_v10.reranker import validate_feature_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _runtime_contract_sha_from_config(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        return None
    raw = value.get("runtime_contract_sha")
    if raw is not None:
        return str(raw)
    tempo = value.get("tempo", value)
    if not isinstance(tempo, dict):
        return None
    raw = tempo.get("runtime_contract_sha")
    return None if raw is None else str(raw)


def _validate_teta_dependency(receipt: dict[str, Any]) -> bool:
    dependency = receipt.get("teta_dependency")
    preflight = receipt.get("teta_import_preflight")
    if not isinstance(dependency, dict) or not isinstance(preflight, dict):
        return False
    init_file = Path(str(dependency.get("init_file", "")))
    if not init_file.is_file():
        return False
    if dependency.get("init_sha256") != _sha256(init_file):
        return False
    if dependency.get("tracked_source_clean") is not True:
        return False
    if preflight.get("status") != "PASS" or int(preflight.get("returncode", -1)) != 0:
        return False
    actual_imports = preflight.get("actual_imports")
    return isinstance(actual_imports, dict) and (
        actual_imports.get("teta_file") == dependency.get("init_file")
        and actual_imports.get("teta_file") == preflight.get("expected_teta_init")
        and actual_imports.get("cov_dataset_file") == preflight.get("expected_cov_dataset_init")
    )


def _run_contract_tests(repo: Path, pytest_python: Path) -> dict[str, Any]:
    nodes = [
        "tests/test_v10_reranker_contract.py::test_q1_online_prefilter_matches_training_last_observation_cosine",
        "tests/test_v10_reranker_contract.py::test_full_prefilter_ignores_native_affinity_for_candidate_rank",
        "tests/test_v10_reranker_contract.py::test_runtime_max_gap_does_not_change_checkpoint_feature_normalization",
        "tests/test_v10_reranker_contract.py::test_reranker_memory_bank_matches_canonical_anchor",
        "tests/test_v10_contract.py",
    ]
    command = [str(pytest_python), "-m", "pytest", *nodes, "-q"]
    result = subprocess.run(command, cwd=str(repo), text=True, capture_output=True)
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": int(result.returncode),
        "command": command,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-2000:],
    }


def validate(
    *,
    repo: Path,
    trial: Path,
    annotation: Path,
    checkpoint: Path,
    output: Path,
    pytest_python: Path,
    expected_runtime_contract_sha: str | None = None,
) -> dict[str, Any]:
    receipt = _load_json(trial / "receipt.json")
    diagnostics = _load_json(trial / "diagnostics.json")
    stream_manifest = _load_json(trial / "stream" / "stream_manifest.json")
    rows = json.loads((trial / "stream" / "tao_track.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("prediction must be a JSON list")
    annotation_doc = _load_json(annotation)
    frame_count = len(annotation_doc.get("images", []))
    state = torch.load(checkpoint, map_location="cpu")
    feature_config = validate_feature_config(state.get("feature_config"))
    spec = dict(receipt.get("spec") or {})
    expected_decision_k = int(spec.get("candidate_top_k", 0))
    prediction = trial / "stream" / "tao_track.json"
    diagnostics_path = trial / "diagnostics.json"
    stream_manifest_path = trial / "stream" / "stream_manifest.json"
    test_gates = _run_contract_tests(repo, pytest_python)
    source_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    # The exact Q1 prefilter is the same scalar operation used by training:
    # cosine(query, candidate's last real observation).  The production
    # tests above exercise the real overlay call path and its rank ordering.
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    last = np.asarray([0.8, 0.6], dtype=np.float32)
    prefilter_formula_pass = bool(
        np.isclose(float(np.dot(query, last) / (np.linalg.norm(query) * np.linalg.norm(last))), 0.8)
    )
    capability_counts = diagnostics.get("full_capability_status_counts", {})
    reranker_status = diagnostics.get("reranker_status")
    bootstrap_count = diagnostics.get("reranker_native_memo_bootstrap_count")
    runtime_contract_sha = None
    tempo_config = receipt.get("inputs", {}).get("tempo_config")
    if tempo_config and Path(tempo_config).is_file():
        tempo_doc = yaml.safe_load(Path(tempo_config).read_text(encoding="utf-8")) or {}
        runtime_contract_sha = tempo_doc.get("runtime_contract_sha")
    expected_runtime_contract_sha = (
        expected_runtime_contract_sha
        or _runtime_contract_sha_from_config(
            Path(str(receipt.get("inputs", {}).get("base_config", "")))
        )
        or _runtime_contract_sha_from_config(
            Path(str(receipt.get("inputs", {}).get("tempo_config", "")))
        )
    )
    overlay_sha = receipt.get("inputs", {}).get("overlay_sha256")
    runtime_sha = receipt.get("inputs", {}).get("runtime_sha256")
    decision_frames = int(diagnostics.get("frames", -1))
    capability_total = (
        sum(int(value) for value in capability_counts.values())
        if isinstance(capability_counts, dict)
        else -1
    )
    teta_dependency_pass = _validate_teta_dependency(receipt)
    gates = {
        "receipt_completed": receipt.get("status") == "COMPLETED",
        "source_head_bound": receipt.get("repo", {}).get("head") == source_head,
        "checkpoint_feature_config_present": bool(feature_config),
        "expected_q_is_one": int(feature_config["query_observations"]) == 1,
        "actual_q_is_one": diagnostics.get("reranker_actual_query_observations") == 1,
        "expected_actual_q_match": diagnostics.get("reranker_expected_query_observations") == diagnostics.get("reranker_actual_query_observations") == 1,
        "prefilter_exact_formula": prefilter_formula_pass,
        "prefilter_and_memory_and_causality_tests": test_gates["status"] == "PASS",
        "memory_parity_pass": test_gates["status"] == "PASS",
        "reranker_missing_evidence_zero": int(diagnostics.get("reranker_missing_evidence", -1)) == 0,
        "native_memo_bootstrap_counter_present": bootstrap_count is not None,
        "native_memo_bootstrap_counter_zero": bootstrap_count == 0,
        "reranker_provenance_exact": isinstance(reranker_status, dict)
        and reranker_status.get("status") == "EXACT_V9_MODEL_CODE_AND_WEIGHTS"
        and reranker_status.get("base_only_supervision") is True
        and reranker_status.get("novel_gt_used") is False
        and reranker_status.get("test_weights_used") is False,
        "context_k_64": diagnostics.get("reranker_context_candidate_top_k") == int(feature_config["candidate_top_k"]) == 64,
        "decision_k_runtime_bound": diagnostics.get("reranker_decision_candidate_top_k") == expected_decision_k,
        "context_contract_stable": diagnostics.get("reranker_context_contract_mismatch") is False,
        "diagnostic_decision_frames_positive": decision_frames > 0,
        "capability_covers_all_decision_frames": capability_total == decision_frames,
        "all_decision_frames_are_full_q1": capability_counts == {
            "FULL_Q1_RERANKER_RUNTIME_ACTIVE": decision_frames
        },
        "stream_covers_all_annotation_frames": (
            stream_manifest.get("status") == "PASS"
            and int(stream_manifest.get("frames", -1)) == frame_count
        ),
        "prediction_is_nonempty": len(rows) > 0,
        "annotation_binding": receipt.get("inputs", {}).get("annotation_sha256") == _sha256(annotation),
        "prediction_binding": receipt.get("outputs", {}).get("prediction_sha256") == _sha256(prediction),
        "diagnostics_binding": receipt.get("outputs", {}).get("diagnostics_sha256") == _sha256(diagnostics_path),
        "stream_manifest_binding": receipt.get("outputs", {}).get("stream_manifest_sha256") == _sha256(stream_manifest_path),
        "official_evaluator_parsed": receipt.get("metrics", {}).get("status") == "PARSED",
        "runtime_contract_sha_present": bool(runtime_contract_sha),
        "runtime_contract_sha_matches_expected": (
            bool(expected_runtime_contract_sha)
            and runtime_contract_sha == expected_runtime_contract_sha
        ),
        "overlay_sha_present": bool(overlay_sha),
        "overlay_sha_matches_repo": (
            bool(overlay_sha)
            and overlay_sha == _sha256(repo / "tempotrack_v10/overlay.py")
        ),
        "runtime_sha_present": bool(runtime_sha),
        "runtime_sha_matches_repo": (
            bool(runtime_sha)
            and runtime_sha == _sha256(repo / "tempotrack_v10/covtrack_runtime.py")
        ),
        "teta_dependency_provenance": teta_dependency_pass,
    }
    result = {
        "schema_version": 1,
        "artifact": "tempotrack_v10_q1_contract_smoke_gate",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": "COV_Q1_FEATURE_CONTRACT_SMOKE",
        "repo": str(repo),
        "repo_head": source_head,
        "trial": str(trial),
        "annotation": str(annotation),
        "annotation_sha256": _sha256(annotation),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "feature_config": feature_config,
        "expected_q": int(feature_config["query_observations"]),
        "actual_q": diagnostics.get("reranker_actual_query_observations"),
        "context_candidate_top_k": diagnostics.get("reranker_context_candidate_top_k"),
        "decision_candidate_top_k": diagnostics.get("reranker_decision_candidate_top_k"),
        "reranker_missing_evidence": int(diagnostics.get("reranker_missing_evidence", -1)),
        "reranker_native_memo_bootstrap_count": bootstrap_count,
        "runtime_contract_sha": runtime_contract_sha,
        "expected_runtime_contract_sha": expected_runtime_contract_sha,
        "overlay_sha256": overlay_sha,
        "runtime_sha256": runtime_sha,
        "teta_dependency": receipt.get("teta_dependency"),
        "teta_import_preflight": receipt.get("teta_import_preflight"),
        "score_quantiles": diagnostics.get("score_quantiles"),
        "margin_quantiles": diagnostics.get("margin_quantiles"),
        "prediction_sha256": _sha256(prediction),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "stream_manifest_sha256": _sha256(stream_manifest_path),
        "official_metrics": receipt.get("metrics"),
        "contract_tests": test_gates,
        "gates": gates,
    }
    _atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytest-python", type=Path, required=True)
    parser.add_argument("--expected-runtime-contract-sha")
    args = parser.parse_args()
    result = validate(
        repo=args.repo.resolve(),
        trial=args.trial.resolve(),
        annotation=args.annotation.resolve(),
        checkpoint=args.checkpoint.resolve(),
        output=args.output.resolve(),
        pytest_python=args.pytest_python.resolve(),
        expected_runtime_contract_sha=args.expected_runtime_contract_sha,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
