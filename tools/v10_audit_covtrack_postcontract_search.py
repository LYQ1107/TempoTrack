#!/usr/bin/env python3
"""Audit the already-launched V10.4 COV Q1 search without editing receipts.

The first search wave was launched before the later hardening changes.  This
tool deliberately treats its receipts as immutable evidence and writes only
selection sidecars.  A trial is eligible for ranking only when every input,
output, runtime-contract, and Q1 checkpoint binding is independently proven.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

try:
    from v10_search_covtrack_full_test import _load_search_plan, _sha256, _validate_contract_gate
except ModuleNotFoundError:  # import-safe when loaded by pytest from the repo root
    import importlib.util
    import sys

    _sibling_spec = importlib.util.spec_from_file_location(
        "_v10_search_covtrack_full_test", Path(__file__).with_name("v10_search_covtrack_full_test.py")
    )
    if _sibling_spec is None or _sibling_spec.loader is None:
        raise ImportError("cannot load sibling v10_search_covtrack_full_test.py")
    _sibling = importlib.util.module_from_spec(_sibling_spec)
    sys.modules[_sibling_spec.name] = _sibling
    _sibling_spec.loader.exec_module(_sibling)
    _load_search_plan = _sibling._load_search_plan
    _sha256 = _sibling._sha256
    _validate_contract_gate = _sibling._validate_contract_gate


LAUNCH_CHECKPOINT_SHA256 = "ed2524af31c22d17b6fcb61dd118095274b1bb56ad9329b993f9cdc4579aa80f"
LAUNCH_EXTERNAL_COMMIT = "9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b"
LAUNCH_HEAD_DEFAULT = "e24b2db4e2297c51ad6718e70b3e1fd520ee6ff8"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    inputs = receipt.get("inputs", {})
    source = receipt.get("external_source", {})
    return {
        "annotation_sha256": inputs.get("annotation_sha256"),
        "external_commit": source.get("commit"),
        "external_checkpoint_sha256": inputs.get("external_checkpoint_sha256"),
        "external_config_sha256": inputs.get("external_config_sha256"),
        "base_config_sha256": inputs.get("base_config_sha256"),
    }


def _check_file_hash(
    receipt: Mapping[str, Any],
    section: str,
    path_key: str,
    hash_key: str,
    reasons: list[str],
) -> dict[str, Any]:
    section_value = receipt.get(section, {})
    raw_path = section_value.get(path_key)
    expected = section_value.get(hash_key)
    result = {"path": raw_path, "expected_sha256": expected, "actual_sha256": None}
    if not raw_path or not Path(raw_path).is_file():
        reasons.append(f"{section.upper()}_{path_key.upper()}_MISSING")
        return result
    actual = _sha256(Path(raw_path))
    result["actual_sha256"] = actual
    if not expected or actual != expected:
        reasons.append(f"{section.upper()}_{path_key.upper()}_HASH_MISMATCH")
    return result


def _find_receipt(root: Path, requested_id: str) -> Path | None:
    direct = root / requested_id / "receipt.json"
    if direct.is_file():
        return direct
    for path in sorted(root.glob("*/receipt.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(value.get("requested_trial_id", value.get("trial_id", ""))) == requested_id:
            return path
    return None


def _audit_trial(
    *,
    receipt_path: Path | None,
    expected_spec: Mapping[str, Any],
    root: Path,
    annotation_sha256: str,
    launch_head: str,
    external_commit: str,
    checkpoint_sha256: str,
    external_config_sha256: str | None,
) -> dict[str, Any]:
    trial_id = str(expected_spec["trial_id"])
    reasons: list[str] = []
    receipt: dict[str, Any] = {}
    if receipt_path is None:
        reasons.append("RECEIPT_MISSING")
    else:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reasons.append("RECEIPT_INVALID")
    if receipt:
        if receipt.get("status") != "COMPLETED":
            reasons.append("RECEIPT_NOT_COMPLETED")
        if receipt.get("repo", {}).get("head") != launch_head:
            reasons.append("LAUNCH_HEAD_MISMATCH")
        actual_spec = dict(receipt.get("spec", {}))
        wanted_spec = dict(expected_spec)
        if actual_spec != wanted_spec:
            reasons.append("TRIAL_SPEC_MISMATCH")
        binding = _binding(receipt)
        if binding["annotation_sha256"] != annotation_sha256:
            reasons.append("ANNOTATION_HASH_MISMATCH")
        if binding["external_commit"] != external_commit:
            reasons.append("EXTERNAL_COMMIT_MISMATCH")
        if binding["external_checkpoint_sha256"] != checkpoint_sha256:
            reasons.append("EXTERNAL_CHECKPOINT_HASH_MISMATCH")
        if external_config_sha256 is not None and binding["external_config_sha256"] != external_config_sha256:
            reasons.append("EXTERNAL_CONFIG_HASH_MISMATCH")
        inputs = receipt.get("inputs", {})
        for key in ("external_checkpoint", "external_config", "base_config"):
            raw = inputs.get(key)
            hash_key = f"{key}_sha256"
            if not raw or not Path(raw).is_file():
                reasons.append(f"INPUT_{key.upper()}_MISSING")
            elif inputs.get(hash_key) != _sha256(Path(raw)):
                reasons.append(f"INPUT_{key.upper()}_HASH_MISMATCH")
        _check_file_hash(receipt, "inputs", "tempo_config", "tempo_config_sha256", reasons)
        outputs = receipt.get("outputs", {})
        for key in ("prediction", "diagnostics", "summary"):
            raw = outputs.get(key)
            expected_hash = outputs.get(f"{key}_sha256")
            if not raw or not Path(raw).is_file():
                reasons.append(f"OUTPUT_{key.upper()}_MISSING")
            elif not expected_hash or _sha256(Path(raw)) != expected_hash:
                reasons.append(f"OUTPUT_{key.upper()}_HASH_MISMATCH")
        diagnostics_path = Path(outputs.get("diagnostics", ""))
        diagnostics: dict[str, Any] = {}
        if diagnostics_path.is_file():
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if int(diagnostics.get("reranker_expected_query_observations", -1)) != 1:
                reasons.append("RUNTIME_EXPECTED_Q_MISMATCH")
            if int(diagnostics.get("reranker_actual_query_observations", -1)) != 1:
                reasons.append("RUNTIME_ACTUAL_Q_MISMATCH")
            if int(diagnostics.get("reranker_context_candidate_top_k", -1)) != 64:
                reasons.append("RUNTIME_CONTEXT_K_MISMATCH")
            if int(diagnostics.get("reranker_decision_candidate_top_k", -1)) != int(expected_spec["candidate_top_k"]):
                reasons.append("RUNTIME_DECISION_K_MISMATCH")
            if bool(diagnostics.get("reranker_context_contract_mismatch", True)):
                reasons.append("RUNTIME_CONTEXT_CONTRACT_MISMATCH")
            if int(diagnostics.get("reranker_missing_evidence", -1)) != 0:
                reasons.append("RUNTIME_MISSING_EVIDENCE")
            if (
                "reranker_native_memo_bootstrap_count" in diagnostics
                and int(diagnostics["reranker_native_memo_bootstrap_count"]) != 0
            ):
                reasons.append("RUNTIME_NATIVE_MEMO_BOOTSTRAP")
            counts = dict(diagnostics.get("full_capability_status_counts", {}))
            if not counts or any(key != "FULL_Q1_RERANKER_RUNTIME_ACTIVE" for key in counts):
                reasons.append("RUNTIME_CAPABILITY_MISMATCH")
            if int(sum(int(value) for value in counts.values())) != int(diagnostics.get("frames", -1)):
                reasons.append("RUNTIME_CAPABILITY_FRAME_COVERAGE")
            status = diagnostics.get("reranker_status")
            if not isinstance(status, Mapping):
                reasons.append("RERANKER_PROVENANCE_MISSING")
            else:
                if status.get("status") != "EXACT_V9_MODEL_CODE_AND_WEIGHTS":
                    reasons.append("RERANKER_PROVENANCE_STATUS_MISMATCH")
                if status.get("checkpoint_sha256") != LAUNCH_CHECKPOINT_SHA256:
                    reasons.append("RERANKER_CHECKPOINT_HASH_MISMATCH")
                for key, expected in (
                    ("base_only_supervision", True),
                    ("novel_gt_used", False),
                    ("test_weights_used", False),
                ):
                    if status.get(key) is not expected:
                        reasons.append(f"RERANKER_{key.upper()}_MISMATCH")
        else:
            reasons.append("RUNTIME_DIAGNOSTICS_MISSING")
    binding = _binding(receipt) if receipt else {}
    hash_binding = {
        "prediction": _check_file_hash(receipt, "outputs", "prediction", "prediction_sha256", reasons) if receipt else {},
        "diagnostics": _check_file_hash(receipt, "outputs", "diagnostics", "diagnostics_sha256", reasons) if receipt else {},
        "summary": _check_file_hash(receipt, "outputs", "summary", "summary_sha256", reasons) if receipt else {},
    }
    audit = {
        "status": "PASS" if not reasons else "FAIL",
        "usage": "SEARCH_SELECTION" if not reasons else "REJECT_FROM_SELECTION",
        "trial_id": trial_id,
        "receipt": str(receipt_path) if receipt_path else None,
        "search_plan_sha256": None,
        "contract_gate_sha256": None,
        "launch_head": launch_head,
        "runtime_contract": {
            "status": "PASS" if not any(reason.startswith("RUNTIME_") for reason in reasons) else "FAIL",
            "failures": [reason for reason in reasons if reason.startswith("RUNTIME_")],
        },
        "input_binding": binding,
        "hash_binding": hash_binding,
        "failure_reasons": sorted(set(reasons)),
    }
    return audit


def audit(args: argparse.Namespace) -> int:
    search_root = Path(args.search_root).resolve()
    plan_path = Path(args.search_plan).resolve()
    plan = _load_search_plan(plan_path)
    gate = _validate_contract_gate(plan)
    if args.contract_gate:
        explicit_gate = Path(args.contract_gate).resolve()
        if str(explicit_gate) != plan.contract_gate:
            raise RuntimeError("SEARCH_CONTRACT_GATE_PATH_MISMATCH")
        if _sha256(explicit_gate) != plan.contract_gate_sha256:
            raise RuntimeError("SEARCH_CONTRACT_GATE_HASH_MISMATCH")
    annotation = Path(args.annotation).resolve()
    annotation_sha256 = _sha256(annotation)
    checkpoint_sha256 = args.checkpoint_sha256 or LAUNCH_CHECKPOINT_SHA256
    external_commit = args.external_commit or LAUNCH_EXTERNAL_COMMIT
    external_config_sha256 = args.external_config_sha256
    if external_config_sha256 is None:
        for receipt_path in sorted(search_root.glob("*/receipt.json")):
            try:
                value = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            value_hash = value.get("inputs", {}).get("external_config_sha256")
            if value_hash:
                external_config_sha256 = str(value_hash)
                break
    launch_head = args.launch_head or LAUNCH_HEAD_DEFAULT
    trials: dict[str, dict[str, Any]] = {}
    for spec in plan.trials:
        requested_id = str(spec["trial_id"])
        receipt_path = _find_receipt(search_root, requested_id)
        result = _audit_trial(
            receipt_path=receipt_path,
            expected_spec=spec,
            root=search_root,
            annotation_sha256=annotation_sha256,
            launch_head=launch_head,
            external_commit=external_commit,
            checkpoint_sha256=checkpoint_sha256,
            external_config_sha256=external_config_sha256,
        )
        result["search_plan_sha256"] = plan.sha256
        result["contract_gate_sha256"] = plan.contract_gate_sha256
        trials[requested_id] = result
        if receipt_path is not None:
            _write_json(receipt_path.parent / "selection_audit.json", result)
    passed = sum(value["status"] == "PASS" for value in trials.values())
    global_result = {
        "schema_version": 1,
        "artifact": "tempotrack_v10_postcontract_search_audit",
        "status": "PASS" if passed == len(trials) and trials else "INCOMPLETE_OR_FAILED",
        "usage": "SEARCH_SELECTION" if passed == len(trials) and trials else "REJECT_FROM_SELECTION",
        "search_root": str(search_root),
        "search_plan": {"path": str(plan_path), "sha256": plan.sha256},
        "contract_gate": {"path": plan.contract_gate, "sha256": plan.contract_gate_sha256, "status": gate.get("status")},
        "launch_head": launch_head,
        "annotation": {"path": str(annotation), "sha256": annotation_sha256},
        "external_commit": external_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "external_config_sha256": external_config_sha256,
        "trials": trials,
        "counts": {"total": len(trials), "pass": passed, "fail": len(trials) - passed},
    }
    output = Path(args.output).resolve() if args.output else search_root / "postcontract_search_audit.json"
    _write_json(output, global_result)
    print(json.dumps({"status": global_result["status"], "pass": passed, "total": len(trials), "output": str(output)}))
    return 0 if global_result["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--search-plan", required=True)
    parser.add_argument("--contract-gate", help="Optional explicit gate path; plan binding is authoritative")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--launch-head", default=LAUNCH_HEAD_DEFAULT)
    parser.add_argument("--external-commit")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--external-config-sha256")
    parser.add_argument("--output")
    args = parser.parse_args()
    return audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
