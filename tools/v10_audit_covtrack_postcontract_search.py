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
import tempfile
from typing import Any, Mapping

try:
    from v10_search_covtrack_full_test import (
        _load_search_plan,
        _materialize_config,
        _sha256,
        _validate_contract_gate,
    )
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
    _materialize_config = _sibling._materialize_config
    _sha256 = _sibling._sha256
    _validate_contract_gate = _sibling._validate_contract_gate


LAUNCH_CHECKPOINT_SHA256 = "ed2524af31c22d17b6fcb61dd118095274b1bb56ad9329b993f9cdc4579aa80f"
LAUNCH_EXTERNAL_CHECKPOINT_SHA256 = "e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c"
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
    candidates: list[Path] = []
    direct = root / requested_id / "receipt.json"
    if direct.is_file():
        candidates.append(direct)
    candidates.extend(sorted(root.glob(f"{requested_id}__retry*/receipt.json")))
    for path in sorted(root.glob("*/receipt.json")):
        if path in candidates:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(value.get("requested_trial_id", "")) == requested_id:
            candidates.append(path)
    completed: list[Path] = []
    for path in candidates:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("status") == "COMPLETED":
            completed.append(path)
    if len(completed) == 1:
        return completed[0]
    if len(completed) > 1:
        raise RuntimeError(
            "MULTIPLE_COMPLETED_ATTEMPTS_FOR_REQUESTED_TRIAL: " + requested_id
        )
    return candidates[-1] if candidates else None


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
    external_checkpoint_sha256: str | None = None,
    expected_base_config: Path,
    expected_base_config_sha256: str,
) -> dict[str, Any]:
    trial_id = str(expected_spec["trial_id"])
    reasons: list[str] = []
    receipt: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    base_config_reconstruction_status = "NOT_APPLICABLE"
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
        effective_trial_id = str(actual_spec.pop("trial_id", receipt.get("trial_id", "")))
        requested_trial_id = str(wanted_spec.pop("trial_id", ""))
        if actual_spec != wanted_spec:
            reasons.append("TRIAL_SPEC_MISMATCH")
        binding = _binding(receipt)
        if binding["annotation_sha256"] != annotation_sha256:
            reasons.append("ANNOTATION_HASH_MISMATCH")
        if binding["external_commit"] != external_commit:
            reasons.append("EXTERNAL_COMMIT_MISMATCH")
        if (
            external_checkpoint_sha256 is not None
            and binding["external_checkpoint_sha256"] != external_checkpoint_sha256
        ):
            reasons.append("EXTERNAL_CHECKPOINT_HASH_MISMATCH")
        if external_config_sha256 is not None and binding["external_config_sha256"] != external_config_sha256:
            reasons.append("EXTERNAL_CONFIG_HASH_MISMATCH")
        inputs = receipt.get("inputs", {})
        for key in ("external_checkpoint", "external_config"):
            raw = inputs.get(key)
            hash_key = f"{key}_sha256"
            if not raw or not Path(raw).is_file():
                reasons.append(f"INPUT_{key.upper()}_MISSING")
            elif inputs.get(hash_key) != _sha256(Path(raw)):
                reasons.append(f"INPUT_{key.upper()}_HASH_MISMATCH")
        receipt_base_config = inputs.get("base_config")
        receipt_base_config_sha = inputs.get("base_config_sha256")
        legacy_base_config_reconstructed = receipt_base_config is None
        base_config_reconstruction_status = "NOT_APPLICABLE"
        if receipt_base_config is not None:
            receipt_base_path = Path(str(receipt_base_config)).expanduser().resolve()
            if not receipt_base_path.is_file():
                reasons.append("INPUT_BASE_CONFIG_MISSING")
                base_config_reconstruction_status = "FAIL"
            else:
                actual_receipt_base_sha = _sha256(receipt_base_path)
                if actual_receipt_base_sha != receipt_base_config_sha:
                    reasons.append("INPUT_BASE_CONFIG_HASH_MISMATCH")
                if actual_receipt_base_sha != expected_base_config_sha256:
                    reasons.append("BASE_CONFIG_EXPECTED_HASH_MISMATCH")
                base_config_reconstruction_status = (
                    "PASS"
                    if actual_receipt_base_sha == expected_base_config_sha256
                    and actual_receipt_base_sha == receipt_base_config_sha
                    else "FAIL"
                )
        else:
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    reconstructed = Path(temporary) / "tempo.yaml"
                    _materialize_config(
                        expected_base_config,
                        reconstructed,
                        expected_spec,
                        disabled=False,
                    )
                    reconstructed_sha = _sha256(reconstructed)
                if reconstructed_sha != inputs.get("tempo_config_sha256"):
                    reasons.append("LEGACY_BASE_CONFIG_RECONSTRUCTION_MISMATCH")
                    base_config_reconstruction_status = "FAIL"
                else:
                    base_config_reconstruction_status = "PASS"
            except Exception:
                reasons.append("LEGACY_BASE_CONFIG_RECONSTRUCTION_MISMATCH")
                base_config_reconstruction_status = "FAIL"
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
                if status.get("checkpoint_sha256") != checkpoint_sha256:
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
        "contract_classification": (
            "SEARCH_SELECTION_LEGACY_CONTRACT"
            if "reranker_native_memo_bootstrap_count" not in diagnostics
            else "SEARCH_SELECTION_HARDENED_CONTRACT"
        ),
        "trial_id": trial_id,
        "requested_trial_id": requested_trial_id if receipt else trial_id,
        "effective_trial_id": effective_trial_id if receipt else None,
        "receipt": str(receipt_path) if receipt_path else None,
        "search_plan_sha256": None,
        "contract_gate_sha256": None,
        "launch_head": launch_head,
        "runtime_contract": {
            "status": "PASS" if not any(reason.startswith("RUNTIME_") for reason in reasons) else "FAIL",
            "failures": [reason for reason in reasons if reason.startswith("RUNTIME_")],
        },
        "native_memo_bootstrap": (
            {
                "observed": True,
                "value": int(diagnostics.get("reranker_native_memo_bootstrap_count")),
                "status": "PASS"
                if int(diagnostics.get("reranker_native_memo_bootstrap_count")) == 0
                else "FAIL",
            }
            if "reranker_native_memo_bootstrap_count" in diagnostics
            else {
                "observed": False,
                "value": None,
                "status": "LEGACY_NOT_RECORDED",
            }
        ),
        "base_config_binding": {
            "expected_path": str(expected_base_config),
            "expected_sha256": expected_base_config_sha256,
            "receipt_had_base_config": bool(receipt.get("inputs", {}).get("base_config")) if receipt else False,
            "legacy_reconstructed": bool(receipt and "base_config" not in receipt.get("inputs", {})),
            "reconstruction_status": (
                base_config_reconstruction_status if receipt else "NOT_APPLICABLE"
            ),
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
    external_checkpoint_sha256 = (
        args.external_checkpoint_sha256 or LAUNCH_EXTERNAL_CHECKPOINT_SHA256
    )
    external_commit = args.external_commit or LAUNCH_EXTERNAL_COMMIT
    external_config_sha256 = args.external_config_sha256
    base_config = Path(args.base_config).expanduser().resolve()
    if not base_config.is_file():
        raise FileNotFoundError(f"base config missing: {base_config}")
    actual_base_config_sha = _sha256(base_config)
    if args.base_config_sha256 is not None and actual_base_config_sha != args.base_config_sha256:
        raise RuntimeError("BASE_CONFIG_SHA256_MISMATCH")
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
            external_checkpoint_sha256=external_checkpoint_sha256,
            expected_base_config=base_config,
            expected_base_config_sha256=actual_base_config_sha,
        )
        result["search_plan_sha256"] = plan.sha256
        result["contract_gate_sha256"] = plan.contract_gate_sha256
        trials[requested_id] = result
        if receipt_path is not None:
            _write_json(receipt_path.parent / "selection_audit.json", result)
    passed = sum(value["status"] == "PASS" for value in trials.values())
    total = len(trials)
    failed = total - passed
    if total == 0:
        global_status, global_usage = "FAIL", "REJECT_FROM_SELECTION"
    elif passed == total:
        global_status, global_usage = "PASS", "SEARCH_SELECTION"
    elif passed > 0:
        global_status, global_usage = "PARTIAL_PASS", "SEARCH_SELECTION"
    else:
        global_status, global_usage = "FAIL", "REJECT_FROM_SELECTION"
    global_result = {
        "schema_version": 1,
        "artifact": "tempotrack_v10_postcontract_search_audit",
        "status": global_status,
        "usage": global_usage,
        "contract_classification": (
            "SEARCH_SELECTION_LEGACY_CONTRACT"
            if any(
                value.get("contract_classification") == "SEARCH_SELECTION_LEGACY_CONTRACT"
                for value in trials.values()
            )
            else "SEARCH_SELECTION_HARDENED_CONTRACT"
        ),
        "search_root": str(search_root),
        "search_plan": {"path": str(plan_path), "sha256": plan.sha256},
        "contract_gate": {"path": plan.contract_gate, "sha256": plan.contract_gate_sha256, "status": gate.get("status")},
        "launch_head": launch_head,
        "annotation": {"path": str(annotation), "sha256": annotation_sha256},
        "base_config": {"path": str(base_config), "sha256": actual_base_config_sha},
        "external_commit": external_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "external_checkpoint_sha256": external_checkpoint_sha256,
        "external_config_sha256": external_config_sha256,
        "trials": trials,
        "counts": {"total": total, "pass": passed, "fail": failed},
    }
    output = Path(args.output).resolve() if args.output else search_root / "postcontract_search_audit.json"
    _write_json(output, global_result)
    print(json.dumps({"status": global_result["status"], "pass": passed, "total": len(trials), "output": str(output)}))
    return 0 if global_result["status"] in {"PASS", "PARTIAL_PASS"} else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--search-plan", required=True)
    parser.add_argument("--contract-gate", help="Optional explicit gate path; plan binding is authoritative")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--launch-head", default=LAUNCH_HEAD_DEFAULT)
    parser.add_argument("--external-commit")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--external-checkpoint-sha256")
    parser.add_argument("--external-config-sha256")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-config-sha256")
    parser.add_argument("--output")
    args = parser.parse_args()
    return audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
