#!/usr/bin/env python3
"""Auditable staged search over the frozen V10.4 COV contract.

This controller deliberately does not implement another inference path.  It
creates immutable search plans and delegates every trial to the existing
``v10_run_covtrack_search.py`` coordinator and
``v10_search_covtrack_full_test.py`` one-trial runner.  All selections are
made from completed official TETA receipts; failed, partial, or provenance
mismatched observations remain on disk but are never ranking inputs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SEARCH_FIELDS = (
    "max_gap",
    "candidate_top_k",
    "score_threshold",
    "margin_threshold",
)
METRIC_NAMES = ("TETA", "LocA", "AssocA", "ClsA")
STRUCTURE_GAPS = (30, 60, 120, 240, 360)
STRUCTURE_TOPKS = (4, 8, 16, 32, 64)
TERMINAL_COORDINATOR_STATES = {"COMPLETED", "PARTIAL_FAILURE", "FAILED", "BLOCKED"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_branch(repo: Path) -> str | None:
    """Support the older Git installed on the server."""
    return git_value(repo, "symbolic-ref", "--short", "HEAD") or git_value(repo, "branch", "--show-current")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def approx_equal(left: Any, right: Any, *, rel: float = 1e-10, abs_tol: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=rel, abs_tol=abs_tol)
    except (TypeError, ValueError):
        return False


def canonical_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "max_gap": int(spec["max_gap"]),
        "candidate_top_k": int(spec["candidate_top_k"]),
        "score_threshold": float(spec["score_threshold"]),
        "margin_threshold": float(spec["margin_threshold"]),
    }
    if "trial_id" in spec:
        result["trial_id"] = str(spec["trial_id"])
    return result


def spec_key(spec: Mapping[str, Any]) -> tuple[int, int, float, float]:
    value = canonical_spec(spec)
    return (
        value["max_gap"],
        value["candidate_top_k"],
        value["score_threshold"],
        value["margin_threshold"],
    )


def dedupe_specs(specs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int, float, float]] = set()
    for raw in specs:
        value = canonical_spec(raw)
        key = spec_key(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("EXPANDED_SEARCH_INPUT_MISSING:" + ",".join(missing))


def load_policy(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"expanded policy is not an object: {path}")
    if tuple(value.get("search_fields", ())) != SEARCH_FIELDS:
        raise RuntimeError("EXPANDED_SEARCH_SEARCH_FIELDS_MISMATCH")
    structure = value.get("structure")
    if not isinstance(structure, dict):
        raise RuntimeError("EXPANDED_SEARCH_STRUCTURE_POLICY_MISSING")
    if tuple(int(x) for x in structure.get("max_gap", ())) != STRUCTURE_GAPS:
        raise RuntimeError("EXPANDED_SEARCH_GAP_POLICY_MISMATCH")
    if tuple(int(x) for x in structure.get("candidate_top_k", ())) != STRUCTURE_TOPKS:
        raise RuntimeError("EXPANDED_SEARCH_TOPK_POLICY_MISMATCH")
    if value.get("protocol") != "TEST_TUNED_MODEL_SPECIFIC" or value.get("unbiased_test") is not False:
        raise RuntimeError("EXPANDED_SEARCH_POLICY_PROTOCOL_MISMATCH")
    return value


def load_base_plan(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("trials"), list):
        raise RuntimeError("EXPANDED_SEARCH_BASE_PLAN_INVALID")
    if value.get("protocol") != "TEST_TUNED_MODEL_SPECIFIC":
        raise RuntimeError("EXPANDED_SEARCH_BASE_PLAN_PROTOCOL_MISMATCH")
    if value.get("unbiased_test") is not False:
        raise RuntimeError("EXPANDED_SEARCH_BASE_PLAN_UNBIASED_FLAG")
    if value.get("contract_mode") != "hardened":
        raise RuntimeError("EXPANDED_SEARCH_BASE_PLAN_NOT_HARDENED")
    quantiles = value.get("threshold_quantiles")
    expected_quantiles = {
        "score_p05": 1.367088943719864,
        "score_p25": 2.371457517147064,
        "score_p50": 3.5821245908737183,
        "margin_p05": 0.1311543583869934,
        "margin_p25": 0.6632569432258606,
        "margin_p50": 1.566097915172577,
    }
    if not isinstance(quantiles, dict):
        raise RuntimeError("EXPANDED_SEARCH_THRESHOLD_QUANTILES_MISSING")
    for key, expected in expected_quantiles.items():
        if not approx_equal(quantiles.get(key), expected):
            raise RuntimeError(f"EXPANDED_SEARCH_THRESHOLD_QUANTILE_MISMATCH:{key}")
    expected_inputs = value.get("expected_inputs")
    if not isinstance(expected_inputs, dict):
        raise RuntimeError("EXPANDED_SEARCH_EXPECTED_INPUTS_MISSING")
    required = (
        "subset_annotation_sha256",
        "full_test_annotation_sha256",
        "reranker_checkpoint_sha256",
        "external_cov_commit",
        "external_checkpoint_sha256",
        "external_config_sha256",
        "base_config_sha256",
        "teta_source_root",
        "teta_init_sha256",
        "teta_git_commit",
    )
    missing = [key for key in required if not expected_inputs.get(key)]
    if missing:
        raise RuntimeError("EXPANDED_SEARCH_EXPECTED_INPUT_FIELD_MISSING:" + ",".join(missing))
    return value


def validate_contract_inputs(
    *,
    repo: Path,
    base_plan_path: Path,
    base_plan: Mapping[str, Any],
    subset_annotation: Path,
    full_annotation: Path,
    source: Path,
    external_config: Path,
    external_checkpoint: Path,
    base_config: Path,
    teta_source_root: Path,
) -> dict[str, Any]:
    expected = dict(base_plan["expected_inputs"])
    gate_path = Path(str(base_plan["contract_gate"])).expanduser().resolve()
    _require_files(
        (
            base_plan_path,
            subset_annotation,
            full_annotation,
            external_config,
            external_checkpoint,
            base_config,
            gate_path,
            teta_source_root / "teta" / "__init__.py",
            repo / "tempotrack_v10" / "overlay.py",
            repo / "tempotrack_v10" / "covtrack_runtime.py",
        )
    )
    checks = {
        "subset_annotation_sha256": sha256(subset_annotation),
        "full_test_annotation_sha256": sha256(full_annotation),
        "external_checkpoint_sha256": sha256(external_checkpoint),
        "external_config_sha256": sha256(external_config),
        "base_config_sha256": sha256(base_config),
        "teta_init_sha256": sha256(teta_source_root / "teta" / "__init__.py"),
    }
    for key, actual in checks.items():
        if expected.get(key) != actual:
            raise RuntimeError(f"EXPANDED_SEARCH_INPUT_HASH_MISMATCH:{key}")
    source_commit = git_value(source, "rev-parse", "HEAD")
    if source_commit != expected.get("external_cov_commit"):
        raise RuntimeError("EXPANDED_SEARCH_EXTERNAL_SOURCE_COMMIT_MISMATCH")
    teta_commit = git_value(teta_source_root, "rev-parse", "HEAD")
    if teta_commit != expected.get("teta_git_commit"):
        raise RuntimeError("EXPANDED_SEARCH_TETA_COMMIT_MISMATCH")
    gate_sha = sha256(gate_path)
    if gate_sha != base_plan.get("contract_gate_sha256"):
        raise RuntimeError("EXPANDED_SEARCH_CONTRACT_GATE_HASH_MISMATCH")
    gate = read_json(gate_path)
    if not isinstance(gate, dict) or gate.get("status") != "PASS":
        raise RuntimeError("EXPANDED_SEARCH_CONTRACT_GATE_NOT_PASS")
    if int(gate.get("expected_q", -1)) != 1 or int(gate.get("actual_q", -1)) != 1:
        raise RuntimeError("EXPANDED_SEARCH_CONTRACT_Q_NOT_ONE")
    if int(gate.get("context_candidate_top_k", -1)) != 64:
        raise RuntimeError("EXPANDED_SEARCH_CONTRACT_CONTEXT_K_NOT_64")
    if int(gate.get("reranker_missing_evidence", -1)) != 0:
        raise RuntimeError("EXPANDED_SEARCH_CONTRACT_MISSING_EVIDENCE")
    return {
        "expected_inputs": expected,
        "contract_gate": str(gate_path),
        "contract_gate_sha256": gate_sha,
        "source_commit": source_commit,
        "teta_commit": teta_commit,
        "repo_head_at_preflight": git_value(repo, "rev-parse", "HEAD"),
        "repo_branch_at_preflight": git_branch(repo),
        "overlay_sha256": sha256(repo / "tempotrack_v10" / "overlay.py"),
        "runtime_sha256": sha256(repo / "tempotrack_v10" / "covtrack_runtime.py"),
        "base_plan_sha256": sha256(base_plan_path),
    }


def threshold_grid(quantiles: Mapping[str, Any], prefix: str) -> list[float]:
    p05 = float(quantiles[f"{prefix}_p05"])
    p25 = float(quantiles[f"{prefix}_p25"])
    p50 = float(quantiles[f"{prefix}_p50"])
    return [
        0.0,
        0.5 * p05,
        p05,
        0.5 * (p05 + p25),
        p25,
        0.5 * (p25 + p50),
        p50,
        1.25 * p50,
        1.50 * p50,
    ]


def stage1_specs(base_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    scores = threshold_grid(base_plan["threshold_quantiles"], "score")
    margins = threshold_grid(base_plan["threshold_quantiles"], "margin")
    output = []
    for si, score in enumerate(scores):
        for mi, margin in enumerate(margins):
            output.append(
                {
                    "trial_id": f"s{si:02d}_m{mi:02d}",
                    "max_gap": 360,
                    "candidate_top_k": 8,
                    "score_threshold": score,
                    "margin_threshold": margin,
                }
            )
    if len(output) != 81:
        raise RuntimeError("EXPANDED_SEARCH_STAGE1_NOT_81")
    return output


def structure_neighbors(value: int, values: Sequence[int]) -> list[int]:
    if value not in values:
        raise RuntimeError(f"EXPANDED_SEARCH_UNKNOWN_STRUCTURE_VALUE:{value}")
    index = values.index(value)
    return list(values[max(0, index - 1) : min(len(values), index + 2)])


def local_threshold_values(value: float, coarse: Sequence[float], *, multipliers: tuple[float, ...] | None = None) -> list[float]:
    if multipliers is not None:
        return sorted({max(0.0, value * factor) for factor in multipliers})
    index = min(range(len(coarse)), key=lambda i: abs(float(coarse[i]) - value))
    if not approx_equal(coarse[index], value):
        raise RuntimeError("EXPANDED_SEARCH_LOCAL_VALUE_NOT_ON_COARSE_GRID")
    left = coarse[index - 1] if index > 0 else None
    right = coarse[index + 1] if index + 1 < len(coarse) else None
    values = [value]
    if left is not None:
        values.append(0.5 * (left + value))
    if right is not None:
        values.append(0.5 * (value + right))
    return sorted({max(0.0, float(x)) for x in values})


def _row_spec(row: Mapping[str, Any]) -> dict[str, Any]:
    spec = row.get("spec")
    if not isinstance(spec, Mapping):
        raise RuntimeError("EXPANDED_SEARCH_ROW_SPEC_MISSING")
    return canonical_spec(spec)


def ranked(rows: Sequence[Mapping[str, Any]], *metric_path: str) -> list[dict[str, Any]]:
    def value(row: Mapping[str, Any]) -> float:
        current: Any = row
        for key in metric_path:
            current = current.get(key) if isinstance(current, Mapping) else None
        return float(current) if finite(current) else float("-inf")

    return sorted((dict(row) for row in rows), key=value, reverse=True)


def _stage3_specs(
    seeds: Sequence[Mapping[str, Any]],
    score_coarse: Sequence[float],
    margin_coarse: Sequence[float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for seed_index, row in enumerate(seeds):
        current = _row_spec(row)
        score_values = local_threshold_values(current["score_threshold"], score_coarse)
        margin_values = local_threshold_values(current["margin_threshold"], margin_coarse)
        for score in score_values:
            for margin in margin_values:
                candidates.append(
                    {
                        "trial_id": f"r{seed_index:02d}_s{len(candidates):03d}",
                        "max_gap": current["max_gap"],
                        "candidate_top_k": current["candidate_top_k"],
                        "score_threshold": score,
                        "margin_threshold": margin,
                    }
                )
        for gap in structure_neighbors(current["max_gap"], STRUCTURE_GAPS):
            if gap != current["max_gap"]:
                candidates.append(
                    {
                        "trial_id": f"r{seed_index:02d}_gap{gap}",
                        "max_gap": gap,
                        "candidate_top_k": current["candidate_top_k"],
                        "score_threshold": current["score_threshold"],
                        "margin_threshold": current["margin_threshold"],
                    }
                )
        for topk in structure_neighbors(current["candidate_top_k"], STRUCTURE_TOPKS):
            if topk != current["candidate_top_k"]:
                candidates.append(
                    {
                        "trial_id": f"r{seed_index:02d}_topk{topk}",
                        "max_gap": current["max_gap"],
                        "candidate_top_k": topk,
                        "score_threshold": current["score_threshold"],
                        "margin_threshold": current["margin_threshold"],
                    }
                )
    unique = dedupe_specs(candidates)
    for index, value in enumerate(unique):
        value["trial_id"] = f"stage3_{index:03d}"
    return unique


def _stage_full_refine_specs(
    champions: Sequence[Mapping[str, Any]],
    *,
    factors: tuple[float, ...],
    prefix: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for champion_index, row in enumerate(champions):
        current = _row_spec(row)
        for score_factor in factors:
            for margin_factor in factors:
                candidates.append(
                    {
                        "trial_id": f"{prefix}_{champion_index:02d}_{len(candidates):03d}",
                        "max_gap": current["max_gap"],
                        "candidate_top_k": current["candidate_top_k"],
                        "score_threshold": max(0.0, current["score_threshold"] * score_factor),
                        "margin_threshold": max(0.0, current["margin_threshold"] * margin_factor),
                    }
                )
        for gap in structure_neighbors(current["max_gap"], STRUCTURE_GAPS):
            if gap != current["max_gap"]:
                candidates.append(
                    {
                        "trial_id": f"{prefix}_{champion_index:02d}_gap{gap}",
                        "max_gap": gap,
                        "candidate_top_k": current["candidate_top_k"],
                        "score_threshold": current["score_threshold"],
                        "margin_threshold": current["margin_threshold"],
                    }
                )
        for topk in structure_neighbors(current["candidate_top_k"], STRUCTURE_TOPKS):
            if topk != current["candidate_top_k"]:
                candidates.append(
                    {
                        "trial_id": f"{prefix}_{champion_index:02d}_topk{topk}",
                        "max_gap": current["max_gap"],
                        "candidate_top_k": topk,
                        "score_threshold": current["score_threshold"],
                        "margin_threshold": current["margin_threshold"],
                    }
                )
    unique = dedupe_specs(candidates)
    for index, value in enumerate(unique):
        value["trial_id"] = f"{prefix}_{index:03d}"
    return unique


def build_stage_plan(base_plan: Mapping[str, Any], stage: str, specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    inherited_keys = (
        "protocol",
        "unbiased_test",
        "contract_mode",
        "contract_gate",
        "contract_gate_sha256",
        "threshold_source",
        "threshold_quantiles",
        "expected_inputs",
    )
    plan = {key: base_plan[key] for key in inherited_keys}
    plan["controller_stage"] = stage
    plan["search_fields"] = list(SEARCH_FIELDS)
    plan["trials"] = [canonical_spec(spec) for spec in specs]
    return plan


def write_immutable_plan(stage_dir: Path, plan: Mapping[str, Any]) -> tuple[Path, str]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / "search_plan.json"
    serialized = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    expected_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    hash_path = stage_dir / "search_plan.sha256"
    if path.exists():
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"EXPANDED_SEARCH_PLAN_IMMUTABLE_MISMATCH:{path}")
        if hash_path.is_file() and hash_path.read_text(encoding="utf-8").strip() != expected_sha:
            raise RuntimeError(f"EXPANDED_SEARCH_PLAN_HASH_FILE_MISMATCH:{hash_path}")
        if not hash_path.is_file():
            hash_path.write_text(expected_sha + "\n", encoding="utf-8")
        return path, expected_sha
    path.write_text(serialized, encoding="utf-8")
    hash_path.write_text(expected_sha + "\n", encoding="utf-8")
    return path, expected_sha


def coordinator_status(stage_dir: Path) -> dict[str, Any]:
    path = stage_dir / "coordinator_status.json"
    if not path.is_file():
        return {"status": "NOT_STARTED", "jobs": {}, "completed": 0, "failed": 0}
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {"status": "INVALID", "jobs": {}}
    return value if isinstance(value, dict) else {"status": "INVALID", "jobs": {}}


def _metrics_from_receipt(receipt: Mapping[str, Any]) -> bool:
    metrics = receipt.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    for split in ("base", "novel", "overall"):
        values = metrics.get(split)
        if not isinstance(values, Mapping) or any(not finite(values.get(name)) for name in METRIC_NAMES):
            return False
    return True


def _receipt_output_hashes_match(receipt: Mapping[str, Any]) -> bool:
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        return False
    for path_key, hash_key in (("prediction", "prediction_sha256"), ("summary", "summary_sha256")):
        path = Path(str(outputs.get(path_key, "")))
        expected = outputs.get(hash_key)
        if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
            return False
    return True


def valid_current_receipt(
    receipt: Mapping[str, Any],
    *,
    stage: str,
    plan_sha: str,
    annotation_sha: str,
    expected: Mapping[str, Any],
) -> bool:
    if receipt.get("status") != "COMPLETED" or receipt.get("stage") != stage:
        return False
    spec = receipt.get("spec")
    if not isinstance(spec, Mapping):
        return False
    try:
        canonical_spec(spec)
    except (KeyError, TypeError, ValueError):
        return False
    plan_binding = receipt.get("search_plan")
    if not isinstance(plan_binding, Mapping) or plan_binding.get("sha256") != plan_sha:
        return False
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    if inputs.get("annotation_sha256") != annotation_sha:
        return False
    for receipt_key, expected_key in (
        ("external_checkpoint_sha256", "external_checkpoint_sha256"),
        ("external_config_sha256", "external_config_sha256"),
        ("base_config_sha256", "base_config_sha256"),
        ("teta_init_sha256", "teta_init_sha256"),
    ):
        if expected.get(expected_key) and inputs.get(receipt_key) != expected.get(expected_key):
            return False
    source = receipt.get("external_source")
    if not isinstance(source, Mapping) or source.get("commit") != expected.get("external_cov_commit"):
        return False
    if not _metrics_from_receipt(receipt) or not _receipt_output_hashes_match(receipt):
        return False
    protocol = receipt.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("test_tuned_model_specific") is not True or protocol.get("unbiased_test") is not False:
        return False
    contract = receipt.get("runtime_contract")
    if not isinstance(contract, Mapping) or contract.get("status") != "PASS":
        return False
    return True


def read_rows(
    stage_dir: Path,
    *,
    stage: str,
    plan_sha: str,
    annotation_sha: str,
    expected: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for path in sorted(stage_dir.glob("**/receipt.json")):
        try:
            receipt = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append({"receipt": str(path), "reason": f"PARSE:{type(exc).__name__}"})
            continue
        if not isinstance(receipt, dict):
            invalid.append({"receipt": str(path), "reason": "NOT_OBJECT"})
            continue
        if not valid_current_receipt(
            receipt,
            stage=stage,
            plan_sha=plan_sha,
            annotation_sha=annotation_sha,
            expected=expected,
        ):
            if receipt.get("status") == "COMPLETED":
                invalid.append({"receipt": str(path), "reason": "PROVENANCE_OR_METRICS_INVALID"})
            continue
        metrics = receipt["metrics"]
        rows.append(
            {
                "stage": stage,
                "trial_id": receipt.get("trial_id", path.parent.name),
                "spec": canonical_spec(receipt["spec"]),
                "base": dict(metrics["base"]),
                "novel": dict(metrics["novel"]),
                "overall": dict(metrics["overall"]),
                "receipt": str(path),
                "receipt_sha256": sha256(path),
                "prediction_sha256": receipt.get("outputs", {}).get("prediction_sha256"),
                "status": "COMPLETED",
                "source": "new_stage",
            }
        )
    return rows, invalid


def valid_old_full_receipt(
    receipt: Mapping[str, Any],
    *,
    full_annotation_sha: str,
    expected: Mapping[str, Any],
    contract_gate_sha: str,
    overlay_sha: str,
    runtime_sha: str,
) -> bool:
    if receipt.get("status") != "COMPLETED" or receipt.get("stage") != "full":
        return False
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("annotation_sha256") != full_annotation_sha:
        return False
    for key in ("external_checkpoint_sha256", "external_config_sha256", "base_config_sha256", "teta_init_sha256"):
        if expected.get(key) and inputs.get(key) != expected.get(key):
            return False
    source = receipt.get("external_source")
    if not isinstance(source, Mapping) or source.get("commit") != expected.get("external_cov_commit"):
        return False
    protocol = receipt.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("test_tuned_model_specific") is not True or protocol.get("unbiased_test") is not False:
        return False
    plan_binding = receipt.get("search_plan")
    if not isinstance(plan_binding, Mapping):
        return False
    if plan_binding.get("contract_gate_sha256") != contract_gate_sha:
        return False
    plan_expected = plan_binding.get("expected_inputs")
    if isinstance(plan_expected, Mapping) and expected.get("reranker_checkpoint_sha256"):
        if plan_expected.get("reranker_checkpoint_sha256") != expected.get("reranker_checkpoint_sha256"):
            return False
    if inputs.get("overlay_sha256") != overlay_sha or inputs.get("runtime_sha256") != runtime_sha:
        return False
    contract = receipt.get("runtime_contract")
    if not isinstance(contract, Mapping) or contract.get("status") != "PASS":
        return False
    if not _metrics_from_receipt(receipt) or not _receipt_output_hashes_match(receipt):
        return False
    return True


def old_full_rows(
    roots: Sequence[Path],
    *,
    full_annotation_sha: str,
    expected: Mapping[str, Any],
    contract_gate_sha: str,
    overlay_sha: str,
    runtime_sha: str,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    rejected = 0
    for root in roots:
        for path in sorted(root.glob("**/receipt.json")):
            try:
                receipt = read_json(path)
            except (OSError, json.JSONDecodeError):
                rejected += 1
                continue
            if not isinstance(receipt, dict) or not valid_old_full_receipt(
                receipt,
                full_annotation_sha=full_annotation_sha,
                expected=expected,
                contract_gate_sha=contract_gate_sha,
                overlay_sha=overlay_sha,
                runtime_sha=runtime_sha,
            ):
                if isinstance(receipt, dict) and receipt.get("status") == "COMPLETED":
                    rejected += 1
                continue
            rows.append(
                {
                    "stage": "full",
                    "trial_id": receipt.get("trial_id", path.parent.name),
                    "spec": canonical_spec(receipt["spec"]),
                    "base": dict(receipt["metrics"]["base"]),
                    "novel": dict(receipt["metrics"]["novel"]),
                    "overall": dict(receipt["metrics"]["overall"]),
                    "receipt": str(path),
                    "receipt_sha256": sha256(path),
                    "prediction_sha256": receipt.get("outputs", {}).get("prediction_sha256"),
                    "status": "COMPLETED",
                    "source": "reused_old_full",
                }
            )
    unique: dict[tuple[int, int, float, float], dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(spec_key(row["spec"]), row)
    return list(unique.values()), rejected


def write_leaderboard(path_root: Path, stage: str, rows: Sequence[Mapping[str, Any]], invalid: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    leaderboard = path_root / "leaderboard"
    leaderboard.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "stage": stage,
        "generated_at": now_iso(),
        "completed_count": len(rows),
        "invalid_completed_count": len(invalid),
        "rows": list(rows),
        "invalid": list(invalid),
        "best_overall_teta": ranked(rows, "overall", "TETA")[:1],
        "best_novel_teta": ranked(rows, "novel", "TETA")[:1],
        "best_novel_assoc": ranked(rows, "novel", "AssocA")[:1],
    }
    atomic_json(leaderboard / f"{stage}.json", data)
    fields = [
        "stage", "trial_id", "source", "receipt", "receipt_sha256", "prediction_sha256",
        "max_gap", "candidate_top_k", "score_threshold", "margin_threshold",
        *[f"base_{name}" for name in METRIC_NAMES],
        *[f"novel_{name}" for name in METRIC_NAMES],
        *[f"overall_{name}" for name in METRIC_NAMES],
    ]
    with (leaderboard / f"{stage}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ranked(rows, "overall", "TETA"):
            spec = row["spec"]
            value = {
                "stage": row.get("stage"),
                "trial_id": row.get("trial_id"),
                "source": row.get("source"),
                "receipt": row.get("receipt"),
                "receipt_sha256": row.get("receipt_sha256"),
                "prediction_sha256": row.get("prediction_sha256"),
                **spec,
            }
            for split in ("base", "novel", "overall"):
                for name in METRIC_NAMES:
                    value[f"{split}_{name}"] = row[split][name]
            writer.writerow(value)
    lines = [
        f"# V10.4 expanded search leaderboard — {stage}",
        "",
        f"Generated: `{now_iso()}`",
        f"Completed official receipts: `{len(rows)}`; invalid completed receipts: `{len(invalid)}`.",
        "",
        "| rank | trial | max_gap | topK | score | margin | Base TETA | Base AssocA | Novel TETA | Novel AssocA | Overall TETA | receipt |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(ranked(rows, "overall", "TETA")[:20], 1):
        spec = row["spec"]
        lines.append(
            f"| {index} | `{row['trial_id']}` | {spec['max_gap']} | {spec['candidate_top_k']} | "
            f"{spec['score_threshold']:.6g} | {spec['margin_threshold']:.6g} | "
            f"{row['base']['TETA']:.6f} | {row['base']['AssocA']:.6f} | "
            f"{row['novel']['TETA']:.6f} | {row['novel']['AssocA']:.6f} | "
            f"{row['overall']['TETA']:.6f} | `{row['receipt']}` |"
        )
    (leaderboard / f"{stage}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


def _state_best(rows: Sequence[Mapping[str, Any]], metric_path: tuple[str, str]) -> dict[str, Any] | None:
    values = ranked(rows, *metric_path)
    return values[0] if values else None


def update_controller_state(
    state_path: Path,
    *,
    current_stage: str,
    stage_status: str,
    plans: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    failed_trials: Sequence[Mapping[str, Any]],
    next_action: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    value: dict[str, Any] = {
        "schema_version": 1,
        "current_stage": current_stage,
        "stage_status": stage_status,
        "plans": dict(plans),
        "plan_sha256": {key: item.get("sha256") for key, item in plans.items() if isinstance(item, Mapping)},
        "completed_trials": [row.get("trial_id") for row in rows],
        "failed_trials": list(failed_trials),
        "current_best_overall": _state_best(rows, ("overall", "TETA")),
        "current_best_novel": _state_best(rows, ("novel", "TETA")),
        "current_best_novel_assoc": _state_best(rows, ("novel", "AssocA")),
        "last_update": now_iso(),
        "next_action": next_action,
    }
    if extra:
        value.update(extra)
    atomic_json(state_path, value)


@dataclass
class ControllerArgs:
    repo: Path
    source: Path
    subset_annotation: Path
    full_annotation: Path
    img_prefix: Path
    external_config: Path
    external_checkpoint: Path
    base_config: Path
    base_search_plan: Path
    policy: Path
    output_root: Path
    teta_source_root: Path
    gpus: str
    max_workers: int
    poll_seconds: float
    evaluator_cores: int
    min_available_ram_gb: float
    launch_reserve_ram_gb: float
    coordinator_python: str
    stream_python: str
    evaluator_python: str
    old_full_roots: list[Path]
    max_micro_rounds: int


def _coordinator_command(
    args: ControllerArgs,
    *,
    annotation: Path,
    stage: str,
    stage_dir: Path,
    plan_path: Path,
) -> list[str]:
    return [
        args.coordinator_python,
        str(args.repo / "tools" / "v10_run_covtrack_search.py"),
        "--repo", str(args.repo),
        "--source", str(args.source),
        "--annotation", str(annotation),
        "--img-prefix", str(args.img_prefix),
        "--external-config", str(args.external_config),
        "--external-checkpoint", str(args.external_checkpoint),
        "--base-config", str(args.base_config),
        "--output-root", str(stage_dir),
        "--spec-file", str(plan_path),
        "--stage", stage,
        "--gpus", args.gpus,
        "--max-workers", str(args.max_workers),
        "--poll-seconds", str(args.poll_seconds),
        "--evaluator-cores", str(args.evaluator_cores),
        "--min-available-ram-gb", str(args.min_available_ram_gb),
        "--launch-reserve-ram-gb", str(args.launch_reserve_ram_gb),
        "--stream-python", args.stream_python,
        "--evaluator-python", args.evaluator_python,
        "--teta-source-root", str(args.teta_source_root),
        "--resume",
    ]


def run_coordinator(
    args: ControllerArgs,
    *,
    stage: str,
    stage_dir: Path,
    annotation: Path,
    plan_path: Path,
    state_path: Path,
    plans: dict[str, Any],
    all_rows: list[dict[str, Any]],
    failed_trials: list[dict[str, Any]],
) -> str:
    status = coordinator_status(stage_dir)
    if status.get("status") in TERMINAL_COORDINATOR_STATES:
        return str(status["status"])
    command = _coordinator_command(
        args,
        annotation=annotation,
        stage=stage,
        stage_dir=stage_dir,
        plan_path=plan_path,
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / "coordinator.log"
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "V10_EXPANDED_SEARCH_CONTROLLER": str(state_path),
        }
    )
    pythonpath = [str(args.repo), str(args.source), str(args.teta_source_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now_iso()}] launch {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(args.repo),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        plans[stage]["coordinator_pid"] = process.pid
        update_controller_state(
            state_path,
            current_stage=stage,
            stage_status="RUNNING",
            plans=plans,
            rows=all_rows,
            failed_trials=failed_trials,
            next_action=f"wait for {stage} coordinator; pid={process.pid}",
        )
        while process.poll() is None:
            current = coordinator_status(stage_dir)
            current_jobs = current.get("jobs", {})
            if isinstance(current_jobs, Mapping):
                stage_failed = [
                    {"trial_id": key, "status": item.get("state"), "returncode": item.get("returncode")}
                    for key, item in current_jobs.items()
                    if isinstance(item, Mapping) and item.get("state") == "FAILED"
                ]
            else:
                stage_failed = []
            update_controller_state(
                state_path,
                current_stage=stage,
                stage_status=str(current.get("status", "RUNNING")),
                plans=plans,
                rows=all_rows,
                failed_trials=[*failed_trials, *stage_failed],
                next_action=f"coordinator active; stage={stage}; completed={current.get('completed', 0)} failed={current.get('failed', 0)}",
            )
            time.sleep(max(5.0, args.poll_seconds * 3.0))
    returncode = int(process.returncode or 0)
    current = coordinator_status(stage_dir)
    if returncode != 0 and current.get("status") not in TERMINAL_COORDINATOR_STATES:
        current["status"] = "FAILED"
        atomic_json(stage_dir / "coordinator_status_controller_receipt.json", {
            "status": "FAILED",
            "returncode": returncode,
            "stage": stage,
            "timestamp": now_iso(),
        })
    return str(current.get("status", "FAILED"))


def _promoted_specs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for path, count in ((("overall", "TETA"), 10), (("novel", "TETA"), 5), (("novel", "AssocA"), 5)):
        selected.extend(_row_spec(row) for row in ranked(rows, *path)[:count])
    return dedupe_specs(selected)


def _stage2_specs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[tuple[float, float]] = []
    for row in ranked(rows, "overall", "TETA")[:3] + ranked(rows, "novel", "TETA")[:1]:
        spec = _row_spec(row)
        pair = (spec["score_threshold"], spec["margin_threshold"])
        if pair not in pairs:
            pairs.append(pair)
    specs = []
    for pair_index, (score, margin) in enumerate(pairs):
        for gap in STRUCTURE_GAPS:
            for topk in STRUCTURE_TOPKS:
                specs.append({
                    "trial_id": f"stage2_{pair_index:02d}_{gap}_{topk}",
                    "max_gap": gap,
                    "candidate_top_k": topk,
                    "score_threshold": score,
                    "margin_threshold": margin,
                })
    return dedupe_specs(specs)


def _stage3_seeds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for path, count in ((("overall", "TETA"), 4), (("novel", "TETA"), 1), (("novel", "AssocA"), 1)):
        selected.extend(ranked(rows, *path)[:count])
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, int, float, float]] = set()
    for row in selected:
        key = spec_key(_row_spec(row))
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result[:6]


def _old_reuse_specs(rows: Sequence[Mapping[str, Any]]) -> set[tuple[int, int, float, float]]:
    return {spec_key(row["spec"]) for row in rows if row.get("source") == "reused_old_full"}


def _aggregate_rows(stage_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rows in stage_rows.values():
        output.extend(dict(row) for row in rows)
    return output


def _write_control_snapshot(args: ControllerArgs, root: Path, provenance: Mapping[str, Any]) -> None:
    control = {
        "schema_version": 1,
        "status": "REFERENCE_ONLY",
        "description": "Expanded search does not rerun native COV control; existing control artifacts remain immutable.",
        "repo_head": provenance.get("repo_head_at_preflight"),
        "source_commit": provenance.get("source_commit"),
        "base_search_plan": str(args.base_search_plan),
        "base_search_plan_sha256": sha256(args.base_search_plan),
        "created_at": now_iso(),
    }
    atomic_json(root / "00_control" / "control_reference.json", control)


def final_report(
    args: ControllerArgs,
    *,
    root: Path,
    state: Mapping[str, Any],
    stage_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    provenance: Mapping[str, Any],
    reused_old_count: int,
    micro_rounds: int,
    stop_reason: str,
) -> Path:
    report = args.repo / "reports" / "tempotrack_v10" / "V10_4_EXPANDED_SEARCH_FINAL.md"
    rows = _aggregate_rows(stage_rows)
    best_overall = _state_best(rows, ("overall", "TETA"))
    best_novel = _state_best(rows, ("novel", "TETA"))
    best_novel_assoc = _state_best(rows, ("novel", "AssocA"))
    lines = [
        "# TempoTrack V10.4 expanded COV parameter search",
        "",
        f"Generated: `{now_iso()}`",
        "",
        "> `TEST_TUNED_MODEL_SPECIFIC` / `NOT_UNBIASED_TEST`. Test subset/full annotations were used for model-specific search selection; this is not an unbiased held-out Test claim.",
        "",
        "## Frozen contract and provenance",
        "",
        f"- Source repo: `{args.repo}`; HEAD `{git_value(args.repo, 'rev-parse', 'HEAD')}`; branch `{git_branch(args.repo)}`.",
        f"- External COV commit: `{provenance.get('source_commit')}`.",
        f"- Base search plan SHA256: `{provenance.get('base_plan_sha256')}`.",
        f"- Contract gate: `{provenance.get('contract_gate')}` SHA256 `{provenance.get('contract_gate_sha256')}`.",
        f"- Overlay SHA256: `{provenance.get('overlay_sha256')}`; runtime SHA256: `{provenance.get('runtime_sha256')}`.",
        f"- Subset annotation: `{args.subset_annotation}` SHA256 `{sha256(args.subset_annotation)}`.",
        f"- Full Test annotation: `{args.full_annotation}` SHA256 `{sha256(args.full_annotation)}`.",
        f"- TETA commit: `{provenance.get('teta_commit')}`.",
        "- Only the four legal fields `max_gap`, `candidate_top_k`, `score_threshold`, and `margin_threshold` were varied; detector, COV checkpoint, Q1 contract, context Top-64, reranker weights, and native COV settings were frozen.",
        "",
        "## Stop condition and counts",
        "",
        f"- Stop reason: `{stop_reason}`; micro rounds executed: `{micro_rounds}`.",
        f"- Total receipt rows considered: `{len(rows)}`; reused old exact Full-Test receipts: `{reused_old_count}`.",
        "",
        "## Champions",
        "",
        "| leaderboard | trial | source | config | Base TETA / LocA / AssocA / ClsA | Novel TETA / LocA / AssocA / ClsA | Overall TETA / LocA / AssocA / ClsA | prediction SHA | receipt |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, row in (("Best Overall TETA", best_overall), ("Best Novel TETA", best_novel), ("Best Novel AssocA", best_novel_assoc)):
        if not row:
            lines.append(f"| {label} | — | — | — | — | — | — | — | — |")
            continue
        spec = row["spec"]
        cfg = f"gap={spec['max_gap']},K={spec['candidate_top_k']},score={spec['score_threshold']:.8g},margin={spec['margin_threshold']:.8g}"
        fmt = lambda split: " / ".join(f"{row[split][name]:.6f}" for name in METRIC_NAMES)
        lines.append(
            f"| {label} | `{row['trial_id']}` | `{row.get('source')}` | `{cfg}` | {fmt('base')} | {fmt('novel')} | {fmt('overall')} | `{row.get('prediction_sha256')}` | `{row.get('receipt')}` |"
        )
    lines.extend([
        "",
        "## Stage artifacts",
        "",
        f"- Controller state: `{root / 'controller_state.json'}`.",
        "- Immutable plans and stage leaderboards are under `01_subset_threshold_grid` through `06_full_micro_refine`.",
        "- Old 10-way roots and their manual stop receipts were preserved; partial/failed observations are not promoted.",
        "",
        "## Full trial table",
        "",
        "| stage | trial | source | Base AssocA | Novel AssocA | Overall TETA | prediction SHA | receipt |",
        "|---|---|---|---:|---:|---:|---|---|",
    ])
    for row in ranked(rows, "overall", "TETA"):
        lines.append(
            f"| `{row.get('stage')}` | `{row.get('trial_id')}` | `{row.get('source')}` | {row['base']['AssocA']:.6f} | {row['novel']['AssocA']:.6f} | {row['overall']['TETA']:.6f} | `{row.get('prediction_sha256')}` | `{row.get('receipt')}` |"
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(root / "leaderboard" / "all_trials.json", {"generated_at": now_iso(), "rows": rows})
    with (root / "leaderboard" / "all_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "trial_id", "source", "max_gap", "candidate_top_k", "score_threshold", "margin_threshold", "base_TETA", "base_LocA", "base_AssocA", "base_ClsA", "novel_TETA", "novel_LocA", "novel_AssocA", "novel_ClsA", "overall_TETA", "overall_LocA", "overall_AssocA", "overall_ClsA", "prediction_sha256", "receipt"])
        for row in rows:
            spec = row["spec"]
            writer.writerow([
                row.get("stage"), row.get("trial_id"), row.get("source"), spec["max_gap"], spec["candidate_top_k"], spec["score_threshold"], spec["margin_threshold"],
                *[row["base"][name] for name in METRIC_NAMES], *[row["novel"][name] for name in METRIC_NAMES], *[row["overall"][name] for name in METRIC_NAMES], row.get("prediction_sha256"), row.get("receipt"),
            ])
    for filename, row in (("best_overall.json", best_overall), ("best_novel_teta.json", best_novel), ("best_novel_assoc.json", best_novel_assoc)):
        atomic_json(root / "leaderboard" / filename, {"row": row})
    atomic_json(root / "leaderboard" / "pareto_front.json", {"rows": rows})
    return report


def run(args: ControllerArgs) -> int:
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "controller_state.json"
    lock_path = root / "controller.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"EXPANDED_SEARCH_CONTROLLER_ALREADY_RUNNING:{lock_path}") from exc
        policy = load_policy(args.policy)
        base_plan = load_base_plan(args.base_search_plan)
        provenance = validate_contract_inputs(
            repo=args.repo,
            base_plan_path=args.base_search_plan,
            base_plan=base_plan,
            subset_annotation=args.subset_annotation,
            full_annotation=args.full_annotation,
            source=args.source,
            external_config=args.external_config,
            external_checkpoint=args.external_checkpoint,
            base_config=args.base_config,
            teta_source_root=args.teta_source_root,
        )
        _write_control_snapshot(args, root, provenance)
        plans: dict[str, Any] = {}
        stage_rows: dict[str, list[dict[str, Any]]] = {}
        failed_trials: list[dict[str, Any]] = []
        update_controller_state(
            state_path,
            current_stage="preflight",
            stage_status="PASS",
            plans=plans,
            rows=[],
            failed_trials=[],
            next_action="build Stage1 immutable plan",
            extra={"preflight": provenance, "policy_sha256": sha256(args.policy)},
        )

        # Stage 1: threshold-only 9x9 subset grid.
        stage = "01_subset_threshold_grid"
        stage_dir = root / stage
        plan_path, plan_sha = write_immutable_plan(stage_dir, build_stage_plan(base_plan, "subset", stage1_specs(base_plan)))
        plans[stage] = {"path": str(plan_path), "sha256": plan_sha, "stage": "subset", "trial_count": 81}
        stage_status = run_coordinator(
            args, stage="subset", stage_dir=stage_dir, annotation=args.subset_annotation,
            plan_path=plan_path, state_path=state_path, plans=plans, all_rows=[], failed_trials=failed_trials,
        )
        rows, invalid = read_rows(
            stage_dir, stage="subset", plan_sha=plan_sha, annotation_sha=sha256(args.subset_annotation), expected=base_plan["expected_inputs"],
        )
        stage_rows[stage] = rows
        write_leaderboard(root / stage, "stage1", rows, invalid)
        if not rows:
            update_controller_state(state_path, current_stage=stage, stage_status="BLOCKED", plans=plans, rows=[], failed_trials=invalid, next_action="no valid Stage1 official receipt; fail closed")
            return 2
        update_controller_state(state_path, current_stage=stage, stage_status=stage_status, plans=plans, rows=rows, failed_trials=invalid, next_action="build Stage2 structure grid")

        # Stage 2: up to four threshold pairs x 5 x 5 structures.
        stage = "02_subset_structure_grid"
        stage_dir = root / stage
        specs = _stage2_specs(rows)
        plan_path, plan_sha = write_immutable_plan(stage_dir, build_stage_plan(base_plan, "subset", specs))
        plans[stage] = {"path": str(plan_path), "sha256": plan_sha, "stage": "subset", "trial_count": len(specs)}
        stage_status = run_coordinator(args, stage="subset", stage_dir=stage_dir, annotation=args.subset_annotation, plan_path=plan_path, state_path=state_path, plans=plans, all_rows=rows, failed_trials=failed_trials)
        stage_rows[stage], invalid = read_rows(stage_dir, stage="subset", plan_sha=plan_sha, annotation_sha=sha256(args.subset_annotation), expected=base_plan["expected_inputs"])
        write_leaderboard(root / stage, "stage2", stage_rows[stage], invalid)
        if not stage_rows[stage]:
            update_controller_state(state_path, current_stage=stage, stage_status="BLOCKED", plans=plans, rows=_aggregate_rows(stage_rows), failed_trials=invalid, next_action="no valid Stage2 official receipt; fail closed")
            return 2
        subset_rows = _aggregate_rows({"stage1": stage_rows["01_subset_threshold_grid"], "stage2": stage_rows[stage]})
        update_controller_state(state_path, current_stage=stage, stage_status=stage_status, plans=plans, rows=subset_rows, failed_trials=invalid, next_action="build Stage3 local subset refine")

        # Stage 3: bounded local threshold/structure refine.
        stage = "03_subset_local_refine"
        stage_dir = root / stage
        seeds = _stage3_seeds(subset_rows)
        specs = _stage3_specs(
            seeds,
            threshold_grid(base_plan["threshold_quantiles"], "score"),
            threshold_grid(base_plan["threshold_quantiles"], "margin"),
        )
        plan_path, plan_sha = write_immutable_plan(stage_dir, build_stage_plan(base_plan, "subset", specs))
        plans[stage] = {"path": str(plan_path), "sha256": plan_sha, "stage": "subset", "trial_count": len(specs), "seed_count": len(seeds)}
        stage_status = run_coordinator(args, stage="subset", stage_dir=stage_dir, annotation=args.subset_annotation, plan_path=plan_path, state_path=state_path, plans=plans, all_rows=subset_rows, failed_trials=failed_trials)
        stage_rows[stage], invalid = read_rows(stage_dir, stage="subset", plan_sha=plan_sha, annotation_sha=sha256(args.subset_annotation), expected=base_plan["expected_inputs"])
        write_leaderboard(root / stage, "stage3", stage_rows[stage], invalid)
        subset_rows = _aggregate_rows({"stage1": stage_rows["01_subset_threshold_grid"], "stage2": stage_rows["02_subset_structure_grid"], "stage3": stage_rows[stage]})
        if not stage_rows[stage]:
            update_controller_state(state_path, current_stage=stage, stage_status="BLOCKED", plans=plans, rows=subset_rows, failed_trials=invalid, next_action="no valid Stage3 official receipt; fail closed")
            return 2
        update_controller_state(state_path, current_stage=stage, stage_status=stage_status, plans=plans, rows=subset_rows, failed_trials=invalid, next_action="build Full-Test promotion pool")

        # Stage 4: promote subset winners plus exact old Full-Test receipts.
        reused_rows, reused_rejected = old_full_rows(
            args.old_full_roots,
            full_annotation_sha=sha256(args.full_annotation),
            expected=base_plan["expected_inputs"],
            contract_gate_sha=str(base_plan["contract_gate_sha256"]),
            overlay_sha=provenance["overlay_sha256"],
            runtime_sha=provenance["runtime_sha256"],
        )
        stage = "04_full_promotion"
        stage_dir = root / stage
        candidate_specs = _promoted_specs(subset_rows)
        reused_keys = _old_reuse_specs(reused_rows)
        candidate_specs = [spec for spec in candidate_specs if spec_key(spec) not in reused_keys]
        for index, spec in enumerate(candidate_specs):
            spec["trial_id"] = f"full_promo_{index:03d}"
        plan_path, plan_sha = write_immutable_plan(stage_dir, build_stage_plan(base_plan, "full", candidate_specs))
        plans[stage] = {"path": str(plan_path), "sha256": plan_sha, "stage": "full", "trial_count": len(candidate_specs), "reused_old_count": len(reused_rows), "reused_rejected_count": reused_rejected}
        if candidate_specs:
            stage_status = run_coordinator(args, stage="full", stage_dir=stage_dir, annotation=args.full_annotation, plan_path=plan_path, state_path=state_path, plans=plans, all_rows=subset_rows, failed_trials=failed_trials)
            new_full_rows, invalid = read_rows(stage_dir, stage="full", plan_sha=plan_sha, annotation_sha=sha256(args.full_annotation), expected=base_plan["expected_inputs"])
        else:
            stage_status = "COMPLETED_REUSED"
            new_full_rows, invalid = [], []
        stage_rows[stage] = [*reused_rows, *new_full_rows]
        write_leaderboard(root / stage, "stage4", stage_rows[stage], invalid)
        if not stage_rows[stage]:
            update_controller_state(state_path, current_stage=stage, stage_status="BLOCKED", plans=plans, rows=subset_rows, failed_trials=invalid, next_action="no valid Full-Test receipt after promotion; fail closed")
            return 2
        full_rows = stage_rows[stage]
        update_controller_state(state_path, current_stage=stage, stage_status=stage_status, plans=plans, rows=full_rows, failed_trials=invalid, next_action="build Full-Test champion refine")

        # Stage 5: local hill climbing around the three independent champions.
        stage = "05_full_refine"
        stage_dir = root / stage
        champions: list[dict[str, Any]] = []
        for metric_path in (("overall", "TETA"), ("novel", "TETA"), ("novel", "AssocA")):
            row = _state_best(full_rows, metric_path)
            if row and spec_key(row["spec"]) not in {spec_key(item["spec"]) for item in champions}:
                champions.append(row)
        specs = _stage_full_refine_specs(champions, factors=(0.90, 1.0, 1.10), prefix="full_refine")
        existing_full_keys = {spec_key(row["spec"]) for row in full_rows}
        specs = [spec for spec in specs if spec_key(spec) not in existing_full_keys]
        plan_path, plan_sha = write_immutable_plan(stage_dir, build_stage_plan(base_plan, "full", specs))
        plans[stage] = {"path": str(plan_path), "sha256": plan_sha, "stage": "full", "trial_count": len(specs), "champion_count": len(champions)}
        if specs:
            stage_status = run_coordinator(args, stage="full", stage_dir=stage_dir, annotation=args.full_annotation, plan_path=plan_path, state_path=state_path, plans=plans, all_rows=full_rows, failed_trials=failed_trials)
            stage_rows[stage], invalid = read_rows(stage_dir, stage="full", plan_sha=plan_sha, annotation_sha=sha256(args.full_annotation), expected=base_plan["expected_inputs"])
        else:
            stage_status, invalid, stage_rows[stage] = "COMPLETED_NO_NEW_SPECS", [], []
        write_leaderboard(root / stage, "stage5", stage_rows[stage], invalid)
        full_rows = _aggregate_rows({"stage4": stage_rows["04_full_promotion"], "stage5": stage_rows[stage]})
        update_controller_state(state_path, current_stage=stage, stage_status=stage_status, plans=plans, rows=full_rows, failed_trials=invalid, next_action="build Full-Test micro refine")

        # Stage 6: at most two 5% micro-refine rounds, stopping on <0.02 gain.
        previous_best = _state_best(full_rows, ("overall", "TETA"))
        executed_micro = 0
        stop_reason = "NO_NEW_MICRO_SPECS"
        for round_index in range(1, args.max_micro_rounds + 1):
            if not previous_best:
                stop_reason = "NO_VALID_BEST_FOR_MICRO_REFINE"
                break
            stage = f"06_full_micro_refine/round{round_index:02d}"
            stage_dir = root / stage
            specs = _stage_full_refine_specs([previous_best], factors=(0.95, 1.0, 1.05), prefix=f"micro{round_index:02d}")
            existing_keys = {spec_key(row["spec"]) for row in full_rows}
            specs = [spec for spec in specs if spec_key(spec) not in existing_keys]
            plan_path, plan_sha = write_immutable_plan(stage_dir, build_stage_plan(base_plan, "full", specs))
            plans[stage] = {"path": str(plan_path), "sha256": plan_sha, "stage": "full", "trial_count": len(specs), "round": round_index}
            if not specs:
                stop_reason = "NO_NEW_MICRO_SPECS"
                break
            stage_status = run_coordinator(args, stage="full", stage_dir=stage_dir, annotation=args.full_annotation, plan_path=plan_path, state_path=state_path, plans=plans, all_rows=full_rows, failed_trials=failed_trials)
            stage_rows[stage], invalid = read_rows(stage_dir, stage="full", plan_sha=plan_sha, annotation_sha=sha256(args.full_annotation), expected=base_plan["expected_inputs"])
            write_leaderboard(root / stage, f"stage6_round{round_index:02d}", stage_rows[stage], invalid)
            old_best_value = float(previous_best["overall"]["TETA"])
            full_rows = _aggregate_rows({**{key: value for key, value in stage_rows.items() if key.startswith("04_") or key.startswith("05_") or key.startswith("06_")}, "current": stage_rows[stage]})
            new_best = _state_best(full_rows, ("overall", "TETA"))
            executed_micro = round_index
            improvement = float(new_best["overall"]["TETA"]) - old_best_value if new_best else float("-inf")
            update_controller_state(state_path, current_stage=stage, stage_status=stage_status, plans=plans, rows=full_rows, failed_trials=invalid, next_action=f"micro round {round_index} complete; improvement={improvement:.6f}")
            if new_best is None or improvement < 0.02:
                stop_reason = f"MICRO_IMPROVEMENT_LT_0_02:{improvement:.6f}"
                break
            previous_best = new_best
            stop_reason = "MICRO_ROUND_LIMIT_REACHED"

        final_rows = _aggregate_rows(stage_rows)
        report = final_report(args, root=root, state={}, stage_rows=stage_rows, provenance=provenance, reused_old_count=len(reused_rows), micro_rounds=executed_micro, stop_reason=stop_reason)
        update_controller_state(
            state_path,
            current_stage="complete",
            stage_status="COMPLETED",
            plans=plans,
            rows=final_rows,
            failed_trials=failed_trials,
            next_action=f"final report written: {report}",
            extra={"final_report": str(report), "stop_reason": stop_reason, "micro_rounds": executed_micro, "policy_sha256": sha256(args.policy)},
        )
        print(json.dumps({"status": "COMPLETED", "report": str(report), "controller_state": str(state_path), "rows": len(final_rows)}, ensure_ascii=False))
        return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repo", required=True, type=Path)
    run_parser.add_argument("--source", required=True, type=Path)
    run_parser.add_argument("--subset-annotation", required=True, type=Path)
    run_parser.add_argument("--full-annotation", required=True, type=Path)
    run_parser.add_argument("--img-prefix", required=True, type=Path)
    run_parser.add_argument("--external-config", required=True, type=Path)
    run_parser.add_argument("--external-checkpoint", required=True, type=Path)
    run_parser.add_argument("--base-config", required=True, type=Path)
    run_parser.add_argument("--base-search-plan", required=True, type=Path)
    run_parser.add_argument("--policy", required=True, type=Path)
    run_parser.add_argument("--output-root", required=True, type=Path)
    run_parser.add_argument("--teta-source-root", required=True, type=Path)
    run_parser.add_argument("--gpus", required=True)
    run_parser.add_argument("--max-workers", type=int, default=10)
    run_parser.add_argument("--poll-seconds", type=float, default=5.0)
    run_parser.add_argument("--evaluator-cores", type=int, default=2)
    run_parser.add_argument("--min-available-ram-gb", type=float, default=20.0)
    run_parser.add_argument("--launch-reserve-ram-gb", type=float, default=8.0)
    run_parser.add_argument("--coordinator-python", default="/home/lwr/anaconda3/envs/masaenv/bin/python")
    run_parser.add_argument("--stream-python", default="/home/lwr/anaconda3/envs/ovtr/bin/python")
    run_parser.add_argument("--evaluator-python", default="/home/lwr/anaconda3/envs/masaenv/bin/python")
    run_parser.add_argument("--old-full-root", action="append", dest="old_full_roots", type=Path, default=[])
    run_parser.add_argument("--max-micro-rounds", type=int, default=2)
    return root


def main() -> int:
    arguments = parser().parse_args()
    if arguments.action != "run":
        raise RuntimeError(f"unsupported action: {arguments.action}")
    if not arguments.old_full_roots:
        arguments.old_full_roots = [
            Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_test_full_20260914_10way"),
            Path("/data2/usr_for_deadline/tempotrack_v10_unified/search/covtrack_test_full_20260914_10way_supplement"),
        ]
    args = ControllerArgs(
        repo=arguments.repo.resolve(),
        source=arguments.source.resolve(),
        subset_annotation=arguments.subset_annotation.resolve(),
        full_annotation=arguments.full_annotation.resolve(),
        img_prefix=arguments.img_prefix.resolve(),
        external_config=arguments.external_config.resolve(),
        external_checkpoint=arguments.external_checkpoint.resolve(),
        base_config=arguments.base_config.resolve(),
        base_search_plan=arguments.base_search_plan.resolve(),
        policy=arguments.policy.resolve(),
        output_root=arguments.output_root.resolve(),
        teta_source_root=arguments.teta_source_root.resolve(),
        gpus=arguments.gpus,
        max_workers=max(1, int(arguments.max_workers)),
        poll_seconds=max(1.0, float(arguments.poll_seconds)),
        evaluator_cores=max(1, int(arguments.evaluator_cores)),
        min_available_ram_gb=float(arguments.min_available_ram_gb),
        launch_reserve_ram_gb=float(arguments.launch_reserve_ram_gb),
        coordinator_python=str(arguments.coordinator_python),
        stream_python=str(arguments.stream_python),
        evaluator_python=str(arguments.evaluator_python),
        old_full_roots=[path.resolve() for path in arguments.old_full_roots],
        max_micro_rounds=min(2, max(0, int(arguments.max_micro_rounds))),
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
