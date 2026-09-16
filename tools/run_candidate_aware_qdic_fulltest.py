#!/usr/bin/env python3
"""Run one gated, auditable candidate-aware QDIC Full-Test evaluation.

The structural six-trial search is an explicit prerequisite.  This runner is
intentionally single-candidate so an external scheduler can enforce the
project-wide maximum of two concurrent Full-Test candidates without sharing
mutable state between architecture or loss-search runs.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_local_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load repository module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_local_module(
    REPO / "tools" / "v11_search_covtrack_full_test.py",
    "tempotrack_v11_candidate_fulltest_base",
)

from tempotrack_v10.candidate_aware_qdic_loader import (  # noqa: E402
    CANDIDATE_AWARE_STATUS,
)


DEFAULT_ANNOTATION = base.DEFAULT_ANNOTATION
DEFAULT_SHARD_MANIFEST = base.DEFAULT_SHARD_MANIFEST
DEFAULT_COV_SOURCE = base.DEFAULT_COV_SOURCE
DEFAULT_COV_CONFIG = base.DEFAULT_COV_CONFIG
DEFAULT_COV_CHECKPOINT = base.DEFAULT_COV_CHECKPOINT
DEFAULT_IMG_PREFIX = base.DEFAULT_IMG_PREFIX
DEFAULT_TETA_SOURCE_ROOT = base.DEFAULT_TETA_SOURCE_ROOT
DEFAULT_RUNTIME_RECEIPT = base.DEFAULT_V10_RUNTIME_RECEIPT
DEFAULT_BASE_CONFIG = REPO / "configs/research/v11/covtrack_qdic_test.yaml"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_qdic_preflight(
    *,
    python: str,
    repo: Path,
    checkpoint: Path,
    scalabel_root: Path,
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    code = r'''
import json
import sys
from tempotrack_v10.candidate_aware_qdic_loader import load_candidate_aware_checkpoint
artifact = load_candidate_aware_checkpoint(sys.argv[1], device="cpu")
print(json.dumps(artifact.provenance, sort_keys=True))
'''
    result = base._run_import_preflight(
        python=python,
        repo=repo,
        teta_root=Path("/nonexistent"),
        scalabel_root=scalabel_root,
        runtime_environment=runtime_environment,
        code=code,
        arguments=[str(checkpoint)],
    )
    if result.get("status") != CANDIDATE_AWARE_STATUS:
        raise RuntimeError("candidate-aware checkpoint provenance status is invalid")
    if result.get("checkpoint_sha256") != base._sha256(checkpoint):
        raise RuntimeError("candidate-aware checkpoint preflight hash mismatch")
    if result.get("training_protocol") != "QDIC_V11_BASE_ONLY_CANDIDATE_AWARE_TRAINING":
        raise RuntimeError("candidate-aware checkpoint is not the Base-only training protocol")
    if result.get("base_only_supervision") is not True:
        raise RuntimeError("candidate-aware checkpoint is not Base-only")
    if result.get("novel_gt_used") is not False or result.get("test_gt_used_for_optimizer") is not False:
        raise RuntimeError("candidate-aware checkpoint has forbidden supervision provenance")
    return result


def _canonical_architecture(value: str) -> str:
    name = str(value).strip().upper()
    if name == "A0":
        return "A0"
    if name in {"A0-D", "A0_D", "A0DIST", "A0-DISTLOSS"}:
        return "A0-D"
    if name in {"A1", "A1-DS-QDIC", "DS-QDIC"}:
        return "A1-DS-QDIC"
    if name in {"A2", "A2-DGSA-QDIC", "DGSA-QDIC"}:
        return "A2-DGSA-QDIC"
    raise ValueError(f"unsupported candidate-aware architecture: {value!r}")


def _validate_structure_gate(root: Path) -> dict[str, Any]:
    """Require all six structural trials and the behavior sanity artifact."""
    state_path = root / "search_state.json"
    sanity_path = root / "structural_behavior_sanity.json"
    if not state_path.is_file() or not sanity_path.is_file():
        raise RuntimeError("BLOCKED_STRUCTURE_GATE_ARTIFACT_MISSING")
    state = _read_json(state_path)
    sanity = _read_json(sanity_path)
    expected = {
        "ST1_K16_G360",
        "ST2_K32_G360",
        "ST3_K64_G360",
        "ST4_K32_G180",
        "ST5_K32_G720",
        "ST6_K64_G720",
    }
    trials = {
        str(item.get("trial_id")): item
        for item in state.get("trials", [])
        if isinstance(item, Mapping)
    }
    missing = sorted(expected - set(trials))
    incomplete = sorted(
        trial_id
        for trial_id in expected & set(trials)
        if trials[trial_id].get("status") != "COMPLETED"
        or not trials[trial_id].get("full_test_end_unix")
    )
    if missing or incomplete:
        raise RuntimeError(
            "BLOCKED_STRUCTURE_GATE_INCOMPLETE: "
            f"missing={missing}, incomplete={incomplete}"
        )
    if state.get("structural_behavior_sanity_status") != "PASS":
        raise RuntimeError("BLOCKED_STRUCTURE_GATE_BEHAVIOR_SANITY_STATUS")
    if sanity.get("status") != "PASS":
        raise RuntimeError("BLOCKED_STRUCTURE_GATE_BEHAVIOR_SANITY")
    if sanity.get("preflight_sha256") != base._sha256(root / "preflight.json"):
        raise RuntimeError("BLOCKED_STRUCTURE_GATE_PREFLIGHT_HASH")
    return {
        "root": str(root),
        "state_sha256": base._sha256(state_path),
        "sanity_sha256": base._sha256(sanity_path),
        "state_status": state.get("status"),
        "structural_behavior_sanity_status": state.get("structural_behavior_sanity_status"),
        "trial_ids": sorted(expected),
    }


def _preflight(args: argparse.Namespace, root: Path, repo: Path) -> dict[str, Any]:
    structure_gate = _validate_structure_gate(base._path(args.structure_search_root))
    annotation = base._path(args.annotation)
    shard_manifest = base._path(args.shard_manifest)
    cov_source = base._path(args.cov_source)
    cov_config = base._path(args.external_config)
    cov_checkpoint = base._path(args.external_checkpoint)
    base_config = base._path(args.base_config)
    img_prefix = base._path(args.img_prefix)
    teta_root = base._path(args.teta_source_root)
    checkpoint = base._path(args.qdic_checkpoint)

    full = base._annotation_summary(annotation)
    if full["sha256"] != base.EXPECTED_ANNOTATION_SHA256:
        raise RuntimeError("FULL_TEST_ANNOTATION hash is not the audited source")
    shards = base._validate_shards(
        full_annotation=annotation,
        shard_manifest=shard_manifest,
        full=full,
    )
    if not cov_source.is_dir() or base._git_value(cov_source, "rev-parse", "HEAD") != base.EXPECTED_COV_COMMIT:
        raise RuntimeError("COV_PROVENANCE commit mismatch")
    if base._git_source_status(cov_source) != "":
        raise RuntimeError("COV_PROVENANCE tracked source is dirty")
    if not cov_config.is_file() or base._sha256(cov_config) != base.EXPECTED_COV_CONFIG_SHA256:
        raise RuntimeError("COV_PROVENANCE external config hash mismatch")
    if not cov_checkpoint.is_file() or base._sha256(cov_checkpoint) != base.EXPECTED_COV_CHECKPOINT_SHA256:
        raise RuntimeError("COV_PROVENANCE external checkpoint hash mismatch")
    if not img_prefix.is_dir() or not base_config.is_file():
        raise FileNotFoundError("Full-Test input path is missing")
    base._validate_base_config(base_config)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"candidate-aware checkpoint is missing: {checkpoint}")
    if base._git_status(repo, include_untracked=True) != "":
        raise RuntimeError("candidate Full-Test requires a clean repository")

    runtime_environment = base._audit_v10_runtime_environment(args=args, repo=repo)
    scalabel_root = base._path(str(runtime_environment["scalabel_root"]))
    qdic_provenance = _candidate_qdic_preflight(
        python=args.stream_python,
        repo=repo,
        checkpoint=checkpoint,
        scalabel_root=scalabel_root,
        runtime_environment=runtime_environment,
    )
    expected_architecture = _canonical_architecture(args.architecture)
    if qdic_provenance.get("architecture_name") != expected_architecture:
        raise RuntimeError(
            "candidate architecture mismatch: "
            f"checkpoint={qdic_provenance.get('architecture_name')!r}, "
            f"requested={expected_architecture!r}"
        )
    for key, expected in (
        ("lambda_dist", float(args.lambda_dist)),
        ("lambda_hard", float(args.lambda_hard)),
    ):
        if key in qdic_provenance and float(qdic_provenance[key]) != expected:
            raise RuntimeError(
                f"candidate checkpoint {key} mismatch: "
                f"{qdic_provenance[key]!r} != {expected!r}"
            )
    teta_provenance = base._teta_preflight(
        args.evaluator_python,
        repo,
        teta_root,
        scalabel_root=scalabel_root,
        runtime_environment=runtime_environment,
    )
    preflight = {
        "schema_version": 1,
        "status": "PASS",
        "artifact": "candidate_aware_qdic_full_test_preflight",
        "labels": ["TEST_TUNED_MODEL_SPECIFIC", "NOT_UNBIASED_TEST"],
        "repository": {
            "path": str(repo),
            "branch": base._git_branch(repo),
            "head": base._git_value(repo, "rev-parse", "HEAD"),
            "status": "CLEAN",
        },
        "structure_gate": structure_gate,
        "full_test_annotation": full,
        "shards": {
            "status": "PASS",
            "count": len(shards),
            "items": shards,
            "manifest": str(shard_manifest),
            "manifest_sha256": base._sha256(shard_manifest),
        },
        "qdic": {
            "status": "PASS",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": base._sha256(checkpoint),
            "loader_provenance": qdic_provenance,
        },
        "cov": {
            "status": "PASS",
            "source": str(cov_source),
            "commit": base._git_value(cov_source, "rev-parse", "HEAD"),
            "config": str(cov_config),
            "config_sha256": base._sha256(cov_config),
            "checkpoint": str(cov_checkpoint),
            "checkpoint_sha256": base._sha256(cov_checkpoint),
            "tracked_source_status": "BYTECODE_ONLY_OR_CLEAN",
        },
        "teta": {
            "status": "PASS",
            "source_root": str(teta_root),
            "provenance": teta_provenance,
        },
        "base_config": {
            "path": str(base_config),
            "sha256": base._sha256(base_config),
        },
        "runtime_environment": runtime_environment,
        "runtime": {
            "fixed_fields": {
                "alpha_fast": 0.70,
                "alpha_slow": 0.15,
                "min_gap": 0,
                "memory_capacity": 64,
                "qdic_recent_k": 8,
                "qdic_context_top_k": 64,
                "qdic_feature_decision_candidate_top_k": 8,
                "qdic_feature_max_gap": 360,
                "qdic_weight": 1.0,
                "reranker_weight": 0.0,
            },
            "search_fields": ["architecture", "lambda_dist", "lambda_hard"],
            "frontend_cache_used": False,
        },
        "created_at_unix": time.time(),
    }
    base._validate_runtime_environment(args=args, repo=repo, preflight=preflight)
    _write_json(root / "preflight.json", preflight)
    return preflight


def _plan(args: argparse.Namespace, root: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    plan = {
        "schema_version": 1,
        "artifact": "candidate_aware_qdic_full_test_plan",
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "search_fields": ["architecture", "lambda_dist", "lambda_hard"],
        "fixed_runtime": dict(preflight["runtime"]["fixed_fields"]),
        "candidate": {
            "trial_id": str(args.trial_id),
            "architecture": str(args.architecture),
            "qdic_checkpoint": str(base._path(args.qdic_checkpoint)),
            "qdic_checkpoint_sha256": base._sha256(base._path(args.qdic_checkpoint)),
            "score_threshold": float(args.score_threshold),
            "margin_threshold": float(args.margin_threshold),
            "candidate_top_k": int(args.candidate_top_k),
            "max_gap": int(args.max_gap),
            "lambda_dist": float(args.lambda_dist),
            "lambda_hard": float(args.lambda_hard),
        },
        "structure_gate": preflight["structure_gate"],
        "created_at_unix": time.time(),
    }
    _write_json(root / "search_plan.json", plan)
    return plan


def _make_args_namespace(args: argparse.Namespace, root: Path) -> argparse.Namespace:
    """Add the fields expected by the shared Full-Test worker controller."""
    args.root = str(root)
    args.repo = str(REPO)
    args.qdic_fast_root = str(root)
    args.qdic_checkpoint = str(base._path(args.qdic_checkpoint))
    args.worker_python = args.worker_python or sys.executable
    args.report_root = str(base._path(args.report_root))
    return args


def _run_diagnostics(
    *, args: argparse.Namespace, preflight: Mapping[str, Any], trial: dict[str, Any], root: Path
) -> dict[str, Any]:
    candidate_root = base._candidate_root(root, str(trial["trial_id"]))
    output = candidate_root / "diagnostic_statistics.json"
    command = [
        str(args.worker_python),
        str(REPO / "tools" / "v11_structure_diagnostics.py"),
        "--annotation",
        str(preflight["full_test_annotation"]["path"]),
        "--candidate-root",
        str(candidate_root),
        "--output",
        str(output),
    ]
    log_path = candidate_root / "diagnostics.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=str(REPO),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(f"posthoc structural diagnostics failed; see {log_path}")
    diagnostics = _read_json(output)
    if diagnostics.get("status") not in {"PASS", "UNAVAILABLE_NO_EVENT_ROWS"}:
        raise RuntimeError(f"candidate diagnostics did not pass: {diagnostics.get('status')}")
    trial["diagnostics_command"] = command
    trial["diagnostics_log"] = str(log_path)
    trial["diagnostics_path"] = str(output)
    trial["diagnostics"] = diagnostics
    return diagnostics


def _write_result(
    *,
    args: argparse.Namespace,
    root: Path,
    preflight: Mapping[str, Any],
    plan: Mapping[str, Any],
    trial: Mapping[str, Any],
) -> None:
    report_root = base._path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "artifact": "candidate_aware_qdic_full_test_result",
        "status": "PASS" if trial.get("status") == "COMPLETED" else trial.get("status"),
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "trial_id": trial.get("trial_id"),
        "architecture": args.architecture,
        "lambda_dist": float(args.lambda_dist),
        "lambda_hard": float(args.lambda_hard),
        "checkpoint": str(base._path(args.qdic_checkpoint)),
        "checkpoint_sha256": base._sha256(base._path(args.qdic_checkpoint)),
        "preflight": str(root / "preflight.json"),
        "preflight_sha256": base._sha256(root / "preflight.json"),
        "plan": str(root / "search_plan.json"),
        "plan_sha256": base._sha256(root / "search_plan.json"),
        "structure_gate": preflight["structure_gate"],
        "metrics": trial.get("metrics"),
        "score_distribution": trial.get("score_distribution"),
        "diagnostics": trial.get("diagnostics"),
        "candidate_receipt": str(base._candidate_root(root, str(trial["trial_id"])) / "receipt.json"),
        "candidate_receipt_sha256": base._sha256(
            base._candidate_root(root, str(trial["trial_id"])) / "receipt.json"
        ),
        "repository": preflight["repository"],
        "runtime_environment": preflight["runtime_environment"],
        "created_at_unix": time.time(),
    }
    _write_json(root / "result.json", result)
    _write_json(report_root / f"{args.trial_id}.json", result)


def run(args: argparse.Namespace) -> int:
    root = base._path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "result.json").is_file():
        existing = _read_json(root / "result.json")
        if existing.get("status") == "PASS":
            print(json.dumps(existing, ensure_ascii=False, indent=2), flush=True)
            return 0
        raise RuntimeError(f"refusing to reuse a non-PASS candidate root: {root}")
    args = _make_args_namespace(args, root)
    preflight = _preflight(args, root, REPO)
    plan = _plan(args, root, preflight)
    trial: dict[str, Any] = {
        "trial_id": str(args.trial_id),
        "wave": "ARCHITECTURE",
        "score_threshold": float(args.score_threshold),
        "margin_threshold": float(args.margin_threshold),
        "candidate_top_k": int(args.candidate_top_k),
        "max_gap": int(args.max_gap),
        "master_port": int(args.master_port),
        "collect_event_diagnostics": True,
        "architecture": str(args.architecture),
        "lambda_dist": float(args.lambda_dist),
        "lambda_hard": float(args.lambda_hard),
        "status": "PENDING",
    }
    state: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "candidate_aware_qdic_full_test_state",
        "status": "RUNNING",
        "root": str(root),
        "selected_gpus": base._parse_gpu_ids(args.gpus),
        "resource_policy": base._resource_policy(args),
        "candidate_parallelism": 1,
        "preflight_sha256": base._sha256(root / "preflight.json"),
        "search_plan_sha256": base._sha256(root / "search_plan.json"),
        "trials": [trial],
        "events": [],
        "created_at_unix": time.time(),
    }
    state["resource_gate"] = base._resource_gate(
        args, state["selected_gpus"], purpose=str(args.trial_id)
    )
    _write_json(root / "search_state.json", state)
    try:
        if not base._run_candidate(
            args=args,
            state=state,
            preflight=preflight,
            trial=trial,
            root=root,
            persist_state=True,
        ):
            raise RuntimeError("candidate Full-Test shards/merge failed")
        jobs: dict[str, Any] = {}
        process = base._start_evaluation(
            args=args, state=state, preflight=preflight, trial=trial
        )
        if process is not None:
            jobs[str(trial["trial_id"])] = process
        base._wait_for_evaluations(
            args=args, state=state, preflight=preflight, jobs=jobs
        )
        if trial.get("status") != "COMPLETED":
            raise RuntimeError("candidate official TETA evaluation did not complete")
        _run_diagnostics(args=args, preflight=preflight, trial=trial, root=root)
        trial["score_distribution"] = base._score_distribution(trial)
        trial["architecture"] = str(args.architecture)
        trial["lambda_dist"] = float(args.lambda_dist)
        trial["lambda_hard"] = float(args.lambda_hard)
        trial["status"] = "COMPLETED"
        candidate_receipt = dict(_read_json(base._candidate_root(root, str(trial["trial_id"])) / "receipt.json"))
        candidate_receipt.update(
            {
                "architecture": str(args.architecture),
                "lambda_dist": float(args.lambda_dist),
                "lambda_hard": float(args.lambda_hard),
                "diagnostics_path": trial["diagnostics_path"],
                "diagnostics": trial["diagnostics"],
                "score_distribution": trial["score_distribution"],
                "candidate_checkpoint": str(base._path(args.qdic_checkpoint)),
                "candidate_checkpoint_sha256": base._sha256(base._path(args.qdic_checkpoint)),
                "candidate_aware_status": CANDIDATE_AWARE_STATUS,
            }
        )
        _write_json(
            base._candidate_root(root, str(trial["trial_id"])) / "receipt.json",
            candidate_receipt,
        )
        state["status"] = "COMPLETED"
        state["updated_at_unix"] = time.time()
        state["trials"] = [trial]
        _write_json(root / "search_state.json", state)
        _write_result(
            args=args,
            root=root,
            preflight=preflight,
            plan=plan,
            trial=trial,
        )
        print(
            f"CANDIDATE_AWARE_FULL_TEST_COMPLETED trial={args.trial_id} "
            f"architecture={args.architecture} "
            f"overall_teta={trial.get('metrics', {}).get('overall', {}).get('TETA')}",
            flush=True,
        )
        return 0
    except Exception as exc:
        state["status"] = "FAILED"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["traceback"] = traceback.format_exc()
        state["updated_at_unix"] = time.time()
        _write_json(root / "search_state.json", state)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-search-root", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--qdic-checkpoint", required=True)
    parser.add_argument("--lambda-dist", type=float)
    parser.add_argument("--lambda-hard", type=float, default=0.2)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--margin-threshold", type=float, default=0.37210235595703123)
    parser.add_argument("--candidate-top-k", type=int, default=8)
    parser.add_argument("--max-gap", type=int, default=360)
    parser.add_argument("--master-port", type=int, default=29600)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--allow-gpu-overlap", action="store_true")
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument("--annotation", default=str(DEFAULT_ANNOTATION))
    parser.add_argument("--shard-manifest", default=str(DEFAULT_SHARD_MANIFEST))
    parser.add_argument("--cov-source", default=str(DEFAULT_COV_SOURCE))
    parser.add_argument("--external-config", default=str(DEFAULT_COV_CONFIG))
    parser.add_argument("--external-checkpoint", default=str(DEFAULT_COV_CHECKPOINT))
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--img-prefix", default=str(DEFAULT_IMG_PREFIX))
    parser.add_argument("--teta-source-root", default=str(DEFAULT_TETA_SOURCE_ROOT))
    parser.add_argument("--v10-runtime-receipt", default=str(DEFAULT_RUNTIME_RECEIPT))
    parser.add_argument("--v10-runtime-env-capture")
    parser.add_argument("--stream-python", default=base.DEFAULT_STREAM_PYTHON)
    parser.add_argument("--evaluator-python", default=base.DEFAULT_EVALUATOR_PYTHON)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--evaluator-cores", type=int, default=32)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.lambda_dist is None:
        args.lambda_dist = 0.0 if _canonical_architecture(args.architecture) == "A0" else 1.0
    if args.allow_gpu_overlap and args.wait_for_gpus:
        raise ValueError("--allow-gpu-overlap and --wait-for-gpus are mutually exclusive")
    if int(args.candidate_top_k) < 1 or int(args.candidate_top_k) > 64:
        raise ValueError("candidate_top_k must be in [1,64]")
    if int(args.max_gap) < 0:
        raise ValueError("max_gap must be non-negative")
    if float(args.lambda_dist) < 0.0 or float(args.lambda_hard) < 0.0:
        raise ValueError("loss weights must be non-negative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
