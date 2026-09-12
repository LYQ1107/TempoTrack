#!/usr/bin/env python3
"""Run one auditable COVTrack V10 Tempo trial and official TETA evaluation.

Each invocation owns a complete output directory.  It launches the pinned
COVTrack runner from the external checkout, writes the real stream and
diagnostics, evaluates that exact prediction, and records all source/config/
prediction/evaluator hashes in ``receipt.json``.  The script is intentionally
one-trial-at-a-time; ``v10_run_covtrack_search.py`` supplies independent GPU
workers around it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


SEARCH_FIELDS = (
    "max_gap",
    "candidate_top_k",
    "score_threshold",
    "margin_threshold",
)


@dataclass(frozen=True)
class SearchPlan:
    """Search trials plus the contract that makes them selectable."""

    path: str | None
    sha256: str | None
    protocol: str
    unbiased_test: bool
    contract_mode: str
    contract_gate: str | None
    contract_gate_sha256: str | None
    threshold_source: str | None
    threshold_quantiles: dict[str, float]
    expected_inputs: dict[str, Any]
    trials: tuple[dict[str, Any], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_value(path: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve_teta_source_root(value: str | Path | None) -> tuple[Path, Path] | None:
    """Resolve the import parent and real ``teta/__init__.py``.

    The audited checkout is a repository at ``.../tet`` whose importable
    package is nested at ``.../tet/teta/teta``.  The CLI therefore receives
    the directory that contains the importable ``teta`` package, not the git
    repository root.  Keeping this check in the runner prevents a namespace
    package or an unrelated installed ``teta`` from silently satisfying the
    dependency.
    """

    if value is None or str(value).strip() == "":
        return None
    root = Path(value).expanduser().resolve()
    init_file = root / "teta" / "__init__.py"
    if not init_file.is_file():
        raise RuntimeError(
            "TETA_SOURCE_ROOT_INVALID: expected importable package at "
            f"{init_file}"
        )
    return root, init_file


def _teta_dependency(value: str | Path | None) -> dict[str, Any] | None:
    resolved = _resolve_teta_source_root(value)
    if resolved is None:
        return None
    root, init_file = resolved
    git_root_raw = _git_value(root, "rev-parse", "--show-toplevel")
    git_root = None if not git_root_raw else Path(git_root_raw).resolve()
    git_commit = _git_value(root, "rev-parse", "HEAD")
    git_status = _git_value(root, "status", "--porcelain")
    tracked_git_status = (
        None
        if git_root is None
        else _git_value(
            git_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    )
    return {
        "source_root": str(root),
        "init_file": str(init_file),
        "init_sha256": _sha256(init_file),
        "git_root": None if git_root is None else str(git_root),
        "git_commit": git_commit,
        "git_status": git_status,
        "tracked_git_status": tracked_git_status,
        "tracked_source_clean": tracked_git_status == "",
    }


def _run_teta_import_preflight(
    *,
    stream_python: str | Path,
    source: Path,
    teta_source_root: str | Path,
) -> dict[str, Any]:
    """Use the exact stream interpreter to import TETA and COV datasets."""

    resolved = _resolve_teta_source_root(teta_source_root)
    if resolved is None:  # pragma: no cover - caller enforces this
        raise RuntimeError("TETA_SOURCE_ROOT_REQUIRED")
    root, teta_init_file = resolved
    expected_teta_init = teta_init_file.resolve()
    expected_cov_dataset_init = (source / "ovtrack" / "datasets" / "__init__.py").resolve()
    if not expected_cov_dataset_init.is_file():
        raise RuntimeError(f"COV_DATASET_INIT_MISSING: {expected_cov_dataset_init}")
    preflight_code = r'''
from pathlib import Path
import json
import sys

import teta
import ovtrack.datasets

actual_teta = Path(teta.__file__).resolve()
actual_cov = Path(ovtrack.datasets.__file__).resolve()
expected_teta = Path(sys.argv[1]).resolve()
expected_cov = Path(sys.argv[2]).resolve()
if actual_teta != expected_teta:
    raise RuntimeError(
        "TETA_IMPORT_PATH_MISMATCH: "
        f"{actual_teta} != {expected_teta}"
    )
if actual_cov != expected_cov:
    raise RuntimeError(
        "COV_IMPORT_PATH_MISMATCH: "
        f"{actual_cov} != {expected_cov}"
    )
print(json.dumps({
    "teta_file": str(actual_teta),
    "cov_dataset_file": str(actual_cov),
}, sort_keys=True))
'''
    command = [
        str(stream_python),
        "-c",
        preflight_code,
        str(expected_teta_init),
        str(expected_cov_dataset_init),
    ]
    env = os.environ.copy()
    path_entries = [str(root), str(source)]
    old = env.get("PYTHONPATH")
    if old:
        path_entries.append(old)
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    result = subprocess.run(
        command,
        cwd=str(source),
        env=env,
        capture_output=True,
        text=True,
    )
    output = {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": int(result.returncode),
        "command": command,
        "cwd": str(source),
        "expected_teta_init": str(expected_teta_init),
        "expected_cov_dataset_init": str(expected_cov_dataset_init),
        "actual_imports": None,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }
    if result.returncode != 0:
        raise RuntimeError(
            "TETA_IMPORT_PREFLIGHT_FAILED: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        output["actual_imports"] = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("TETA_IMPORT_PREFLIGHT_NON_JSON_OUTPUT") from exc
    if output["actual_imports"].get("teta_file") != str(expected_teta_init):
        raise RuntimeError("TETA_IMPORT_PATH_MISMATCH")
    if output["actual_imports"].get("cov_dataset_file") != str(expected_cov_dataset_init):
        raise RuntimeError("COV_IMPORT_PATH_MISMATCH")
    return output


def _resource_snapshot(gpu: str) -> dict[str, Any]:
    result: dict[str, Any] = {"gpu": gpu, "timestamp_unix": time.time()}
    try:
        result["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        result["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    try:
        result["free"] = subprocess.check_output(["free", "-b"], text=True).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        result["free_error"] = f"{type(exc).__name__}: {exc}"
    return result


@dataclass(frozen=True)
class _CategoryProtocol:
    benchmark_categories: tuple[dict[str, Any], ...]
    base_ids: frozenset[int]
    novel_ids: frozenset[int]

    def content_hash(self) -> str:
        payload = {
            "categories": list(self.benchmark_categories),
            "base_ids": sorted(self.base_ids),
            "novel_ids": sorted(self.novel_ids),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_annotation(path: Path) -> tuple[int, _CategoryProtocol]:
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = tuple(dict(item) for item in data.get("categories", []))
    base = frozenset(int(item["id"]) for item in categories if item.get("frequency", "f") != "r")
    novel = frozenset(int(item["id"]) for item in categories if item.get("frequency", "f") == "r")
    return len(data.get("images", [])), _CategoryProtocol(categories, base, novel)


def default_trial_specs() -> list[dict[str, Any]]:
    """A bounded coordinate set selected from the real smoke score range."""

    anchor = {
        "max_gap": 60,
        "candidate_top_k": 8,
        # Thresholds are regenerated from the post-contract Q1 smoke.  This
        # zero is only a structural placeholder and is never a replacement
        # for the smoke-derived quantiles.
        "score_threshold": 0.0,
        "margin_threshold": 0.0,
    }
    rows: list[dict[str, Any]] = []

    def add(name: str, **updates: Any) -> None:
        row = dict(anchor)
        row.update(updates)
        row["trial_id"] = name
        rows.append(row)

    add("anchor")
    add("gap_30", max_gap=30)
    add("gap_60", max_gap=60)
    add("gap_120", max_gap=120)
    add("gap_240", max_gap=240)
    add("gap_360", max_gap=360)
    add("topk_4", candidate_top_k=4)
    add("topk_8", candidate_top_k=8)
    add("topk_16", candidate_top_k=16)
    add("topk_32", candidate_top_k=32)
    add("topk_64", candidate_top_k=64)
    return rows


def _resolve_plan_reference(value: Any, plan_path: Path) -> str | None:
    if value is None:
        return None
    reference = Path(str(value)).expanduser()
    if not reference.is_absolute():
        reference = plan_path.parent / reference
    return str(reference.resolve())


def _load_search_plan(path: Path | None) -> SearchPlan:
    if path is None:
        return SearchPlan(
            path=None,
            sha256=None,
            protocol="UNBOUND_DEFAULTS",
            unbiased_test=False,
            contract_mode="legacy",
            contract_gate=None,
            contract_gate_sha256=None,
            threshold_source=None,
            threshold_quantiles={},
            expected_inputs={},
            trials=tuple(default_trial_specs()),
        )
    path = path.expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return SearchPlan(
            path=str(path),
            sha256=_sha256(path),
            protocol="LEGACY_LIST",
            unbiased_test=False,
            contract_mode="legacy",
            contract_gate=None,
            contract_gate_sha256=None,
            threshold_source=None,
            threshold_quantiles={},
            expected_inputs={},
            trials=tuple(dict(item) for item in raw),
        )
    if not isinstance(raw, Mapping):
        raise ValueError(f"invalid search plan: {path}")
    trials = raw.get("trials")
    if not isinstance(trials, list):
        raise ValueError(f"search plan must contain trials[]: {path}")
    quantiles = raw.get("threshold_quantiles", {})
    if not isinstance(quantiles, Mapping):
        raise ValueError("search plan threshold_quantiles must be a mapping")
    expected_inputs = raw.get("expected_inputs", {})
    if not isinstance(expected_inputs, Mapping):
        raise ValueError("search plan expected_inputs must be a mapping")
    return SearchPlan(
        path=str(path),
        sha256=_sha256(path),
        protocol=str(raw.get("protocol", "")),
        unbiased_test=bool(raw.get("unbiased_test", False)),
        contract_mode=str(raw.get("contract_mode", "legacy")),
        contract_gate=_resolve_plan_reference(raw.get("contract_gate"), path),
        contract_gate_sha256=(
            None if raw.get("contract_gate_sha256") is None else str(raw["contract_gate_sha256"])
        ),
        threshold_source=(
            None if raw.get("threshold_source") is None else str(raw["threshold_source"])
        ),
        threshold_quantiles={str(key): float(value) for key, value in quantiles.items()},
        expected_inputs={str(key): value for key, value in expected_inputs.items()},
        trials=tuple(dict(item) for item in trials),
    )


def _load_specs(path: Path | None) -> list[dict[str, Any]]:
    """Compatibility wrapper; coordinator uses the full SearchPlan."""
    return list(_load_search_plan(path).trials)


def _validate_contract_gate(plan: SearchPlan) -> dict[str, Any]:
    if plan.contract_mode not in {"legacy", "hardened"}:
        raise RuntimeError(f"SEARCH_CONTRACT_MODE_INVALID: {plan.contract_mode}")
    if not plan.contract_gate:
        raise RuntimeError("SEARCH_CONTRACT_GATE_MISSING")
    gate_path = Path(plan.contract_gate).resolve()
    if not gate_path.is_file():
        raise RuntimeError(f"SEARCH_CONTRACT_GATE_NOT_FOUND: {gate_path}")
    actual = _sha256(gate_path)
    if not plan.contract_gate_sha256 or actual != plan.contract_gate_sha256:
        raise RuntimeError("SEARCH_CONTRACT_GATE_HASH_MISMATCH")
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SEARCH_CONTRACT_GATE_INVALID: {gate_path}") from exc
    if gate.get("status") != "PASS":
        raise RuntimeError("SEARCH_CONTRACT_GATE_NOT_PASS")
    if int(gate.get("expected_q", -1)) != 1:
        raise RuntimeError("SEARCH_CONTRACT_Q_NOT_ONE")
    if int(gate.get("actual_q", -1)) != 1:
        raise RuntimeError("SEARCH_CONTRACT_RUNTIME_Q_NOT_ONE")
    if int(gate.get("context_candidate_top_k", -1)) != 64:
        raise RuntimeError("SEARCH_CONTRACT_CONTEXT_K_NOT_64")
    if int(gate.get("reranker_missing_evidence", -1)) != 0:
        raise RuntimeError("SEARCH_CONTRACT_MISSING_EVIDENCE")
    bootstrap_key = "reranker_native_memo_bootstrap_count"
    if bootstrap_key not in gate:
        if plan.contract_mode == "hardened":
            raise RuntimeError("SEARCH_CONTRACT_BOOTSTRAP_COUNTER_MISSING")
    elif int(gate[bootstrap_key]) != 0:
        raise RuntimeError("SEARCH_CONTRACT_NATIVE_MEMO_BOOTSTRAP")
    if plan.contract_mode == "hardened":
        for key, error in (
            ("teta_dependency_provenance", "SEARCH_CONTRACT_TETA_PROVENANCE_MISSING"),
            ("runtime_contract_sha_matches_expected", "SEARCH_CONTRACT_RUNTIME_REVISION_MISMATCH"),
            ("overlay_sha_matches_repo", "SEARCH_CONTRACT_OVERLAY_HASH_MISMATCH"),
            ("runtime_sha_matches_repo", "SEARCH_CONTRACT_RUNTIME_HASH_MISMATCH"),
            ("stream_covers_all_annotation_frames", "SEARCH_CONTRACT_STREAM_FRAME_COVERAGE"),
        ):
            if gate.get(key) is not True:
                raise RuntimeError(error)
    return gate


def _spec_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.spec_json:
        spec = json.loads(args.spec_json)
        if not isinstance(spec, Mapping):
            raise ValueError("--spec-json must be a JSON object")
        result = dict(spec)
    else:
        specs = _load_specs(Path(args.spec_file) if args.spec_file else None)
        matches = [item for item in specs if str(item.get("trial_id")) == args.trial_id]
        if len(matches) != 1:
            raise ValueError(f"expected one trial_id={args.trial_id!r}, found {len(matches)}")
        result = dict(matches[0])
    result["trial_id"] = args.trial_id
    unknown = sorted(set(result) - set(SEARCH_FIELDS) - {"trial_id"})
    if unknown:
        raise ValueError(f"unsupported search fields: {unknown}")
    return result


def _materialize_config(base_path: Path, output_path: Path, spec: Mapping[str, Any], disabled: bool) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Tempo config must be a mapping: {base_path}")
    tempo = raw.setdefault("tempo", {})
    if not isinstance(tempo, dict):
        raise ValueError("tempo config must be a mapping")
    for field in SEARCH_FIELDS:
        if field in spec:
            tempo[field] = spec[field]
    if disabled:
        tempo["enabled"] = False
        tempo["reranker_weight"] = 0.0
        tempo["reranker_checkpoint"] = None
    raw["protocol"] = {
        "test_tuned_model_specific": True,
        "unbiased_test": False,
        "search_fields": list(SEARCH_FIELDS),
        "disabled_overlay_control": bool(disabled),
    }
    text = "# TEST_TUNED_MODEL_SPECIFIC\n# NOT_UNBIASED_TEST\n" + yaml.safe_dump(raw, sort_keys=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return raw


def _run_logged(command: list[str], *, cwd: Path, env: Mapping[str, str], log_path: Path) -> tuple[int, int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(cwd), env=dict(env), stdout=log, stderr=subprocess.STDOUT, text=True)
        pid = process.pid
        return_code = process.wait()
    return pid, int(return_code), time.time() - started


def _runtime_env(args: argparse.Namespace, repo: Path, source: Path, trial_root: Path, config: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LD_PRELOAD": os.environ.get("LD_PRELOAD", "/home/lwr/anaconda3/envs/masaenv/lib/libsqlite3.so.3.52.0"),
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "V10_COV_SOURCE": str(source),
            "V10_WORK_DIR": str(trial_root / "work"),
            "V10_STREAM_RESULTS_DIR": str(trial_root / "stream"),
            "V10_COV_TEMPO_CONFIG": str(config),
            "V10_COV_TEMPO_DIAGNOSTICS": str(trial_root / "diagnostics.json"),
            "V10_TAO_FRAMES_ROOT": str(Path(args.img_prefix).resolve()),
        }
    )
    pythonpath = [str(repo), str(source), "/data1/LWR/vranlee/LLM/scalabel-scalabel-evalAPI"]
    teta_source_root = getattr(args, "teta_source_root", None)
    if teta_source_root:
        resolved_teta = _resolve_teta_source_root(teta_source_root)
        if resolved_teta is None:  # pragma: no cover - guarded above
            raise RuntimeError("TETA_SOURCE_ROOT_REQUIRED")
        pythonpath.insert(0, str(resolved_teta[0]))
    old = env.get("PYTHONPATH")
    if old:
        pythonpath.append(old)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _stream_command(args: argparse.Namespace, repo: Path, source: Path, trial_root: Path) -> list[str]:
    return [
        args.stream_python,
        str(repo / "tools/v10_covtrack_test_tempo_stream.py"),
        str(args.external_config),
        str(args.external_checkpoint),
        "--out",
        str(trial_root / "native_results.pkl"),
        "--eval-options",
        f"resfile_path={trial_root / 'internal_results.pth'}",
        "--cfg-options",
        f"data.test.ann_file={Path(args.annotation).resolve()}",
        f"data.test.img_prefix={Path(args.img_prefix).resolve()}/",
        "data.workers_per_gpu=1",
        "model.roi_head.only_validation_categories=False",
        "model.roi_head.only_test_categories=True",
        "model.tracker.match_score_thr=0.37",
        "model.tracker.memo_frames=50",
        "model.tracker.momentum_embed=0.4",
        "model.tracker.confused_features=True",
        "model.tracker.vis=False",
        "model.test_cfg.rcnn.max_per_img=80",
        "model.roi_head.feature_fusion_head.max_fusion_ratio=2.0",
    ]


def _evaluate_command(args: argparse.Namespace, repo: Path, trial_root: Path) -> list[str]:
    return [
        args.evaluator_python,
        str(repo / "tools/eval_ovmot_teta.py"),
        "--gt",
        str(Path(args.annotation).resolve()),
        "--pred",
        str(trial_root / "stream/tao_track.json"),
        "--out",
        str(trial_root / "evaluation"),
        "--name",
        args.evaluator_name,
        "--cores",
        str(args.evaluator_cores),
    ]


def _hash_if_file(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _validate_runtime_contract(
    *,
    diagnostics_path: Path,
    spec: Mapping[str, Any],
    expected_checkpoint_sha: str | None = None,
    require_native_memo_bootstrap_counter: bool = True,
) -> dict[str, Any]:
    """Validate the runtime contract before allowing official evaluation."""
    if not diagnostics_path.is_file():
        raise RuntimeError("RUNTIME_DIAGNOSTICS_MISSING")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if int(diagnostics.get("reranker_expected_query_observations", -1)) != 1:
        failures.append("expected_q")
    if int(diagnostics.get("reranker_actual_query_observations", -1)) != 1:
        failures.append("actual_q")
    if int(diagnostics.get("reranker_context_candidate_top_k", -1)) != 64:
        failures.append("context_k")
    if int(diagnostics.get("reranker_decision_candidate_top_k", -1)) != int(spec["candidate_top_k"]):
        failures.append("decision_k")
    if bool(diagnostics.get("reranker_context_contract_mismatch", True)):
        failures.append("context_contract")
    if int(diagnostics.get("reranker_missing_evidence", -1)) != 0:
        failures.append("missing_evidence")
    bootstrap_key = "reranker_native_memo_bootstrap_count"
    if bootstrap_key not in diagnostics:
        if require_native_memo_bootstrap_counter:
            failures.append("native_memo_bootstrap_counter_missing")
    elif int(diagnostics[bootstrap_key]) != 0:
        failures.append("native_memo_bootstrap")

    status = diagnostics.get("reranker_status")
    if not isinstance(status, Mapping):
        failures.append("reranker_provenance")
    else:
        if status.get("status") != "EXACT_V9_MODEL_CODE_AND_WEIGHTS":
            failures.append("reranker_provenance_status")
        if status.get("base_only_supervision") is not True:
            failures.append("reranker_not_base_only")
        if status.get("novel_gt_used") is not False:
            failures.append("reranker_novel_gt_used")
        if status.get("test_weights_used") is not False:
            failures.append("reranker_test_weights_used")
        if expected_checkpoint_sha is not None and status.get("checkpoint_sha256") != expected_checkpoint_sha:
            failures.append("reranker_checkpoint")
    capability = dict(diagnostics.get("full_capability_status_counts", {}))
    if not capability or any(
        name != "FULL_Q1_RERANKER_RUNTIME_ACTIVE" for name in capability
    ):
        failures.append("capability")
    if int(sum(int(value) for value in capability.values())) != int(diagnostics.get("frames", -1)):
        failures.append("capability_frame_coverage")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "diagnostics": diagnostics,
    }


def _worker_search_plan_binding(args: argparse.Namespace) -> dict[str, Any] | None:
    """Validate and return the immutable search-plan binding for a worker."""
    if args.disabled_overlay:
        if args.search_plan is None and args.search_plan_sha256 is None:
            return None
        if args.search_plan is None or args.search_plan_sha256 is None:
            raise RuntimeError("SEARCH_PLAN_BINDING_INCOMPLETE")
        plan_path = Path(args.search_plan).expanduser().resolve()
        plan = _load_search_plan(plan_path)
        if plan.sha256 != args.search_plan_sha256:
            raise RuntimeError("SEARCH_PLAN_HASH_MISMATCH")
        if plan.contract_mode == "hardened" and not getattr(args, "teta_source_root", None):
            raise RuntimeError("TETA_SOURCE_ROOT_REQUIRED")
        return {
            "path": str(plan_path),
            "sha256": str(plan.sha256),
            "contract_gate": None,
            "contract_gate_sha256": None,
            "contract_mode": plan.contract_mode,
            "threshold_source": plan.threshold_source,
            "protocol": plan.protocol,
            "unbiased_test": plan.unbiased_test,
            "expected_inputs": dict(plan.expected_inputs),
        }
    fields = (
        args.search_plan,
        args.search_plan_sha256,
        args.contract_gate,
        args.contract_gate_sha256,
    )
    if all(value is None for value in fields):
        return None
    if any(value is None for value in fields):
        raise RuntimeError("SEARCH_PLAN_BINDING_INCOMPLETE")
    plan_path = Path(args.search_plan).expanduser().resolve()
    plan = _load_search_plan(plan_path)
    if plan.contract_mode == "hardened" and not getattr(args, "teta_source_root", None):
        raise RuntimeError("TETA_SOURCE_ROOT_REQUIRED")
    if plan.sha256 != args.search_plan_sha256:
        raise RuntimeError("SEARCH_PLAN_HASH_MISMATCH")
    gate_path = Path(args.contract_gate).expanduser().resolve()
    if plan.contract_gate != str(gate_path):
        raise RuntimeError("SEARCH_PLAN_GATE_PATH_MISMATCH")
    if plan.contract_gate_sha256 != args.contract_gate_sha256:
        raise RuntimeError("SEARCH_PLAN_GATE_HASH_MISMATCH")
    _validate_contract_gate(plan)
    if args.threshold_source != plan.threshold_source:
        raise RuntimeError("SEARCH_PLAN_THRESHOLD_SOURCE_MISMATCH")
    return {
        "path": str(plan_path),
        "sha256": str(plan.sha256),
        "contract_gate": str(gate_path),
        "contract_gate_sha256": str(plan.contract_gate_sha256),
        "contract_mode": plan.contract_mode,
        "threshold_source": plan.threshold_source,
        "protocol": plan.protocol,
        "unbiased_test": plan.unbiased_test,
        "expected_inputs": dict(plan.expected_inputs),
    }


def _parse_summary_with_evaluator(
    args: argparse.Namespace,
    summary: Path,
    annotation: Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Parse with the same masaenv that ran the official TETA evaluator."""

    code = r'''
import hashlib
import json
import sys
from pathlib import Path
from tempotrack_research.evaluation.teta_parser import parse_teta_summary

class Protocol:
    def __init__(self, categories):
        self.benchmark_categories = tuple(categories)
        self.base_ids = frozenset(int(item["id"]) for item in categories if item.get("frequency", "f") != "r")
        self.novel_ids = frozenset(int(item["id"]) for item in categories if item.get("frequency", "f") == "r")
    def content_hash(self):
        value = {"categories": list(self.benchmark_categories), "base_ids": sorted(self.base_ids), "novel_ids": sorted(self.novel_ids)}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()

annotation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
protocol = Protocol(annotation.get("categories", []))
print(json.dumps(parse_teta_summary(Path(sys.argv[1]), category_protocol=protocol)))
'''
    result = subprocess.run(
        [args.evaluator_python, "-c", code, str(summary), str(annotation)],
        cwd=str(Path(args.repo).resolve()),
        env=dict(env),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "summary parsing failed in evaluator environment: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"summary parser returned non-JSON output: {result.stdout[-1000:]}") from exc


def run_trial(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    source = Path(args.source).resolve()
    annotation = Path(args.annotation).resolve()
    external_config = Path(args.external_config).resolve()
    external_checkpoint = Path(args.external_checkpoint).resolve()
    base_config = Path(args.base_config).resolve()
    output_root = Path(args.output_root).resolve()
    search_plan_binding = _worker_search_plan_binding(args)
    trial_root = output_root / args.trial_id
    receipt_path = trial_root / "receipt.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            print(json.dumps({"status": "REUSED", "receipt": str(receipt_path)}))
            return 0
        raise FileExistsError(f"refusing to overwrite non-completed trial: {trial_root}")
    if trial_root.exists() and any(trial_root.iterdir()):
        raise FileExistsError(f"refusing to reuse partial trial directory: {trial_root}")
    trial_root.mkdir(parents=True, exist_ok=False)

    image_count, protocol = _load_annotation(annotation)
    spec = _spec_from_args(args)
    config_path = trial_root / "tempo.yaml"
    config_data = _materialize_config(base_config, config_path, spec, args.disabled_overlay)
    teta_source_root = getattr(args, "teta_source_root", None)
    teta_dependency = _teta_dependency(teta_source_root)
    teta_preflight = None
    if teta_source_root:
        teta_preflight = _run_teta_import_preflight(
            stream_python=args.stream_python,
            source=source,
            teta_source_root=teta_source_root,
        )
    env = _runtime_env(args, repo, source, trial_root, config_path)
    stream_command = _stream_command(args, repo, source, trial_root)
    evaluate_command = _evaluate_command(args, repo, trial_root)
    started = time.time()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "tempotrack_v10_covtrack_test_trial",
        "status": "RUNNING",
        "trial_id": args.trial_id,
        "stage": args.stage,
        "spec": spec,
        "protocol": {
            "test_tuned_model_specific": True,
            "unbiased_test": False,
            "disabled_overlay_control": bool(args.disabled_overlay),
            "annotation_image_count": image_count,
            "category_protocol_hash": protocol.content_hash(),
        },
        "repo": {
            "path": str(repo),
            "head": _git_value(repo, "rev-parse", "HEAD"),
            "branch": _git_value(repo, "branch", "--show-current"),
        },
        "external_source": {
            "path": str(source),
            "commit": _git_value(source, "rev-parse", "HEAD"),
        },
        "teta_dependency": teta_dependency,
        "teta_import_preflight": teta_preflight,
        "inputs": {
            "annotation": str(annotation),
            "annotation_sha256": _sha256(annotation),
            "external_config": str(external_config),
            "external_config_sha256": _sha256(external_config),
            "external_checkpoint": str(external_checkpoint),
            "external_checkpoint_sha256": _sha256(external_checkpoint),
            "base_config": str(base_config),
            "base_config_sha256": _sha256(base_config),
            "tempo_config": str(config_path),
            "tempo_config_sha256": _sha256(config_path),
            "evaluator": str(repo / "tools/eval_ovmot_teta.py"),
            "evaluator_sha256": _sha256(repo / "tools/eval_ovmot_teta.py"),
            "overlay_sha256": _sha256(repo / "tempotrack_v10/overlay.py"),
            "runtime_sha256": _sha256(repo / "tempotrack_v10/covtrack_runtime.py"),
            "stream_sha256": _sha256(repo / "tools/v10_covtrack_test_tempo_stream.py"),
        },
        "commands": {
            "stream": stream_command,
            "evaluate": evaluate_command,
            "stream_cwd": str(source),
            "evaluate_cwd": str(repo),
            "teta_import_preflight": (
                None if teta_preflight is None else teta_preflight["command"]
            ),
        },
        "gpu": str(args.gpu),
        "python": {
            "stream": args.stream_python,
            "evaluator": args.evaluator_python,
        },
        "resources_start": _resource_snapshot(str(args.gpu)),
        "started_at_unix": started,
    }
    if teta_dependency is not None:
        receipt["inputs"].update(
            {
                "teta_source_root": teta_dependency["source_root"],
                "teta_init_file": teta_dependency["init_file"],
                "teta_init_sha256": teta_dependency["init_sha256"],
                "teta_git_commit": teta_dependency["git_commit"],
            }
        )
    if args.requested_trial_id:
        receipt["requested_trial_id"] = args.requested_trial_id
    if search_plan_binding is not None:
        receipt["search_plan"] = search_plan_binding
    _write_json(receipt_path, receipt)
    try:
        stream_pid, stream_rc, stream_seconds = _run_logged(
            stream_command,
            cwd=source,
            env=env,
            log_path=trial_root / "stream.log",
        )
        receipt["stream"] = {"pid": stream_pid, "returncode": stream_rc, "seconds": stream_seconds}
        if stream_rc != 0:
            raise RuntimeError(f"COV stream failed with returncode={stream_rc}; see {trial_root / 'stream.log'}")
        stream_manifest = trial_root / "stream/stream_manifest.json"
        prediction = trial_root / "stream/tao_track.json"
        if not stream_manifest.is_file() or not prediction.is_file():
            raise FileNotFoundError("stream did not produce stream_manifest.json and tao_track.json")
        manifest = json.loads(stream_manifest.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or int(manifest.get("frames", -1)) != image_count:
            raise RuntimeError(f"stream manifest contract failed: {manifest}")
        if args.disabled_overlay:
            runtime_contract = {"status": "NOT_APPLICABLE", "failures": []}
        else:
            runtime_contract = _validate_runtime_contract(
                diagnostics_path=trial_root / "diagnostics.json",
                spec=spec,
                expected_checkpoint_sha=(
                    _sha256(Path(str(config_data["tempo"]["reranker_checkpoint"])).resolve())
                    if config_data.get("tempo", {}).get("reranker_checkpoint")
                    and Path(str(config_data["tempo"]["reranker_checkpoint"])).resolve().is_file()
                    else None
                ),
                require_native_memo_bootstrap_counter=True,
            )
        receipt["runtime_contract"] = {
            "status": runtime_contract["status"],
            "failures": runtime_contract["failures"],
        }
        _write_json(receipt_path, receipt)
        if not args.disabled_overlay and runtime_contract["status"] != "PASS":
            raise RuntimeError(
                "RUNTIME_CONTRACT_FAILED: " + ",".join(runtime_contract["failures"])
            )
        eval_pid, eval_rc, eval_seconds = _run_logged(
            evaluate_command,
            cwd=repo,
            env=env,
            log_path=trial_root / "evaluation.log",
        )
        receipt["evaluate"] = {"pid": eval_pid, "returncode": eval_rc, "seconds": eval_seconds}
        if eval_rc != 0:
            raise RuntimeError(f"official TETA failed with returncode={eval_rc}; see {trial_root / 'evaluation.log'}")
        summary = trial_root / "evaluation" / args.evaluator_name / "teta_summary_results.pth"
        if not summary.is_file():
            raise FileNotFoundError(f"official TETA summary missing: {summary}")
        parsed = _parse_summary_with_evaluator(args, summary, annotation, env)
        diagnostics = trial_root / "diagnostics.json"
        receipt["outputs"] = {
            "stream_manifest": str(stream_manifest),
            "stream_manifest_sha256": _sha256(stream_manifest),
            "prediction": str(prediction),
            "prediction_sha256": _sha256(prediction),
            "prediction_bytes": prediction.stat().st_size,
            "diagnostics": str(diagnostics),
            "diagnostics_sha256": _hash_if_file(diagnostics),
            "summary": str(summary),
            "summary_sha256": _sha256(summary),
        }
        receipt["metrics"] = parsed
        receipt["status"] = "COMPLETED"
        receipt["resources_end"] = _resource_snapshot(str(args.gpu))
        receipt["ended_at_unix"] = time.time()
        receipt["duration_seconds"] = receipt["ended_at_unix"] - started
        _write_json(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "trial_id": args.trial_id, "receipt": str(receipt_path), "base": parsed.get("base"), "novel": parsed.get("novel")}))
        return 0
    except Exception as exc:
        receipt["status"] = "FAILED"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        receipt["resources_end"] = _resource_snapshot(str(args.gpu))
        receipt["ended_at_unix"] = time.time()
        receipt["duration_seconds"] = receipt["ended_at_unix"] - started
        _write_json(receipt_path, receipt)
        print(json.dumps({"status": "FAILED", "trial_id": args.trial_id, "receipt": str(receipt_path), "error": receipt["error"]}), file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source", required=True, help="Pinned external COVTrack checkout")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--img-prefix", required=True)
    parser.add_argument("--external-config", required=True)
    parser.add_argument("--external-checkpoint", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--requested-trial-id")
    parser.add_argument("--spec-file")
    parser.add_argument("--spec-json")
    parser.add_argument("--stage", choices=("subset", "full"), default="subset")
    parser.add_argument("--gpu", required=True)
    parser.add_argument(
        "--stream-python",
        default="/home/lwr/anaconda3/envs/ovtr/bin/python",
    )
    parser.add_argument(
        "--evaluator-python",
        default="/home/lwr/anaconda3/envs/masaenv/bin/python",
    )
    parser.add_argument(
        "--teta-source-root",
        help="Import parent containing the pinned teta package (for example .../tet/teta)",
    )
    parser.add_argument("--evaluator-name", default="COV_V10_TEMPO")
    parser.add_argument("--evaluator-cores", type=int, default=8)
    parser.add_argument("--disabled-overlay", action="store_true")
    parser.add_argument("--search-plan")
    parser.add_argument("--search-plan-sha256")
    parser.add_argument("--contract-gate")
    parser.add_argument("--contract-gate-sha256")
    parser.add_argument("--threshold-source")
    parser.add_argument("--list-defaults", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_defaults:
        print(json.dumps(default_trial_specs(), indent=2))
        return 0
    return run_trial(args)


if __name__ == "__main__":
    raise SystemExit(main())
