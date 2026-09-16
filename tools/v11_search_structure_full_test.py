#!/usr/bin/env python3
"""Run the six-candidate V11 QDIC structural Full-Test search.

This controller deliberately keeps the earlier threshold-search controller
out of the policy loop.  It reuses its audited preflight, shard worker,
complete-video merge, and official TETA evaluator, while the only formal
search axes here are runtime ``candidate_top_k`` and runtime ``max_gap``.
The QDIC checkpoint feature contract remains K=8/max_gap=360.
"""

from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_local_module(path: Path, name: str) -> Any:
    """Load a repository script without being shadowed by site-packages/tools."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load repository module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_local_module(REPO / "tools" / "v11_search_covtrack_full_test.py", "tempotrack_v11_base_search")


ROOT_DEFAULT = Path("/data2/usr_for_deadline/tempotrack_v11_qdic_structure_search_20260916")
REPORT_DEFAULT = REPO / "reports/tempotrack_v11/structure_search_6x_20260916"
ANCHOR_DEFAULT = Path(
    "/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_recovery_20260915_racefix/"
    "full_results.json"
)
MARGIN_DEFAULT = 0.37210235595703123
EVENT_DIAGNOSTICS_TOOL = REPO / "tools" / "v11_structure_diagnostics.py"


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _six_trials() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    raw = [
        ("ST1_K16_G360", "W1", 16, 360, 29502),
        ("ST2_K32_G360", "W3", 32, 360, 29506),
        ("ST3_K64_G360", "W2", 64, 360, 29503),
        ("ST4_K32_G180", "W2", 32, 180, 29504),
        ("ST5_K32_G720", "W3", 32, 720, 29505),
        ("ST6_K64_G720", "W1", 64, 720, 29501),
    ]
    trials = [
        {
            "trial_id": trial_id,
            "wave": wave,
            "candidate_top_k": k,
            "max_gap": gap,
            "score_threshold": 0.0,
            "margin_threshold": MARGIN_DEFAULT,
            "master_port": port,
            "collect_event_diagnostics": True,
        }
        for trial_id, wave, k, gap, port in raw
    ]
    waves = {
        "W1": ["ST6_K64_G720", "ST1_K16_G360"],
        "W2": ["ST3_K64_G360", "ST4_K32_G180"],
        "W3": ["ST5_K32_G720", "ST2_K32_G360"],
    }
    return trials, waves


def _fixed_runtime() -> dict[str, Any]:
    return {
        "candidate_top_k_anchor": 8,
        "max_gap_anchor": 360,
        "qdic_context_top_k": 64,
        "top_r": 3,
        "memory_capacity": 64,
        "qdic_recent_k": 8,
        "qdic_weight": 1.0,
        "reranker_weight": 0.0,
        "alpha_fast": 0.70,
        "alpha_slow": 0.15,
        "min_gap": 0,
        "score_threshold": 0.0,
        "margin_threshold": MARGIN_DEFAULT,
        "qdic_feature_decision_candidate_top_k": 8,
        "qdic_feature_max_gap": 360,
    }


def _structure_plan(preflight: Mapping[str, Any]) -> dict[str, Any]:
    trials, waves = _six_trials()
    return {
        "schema_version": 1,
        "artifact": "v11_qdic_structure_full_test_search_plan",
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "search_fields": ["candidate_top_k", "max_gap"],
        "fixed_runtime": _fixed_runtime(),
        "learned_checkpoint_contract": {
            "decision_candidate_top_k": 8,
            "max_gap": 360,
            "note": "runtime candidate legality/K may vary; QDIC feature construction keeps checkpoint max_gap=360 and learned decision-K metadata=8",
        },
        "policy": {
            "primary_objective": "official Overall TETA on the tuned Test protocol",
            "tie_break": ["Novel TETA", "Overall AssocA", "Novel AssocA"],
            "formal_candidate_count": 6,
            "candidate_parallelism": 2,
            "gpu_policy": "all ten selected GPUs per complete candidate; never split 5+5",
            "frontend_cache_used": False,
            "no_retraining": True,
            "no_gt_online_selection": True,
        },
        "waves": waves,
        "trials": trials,
        "anchor": {
            "trial_id": "ANCHOR_FT_B",
            "candidate_top_k": 8,
            "max_gap": 360,
            "score_threshold": 0.0,
            "margin_threshold": MARGIN_DEFAULT,
            "rerun": False,
            "source": str(ANCHOR_DEFAULT),
        },
        "full_test_annotation_sha256": preflight["full_test_annotation"]["sha256"],
        "qdic_checkpoint_sha256": preflight["qdic"]["checkpoint_sha256"],
        "repo_head": preflight["repository"]["head"],
        "created_at_unix": time.time(),
    }


def _active_v11_processes(repo: Path) -> list[dict[str, Any]]:
    current = os.getpid()
    result: list[dict[str, Any]] = []
    for path in sorted(Path("/proc").glob("[0-9]*"), key=lambda item: int(item.name)):
        try:
            pid = int(path.name)
        except ValueError:
            continue
        if pid == current:
            continue
        command = base._proc_cmdline(pid)
        if not command:
            continue
        # The controller is commonly launched through ``bash -lc`` or
        # ``nohup``.  Those wrapper command lines contain the Python script
        # text but are not V11 workers; only inspect Python executables so a
        # preflight cannot report its own shell wrapper as a stale run.
        executable = Path(command[0]).name.lower()
        if not executable.startswith("python"):
            continue
        joined = " ".join(command)
        if str(repo) not in joined and "v11_search_covtrack_full_test.py" not in joined:
            continue
        if any(name in joined for name in ("v11_search_covtrack_full_test.py", "v11_run_covtrack_full_trial.py", "v11_search_structure_full_test.py")):
            result.append({"pid": pid, "command": command})
    return result


def _disk_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "free_gib": float(usage.free / (1024**3)),
        "warning": bool(usage.free < 35 * 1024**3),
    }


def _write_structure_preflight(
    *, args: argparse.Namespace, root: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run the audited base preflight, then replace only its search plan."""
    residual = _active_v11_processes(repo)
    if residual:
        raise RuntimeError(
            "old V11 Full-Test processes are still active; refusing to overlap them: "
            + ",".join(str(item["pid"]) for item in residual)
        )
    preflight = base._preflight(args, root, repo)
    plan = _structure_plan(preflight)
    plan_path = root / "search_plan.json"
    base._write_json(plan_path, plan)
    preflight = dict(preflight)
    preflight["artifact"] = "v11_qdic_structure_full_test_preflight"
    preflight["search_plan"] = {
        "path": str(plan_path),
        "sha256": base._sha256(plan_path),
        "initial_sha256": base._sha256(plan_path),
        "status": "PASS",
    }
    preflight["runtime"] = dict(preflight.get("runtime", {}))
    preflight["runtime"]["fixed_fields"] = dict(plan["fixed_runtime"])
    preflight["runtime"]["search_fields"] = list(plan["search_fields"])
    preflight["runtime"]["frontend_cache_used"] = False
    preflight["structure_search"] = {
        "status": "PASS",
        "search_plan_artifact": plan["artifact"],
        "search_fields": list(plan["search_fields"]),
        "formal_trials": [dict(item) for item in plan["trials"]],
        "waves": dict(plan["waves"]),
        "anchor": dict(plan["anchor"]),
        "old_v11_processes_at_preflight": residual,
        "disk": _disk_snapshot(root),
        "gpu_snapshot": base._gpu_snapshot(base._parse_gpu_ids(args.gpus)),
    }
    base._write_json(root / "preflight.json", preflight)
    manifest = {
        "schema_version": 1,
        "artifact": "v11_qdic_structure_search_manifest",
        "status": "PREPARED",
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "repository": dict(preflight["repository"]),
        "branch_required": "codex/v11-fulltest-20h-search",
        "preflight": {
            "path": str(root / "preflight.json"),
            "sha256": base._sha256(root / "preflight.json"),
        },
        "search_plan": {
            "path": str(plan_path),
            "sha256": base._sha256(plan_path),
        },
        "full_test_annotation": dict(preflight["full_test_annotation"]),
        "complete_video_shards": dict(preflight["shards"]),
        "qdic": dict(preflight["qdic"]),
        "cov": dict(preflight["cov"]),
        "teta": dict(preflight["teta"]),
        "runtime_environment": dict(preflight["runtime_environment"]),
        "fixed_runtime": dict(plan["fixed_runtime"]),
        "learned_checkpoint_contract": dict(plan["learned_checkpoint_contract"]),
        "anchor": dict(plan["anchor"]),
        "trials": [dict(item) for item in plan["trials"]],
        "waves": dict(plan["waves"]),
        "selected_gpus": base._parse_gpu_ids(args.gpus),
        "resource_policy": base._resource_policy(args),
        "candidate_parallelism": 2,
        "output_root": str(root),
        "report_root": str(Path(args.report_root).resolve()),
        "disk": _disk_snapshot(root),
        "created_at_unix": time.time(),
    }
    base._write_json(root / "structure_search_manifest.json", manifest)
    return preflight, plan, manifest


def _validate_structure_resume(
    *, args: argparse.Namespace, root: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight_path = root / "preflight.json"
    plan_path = root / "search_plan.json"
    if not preflight_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError("resume requires structure preflight.json and search_plan.json")
    preflight = base._read_json(preflight_path)
    plan = base._read_json(plan_path)
    if preflight.get("status") != "PASS" or preflight.get("artifact") != "v11_qdic_structure_full_test_preflight":
        raise RuntimeError("existing structure preflight is not PASS")
    if plan.get("artifact") != "v11_qdic_structure_full_test_search_plan":
        raise RuntimeError("existing search plan is not the structure-search plan")
    if preflight.get("search_plan", {}).get("sha256") != base._sha256(plan_path):
        raise RuntimeError("structure preflight search-plan hash is stale")
    expected_gpus = base._parse_gpu_ids(args.gpus)
    if expected_gpus != [str(value) for value in range(10)]:
        raise RuntimeError("structure search resume requires physical GPUs 0-9")
    annotation = base._path(args.annotation)
    full = base._annotation_summary(annotation)
    if full["sha256"] != base.EXPECTED_ANNOTATION_SHA256 or full["sha256"] != preflight["full_test_annotation"]["sha256"]:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: full Test annotation changed")
    shards = base._validate_shards(
        full_annotation=annotation,
        shard_manifest=base._path(args.shard_manifest),
        full=full,
    )
    if [(item["index"], item["sha256"]) for item in shards] != [
        (item["index"], item["sha256"]) for item in preflight["shards"]["items"]
    ]:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: shard annotations changed")
    cov_source = base._path(args.cov_source)
    if _git_value(cov_source, "rev-parse", "HEAD") != preflight["cov"]["commit"] or base._git_source_status(cov_source) != "":
        raise RuntimeError("RESUME_PROVENANCE_FAIL: COV source changed")
    for key, path_arg in (("config", args.external_config), ("checkpoint", args.external_checkpoint)):
        path = base._path(path_arg)
        if base._path(preflight["cov"][key]) != path or base._sha256(path) != preflight["cov"][f"{key}_sha256"]:
            raise RuntimeError(f"RESUME_PROVENANCE_FAIL: COV {key} changed")
    repo_branch = base._git_branch(repo)
    repo_head = base._git_value(repo, "rev-parse", "HEAD")
    if repo_branch != preflight["repository"]["branch"] or repo_head != preflight["repository"]["head"]:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: V11 branch or HEAD changed")
    runtime = base._validate_runtime_environment(args=args, repo=repo, preflight=preflight)
    checkpoint = base._path(preflight["qdic"]["checkpoint"])
    if base._sha256(checkpoint) != preflight["qdic"]["checkpoint_sha256"]:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: QDIC checkpoint changed")
    base._qdic_preflight(
        args.stream_python,
        repo,
        checkpoint,
        scalabel_root=base._path(runtime["scalabel_root"]),
        runtime_environment=runtime,
    )
    if base._path(args.base_config) != base._path(preflight["base_config"]["path"]) or base._sha256(base._path(args.base_config)) != preflight["base_config"]["sha256"]:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: base config changed")
    if base._resource_policy(args) != preflight["resource_gate"]["policy"]:
        raise RuntimeError("RESUME_PROVENANCE_FAIL: resource policy changed")
    print("STRUCTURE_RESUME_PROVENANCE_PASS", flush=True)
    return dict(preflight), dict(plan)


def _config_diff(plan: Mapping[str, Any], preflight: Mapping[str, Any]) -> str:
    lines = [
        "# V11 QDIC structural Full-Test config diff",
        "",
        "Protocol: `TEST_TUNED_MODEL_SPECIFIC` (`NOT_UNBIASED_TEST`).",
        "",
        "The detector, COV checkpoint/config, QDIC checkpoint, frontend, evaluator, and all fixed runtime fields are held constant. Only the two structural runtime fields below vary.",
        "",
        "| Candidate | candidate_top_k | max_gap | score_threshold | margin_threshold |",
        "|---|---:|---:|---:|---:|",
        f"| Anchor FT_B | 8 | 360 | 0.0 | {MARGIN_DEFAULT:.17g} |",
    ]
    for trial in plan["trials"]:
        lines.append(
            f"| {trial['trial_id']} | {trial['candidate_top_k']} | {trial['max_gap']} | "
            f"{trial['score_threshold']} | {trial['margin_threshold']:.17g} |"
        )
    lines.extend(
        [
            "",
            "## Immutable QDIC learned feature contract",
            "",
            "- Checkpoint decision feature metadata remains `decision_candidate_top_k=8`.",
            "- Checkpoint feature normalization remains `max_gap=360`.",
            "- Runtime `candidate_top_k` changes how many candidates survive the shared Top-64 prefilter into decision scoring.",
            "- Runtime `max_gap` changes only the legal causal candidate horizon. The QDIC payload still receives the checkpoint feature `max_gap=360` for learned feature construction.",
            "- No GT, category labels, or official evaluator result enters inference.",
            "",
            "## Provenance",
            "",
            f"- Repository HEAD pinned by preflight: `{preflight['repository']['head']}`.",
            f"- QDIC checkpoint SHA256: `{preflight['qdic']['checkpoint_sha256']}`.",
            f"- Full Test annotation SHA256: `{preflight['full_test_annotation']['sha256']}`.",
            "- All formal candidates use direct COVTrack execution; no frontend replay cache is used.",
        ]
    )
    return "\n".join(lines) + "\n"


def _new_state(
    *, args: argparse.Namespace, root: Path, report_root: Path,
    preflight: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    trials = []
    for item in plan["trials"]:
        value = dict(item)
        value.update(
            {
                "status": "PENDING",
                "receipt": str(root / "trials" / item["trial_id"] / "receipt.json"),
                "candidate_root": str(root / "trials" / item["trial_id"]),
                "report_root": str(report_root / "per_experiment" / item["trial_id"]),
            }
        )
        trials.append(value)
    return {
        "schema_version": 1,
        "artifact": "v11_qdic_structure_full_test_search_state",
        "status": "PREPARED",
        "phase": "SANITY",
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "root": str(root),
        "report_root": str(report_root),
        "repo_head": preflight["repository"]["head"],
        "repo_branch": preflight["repository"]["branch"],
        "preflight_sha256": base._sha256(root / "preflight.json"),
        "search_plan_sha256": base._sha256(root / "search_plan.json"),
        "selected_gpus": base._parse_gpu_ids(args.gpus),
        "resource_policy": base._resource_policy(args),
        "candidate_parallelism": 2,
        "target_hours": float(args.hours),
        "runtime_sanity_status": "PENDING",
        "runtime_sanity_path": str(root / "sanity.json"),
        "formal_search_started_at_unix": None,
        "formal_search_ended_at_unix": None,
        "waves": {},
        "trials": trials,
        "events": [],
        "created_at_unix": time.time(),
        "updated_at_unix": time.time(),
    }


def _status_view(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "v11_qdic_structure_search_status",
        "status": state.get("status"),
        "phase": state.get("phase"),
        "protocol": state.get("protocol"),
        "unbiased_test": state.get("unbiased_test"),
        "root": state.get("root"),
        "report_root": state.get("report_root"),
        "repo_head": state.get("repo_head"),
        "selected_gpus": state.get("selected_gpus"),
        "resource_policy": state.get("resource_policy"),
        "candidate_parallelism": state.get("candidate_parallelism"),
        "runtime_sanity_status": state.get("runtime_sanity_status"),
        "formal_search_started_at_unix": state.get("formal_search_started_at_unix"),
        "formal_search_ended_at_unix": state.get("formal_search_ended_at_unix"),
        "waves": state.get("waves", {}),
        "trials": [
            {
                key: value
                for key, value in item.items()
                if key
                in {
                    "trial_id", "wave", "candidate_top_k", "max_gap", "score_threshold",
                    "margin_threshold", "master_port", "status", "candidate_root",
                    "report_root", "full_started_at_unix", "full_test_end_unix",
                    "full_duration_seconds", "completed_at_unix", "error", "diagnostics_path",
                    "evaluation", "score_distribution",
                }
            }
            for item in state.get("trials", [])
        ],
        "updated_at_unix": time.time(),
    }


def _persist_state(root: Path, report_root: Path, state: dict[str, Any]) -> None:
    base._write_state(root, state)
    view = _status_view(state)
    base._write_json(root / "structure_search_status.json", view)
    base._write_json(report_root / "structure_search_status.json", view)


def _load_anchor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "trial_id": "ANCHOR_FT_B",
            "status": "UNAVAILABLE",
            "candidate_top_k": 8,
            "max_gap": 360,
            "score_threshold": 0.0,
            "margin_threshold": MARGIN_DEFAULT,
            "metrics": None,
            "source": str(path),
        }
    data = base._read_json(path)
    candidate = next(
        (
            dict(item)
            for item in data.get("candidates", [])
            if str(item.get("trial_id")) == "FT_B"
        ),
        None,
    )
    if candidate is None:
        return {
            "trial_id": "ANCHOR_FT_B", "status": "UNAVAILABLE", "candidate_top_k": 8,
            "max_gap": 360, "score_threshold": 0.0,
            "margin_threshold": MARGIN_DEFAULT, "metrics": None, "source": str(path),
        }
    candidate["trial_id"] = "ANCHOR_FT_B"
    candidate["status"] = "COMPLETED" if candidate.get("status") == "COMPLETED" else candidate.get("status")
    candidate["candidate_top_k"] = 8
    candidate["max_gap"] = 360
    candidate["score_threshold"] = 0.0
    candidate["margin_threshold"] = MARGIN_DEFAULT
    candidate["anchor_source"] = str(path)
    return candidate


def _runtime_sanity_valid(root: Path, preflight: Mapping[str, Any]) -> bool:
    path = root / "sanity.json"
    if not path.is_file():
        return False
    try:
        value = base._read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(value, Mapping)
        and value.get("status") == "PASS"
        and value.get("preflight_sha256") == base._sha256(root / "preflight.json")
        and value.get("repo_head") == preflight["repository"]["head"]
        and {str(item.get("trial_id")) for item in value.get("trials", [])}
        == {"SANITY_K16_G360", "SANITY_K64_G720"}
        and all(item.get("status") == "PASS" for item in value.get("trials", []))
    )


def _run_runtime_sanity(
    *, args: argparse.Namespace, root: Path, report_root: Path,
    preflight: Mapping[str, Any], state: dict[str, Any]
) -> None:
    if _runtime_sanity_valid(root, preflight):
        state["runtime_sanity_status"] = "PASS"
        _persist_state(root, report_root, state)
        print("STRUCTURE_RUNTIME_SANITY: PASS (reused)", flush=True)
        return
    sanity_root = root / "sanity"
    sanity_root.mkdir(parents=True, exist_ok=True)
    source_annotation = base._path(preflight["full_test_annotation"]["path"])
    specs = (("SANITY_K16_G360", 16, 360, 29601), ("SANITY_K64_G720", 64, 720, 29602))
    record: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "v11_qdic_structure_runtime_sanity",
        "status": "RUNNING",
        "protocol": "ACTUAL_RUNTIME_STRUCTURAL_SANITY",
        "selection_use": "NONE",
        "preflight_sha256": base._sha256(root / "preflight.json"),
        "repo_head": preflight["repository"]["head"],
        "source_annotation": str(source_annotation),
        "trials": [],
        "started_at_unix": time.time(),
    }
    base._write_json(root / "sanity.json", record)
    state["runtime_sanity_status"] = "RUNNING"
    _persist_state(root, report_root, state)
    try:
        gpu = str(args.sanity_gpu)
        if gpu not in state["selected_gpus"]:
            raise ValueError(f"sanity GPU {gpu} is not among selected GPUs")
        for trial_id, candidate_top_k, max_gap, port in specs:
            trial_root = sanity_root / trial_id
            trial_root.mkdir(parents=True, exist_ok=True)
            smoke_annotation = trial_root / "annotation.json"
            annotation_summary = base._write_runtime_smoke_annotation(
                full_annotation=source_annotation,
                output=smoke_annotation,
                video_count=int(args.smoke_videos),
            )
            config_path = trial_root / "config.yaml"
            base._materialize_config(
                base_config=base._path(args.base_config),
                output=config_path,
                qdic_checkpoint=base._path(preflight["qdic"]["checkpoint"]),
                trial_id=trial_id,
                score_threshold=0.0,
                margin_threshold=MARGIN_DEFAULT,
                candidate_top_k=candidate_top_k,
                max_gap=max_gap,
                search_fields=["candidate_top_k", "max_gap"],
            )
            shard = {
                "index": 0,
                "path": str(smoke_annotation),
                "sha256": annotation_summary["sha256"],
                "video_count": annotation_summary["videos"],
                "frame_count": annotation_summary["frames"],
            }
            shard_dir, attempt = base._next_shard_dir(trial_root, 0)
            trial = {
                "trial_id": trial_id,
                "wave": "SANITY",
                "score_threshold": 0.0,
                "margin_threshold": MARGIN_DEFAULT,
                "candidate_top_k": candidate_top_k,
                "max_gap": max_gap,
                "master_port": port,
                "collect_event_diagnostics": True,
            }
            existing = base._completed_shard(trial_root, 0)
            if existing is None:
                gate = base._resource_gate(args, [gpu], purpose=trial_id)
                command = base._worker_command(
                    args=args,
                    preflight=preflight,
                    trial=trial,
                    shard=shard,
                    shard_dir=shard_dir,
                    attempt=attempt,
                    gpu=gpu,
                    tempo_config=config_path,
                )
                log_path = trial_root / f"worker_attempt{attempt:02d}.log"
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        command,
                        cwd=str(REPO),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                if process.returncode != 0:
                    raise RuntimeError(f"{trial_id} worker failed rc={process.returncode}; see {log_path}")
            else:
                gate = {"status": "REUSED_COMPLETED_SHARD"}
            existing = base._completed_shard(trial_root, 0)
            if existing is None:
                raise RuntimeError(f"{trial_id} did not produce a completed shard")
            shard_dir, receipt = existing
            base._validate_completed_shard_receipt(
                receipt=receipt, shard=shard, preflight=preflight, expected_spec=trial
            )
            diagnostics = base._read_json(shard_dir / "diagnostics.json")
            config = base._load_yaml(config_path)
            tempo = config.get("tempo", {})
            required = {
                "config_candidate_top_k": int(tempo.get("candidate_top_k", -1)) == candidate_top_k,
                "config_max_gap": int(tempo.get("max_gap", -1)) == max_gap,
                "diag_structural_candidate_top_k": int(diagnostics.get("structural_candidate_top_k", -1)) == candidate_top_k,
                "diag_structural_max_gap": int(diagnostics.get("structural_max_gap", -1)) == max_gap,
                "diag_qdic_decision_candidate_top_k": int(diagnostics.get("qdic_decision_candidate_top_k", -1)) == candidate_top_k,
                "diag_qdic_feature_decision_top_k": int(diagnostics.get("qdic_feature_decision_candidate_top_k", -1)) == 8,
                "diag_qdic_feature_max_gap": int(diagnostics.get("qdic_feature_max_gap", -1)) == 360,
                "event_file": (shard_dir / "event_diagnostics.jsonl").is_file(),
            }
            if not all(required.values()):
                raise RuntimeError(f"{trial_id} structural sanity checks failed: {required}")
            item = {
                "trial_id": trial_id,
                "status": "PASS",
                "candidate_top_k": candidate_top_k,
                "max_gap": max_gap,
                "annotation": annotation_summary,
                "resource_gate": gate,
                "config": str(config_path),
                "config_sha256": base._sha256(config_path),
                "worker_receipt": str(shard_dir / "receipt.json"),
                "diagnostics": str(shard_dir / "diagnostics.json"),
                "checks": required,
            }
            record["trials"].append(item)
            base._write_json(root / "sanity.json", record)
        record["status"] = "PASS"
        record["ended_at_unix"] = time.time()
        record["duration_seconds"] = record["ended_at_unix"] - record["started_at_unix"]
        base._write_json(root / "sanity.json", record)
        state["runtime_sanity_status"] = "PASS"
        state["runtime_sanity"] = record
        _persist_state(root, report_root, state)
        print("STRUCTURE_RUNTIME_SANITY: PASS", flush=True)
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        record["ended_at_unix"] = time.time()
        base._write_json(root / "sanity.json", record)
        state["runtime_sanity_status"] = "FAILED"
        state["error"] = record["error"]
        _persist_state(root, report_root, state)
        raise


def _run_candidate_thread(
    *, args: argparse.Namespace, state: Mapping[str, Any],
    preflight: Mapping[str, Any], trial_template: Mapping[str, Any], root: Path
) -> tuple[str, bool, dict[str, Any], str | None]:
    local_state = copy.deepcopy(dict(state))
    local_trial = base._trial(local_state, str(trial_template["trial_id"]))
    if local_trial is None:
        return str(trial_template["trial_id"]), False, dict(trial_template), "trial missing in local state"
    try:
        ok = base._run_candidate(
            args=args,
            state=local_state,
            preflight=preflight,
            trial=local_trial,
            root=root,
            persist_state=False,
        )
        return str(local_trial["trial_id"]), bool(ok), local_trial, None
    except Exception as exc:
        return str(local_trial["trial_id"]), False, local_trial, f"{type(exc).__name__}: {exc}"


def _run_full_wave(
    *, args: argparse.Namespace, root: Path, report_root: Path,
    preflight: Mapping[str, Any], state: dict[str, Any],
    trial_ids: list[str], wave_name: str
) -> None:
    selected = [
        item for item in state["trials"]
        if str(item.get("trial_id")) in set(trial_ids)
        and item.get("status") != "COMPLETED"
    ]
    if not selected:
        return
    gate = base._resource_gate(args, list(state["selected_gpus"]), purpose=f"{wave_name}_start")
    state["waves"].setdefault(wave_name, {})["resource_gate"] = gate
    state["waves"][wave_name]["status"] = "RUNNING_FULL_TEST"
    state["waves"][wave_name]["trial_ids"] = list(trial_ids)
    state["phase"] = wave_name
    for item in selected:
        item["status"] = "RUNNING_FULL_TEST"
        item.setdefault("full_started_at_unix", time.time())
    _persist_state(root, report_root, state)

    results: dict[str, tuple[bool, dict[str, Any], str | None]] = {}
    with ThreadPoolExecutor(max_workers=min(2, len(selected)), thread_name_prefix="v11-structure") as pool:
        futures = {
            pool.submit(
                _run_candidate_thread,
                args=args,
                state=state,
                preflight=preflight,
                trial_template=item,
                root=root,
            ): str(item["trial_id"])
            for item in selected
        }
        for future in as_completed(futures):
            trial_id, ok, local_trial, error = future.result()
            shared = base._trial(state, trial_id)
            if shared is None:
                continue
            shared.clear()
            shared.update(local_trial)
            if not ok:
                shared["status"] = "FAILED_OVERLAP_RETRY_PENDING"
                if error:
                    shared["error"] = error
            results[trial_id] = (bool(ok), dict(local_trial), error)
            _persist_state(root, report_root, state)

    failed = [trial_id for trial_id, (ok, _trial, _error) in results.items() if not ok]
    if failed:
        state["waves"][wave_name]["overlap_retry"] = True
        state["waves"][wave_name]["overlap_retry_trial_ids"] = failed
        _persist_state(root, report_root, state)
        print(
            f"{wave_name}: parallel candidate failure; retrying failed complete candidates sequentially on all ten GPUs",
            flush=True,
        )
        for trial_id in failed:
            trial = base._trial(state, trial_id)
            if trial is None:
                raise RuntimeError(f"missing failed trial {trial_id}")
            trial["status"] = "RUNNING_FULL_TEST_SEQUENTIAL_RETRY"
            _persist_state(root, report_root, state)
            ok = base._run_candidate(
                args=args, state=state, preflight=preflight, trial=trial,
                root=root, persist_state=True,
            )
            if not ok:
                trial["status"] = "FAILED"
                _persist_state(root, report_root, state)
                raise RuntimeError(f"{wave_name} candidate {trial_id} failed after sequential retry")
    for trial_id in trial_ids:
        trial = base._trial(state, trial_id)
        if trial is None or trial.get("status") == "COMPLETED":
            continue
        if trial.get("status") not in {"FULL_TEST_READY", "RUNNING_FULL_TEST"}:
            raise RuntimeError(f"{wave_name} candidate {trial_id} did not reach FULL_TEST_READY")
    state["waves"][wave_name]["status"] = "FULL_TEST_READY"
    _persist_state(root, report_root, state)


def _write_experiment_artifacts(
    *, args: argparse.Namespace, root: Path, report_root: Path,
    preflight: Mapping[str, Any], trial: dict[str, Any]
) -> None:
    exp_root = report_root / "per_experiment" / str(trial["trial_id"])
    exp_root.mkdir(parents=True, exist_ok=True)
    config = Path(str(trial.get("config", root / "trials" / trial["trial_id"] / "config.yaml")))
    if config.is_file():
        shutil.copy2(config, exp_root / "resolved_config.yaml")
    base._write_json(exp_root / "commands.json", {
        "trial_id": trial.get("trial_id"),
        "shard_commands": [item for item in trial.get("launched_shards", [])],
        "evaluation": trial.get("evaluation"),
        "merge": trial.get("merged"),
    })
    base._write_json(exp_root / "metrics.json", trial.get("metrics", {}))
    base._write_json(exp_root / "runtime.json", {
        "trial_id": trial.get("trial_id"),
        "candidate_top_k": trial.get("candidate_top_k"),
        "max_gap": trial.get("max_gap"),
        "score_threshold": trial.get("score_threshold"),
        "margin_threshold": trial.get("margin_threshold"),
        "full_duration_seconds": trial.get("full_duration_seconds"),
        "shards": trial.get("shards", []),
        "evaluation": trial.get("evaluation"),
        "resource_policy": state_resource_policy(args),
    })
    base._write_json(exp_root / "provenance.json", {
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "repository": preflight["repository"],
        "annotation": preflight["full_test_annotation"],
        "qdic": preflight["qdic"],
        "cov": preflight["cov"],
        "teta": preflight["teta"],
        "runtime_environment": preflight["runtime_environment"],
        "fixed_learned_feature_contract": {
            "decision_candidate_top_k": 8,
            "max_gap": 360,
        },
    })
    base._write_json(exp_root / "log_paths.json", {
        "candidate_root": trial.get("candidate_root"),
        "receipt": trial.get("receipt"),
        "worker_logs": [item.get("log") for item in trial.get("launched_shards", [])],
        "evaluation_log": trial.get("evaluation", {}).get("log") if isinstance(trial.get("evaluation"), Mapping) else None,
        "diagnostics": trial.get("diagnostics_path"),
    })
    trial["report_experiment_root"] = str(exp_root)


def state_resource_policy(args: argparse.Namespace) -> str:
    return base._resource_policy(args)


def _evaluate_and_diagnose(
    *, args: argparse.Namespace, root: Path, report_root: Path,
    preflight: Mapping[str, Any], state: dict[str, Any], trial: dict[str, Any]
) -> None:
    if trial.get("status") == "COMPLETED" and trial.get("metrics"):
        pass
    else:
        jobs: dict[str, subprocess.Popen[Any]] = {}
        process = base._start_evaluation(
            args=args, state=state, preflight=preflight, trial=trial
        )
        if process is not None:
            jobs[str(trial["trial_id"])] = process
        base._wait_for_evaluations(
            args=args, state=state, preflight=preflight, jobs=jobs
        )
        if trial.get("status") != "COMPLETED":
            raise RuntimeError(f"official TETA evaluation did not complete: {trial['trial_id']}")
    exp_root = report_root / "per_experiment" / str(trial["trial_id"])
    diagnostic_path = exp_root / "diagnostic_statistics.json"
    if not diagnostic_path.is_file():
        command = [
            str(args.worker_python or sys.executable),
            str(EVENT_DIAGNOSTICS_TOOL),
            "--annotation", str(preflight["full_test_annotation"]["path"]),
            "--candidate-root", str(trial["candidate_root"]),
            "--output", str(diagnostic_path),
        ]
        diagnostic_log = exp_root / "diagnostics.log"
        with diagnostic_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, text=True
            )
        if result.returncode != 0 or not diagnostic_path.is_file():
            raise RuntimeError(f"posthoc structural diagnostics failed for {trial['trial_id']}; see {diagnostic_log}")
        trial["diagnostics_command"] = command
        trial["diagnostics_log"] = str(diagnostic_log)
    trial["diagnostics_path"] = str(diagnostic_path)
    trial["diagnostics"] = base._read_json(diagnostic_path)
    trial["score_distribution"] = base._score_distribution(trial)
    _write_experiment_artifacts(
        args=args, root=root, report_root=report_root, preflight=preflight, trial=trial
    )
    _persist_state(root, report_root, state)


def _metric(trial: Mapping[str, Any], split: str, name: str) -> float:
    try:
        return float(trial["metrics"][split][name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _delta_metrics(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> dict[str, dict[str, float]] | None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return None
    result: dict[str, dict[str, float]] = {}
    for split in ("overall", "base", "novel"):
        if not isinstance(left.get(split), Mapping) or not isinstance(right.get(split), Mapping):
            continue
        result[split] = {
            name: float(left[split].get(name, float("nan"))) - float(right[split].get(name, float("nan")))
            for name in base.METRIC_NAMES
        }
    return result


def _valid_candidates(items: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        item for item in items
        if item.get("status") == "COMPLETED"
        and isinstance(item.get("metrics"), Mapping)
        and isinstance(item.get("metrics", {}).get("overall"), Mapping)
    ]


def _overall_champion(items: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    valid = _valid_candidates(items)
    if not valid:
        return None
    winner = max(
        valid,
        key=lambda item: (
            _metric(item, "overall", "TETA"),
            _metric(item, "novel", "TETA"),
            _metric(item, "overall", "AssocA"),
            _metric(item, "novel", "AssocA"),
            -int(item.get("candidate_top_k", 0)),
            -int(item.get("max_gap", 0)),
        ),
    )
    return dict(winner)


def _ov_champion(items: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    valid = _valid_candidates(items)
    if not valid:
        return None
    best_overall = max(_metric(item, "overall", "TETA") for item in valid)
    eligible = [item for item in valid if _metric(item, "overall", "TETA") >= best_overall - 0.05]
    winner = max(
        eligible,
        key=lambda item: (
            _metric(item, "novel", "TETA"),
            _metric(item, "novel", "AssocA"),
            _metric(item, "overall", "AssocA"),
            _metric(item, "overall", "TETA"),
            str(item.get("trial_id")),
        ),
    )
    return dict(winner)


def _load_baselines(args: argparse.Namespace, preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        return [dict(item) for item in base._baseline_catalog(args, preflight)]
    except Exception as exc:
        return [{"name": "baseline_catalog", "status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}]


def _bottleneck(candidate: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = candidate.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return {"label": "MIXED", "status": "UNAVAILABLE_DIAGNOSTICS", "evidence": {}}
    overall = diagnostics.get("groups", {}).get("overall", {})
    if not isinstance(overall, Mapping) or int(overall.get("association_events", 0)) == 0:
        return {"label": "MIXED", "status": "NO_RELIABLE_ASSOCIATION_DENOMINATOR", "evidence": {}}
    pre = (overall.get("prefilter_recall_at") or {}).get("64")
    post = (overall.get("qdic_recall_at") or {}).get("64")
    gaps = diagnostics.get("temporal_gap_bins", {})
    short = (gaps.get("gap_le_180") or {}).get("candidate_recall")
    long = (gaps.get("gap_361_720") or {}).get("candidate_recall")
    evidence = {
        "candidate_recall": overall.get("candidate_recall"),
        "prefilter_recall_at_64": pre,
        "qdic_recall_at_64": post,
        "gap_le_180_candidate_recall": short,
        "gap_361_720_candidate_recall": long,
        "accepted_unresolved_rate": (
            float(overall.get("accepted_unresolved", 0)) / float(overall.get("accepted_total", 1))
            if int(overall.get("accepted_total", 0)) else None
        ),
    }
    labels: list[str] = []
    if isinstance(overall.get("candidate_recall"), (int, float)) and float(overall["candidate_recall"]) < 0.70:
        labels.append("CANDIDATE_SUPPLY_LIMITED")
    if isinstance(pre, (int, float)) and isinstance(post, (int, float)) and float(pre) - float(post) > 0.08:
        labels.append("QDIC_RANKING_LIMITED")
    if isinstance(short, (int, float)) and isinstance(long, (int, float)) and float(short) - float(long) > 0.10:
        labels.append("TEMPORAL_HORIZON_LIMITED")
    if isinstance(evidence["accepted_unresolved_rate"], (int, float)) and float(evidence["accepted_unresolved_rate"]) > 0.25:
        labels.append("STALE_MEMORY_LIMITED")
    label = labels[0] if len(labels) == 1 else ("MIXED" if not labels else "MIXED")
    return {"label": label, "status": "DIAGNOSTIC_RULES_APPLIED", "components": labels, "evidence": evidence}


def _previous_margin_distributions() -> dict[str, Any]:
    value: dict[str, Any] = {}
    if not ANCHOR_DEFAULT.is_file():
        return value
    try:
        previous = base._read_json(ANCHOR_DEFAULT)
        for item in previous.get("candidates", []):
            trial_id = str(item.get("trial_id"))
            if trial_id not in {"FT_B", "FT_M80", "FT_M120", "FT_M60", "FT_M145"}:
                continue
            try:
                value[trial_id] = base._score_distribution(item)
            except Exception as exc:
                value[trial_id] = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        value["status"] = "UNAVAILABLE"
        value["error"] = f"{type(exc).__name__}: {exc}"
    return value


def _write_csvs(report_root: Path, anchor: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> None:
    all_items = [anchor] + candidates
    metric_fields = [
        "trial_id", "candidate_top_k", "max_gap", "status",
    ] + [f"{split}_{name}" for split in ("overall", "base", "novel") for name in base.METRIC_NAMES]
    rows = []
    for item in all_items:
        row = {field: "" for field in metric_fields}
        row.update({key: item.get(key, "") for key in ("trial_id", "candidate_top_k", "max_gap", "status")})
        for split in ("overall", "base", "novel"):
            for name in base.METRIC_NAMES:
                value = item.get("metrics", {}).get(split, {}).get(name) if isinstance(item.get("metrics"), Mapping) else None
                row[f"{split}_{name}"] = "" if value is None else value
        rows.append(row)
    for filename in ("candidate_k_analysis.csv", "temporal_gap_analysis.csv"):
        path = report_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=metric_fields)
            writer.writeheader()
            writer.writerows(rows)


def _format_metrics(metrics: Mapping[str, Any] | None) -> str:
    if not isinstance(metrics, Mapping):
        return "指标不可用。"
    lines = [
        "| Split | " + " | ".join(base.METRIC_NAMES) + " |",
        "|---|" + "---:|" * len(base.METRIC_NAMES),
    ]
    for split in ("overall", "base", "novel"):
        values = metrics.get(split, {})
        lines.append(
            f"| {split} | " + " | ".join(
                "NA" if not isinstance(values, Mapping) or values.get(name) is None
                else f"{float(values[name]):.3f}"
                for name in base.METRIC_NAMES
            ) + " |"
        )
    return "\n".join(lines)


def _comparison_table(champion: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    delta = _delta_metrics(champion.get("metrics"), baseline.get("metrics")) or {}
    lines = [
        f"### {champion.get('trial_id')} vs {baseline.get('name')}",
        "",
        "| Split | " + " | ".join(base.METRIC_NAMES) + " |",
        "|---|" + "---:|" * len(base.METRIC_NAMES),
    ]
    for split in ("overall", "base", "novel"):
        values = delta.get(split, {})
        lines.append(
            f"| {split} | " + " | ".join(
                "NA" if name not in values else f"{float(values[name]):+.3f}"
                for name in base.METRIC_NAMES
            ) + " |"
        )
    return "\n".join(lines)


def _write_final_outputs(
    *, args: argparse.Namespace, root: Path, report_root: Path,
    preflight: Mapping[str, Any], plan: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    anchor = _load_anchor(Path(args.anchor_results).resolve())
    candidates = [dict(item) for item in state.get("trials", [])]
    overall = _overall_champion(candidates)
    ov = _ov_champion(candidates)
    baselines = _load_baselines(args, preflight)
    original_baselines = [
        item for item in baselines
        if item.get("status") == "PASS"
        and item.get("comparison_eligible") is True
        and item.get("scope") == "FULL_TEST_ORIGINAL_BASELINE"
    ]
    if overall is None and anchor.get("status") == "COMPLETED":
        overall = dict(anchor)
    if ov is None and anchor.get("status") == "COMPLETED":
        ov = dict(anchor)
    bottleneck = None if overall is None else _bottleneck(overall)
    margin_distributions = _previous_margin_distributions()
    for item in candidates:
        if item.get("score_distribution") is None and item.get("status") == "COMPLETED":
            try:
                item["score_distribution"] = base._score_distribution(item)
            except Exception:
                pass
    results: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "v11_qdic_structure_search_results",
        "status": state.get("status"),
        "protocol": "TEST_TUNED_MODEL_SPECIFIC",
        "unbiased_test": False,
        "objective": "evaluate six runtime candidate_top_k/max_gap structures on complete tuned Test",
        "fixed_runtime": plan["fixed_runtime"],
        "learned_checkpoint_contract": plan["learned_checkpoint_contract"],
        "repository": preflight["repository"],
        "resources": {
            "selected_gpus": state.get("selected_gpus"),
            "resource_policy": state.get("resource_policy"),
            "candidate_parallelism": state.get("candidate_parallelism"),
            "waves": state.get("waves"),
        },
        "inputs": {
            "preflight": str(root / "preflight.json"),
            "preflight_sha256": base._sha256(root / "preflight.json"),
            "full_test_annotation": preflight["full_test_annotation"],
            "qdic": preflight["qdic"],
            "cov": preflight["cov"],
            "teta": preflight["teta"],
        },
        "runtime_sanity": {
            "path": str(root / "sanity.json"),
            "status": state.get("runtime_sanity_status"),
        },
        "anchor": anchor,
        "candidates": candidates,
        "champions": {
            "overall_teta_champion": None if overall is None else {
                "trial_id": overall.get("trial_id"),
                "spec": {key: overall.get(key) for key in ("candidate_top_k", "max_gap", "score_threshold", "margin_threshold")},
                "metrics": overall.get("metrics"),
                "diagnostics": overall.get("diagnostics"),
            },
            "novel_priority_within_0_05": None if ov is None else {
                "trial_id": ov.get("trial_id"),
                "spec": {key: ov.get(key) for key in ("candidate_top_k", "max_gap", "score_threshold", "margin_threshold")},
                "metrics": ov.get("metrics"),
                "diagnostics": ov.get("diagnostics"),
            },
        },
        "baselines": baselines,
        "original_baseline_comparisons": [
            {
                "against": item.get("name"),
                "scope": item.get("scope"),
                "overall_champion_delta": _delta_metrics(overall.get("metrics"), item.get("metrics")) if overall else None,
                "novel_priority_champion_delta": _delta_metrics(ov.get("metrics"), item.get("metrics")) if ov else None,
            }
            for item in original_baselines
        ],
        "bottleneck_diagnosis": bottleneck,
        "margin_score_distributions_from_previous_search": margin_distributions,
        "search": {
            "started_at_unix": state.get("formal_search_started_at_unix"),
            "ended_at_unix": state.get("formal_search_ended_at_unix"),
            "wall_hours": (
                None
                if state.get("formal_search_started_at_unix") is None or state.get("formal_search_ended_at_unix") is None
                else (float(state["formal_search_ended_at_unix"]) - float(state["formal_search_started_at_unix"])) / 3600.0
            ),
        },
    }
    base._write_json(report_root / "structure_search_results.json", results)
    base._write_json(root / "structure_search_results.json", results)
    _write_csvs(report_root, anchor, candidates)

    extension: dict[str, Any]
    overall_margin = None if overall is None else float(overall.get("margin_threshold", MARGIN_DEFAULT))
    ov_margin = None if ov is None else float(ov.get("margin_threshold", MARGIN_DEFAULT))
    if overall_margin is None or ov_margin is None or abs(overall_margin - ov_margin) <= 1e-12:
        extension = {
            "schema_version": 1,
            "artifact": "v11_qdic_structure_post_search_extension_plan",
            "status": "NOT_NEEDED_SINGLE_MARGIN",
            "auto_started": False,
            "reason": "All six structural candidates use the fixed audited Overall margin champion; the two champion definitions therefore resolve to the same margin in this round.",
            "candidates": [],
        }
    else:
        distribution_by_margin = {
            "overall_margin": margin_distributions.get("FT_B"),
            "ov_margin": margin_distributions.get("FT_M120"),
        }
        extension = {
            "schema_version": 1,
            "artifact": "v11_qdic_structure_post_search_extension_plan",
            "status": "PROPOSAL_ONLY",
            "auto_started": False,
            "reason": "Overall and Novel-priority champions selected different margins.",
            "score_distribution_source": distribution_by_margin,
            "candidates": [
                {"candidate_id": "overall_margin_SCORE_OFF", "margin": overall_margin, "score_source": "overall_margin_distribution.winner_score_min"},
                {"candidate_id": "overall_margin_P03", "margin": overall_margin, "score_source": "overall_margin_distribution.p03"},
                {"candidate_id": "ov_margin_SCORE_OFF", "margin": ov_margin, "score_source": "ov_margin_distribution.winner_score_min"},
                {"candidate_id": "ov_margin_P03", "margin": ov_margin, "score_source": "ov_margin_distribution.p03"},
            ],
            "do_not_run_P05_unless": "the first four complete Full-Test results show a clear upward score-threshold trend",
        }
    base._write_json(root / "post_search_extension_plan.json", extension)
    base._write_json(report_root / "post_search_extension_plan.json", extension)

    lines = [
        "# TempoTrack V11 QDIC-MO structural Full-Test search",
        "",
        "> **TEST_TUNED_MODEL_SPECIFIC**  ",
        "> **NOT_UNBIASED_TEST**",
        "",
        "本轮在完整 tuned Test 上比较六个结构候选；Test 指标参与结构选择，因此不能解释为无偏 Test 估计。所有候选均使用直接 COVTrack 前端，未使用 frontend replay cache、GT online selection 或 retraining。",
        "",
        "## Execution contract",
        "",
        f"- Status: `{state.get('status')}`",
        f"- Repository: `{preflight['repository']['branch']}` @ `{preflight['repository']['head']}`",
        f"- Resource policy: `{state.get('resource_policy')}`; selected GPUs: `{','.join(state.get('selected_gpus', []))}`; max complete candidates concurrently: `2`",
        f"- Runtime sanity: `{state.get('runtime_sanity_status')}`",
        f"- QDIC learned feature contract: decision K=`8`, feature max_gap=`360`",
        "",
        "## Candidates",
        "",
        "| Trial | K | max_gap | Overall TETA | Base TETA | Novel TETA | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    if anchor:
        lines.append(
            f"| {anchor.get('trial_id')} (anchor) | {anchor.get('candidate_top_k')} | {anchor.get('max_gap')} | "
            f"{_metric(anchor, 'overall', 'TETA'):.3f} | {_metric(anchor, 'base', 'TETA'):.3f} | {_metric(anchor, 'novel', 'TETA'):.3f} | {anchor.get('status')} |"
        )
    for item in candidates:
        lines.append(
            f"| {item.get('trial_id')} | {item.get('candidate_top_k')} | {item.get('max_gap')} | "
            f"{_metric(item, 'overall', 'TETA'):.3f} | {_metric(item, 'base', 'TETA'):.3f} | {_metric(item, 'novel', 'TETA'):.3f} | {item.get('status')} |"
        )
    lines.extend(["", "## Overall TETA champion", ""])
    if overall:
        lines.extend([
            f"`{overall.get('trial_id')}`: K=`{overall.get('candidate_top_k')}`, max_gap=`{overall.get('max_gap')}`, score=`{overall.get('score_threshold')}`, margin=`{overall.get('margin_threshold')}`.",
            "",
            _format_metrics(overall.get("metrics")),
            "",
        ])
    else:
        lines.append("没有官方评估通过的结构候选。\n")
    lines.extend(["## Novel-priority champion within Overall TETA -0.05", ""])
    if ov:
        lines.extend([f"`{ov.get('trial_id')}`", "", _format_metrics(ov.get("metrics")), ""])
    else:
        lines.append("没有可用候选。\n")
    lines.extend(["## Original-baseline comparisons", "", "以下 delta 全部相对于真正的 original Full-Test baseline；不与子集或 anchor 做差值。", ""])
    for item in original_baselines:
        if overall:
            lines.extend([_comparison_table(overall, item), ""])
    if not original_baselines:
        lines.append("没有可用的 original Full-Test baseline。\n")
    lines.extend(["## Candidate-supply and QDIC-rank diagnostics", "", "| Trial | Group | Association events | Positive | Candidate recall | Prefilter R@64 | QDIC R@64 | Assoc recall | Assoc precision | False merge |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for item in candidates:
        diag = item.get("diagnostics", {})
        group = diag.get("groups", {}).get("overall", {}) if isinstance(diag, Mapping) else {}
        lines.append(
            f"| {item.get('trial_id')} | overall | {group.get('association_events', 'NA')} | {group.get('positive_events', 'NA')} | {fmt(group.get('candidate_recall'))} | {fmt((group.get('prefilter_recall_at') or {}).get('64'))} | {fmt((group.get('qdic_recall_at') or {}).get('64'))} | {fmt(group.get('association_recall'))} | {fmt(group.get('association_precision'))} | {group.get('false_merge', 'NA')} |"
        )
    lines.extend(["", "## Temporal gap diagnostics", ""])
    for item in candidates:
        diag = item.get("diagnostics", {})
        lines.append(f"### {item.get('trial_id')}")
        lines.append("")
        lines.append("| Gap bin | Events | Positive | Candidate recall | Accepted correct | False merge | Assoc recall | Assoc precision |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name in ("gap_le_180", "gap_181_360", "gap_361_720", "gap_gt_720"):
            group = diag.get("temporal_gap_bins", {}).get(name, {}) if isinstance(diag, Mapping) else {}
            lines.append(
                f"| {name} | {group.get('association_events', 'NA')} | {group.get('positive_events', 'NA')} | {fmt(group.get('candidate_recall'))} | {group.get('accepted_correct', 'NA')} | {group.get('false_merge', 'NA')} | {fmt(group.get('association_recall'))} | {fmt(group.get('association_precision'))} |"
            )
        lines.append("")
    lines.extend(["## Bottleneck diagnosis", ""])
    if bottleneck:
        lines.append(f"- Label: `{bottleneck.get('label')}`")
        lines.append(f"- Status: `{bottleneck.get('status')}`")
        lines.append(f"- Evidence: `{json.dumps(bottleneck.get('evidence', {}), ensure_ascii=False, sort_keys=True)}`")
    else:
        lines.append("诊断不可用。")
    lines.extend(["", "## Next research suggestion", ""])
    if bottleneck and bottleneck.get("label") == "CANDIDATE_SUPPLY_LIMITED":
        lines.append("优先分析候选生成/预过滤供给；不要继续把 score/margin threshold 当作全局最优搜索。")
    elif bottleneck and bottleneck.get("label") == "QDIC_RANKING_LIMITED":
        lines.append("优先分析 QDIC rank/feature calibration；候选已供给但排序损失明显。")
    elif bottleneck and bottleneck.get("label") == "TEMPORAL_HORIZON_LIMITED":
        lines.append("优先分析长 gap 的 memory freshness 与 temporal aggregation；结构 max_gap 已表现为主要限制。")
    elif bottleneck and bottleneck.get("label") == "STALE_MEMORY_LIMITED":
        lines.append("优先分析 stale memory/identity-to-category aggregation；不要继续无限增加 threshold 点。")
    else:
        lines.append("当前证据呈 mixed 或不足以唯一归因；下一步应做 targeted model/track classification 与 identity-to-category aggregation 分析。")
    lines.extend(["", "## Output index", "", f"- Manifest: `{root / 'structure_search_manifest.json'}`", f"- Status: `{report_root / 'structure_search_status.json'}`", f"- Results: `{report_root / 'structure_search_results.json'}`", f"- Per-experiment diagnostics: `{report_root / 'per_experiment'}`", f"- Extension proposal (not started): `{report_root / 'post_search_extension_plan.json'}`", ""])
    _write_text(report_root / "structure_search_summary.md", "\n".join(lines))
    return results


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        root=str(ROOT_DEFAULT),
        candidate_parallelism=2,
        max_candidates=6,
        include_op00=False,
        smoke_gpu="1",
    )
    parser.add_argument("--report-root", default=str(REPORT_DEFAULT))
    parser.add_argument("--anchor-results", default=str(ANCHOR_DEFAULT))
    parser.add_argument("--sanity-gpu", default="1")
    parser.add_argument("--sanity-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.root = str(base._path(args.root))
    args.repo = str(base._path(args.repo))
    report_root = base._path(args.report_root)
    repo = base._path(args.repo)
    gpus = base._parse_gpu_ids(args.gpus)
    if gpus != [str(value) for value in range(10)]:
        raise ValueError("this structural search requires all physical GPUs 0-9")
    if int(args.candidate_parallelism) != 2:
        raise ValueError("the formal structural protocol requires candidate_parallelism=2")
    if not args.allow_gpu_overlap:
        raise ValueError("the formal structural protocol requires --allow-gpu-overlap")
    if args.wait_for_gpus:
        raise ValueError("use --allow-gpu-overlap for this structural protocol, not --wait-for-gpus")
    root = base._path(args.root)
    state_path = root / "search_state.json"
    root.mkdir(parents=True, exist_ok=True)
    try:
        if state_path.is_file():
            if not args.resume:
                raise FileExistsError(f"structure search state exists; pass --resume: {state_path}")
            preflight, plan = _validate_structure_resume(args=args, root=root, repo=repo)
            state = base._read_json(state_path)
            if state.get("preflight_sha256") != base._sha256(root / "preflight.json"):
                raise RuntimeError("RESUME_PROVENANCE_FAIL: state preflight hash is stale")
            if state.get("search_plan_sha256") != base._sha256(root / "search_plan.json"):
                raise RuntimeError("RESUME_PROVENANCE_FAIL: state search-plan hash is stale")
            if state.get("selected_gpus") != gpus:
                raise RuntimeError("RESUME_PROVENANCE_FAIL: selected GPU set changed")
        else:
            if args.resume:
                raise FileNotFoundError(f"structure search state missing: {state_path}")
            preflight, plan, _manifest = _write_structure_preflight(
                args=args, root=root, repo=repo
            )
            report_root.mkdir(parents=True, exist_ok=True)
            _write_text(report_root / "structure_search_config_diff.md", _config_diff(plan, preflight))
            base._write_json(report_root / "structure_search_manifest.json", base._read_json(root / "structure_search_manifest.json"))
            state = _new_state(
                args=args, root=root, report_root=report_root,
                preflight=preflight, plan=plan,
            )
            _persist_state(root, report_root, state)
        report_root.mkdir(parents=True, exist_ok=True)
        if not (report_root / "structure_search_config_diff.md").is_file():
            _write_text(report_root / "structure_search_config_diff.md", _config_diff(plan, preflight))
        if not (report_root / "structure_search_manifest.json").is_file() and (root / "structure_search_manifest.json").is_file():
            base._write_json(report_root / "structure_search_manifest.json", base._read_json(root / "structure_search_manifest.json"))
        if state.get("runtime_sanity_status") != "PASS":
            _run_runtime_sanity(
                args=args, root=root, report_root=report_root,
                preflight=preflight, state=state,
            )
        if args.sanity_only:
            state["status"] = "SANITY_COMPLETED"
            _persist_state(root, report_root, state)
            return 0
        if state.get("status") == "COMPLETED":
            _write_final_outputs(
                args=args, root=root, report_root=report_root,
                preflight=preflight, plan=plan, state=state,
            )
            return 0
        start = state.get("formal_search_started_at_unix")
        if start is None:
            start = time.time()
            state["formal_search_started_at_unix"] = start
            state["status"] = "RUNNING"
            state["phase"] = "W1"
            base._append_event(state, "formal_search_started", started_at_unix=start)
            _persist_state(root, report_root, state)
        waves = plan["waves"]
        for wave_name in ("W1", "W2", "W3"):
            trial_ids = [str(item) for item in waves[wave_name]]
            _run_full_wave(
                args=args, root=root, report_root=report_root,
                preflight=preflight, state=state,
                trial_ids=trial_ids, wave_name=wave_name,
            )
            for trial_id in trial_ids:
                trial = base._trial(state, trial_id)
                if trial is None:
                    raise RuntimeError(f"missing planned trial {trial_id}")
                if trial.get("status") == "COMPLETED" and trial.get("diagnostics_path"):
                    continue
                _evaluate_and_diagnose(
                    args=args, root=root, report_root=report_root,
                    preflight=preflight, state=state, trial=trial,
                )
                _persist_state(root, report_root, state)
            state["waves"].setdefault(wave_name, {})["status"] = "COMPLETED"
            _persist_state(root, report_root, state)
        state["status"] = "COMPLETED"
        state["phase"] = "AGGREGATE"
        state["formal_search_ended_at_unix"] = time.time()
        base._append_event(state, "formal_search_completed", ended_at_unix=state["formal_search_ended_at_unix"])
        _persist_state(root, report_root, state)
        _write_final_outputs(
            args=args, root=root, report_root=report_root,
            preflight=preflight, plan=plan, state=state,
        )
        print(f"V11_STRUCTURE_FULL_TEST_SEARCH_COMPLETED root={root}", flush=True)
        return 0
    except Exception as exc:
        if "state" in locals() and isinstance(state, dict):
            state["status"] = "FAILED"
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["traceback"] = traceback.format_exc()
            try:
                _persist_state(root, report_root, state)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
