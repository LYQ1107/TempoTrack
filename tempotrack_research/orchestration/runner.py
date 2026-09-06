"""Real, resumable DAG runner for the repair suite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..config import file_hash, object_hash, load_yaml
from ..evaluation.result_writer import append_jsonl
from ..registry import SCHEMES, RESEARCH_SCHEMES, get_scheme
from .gates import run_repair_audit


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _implementation_hash(repo: Path) -> str:
    files = sorted((repo / "tempotrack_research").rglob("*.py"))
    return object_hash({str(path.relative_to(repo)): file_hash(path) for path in files})


def _append_job(repo: Path, record: Mapping[str, Any]) -> None:
    append_jsonl(record, repo / "reports" / "repair_jobs.jsonl")


def _run_subprocess(repo: Path, command: list[str], *, env: Mapping[str, str] | None = None, log_path: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _now()
    start_clock = time.time()
    child_env = os.environ.copy()
    child_env.update({str(key): str(value) for key, value in (env or {}).items()})
    record = dict(job) | {"status": "RUNNING", "started_at": started, "command": command, "log_path": str(log_path), "pid": None}
    _append_job(repo, record)
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=str(repo), env=child_env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            record["pid"] = process.pid
            _append_job(repo, record)
            returncode = process.wait()
    except OSError as exc:
        returncode = 127
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{type(exc).__name__}: {exc}\n")
    status = "COMPLETED" if returncode == 0 else ("BLOCKED_EXTERNAL" if returncode == 127 else "FAILED")
    result = record | {"status": status, "finished_at": _now(), "exit_code": returncode, "duration_seconds": time.time() - start_clock}
    _append_job(repo, result)
    return result


def _resolved_command(repo: Path, local: Mapping[str, Any], *args: str) -> tuple[list[str], dict[str, str]]:
    python = Path(str(local.get("research_python") or sys.executable))
    command = [str(python), "-m", "tempotrack_research.cli", *args]
    env: dict[str, str] = {}
    preload = local.get("legacy_env", {}).get("ld_preload") if isinstance(local.get("legacy_env"), Mapping) else None
    if preload:
        env["LD_PRELOAD"] = str(preload)
    return command, env


def _scheme_signature(repo: Path, suite: Mapping[str, Any], local: Mapping[str, Any], scheme: str, profile: str) -> str:
    item = get_scheme(scheme)
    data_manifest = local.get("data", {}) if isinstance(local.get("data"), Mapping) else {}
    return object_hash({"scheme": item.name, "frontend": item.frontend, "method": item.method, "phase": item.phase, "profile": profile, "suite_hash": object_hash(suite), "local_hash": object_hash(local), "code_hash": _implementation_hash(repo), "data_manifest": data_manifest})


def _scheme_args(scheme: str, profile: str, local_path: Path, suite_path: Path, *, seed: int = 0, episodes: Path | None = None, bc_checkpoint: Path | None = None, max_steps: int | None = None, phase_override: str | None = None, run_root: Path | None = None) -> list[str]:
    item = get_scheme(scheme)
    args = ["train", "--method", item.method, "--frontend", item.frontend, "--profile", profile, "--seed", str(int(seed)), "--scheme", scheme, "--local", str(local_path), "--suite", str(suite_path), "--resume", "auto"]
    if episodes is not None:
        args.extend(["--episodes", str(episodes)])
    if bc_checkpoint is not None:
        args.extend(["--bc-checkpoint", str(bc_checkpoint)])
    if phase_override or item.phase:
        args.extend(["--phase", phase_override or item.phase])
    if max_steps is not None:
        args.extend(["--max-steps", str(int(max_steps))])
    if run_root is not None:
        args.extend(["--run-root", str(run_root)])
    return args


def _bc_checkpoint(run_root: Path, frontend: str, seed: int = 0) -> Path | None:
    roots = [run_root] if run_root.name == "runs" else [run_root / "runs", run_root]
    candidates = [root / f"{frontend}_s5_rl_edit_bc_seed{int(seed)}" / "last.pt" for root in roots]
    for root in roots:
        candidates.extend(sorted(root.glob(f"{frontend}_s5_rl_edit_bc_seed{int(seed)}/last.pt")))
    return next((path for path in candidates if path.exists()), None)


def _gate_passed(payload: Mapping[str, Any] | None) -> bool:
    if not payload:
        return False
    results = payload.get("results", [])
    return bool(results) and all(item.get("status") == "PASS" for item in results)


def _completed_profile_artifact(profile_root: Path, item: Any, suite: Mapping[str, Any], profile: str, seed: int = 0) -> Path | None:
    """Return a valid completed artifact that can be safely reused."""
    phase = str(item.phase or "train")
    path = profile_root / "runs" / f"{item.frontend}_{item.method}_{phase}_seed{int(seed)}" / "train_result.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        resolved = json.loads((path.parent / "resolved_run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if result.get("status") != "COMPLETED" or resolved.get("profile") != profile:
        return None
    if resolved.get("method") != item.method or resolved.get("frontend") != item.frontend or str(resolved.get("phase")) != phase:
        return None
    training = suite.get("training", {}) if isinstance(suite.get("training", {}), Mapping) else {}
    budgets = training.get("full_budgets" if profile == "full" else "budgets", {})
    if phase == "ppo":
        required = int(budgets.get("s5_ppo_transitions", 0))
        return path if int(result.get("transitions", 0)) >= required > 0 else None
    required = int(budgets.get("s5_bc", 0) if phase == "bc" else budgets.get(item.method, 0))
    return path if int(result.get("optimizer_steps", 0)) >= required > 0 else None


def run_suite(repo: str | Path, suite_path: str | Path, local_path: str | Path, *, stage: str = "all", verification: str = "build", resume: str = "auto", keep_going: bool = True, max_steps: int | None = None, run_root: str | Path | None = None, require_gates: str | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve()
    suite_path = Path(suite_path).resolve()
    local_path = Path(local_path).resolve()
    suite = load_yaml(suite_path)
    local = load_yaml(local_path)
    configured_root = Path(run_root).resolve() if run_root is not None else Path(str(local.get("run_root", repo / "outputs/research_v2"))).resolve()
    output: dict[str, Any] = {"schema_version": 2, "started_at": _now(), "stage": stage, "verification": verification, "run_root": str(configured_root), "required_gates": require_gates, "jobs": [], "blocked": []}
    implementation = _implementation_hash(repo)
    episodes_path = configured_root / "episodes" / "train_base" / "episodes_manifest.json"
    prepared_path = configured_root / "prepared" / "prepared_manifest.json"

    def _finish() -> dict[str, Any]:
        output["finished_at"] = _now()
        output["runner_hash"] = object_hash(output)
        report_path = repo / "reports" / "repair_suite_last.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return output

    def _full_shared_data_ready() -> bool:
        prepared = json.loads(prepared_path.read_text(encoding="utf-8")) if prepared_path.exists() else {}
        manifests = prepared.get("dataset_manifests", {}) if isinstance(prepared, Mapping) else {}
        required = {"train_base", "val_base_internal", "official_validation"}
        if not required.issubset(manifests):
            return False
        train_manifest = Path(str(manifests["train_base"]))
        internal_manifest = Path(str(manifests["val_base_internal"]))
        official_manifest = Path(str(manifests["official_validation"]))
        if not all(path.exists() for path in (train_manifest, internal_manifest, official_manifest)):
            return False
        limits = prepared.get("video_limits", {})
        return (
            int(limits.get("train_base", 0)) > int(suite.get("execution", {}).get("integration_limit_videos", 2))
            and int(limits.get("val_base_internal", 0)) > 1
            and len(json.loads(official_manifest.read_text(encoding="utf-8")).get("video_ids", [])) > 1
        )
    # G0/G1 are always real subprocess/in-process checks and never a READY
    # placeholder.  Later stages reuse their artifacts.
    if stage in {"static", "integration", "trial", "full", "all"}:
        output["g0"] = run_repair_audit(repo, level="static", output=repo / "reports" / "repair_gates.json", run_root=configured_root)
    if stage in {"build", "integration", "trial", "full", "all"}:
        command, env = _resolved_command(repo, local, "build-check", "--repo", str(repo), "--changed-only")
        output["jobs"].append(_run_subprocess(repo, command, env=env, log_path=repo / "reports" / "repair_logs" / "G1_build_check.log", job={"job_id": "G1_build_check", "stage": "G1", "code_hash": implementation}))
        if output["jobs"][-1]["status"] != "COMPLETED" and not keep_going:
            output["finished_at"] = _now(); return output
    if stage in {"integration", "all"}:
        limit = suite.get("execution", {}).get("integration_limit_videos", 2)
        command, env = _resolved_command(repo, local, "prepare", "--repo", str(repo), "--suite", str(suite_path), "--local", str(local_path), "--split", "train_base,val_base_internal", "--limit-videos", str(int(limit)), "--run-root", str(configured_root), "--resume", resume)
        output["jobs"].append(_run_subprocess(repo, command, env=env, log_path=repo / "reports" / "repair_logs" / "G2_prepare.log", job={"job_id": "G2_prepare", "stage": "G2", "code_hash": implementation}))
        prepared = prepared_path
        source_manifest = None
        if prepared.exists():
            try:
                source_manifest = json.loads(prepared.read_text(encoding="utf-8")).get("dataset_manifests", {}).get("train_base")
            except (OSError, ValueError):
                source_manifest = None
        output["g2_gate"] = run_repair_audit(repo, level="integration", source_manifest=source_manifest, output=repo / "reports" / "repair_gates.json", run_root=configured_root)
        if require_gates == "pretrial" and not _gate_passed(output["g2_gate"]):
            output["blocked"].append({"stage": "G4", "status": "BLOCKED_GATE", "evidence": "G2 integration gate did not pass"})
            output["finished_at"] = _now()
            return output
        command, env = _resolved_command(repo, local, "build-episodes", "--repo", str(repo), "--suite", str(suite_path), "--local", str(local_path), "--split", "train_base", "--kinds", "memory,pair,continuation,graph,edit", "--run-root", str(configured_root), "--resume", resume)
        output["jobs"].append(_run_subprocess(repo, command, env=env, log_path=repo / "reports" / "repair_logs" / "G2_build_episodes.log", job={"job_id": "G2_build_episodes", "stage": "G2", "code_hash": implementation}))
    if stage in {"trial", "full"}:
        if not _full_shared_data_ready():
            output["blocked"].append({"stage": "G4" if stage == "trial" else "G5", "status": "BLOCKED_DATA", "evidence": f"full shared train_base/val_base_internal/official_validation export is not ready at {prepared_path}; integration subset cannot satisfy {stage}"})
            return _finish()
        prepared = prepared_path
        source_manifest = None
        if prepared.exists():
            try:
                source_manifest = json.loads(prepared.read_text(encoding="utf-8")).get("dataset_manifests", {}).get("train_base")
            except (OSError, ValueError):
                source_manifest = None
        output["g2_gate"] = run_repair_audit(repo, level="integration", source_manifest=source_manifest, output=repo / "reports" / "repair_gates.json", run_root=configured_root)
        if require_gates == "pretrial" and not _gate_passed(output["g2_gate"]):
            output["blocked"].append({"stage": "G4", "status": "BLOCKED_GATE", "evidence": "G2 integration gate did not pass"})
            return _finish()
        command, env = _resolved_command(repo, local, "build-episodes", "--repo", str(repo), "--suite", str(suite_path), "--local", str(local_path), "--split", "train_base", "--kinds", "memory,pair,continuation,graph,edit", "--run-root", str(configured_root), "--resume", resume)
        output["jobs"].append(_run_subprocess(repo, command, env=env, log_path=repo / "reports" / "repair_logs" / "G2_build_episodes.full_shared.log", job={"job_id": "G2_build_episodes.full_shared", "stage": "G2", "code_hash": implementation}))
    if stage == "trial":
        # Keep integration smoke checkpoints out of the trial lineage, and
        # keep trial checkpoints out of the full lineage.  The episode/data
        # root is shared, but optimizer state and resume cursors are
        # profile-scoped so a stale checkpoint can never be resumed against a
        # different profile or data hash.
        profile_root = configured_root / ("trial" if stage == "trial" else "full")
        for scheme in RESEARCH_SCHEMES:
            item = get_scheme(scheme)
            if not item.trainable:
                continue
            profile = "trial"
            current_signature = _scheme_signature(repo, suite, local, scheme, profile)
            if bool(suite.get("execution", {}).get("skip_completed_same_signature", True)):
                reusable = _completed_profile_artifact(profile_root, item, suite, profile)
                if reusable is not None:
                    output["jobs"].append({"job_id": f"{scheme}.trial.seed0", "scheme": scheme, "stage": "G4", "status": "COMPLETED", "reused": True, "reuse_reason": "completed_profile_artifact_meets_declared_budget", "checkpoint_path": str(reusable.parent / "last.pt"), "result_path": str(reusable), "run_signature": current_signature})
                    continue
            bc_checkpoint = None
            if item.phase == "ppo":
                bc_checkpoint = _bc_checkpoint(profile_root, item.frontend)
                if bc_checkpoint is None:
                    # PPO has a real BC dependency.  Build it with the same
                    # frontend and episode manifest instead of passing a
                    # random/uninitialised policy to the rollout loop.
                    bc_args = _scheme_args(
                        scheme, profile, local_path, suite_path,
                        episodes=episodes_path if episodes_path.exists() else None,
                        phase_override="bc", max_steps=max_steps, run_root=profile_root,
                    )
                    bc_args[bc_args.index("--phase") + 1] = "bc"
                    bc_args.remove("--scheme")
                    bc_args.remove(scheme)
                    bc_command, bc_env = _resolved_command(repo, local, *bc_args)
                    bc_result = _run_subprocess(repo, bc_command, env=bc_env, log_path=repo / "reports" / "repair_logs" / f"{scheme}.bc_dependency.log", job={"job_id": f"{scheme}.bc_dependency.seed0", "scheme": scheme, "stage": "G4", "phase": "bc", "code_hash": implementation})
                    output["jobs"].append(bc_result)
                    bc_checkpoint = _bc_checkpoint(profile_root, item.frontend)
            command, env = _resolved_command(repo, local, *_scheme_args(scheme, profile, local_path, suite_path, episodes=episodes_path if episodes_path.exists() else None, bc_checkpoint=bc_checkpoint, max_steps=max_steps, run_root=profile_root))
            result = _run_subprocess(repo, command, env=env, log_path=repo / "reports" / "repair_logs" / f"{scheme}.trial.log", job={"job_id": f"{scheme}.trial.seed0", "scheme": scheme, "stage": "G4", "run_signature": current_signature, "code_hash": implementation})
            output["jobs"].append(result)
            if result["status"] != "COMPLETED":
                output["blocked"].append({"scheme": scheme, "stage": "G4", "status": result["status"], "log": result["log_path"]})
                if not keep_going:
                    break
        output["g4_gate"] = run_repair_audit(repo, level="trial", source_manifest=source_manifest, output=repo / "reports" / "repair_gates.json", run_root=configured_root)
    if stage in {"full", "all"}:
        # Full budgets are launched only when explicitly requested.  The
        # caller may pass max_steps for a documented bounded experiment; the
        # job still records its actual budget and checkpoint.
        run_full = stage == "full" or bool(suite.get("execution", {}).get("run_full", False))
        if stage == "full":
            # The trial gate is an input to G5, not another full-root
            # training pass.  Re-evaluate it against the shared real
            # manifest so the full lineage can safely reuse the completed
            # trial artifacts without mixing profile metadata.
            output["g4_gate"] = run_repair_audit(
                repo,
                level="trial",
                source_manifest=source_manifest,
                output=repo / "reports" / "repair_gates.json",
                run_root=configured_root,
            )
        existing_gate = output.get("g4_gate")
        if existing_gate is None:
            try:
                existing_gate = json.loads((repo / "reports" / "repair_gates.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing_gate = None
        if run_full and require_gates == "prefull" and not _gate_passed(existing_gate):
            output["blocked"].append({"stage": "G5", "status": "BLOCKED_GATE", "evidence": "G4 trial gate did not pass"})
            run_full = False
        if run_full:
            profile_root = configured_root / "full"
            training = suite.get("training", {}) if isinstance(suite.get("training", {}), Mapping) else {}
            configured_seeds = training.get("full_seeds", [0])
            full_seeds = [int(seed) for seed in configured_seeds] or [0]
            for scheme in RESEARCH_SCHEMES:
                item = get_scheme(scheme)
                if not item.trainable:
                    continue
                for seed in full_seeds:
                    current_signature = _scheme_signature(repo, suite, local, scheme, "full")
                    if bool(suite.get("execution", {}).get("skip_completed_same_signature", True)):
                        reusable = _completed_profile_artifact(profile_root, item, suite, "full", seed)
                        if reusable is not None:
                            output["jobs"].append({"job_id": f"{scheme}.full.seed{seed}", "scheme": scheme, "stage": "G5", "status": "COMPLETED", "reused": True, "reuse_reason": "completed_profile_artifact_meets_declared_budget", "checkpoint_path": str(reusable.parent / "last.pt"), "result_path": str(reusable), "run_signature": current_signature, "seed": seed})
                            continue
                    bc_checkpoint = None
                    if item.phase == "ppo":
                        bc_checkpoint = _bc_checkpoint(profile_root, item.frontend, seed)
                        if bc_checkpoint is None:
                            bc_args = _scheme_args(scheme, "full", local_path, suite_path, seed=seed, episodes=episodes_path if episodes_path.exists() else None, phase_override="bc", max_steps=max_steps, run_root=profile_root)
                            if "--scheme" in bc_args:
                                index = bc_args.index("--scheme")
                                del bc_args[index:index + 2]
                            bc_command, bc_env = _resolved_command(repo, local, *bc_args)
                            bc_result = _run_subprocess(repo, bc_command, env=bc_env, log_path=repo / "reports" / "repair_logs" / f"{scheme}.full.seed{seed}.bc_dependency.log", job={"job_id": f"{scheme}.full_bc_dependency.seed{seed}", "scheme": scheme, "stage": "G5", "phase": "bc", "seed": seed, "code_hash": implementation})
                            output["jobs"].append(bc_result)
                            bc_checkpoint = _bc_checkpoint(profile_root, item.frontend, seed)
                    command, env = _resolved_command(repo, local, *_scheme_args(scheme, "full", local_path, suite_path, seed=seed, episodes=episodes_path if episodes_path.exists() else None, bc_checkpoint=bc_checkpoint, max_steps=max_steps, run_root=profile_root))
                    result = _run_subprocess(repo, command, env=env, log_path=repo / "reports" / "repair_logs" / f"{scheme}.full.seed{seed}.log", job={"job_id": f"{scheme}.full.seed{seed}", "scheme": scheme, "stage": "G5", "seed": seed, "run_signature": current_signature, "code_hash": implementation})
                    output["jobs"].append(result)
                    if result["status"] != "COMPLETED" and not keep_going:
                        break
        else:
            output["blocked"].append({"stage": "G5", "status": "NOT_LAUNCHED", "evidence": "suite.execution.run_full=false; full 60k--120k/2M-transition budgets were not substituted by a status table"})
        output["g5_gate"] = run_repair_audit(repo, level="full", output=repo / "reports" / "repair_gates.json", run_root=configured_root)
    return _finish()


__all__ = ["_implementation_hash", "run_suite"]
