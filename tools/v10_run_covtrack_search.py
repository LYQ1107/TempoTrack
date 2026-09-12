#!/usr/bin/env python3
"""Run independent COVTrack V10 trial workers with auditable heartbeats."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

try:
    from v10_search_covtrack_full_test import (
        _load_search_plan,
        _sha256,
        _validate_contract_gate,
        _resolve_teta_source_root,
        _teta_dependency,
        _run_teta_import_preflight,
        default_trial_specs,
        _git_value,
        _write_json,
    )
except ModuleNotFoundError:  # import-safe when loaded as tools.v10_run_covtrack_search
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "_v10_search_covtrack_full_test", Path(__file__).with_name("v10_search_covtrack_full_test.py")
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("cannot load sibling v10_search_covtrack_full_test.py")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
    _load_search_plan = _module._load_search_plan
    _sha256 = _module._sha256
    _validate_contract_gate = _module._validate_contract_gate
    _resolve_teta_source_root = _module._resolve_teta_source_root
    _teta_dependency = _module._teta_dependency
    _run_teta_import_preflight = _module._run_teta_import_preflight
    default_trial_specs = _module.default_trial_specs
    _git_value = _module._git_value
    _write_json = _module._write_json


def _load_requested_specs(path: Path | None, trial_ids: set[str] | None) -> list[dict[str, Any]]:
    specs = list(_load_search_plan(path).trials) if path is not None else default_trial_specs()
    if trial_ids is None:
        return specs
    return [item for item in specs if str(item.get("trial_id")) in trial_ids]


def _build_command(
    args: argparse.Namespace,
    spec: dict[str, Any],
    gpu: str,
    plan: Any | None = None,
    requested_trial_id: str | None = None,
) -> list[str]:
    script = Path(args.repo).resolve() / "tools/v10_search_covtrack_full_test.py"
    command = [
        args.stream_python,
        str(script),
        "--repo",
        str(Path(args.repo).resolve()),
        "--source",
        str(Path(args.source).resolve()),
        "--annotation",
        str(Path(args.annotation).resolve()),
        "--img-prefix",
        str(Path(args.img_prefix).resolve()),
        "--external-config",
        str(Path(args.external_config).resolve()),
        "--external-checkpoint",
        str(Path(args.external_checkpoint).resolve()),
        "--base-config",
        str(Path(args.base_config).resolve()),
        "--output-root",
        str(Path(args.output_root).resolve()),
        "--trial-id",
        str(spec["trial_id"]),
        "--spec-json",
        json.dumps(spec, separators=(",", ":")),
        "--stage",
        args.stage,
        "--gpu",
        str(gpu),
        "--stream-python",
        args.stream_python,
        "--evaluator-python",
        args.evaluator_python,
        "--evaluator-cores",
        str(args.evaluator_cores),
    ]
    teta_source_root = getattr(args, "teta_source_root", None)
    if teta_source_root:
        command.extend(["--teta-source-root", str(Path(teta_source_root).resolve())])
    if args.disabled_overlay:
        command.append("--disabled-overlay")
    if requested_trial_id and requested_trial_id != str(spec["trial_id"]):
        command.extend(["--requested-trial-id", requested_trial_id])
    if plan is not None and plan.path is not None:
        command.extend(
            [
                "--search-plan",
                str(plan.path),
                "--search-plan-sha256",
                str(plan.sha256),
                "--threshold-source",
                str(plan.threshold_source),
            ]
        )
        if not args.disabled_overlay:
            command.extend(
                [
                    "--contract-gate",
                    str(plan.contract_gate),
                    "--contract-gate-sha256",
                    str(plan.contract_gate_sha256),
                ]
            )
    return command


def _mem_available_bytes() -> int:
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def _mem_available_gib() -> float:
    return _mem_available_bytes() / (1024**3)


def _ram_launch_gate(args: argparse.Namespace) -> tuple[bool, float, float]:
    available_gb = _mem_available_gib()
    required_gb = float(args.min_available_ram_gb) + float(args.launch_reserve_ram_gb)
    return available_gb >= required_gb, available_gb, required_gb


def _plan_binding(plan: Any) -> dict[str, Any]:
    return {
        "path": plan.path,
        "sha256": plan.sha256,
        "protocol": plan.protocol,
        "unbiased_test": plan.unbiased_test,
        "contract_mode": plan.contract_mode,
        "contract_gate": plan.contract_gate,
        "contract_gate_sha256": plan.contract_gate_sha256,
        "threshold_source": plan.threshold_source,
        "expected_inputs": dict(plan.expected_inputs),
        "gate_checkpoint_sha256": None,
    }


def _write_status(
    path: Path,
    status: dict[str, Any],
    *,
    plan: Any | None = None,
    contract_gate: dict[str, Any] | None = None,
    runtime_binding: dict[str, str] | None = None,
) -> None:
    status["mem_available_gb"] = _mem_available_gib()
    if plan is not None:
        status["search_plan"] = _plan_binding(plan)
        status["search_plan"]["gate_checkpoint_sha256"] = (
            None if contract_gate is None else contract_gate.get("checkpoint_sha256")
        )
    if runtime_binding is not None:
        status["runtime_gate_binding"] = dict(runtime_binding)
    status["updated_at_unix"] = time.time()
    _write_json(path, status)


def _next_retry_id(root: Path, trial_id: str) -> str:
    """Return a new directory name; failed/partial trials are immutable."""
    index = 1
    while (root / f"{trial_id}__retry{index:02d}").exists():
        index += 1
    return f"{trial_id}__retry{index:02d}"


def _trial_needs_retry(root: Path, trial_id: str, old_job: dict[str, Any] | None) -> bool:
    if old_job is not None and old_job.get("state") == "COMPLETED":
        return False
    trial_root = root / trial_id
    receipt = trial_root / "receipt.json"
    if receipt.is_file():
        try:
            status = json.loads(receipt.read_text(encoding="utf-8")).get("status")
        except json.JSONDecodeError:
            status = "PARTIAL"
        return status != "COMPLETED"
    return trial_root.exists() and any(trial_root.iterdir())


def _acquire_free_gpu(free_gpus: list[str]) -> str:
    """Pop one genuinely free device; never derive allocation from count."""
    if not free_gpus:
        raise RuntimeError("no free GPU lease")
    return free_gpus.pop(0)


def _release_gpu(free_gpus: list[str], gpu: str, order: list[str]) -> None:
    if gpu in free_gpus:
        raise RuntimeError(f"GPU {gpu} released twice")
    free_gpus.append(gpu)
    free_gpus.sort(key=order.index)


def _base_reranker_checkpoint_binding(base_config: Path) -> tuple[Path, str]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required for search gate binding") from exc
    raw = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("BASE_CONFIG_INVALID")
    tempo = raw.get("tempo", raw)
    if not isinstance(tempo, dict):
        raise RuntimeError("BASE_CONFIG_TEMPO_INVALID")
    value = tempo.get("reranker_checkpoint")
    if not value:
        raise RuntimeError("BASE_CONFIG_RERANKER_CHECKPOINT_MISSING")
    checkpoint = Path(str(value)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint, _sha256(checkpoint)


def _validate_gate_checkpoint_binding(
    contract_gate: dict[str, Any],
    base_config: Path,
) -> tuple[Path, str]:
    checkpoint, checkpoint_sha = _base_reranker_checkpoint_binding(base_config)
    gate_checkpoint_sha = contract_gate.get("checkpoint_sha256")
    if not gate_checkpoint_sha:
        raise RuntimeError("SEARCH_GATE_CHECKPOINT_SHA_MISSING")
    if gate_checkpoint_sha != checkpoint_sha:
        raise RuntimeError("SEARCH_GATE_CHECKPOINT_MISMATCH")
    return checkpoint, checkpoint_sha


def _base_runtime_contract_sha(base_config: Path) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required for runtime contract binding") from exc
    raw = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("BASE_CONFIG_INVALID")
    value = raw.get("runtime_contract_sha")
    if value is None:
        tempo = raw.get("tempo", {})
        if isinstance(tempo, dict):
            value = tempo.get("runtime_contract_sha")
    if value is None or not str(value).strip():
        raise RuntimeError("BASE_RUNTIME_CONTRACT_SHA_MISSING")
    return str(value).strip()


def _validate_gate_runtime_binding(
    *,
    contract_gate: dict[str, Any],
    base_config: Path,
    repo: Path,
) -> dict[str, str]:
    expected_runtime_contract_sha = _base_runtime_contract_sha(base_config)
    gate_runtime_contract_sha = contract_gate.get("runtime_contract_sha")
    if not gate_runtime_contract_sha:
        raise RuntimeError("SEARCH_GATE_RUNTIME_REVISION_MISSING")
    if str(gate_runtime_contract_sha) != expected_runtime_contract_sha:
        raise RuntimeError("SEARCH_GATE_RUNTIME_REVISION_MISMATCH")

    overlay_path = repo / "tempotrack_v10" / "overlay.py"
    runtime_path = repo / "tempotrack_v10" / "covtrack_runtime.py"
    if not overlay_path.is_file():
        raise FileNotFoundError(overlay_path)
    if not runtime_path.is_file():
        raise FileNotFoundError(runtime_path)
    current_overlay_sha = _sha256(overlay_path)
    current_runtime_sha = _sha256(runtime_path)
    gate_overlay_sha = contract_gate.get("overlay_sha256")
    gate_runtime_sha = contract_gate.get("runtime_sha256")
    if not gate_overlay_sha:
        raise RuntimeError("SEARCH_GATE_OVERLAY_SHA_MISSING")
    if not gate_runtime_sha:
        raise RuntimeError("SEARCH_GATE_RUNTIME_SHA_MISSING")
    if gate_overlay_sha != current_overlay_sha:
        raise RuntimeError("SEARCH_GATE_CURRENT_OVERLAY_HASH_MISMATCH")
    if gate_runtime_sha != current_runtime_sha:
        raise RuntimeError("SEARCH_GATE_CURRENT_RUNTIME_HASH_MISMATCH")
    return {
        "runtime_contract_sha": expected_runtime_contract_sha,
        "overlay_sha256": current_overlay_sha,
        "runtime_sha256": current_runtime_sha,
    }


def _validate_expected_inputs(
    plan: Any,
    args: argparse.Namespace,
    *,
    reranker_checkpoint_sha256: str | None,
) -> None:
    expected = dict(plan.expected_inputs)
    if not expected:
        return
    teta_keys = {
        "teta_source_root",
        "teta_init_sha256",
        "teta_git_commit",
        "teta_require_tracked_clean",
    }
    if teta_keys.intersection(expected):
        teta_source_root = getattr(args, "teta_source_root", None)
        if not teta_source_root:
            raise RuntimeError("SEARCH_EXPECTED_INPUT_TETA_SOURCE_ROOT_MISSING")
        dependency = _teta_dependency(teta_source_root)
        if dependency is None:
            raise RuntimeError("SEARCH_EXPECTED_INPUT_TETA_SOURCE_ROOT_INVALID")
        if expected.get("teta_source_root") != dependency["source_root"]:
            raise RuntimeError("SEARCH_EXPECTED_INPUT_TETA_ROOT_MISMATCH")
        if expected.get("teta_init_sha256") != dependency["init_sha256"]:
            raise RuntimeError("SEARCH_EXPECTED_INPUT_TETA_INIT_MISMATCH")
        if (
            expected.get("teta_git_commit") is not None
            and expected.get("teta_git_commit") != dependency["git_commit"]
        ):
            raise RuntimeError("SEARCH_EXPECTED_INPUT_TETA_COMMIT_MISMATCH")
        if bool(expected.get("teta_require_tracked_clean", False)):
            tracked_status = dependency.get("tracked_git_status")
            if tracked_status is None:
                raise RuntimeError("SEARCH_EXPECTED_INPUT_TETA_TRACKED_STATUS_UNAVAILABLE")
            if tracked_status != "":
                raise RuntimeError("SEARCH_EXPECTED_INPUT_TETA_SOURCE_TRACKED_DIRTY")
            if dependency.get("tracked_source_clean") is not True:
                raise RuntimeError("SEARCH_EXPECTED_INPUT_TETA_SOURCE_NOT_CLEAN")
    # A commit hash alone does not prove that the pinned external checkout is
    # the source that the worker will import.  Hardened plans must launch only
    # from a clean checkout; the already-running legacy wave has no
    # expected_inputs block and therefore remains unaffected.
    if "external_config_sha256" in expected or "base_config_sha256" in expected:
        dirty = _git_value(Path(args.source).resolve(), "status", "--porcelain")
        if dirty is None:
            raise RuntimeError("EXTERNAL_COV_SOURCE_STATUS_UNAVAILABLE")
        if dirty:
            raise RuntimeError("EXTERNAL_COV_SOURCE_DIRTY")
    annotation_sha = _sha256(Path(args.annotation).resolve())
    if expected.get("subset_annotation_sha256") != annotation_sha:
        raise RuntimeError("SEARCH_EXPECTED_INPUT_ANNOTATION_MISMATCH")
    external_commit = _git_value(Path(args.source).resolve(), "rev-parse", "HEAD")
    if expected.get("external_cov_commit") != external_commit:
        raise RuntimeError("SEARCH_EXPECTED_INPUT_EXTERNAL_COMMIT_MISMATCH")
    external_checkpoint_sha = _sha256(Path(args.external_checkpoint).resolve())
    if expected.get("external_checkpoint_sha256") != external_checkpoint_sha:
        raise RuntimeError("SEARCH_EXPECTED_INPUT_EXTERNAL_CHECKPOINT_MISMATCH")
    external_config_sha = _sha256(Path(args.external_config).resolve())
    if expected.get("external_config_sha256") != external_config_sha:
        raise RuntimeError("SEARCH_EXPECTED_INPUT_EXTERNAL_CONFIG_MISMATCH")
    base_config_sha = _sha256(Path(args.base_config).resolve())
    if expected.get("base_config_sha256") != base_config_sha:
        raise RuntimeError("SEARCH_EXPECTED_INPUT_BASE_CONFIG_MISMATCH")
    if (
        not bool(getattr(args, "disabled_overlay", False))
        and expected.get("reranker_checkpoint_sha256") != reranker_checkpoint_sha256
    ):
        raise RuntimeError("SEARCH_EXPECTED_INPUT_RERANKER_CHECKPOINT_MISMATCH")


def _build_jobs(
    specs: list[dict[str, Any]],
    root: Path,
    old: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Materialize immutable retry-aware jobs from a prior coordinator.

    A requested trial is identified independently from its effective retry
    directory.  Once any completed retry exists, resume must reuse that
    completed artifact; failed/partial directories are never overwritten.
    """
    old_by_requested: dict[str, list[dict[str, Any]]] = {}
    for effective_id, value in old.items():
        requested_id = str(value.get("requested_trial_id", value.get("trial_id", effective_id)))
        old_by_requested.setdefault(requested_id, []).append(value)
    jobs: dict[str, dict[str, Any]] = {}
    for item in specs:
        requested_id = str(item["trial_id"])
        completed_old = next(
            (
                value for value in old_by_requested.get(requested_id, [])
                if value.get("state") == "COMPLETED"
            ),
            None,
        )
        if completed_old is not None:
            effective_id = str(completed_old.get("trial_id", requested_id))
            spec = dict(item)
            spec["trial_id"] = effective_id
            jobs[effective_id] = dict(completed_old)
            jobs[effective_id]["spec"] = spec
            jobs[effective_id].setdefault("requested_trial_id", requested_id)
            continue
        old_job = next(
            (value for value in old_by_requested.get(requested_id, []) if value.get("trial_id") == requested_id),
            None,
        )
        effective_id = requested_id
        if _trial_needs_retry(root, requested_id, old_job):
            effective_id = _next_retry_id(root, requested_id)
        spec = dict(item)
        spec["trial_id"] = effective_id
        jobs[effective_id] = {
            "trial_id": effective_id,
            "requested_trial_id": requested_id,
            "retry_of": requested_id if effective_id != requested_id else None,
            "spec": spec,
            "state": "COMPLETED" if old_job and old_job.get("state") == "COMPLETED" else "PENDING",
            "gpu": None,
            "pid": None,
            "returncode": None,
        }
        if old_job and old_job.get("state") == "COMPLETED":
            jobs[effective_id].update(old_job)
    return jobs


def run(args: argparse.Namespace) -> int:
    requested = {value for value in args.trial_ids.split(",") if value} if args.trial_ids else None
    plan_path = Path(args.spec_file).resolve() if args.spec_file else None
    plan = _load_search_plan(plan_path)
    teta_dependency = None
    teta_preflight = None
    teta_source_root = getattr(args, "teta_source_root", None)
    if plan.contract_mode == "hardened" and not teta_source_root:
        raise RuntimeError("TETA_SOURCE_ROOT_REQUIRED")
    if teta_source_root:
        _resolve_teta_source_root(teta_source_root)
        teta_dependency = _teta_dependency(teta_source_root)
        teta_preflight = _run_teta_import_preflight(
            stream_python=args.stream_python,
            source=Path(args.source).resolve(),
            teta_source_root=teta_source_root,
        )
    contract_gate = None
    runtime_binding = None
    reranker_checkpoint_sha256 = None
    if not args.disabled_overlay:
        contract_gate = _validate_contract_gate(plan)
        base_config_path = Path(args.base_config).resolve()
        repo_path = Path(args.repo).resolve()
        _, reranker_checkpoint_sha256 = _validate_gate_checkpoint_binding(
            contract_gate,
            base_config_path,
        )
        runtime_binding = _validate_gate_runtime_binding(
            contract_gate=contract_gate,
            base_config=base_config_path,
            repo=repo_path,
        )
    else:
        # A disabled native control has no reranker contract.  It still keeps
        # the immutable search-plan/input provenance when a plan is supplied.
        try:
            _, reranker_checkpoint_sha256 = _base_reranker_checkpoint_binding(
                Path(args.base_config).resolve()
            )
        except (FileNotFoundError, RuntimeError):
            reranker_checkpoint_sha256 = None
    _validate_expected_inputs(
        plan,
        args,
        reranker_checkpoint_sha256=reranker_checkpoint_sha256,
    )
    specs = [dict(item) for item in plan.trials]
    if requested is not None:
        specs = [item for item in specs if str(item.get("trial_id")) in requested]
    if not specs:
        raise ValueError("no trial specs selected")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one physical GPU index")
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if teta_dependency is not None:
        _write_json(
            root / "teta_import_preflight.json",
            {"dependency": teta_dependency, "preflight": teta_preflight},
        )
    status_path = root / "coordinator_status.json"
    if status_path.exists() and not args.resume:
        raise FileExistsError(f"coordinator status already exists; use --resume: {status_path}")

    old: dict[str, Any] = {}
    if args.resume and status_path.is_file():
        old = json.loads(status_path.read_text(encoding="utf-8")).get("jobs", {})
    jobs = _build_jobs(specs, root, old)

    pending = [trial_id for trial_id, job in jobs.items() if job["state"] != "COMPLETED"]
    running: dict[str, tuple[subprocess.Popen[Any], str, Any]] = {}
    free_gpus = list(gpus)
    completed = sum(job["state"] == "COMPLETED" for job in jobs.values())
    failed = 0
    while pending or running:
        waiting_for_ram = False
        while pending and free_gpus and len(running) < min(args.max_workers, len(gpus)):
            can_launch, _, _ = _ram_launch_gate(args)
            if not can_launch:
                waiting_for_ram = True
                break
            trial_id = pending.pop(0)
            gpu = _acquire_free_gpu(free_gpus)
            spec = jobs[trial_id]["spec"]
            # The one-trial harness owns ``root/trial_id`` and deliberately
            # refuses a pre-existing partial directory.  Keep coordinator
            # logs outside that namespace so a launch cannot look like an
            # output artifact or trip the safety check.
            trial_log = root / "worker_logs" / f"{trial_id}.log"
            trial_log.parent.mkdir(parents=True, exist_ok=True)
            command = _build_command(
                args,
                spec,
                gpu,
                plan,
                requested_trial_id=jobs[trial_id].get("requested_trial_id"),
            )
            log = trial_log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=str(Path(args.repo).resolve()),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(args.repo).resolve()), os.environ.get("PYTHONPATH", "")])},
            )
            jobs[trial_id].update({"state": "RUNNING", "gpu": gpu, "pid": process.pid, "command": command, "started_at_unix": time.time()})
            running[trial_id] = (process, gpu, log)

        for trial_id, (process, gpu, log) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            log.close()
            running.pop(trial_id)
            job = jobs[trial_id]
            job["returncode"] = int(returncode)
            job["ended_at_unix"] = time.time()
            job["state"] = "COMPLETED" if returncode == 0 else "FAILED"
            if returncode == 0:
                completed += 1
            else:
                failed += 1
            _release_gpu(free_gpus, gpu, gpus)
            _write_status(
                status_path,
                {
                    "schema_version": 1,
                    "status": "WAITING_FOR_HOST_RAM" if waiting_for_ram else "RUNNING",
                    "jobs": jobs,
                    "completed": completed,
                    "failed": failed,
                    "required_before_launch_gb": (
                        float(args.min_available_ram_gb) + float(args.launch_reserve_ram_gb)
                    ),
                },
                plan=plan,
                contract_gate=contract_gate,
                runtime_binding=runtime_binding,
            )
        _write_status(
            status_path,
            {
                "schema_version": 1,
                "status": "WAITING_FOR_HOST_RAM" if waiting_for_ram else "RUNNING",
                "jobs": jobs,
                "completed": completed,
                "failed": failed,
                "required_before_launch_gb": (
                    float(args.min_available_ram_gb) + float(args.launch_reserve_ram_gb)
                ),
            },
            plan=plan,
            contract_gate=contract_gate,
            runtime_binding=runtime_binding,
        )
        if pending or running:
            time.sleep(max(1.0, float(args.poll_seconds)))

    final_status = "COMPLETED" if failed == 0 else "PARTIAL_FAILURE"
    _write_status(
        status_path,
        {"schema_version": 1, "status": final_status, "jobs": jobs, "completed": completed, "failed": failed},
        plan=plan,
        contract_gate=contract_gate,
        runtime_binding=runtime_binding,
    )
    print(json.dumps({"status": final_status, "completed": completed, "failed": failed, "status_path": str(status_path)}))
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--img-prefix", required=True)
    parser.add_argument("--external-config", required=True)
    parser.add_argument("--external-checkpoint", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--spec-file")
    parser.add_argument("--trial-ids")
    parser.add_argument("--stage", choices=("subset", "full"), default="subset")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--evaluator-cores", type=int, default=8)
    parser.add_argument("--min-available-ram-gb", type=float, default=24.0)
    parser.add_argument("--launch-reserve-ram-gb", type=float, default=4.0)
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
    parser.add_argument("--disabled-overlay", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    raise SystemExit(run(arguments))
