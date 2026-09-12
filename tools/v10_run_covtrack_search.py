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
    from v10_search_covtrack_full_test import _load_specs, default_trial_specs, _write_json
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
    _load_specs = _module._load_specs
    default_trial_specs = _module.default_trial_specs
    _write_json = _module._write_json


def _load_requested_specs(path: Path | None, trial_ids: set[str] | None) -> list[dict[str, Any]]:
    specs = _load_specs(path) if path is not None else default_trial_specs()
    if trial_ids is None:
        return specs
    return [item for item in specs if str(item.get("trial_id")) in trial_ids]


def _build_command(args: argparse.Namespace, spec: dict[str, Any], gpu: str) -> list[str]:
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
    if args.disabled_overlay:
        command.append("--disabled-overlay")
    return command


def _write_status(path: Path, status: dict[str, Any]) -> None:
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
    specs = _load_requested_specs(Path(args.spec_file) if args.spec_file else None, requested)
    if not specs:
        raise ValueError("no trial specs selected")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one physical GPU index")
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
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
        while pending and free_gpus and len(running) < min(args.max_workers, len(gpus)):
            trial_id = pending.pop(0)
            gpu = _acquire_free_gpu(free_gpus)
            spec = jobs[trial_id]["spec"]
            # The one-trial harness owns ``root/trial_id`` and deliberately
            # refuses a pre-existing partial directory.  Keep coordinator
            # logs outside that namespace so a launch cannot look like an
            # output artifact or trip the safety check.
            trial_log = root / "worker_logs" / f"{trial_id}.log"
            trial_log.parent.mkdir(parents=True, exist_ok=True)
            command = _build_command(args, spec, gpu)
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
            _write_status(status_path, {"schema_version": 1, "status": "RUNNING", "jobs": jobs, "completed": completed, "failed": failed})
        _write_status(status_path, {"schema_version": 1, "status": "RUNNING", "jobs": jobs, "completed": completed, "failed": failed})
        if pending or running:
            time.sleep(max(1.0, float(args.poll_seconds)))

    final_status = "COMPLETED" if failed == 0 else "PARTIAL_FAILURE"
    _write_status(status_path, {"schema_version": 1, "status": final_status, "jobs": jobs, "completed": completed, "failed": failed})
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
    parser.add_argument(
        "--stream-python",
        default="/home/lwr/anaconda3/envs/ovtr/bin/python",
    )
    parser.add_argument(
        "--evaluator-python",
        default="/home/lwr/anaconda3/envs/masaenv/bin/python",
    )
    parser.add_argument("--disabled-overlay", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    raise SystemExit(run(arguments))
