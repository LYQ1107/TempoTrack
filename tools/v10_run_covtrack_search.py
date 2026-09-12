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

from v10_search_covtrack_full_test import _load_specs, default_trial_specs, _write_json


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

    jobs = {
        str(item["trial_id"]): {
            "trial_id": str(item["trial_id"]),
            "spec": item,
            "state": "PENDING",
            "gpu": None,
            "pid": None,
            "returncode": None,
        }
        for item in specs
    }
    if args.resume and status_path.is_file():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        for trial_id, value in old.get("jobs", {}).items():
            if trial_id in jobs and value.get("state") == "COMPLETED":
                jobs[trial_id].update(value)

    pending = [trial_id for trial_id, job in jobs.items() if job["state"] != "COMPLETED"]
    running: dict[str, tuple[subprocess.Popen[Any], str, Any]] = {}
    completed = sum(job["state"] == "COMPLETED" for job in jobs.values())
    failed = 0
    while pending or running:
        while pending and len(running) < min(args.max_workers, len(gpus)):
            trial_id = pending.pop(0)
            gpu = gpus[len(running) % len(gpus)]
            spec = jobs[trial_id]["spec"]
            trial_log = root / trial_id / "coordinator.log"
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
