#!/usr/bin/env python3
"""Run an explicit Test-tuned operating-point refinement plan.

The plan is created only after the first Full-Test round identifies the Top-2
MGF cards.  Every candidate has its own config, output root, merge root and
postprocess receipt.  B0 can appear in the same plan as a comparator; the
replay/postprocess contracts keep its BASE_TRAIN checkpoint provenance while
labeling the resulting experiment TEST_TUNED_EXPLORATION.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def capture_environment(capture: Path, repo: Path, cov_source: Path) -> dict[str, str]:
    document = read_json(capture)
    if document.get("capture_source") != "observed_live_proc_environ":
        raise RuntimeError("FAIL_CLOSED_CAPTURE_PROVENANCE")
    captured = document.get("environment", {})
    if not isinstance(captured, Mapping):
        raise RuntimeError("FAIL_CLOSED_CAPTURE_ENVIRONMENT_MISSING")
    environment = dict(os.environ)
    if captured.get("LD_PRELOAD"):
        environment["LD_PRELOAD"] = str(captured["LD_PRELOAD"])
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(repo), str(cov_source), str(captured.get("PYTHONPATH", ""))) if value
    )
    environment["V10_COV_SOURCE"] = str(cov_source)
    return environment


def _safe_name(value: Any, field: str) -> str:
    text = str(value)
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"invalid {field}: {text!r}")
    return text


def validate_plan(path: Path) -> dict[str, Any]:
    plan = read_json(path)
    if not isinstance(plan, dict):
        raise RuntimeError("refinement plan must be a JSON object")
    if plan.get("status") != "PASS" or plan.get("paper_status") != "TEST_TUNED_EXPLORATION":
        raise RuntimeError("refinement plan is not a completed Test-tuned exploration plan")
    if plan.get("paper_valid") is not False or plan.get("diagnostic_only") is not True:
        raise RuntimeError("refinement plan paper-status guard failed")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("refinement plan has no candidates")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RuntimeError("refinement candidate is not an object")
        candidate_id = _safe_name(candidate.get("candidate_id"), "candidate_id")
        card_id = _safe_name(candidate.get("card_id"), "card_id")
        trial_id = _safe_name(candidate.get("trial_id", candidate_id), "trial_id")
        config = Path(str(candidate.get("config", ""))).resolve()
        if candidate_id in seen:
            raise RuntimeError(f"duplicate refinement candidate: {candidate_id}")
        if not config.is_file():
            raise FileNotFoundError(f"refinement config missing: {config}")
        try:
            score_threshold = float(candidate["score_threshold"])
            margin_threshold = float(candidate["margin_threshold"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"refinement thresholds missing: {candidate_id}") from error
        if score_threshold != 0.0 or margin_threshold < 0.0:
            raise RuntimeError(f"illegal refinement thresholds: {candidate_id}")
        seen.add(candidate_id)
        normalized.append(
            {
                **candidate,
                "candidate_id": candidate_id,
                "card_id": card_id,
                "trial_id": trial_id,
                "config": str(config),
                "config_sha256": sha256_file(config),
                "score_threshold": score_threshold,
                "margin_threshold": margin_threshold,
            }
        )
    plan["candidates"] = normalized
    return plan


def _start_batch(
    candidates: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    environment: dict[str, str],
) -> None:
    gpus = [value.strip() for value in str(args.gpus).split(",") if value.strip()]
    if len(gpus) != int(args.shard_count):
        raise ValueError("one GPU is required per frontend shard")
    batch: dict[str, Any] = {
        "candidates": [item["candidate_id"] for item in candidates],
        "status": "RUNNING",
        "started_at_unix": time.time(),
        "workers": [],
    }
    runtime.setdefault("batches", []).append(batch)
    write_json(args.runtime, runtime)
    processes: list[tuple[str, str, subprocess.Popen[Any]]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        trial_id = candidate["trial_id"]
        config = Path(candidate["config"])
        for index, gpu in enumerate(gpus):
            shard = f"{index:02d}"
            cache = args.frontend_root.resolve() / f"shard_{shard}" / "frontend_cache"
            output = args.replay_root.resolve() / candidate_id / trial_id / f"shard_{shard}"
            log = args.replay_root.resolve() / candidate_id / "logs" / f"{trial_id}_shard_{shard}.log"
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"refusing to overwrite refinement replay: {output}")
            child_environment = dict(environment)
            child_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                str(args.python_replay.resolve()), "-u", "-m", "tools.v12_mgf_overlay_replay",
                "--cache", str(cache.resolve()), "--tempo-config", str(config),
                "--output-root", str(output.resolve()), "--trial-id", trial_id,
                "--repo", str(args.repo.resolve()), "--cov-source", str(args.cov_source.resolve()),
                "--cov-config", str(args.cov_config.resolve()),
                "--cov-checkpoint", str(args.cov_checkpoint.resolve()),
                "--device", "cuda:0", "--track-offset-scope", "global",
            ]
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(args.cov_source.resolve()),
                    env=child_environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            worker = {
                "candidate_id": candidate_id,
                "card_id": candidate["card_id"],
                "trial_id": trial_id,
                "shard": f"shard_{shard}",
                "gpu": str(gpu),
                "pid": int(process.pid),
                "cache": str(cache.resolve()),
                "output": str(output.resolve()),
                "log": str(log.resolve()),
                "argv": command,
                "status": "RUNNING",
                "started_at_unix": time.time(),
            }
            batch["workers"].append(worker)
            processes.append((candidate_id, shard, process))
    write_json(args.runtime, runtime)
    remaining = {(candidate_id, shard): process for candidate_id, shard, process in processes}
    while remaining:
        for key, process in list(remaining.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            candidate_id, shard = key
            for worker in batch["workers"]:
                if worker["candidate_id"] == candidate_id and worker["shard"] == f"shard_{shard}":
                    worker.update(
                        {
                            "returncode": int(returncode),
                            "status": "PASS" if returncode == 0 else "FAILED",
                            "ended_at_unix": time.time(),
                        }
                    )
                    break
            del remaining[key]
            write_json(args.runtime, runtime)
        if remaining:
            time.sleep(max(1.0, float(args.poll_seconds)))
    failures = [worker for worker in batch["workers"] if worker.get("status") != "PASS"]
    if failures:
        batch.update({"status": "FAILED", "failures": failures, "ended_at_unix": time.time()})
        write_json(args.runtime, runtime)
        raise RuntimeError(f"refinement replay worker failure: {failures}")
    batch.update({"status": "REPLAY_PASS", "replay_completed_at_unix": time.time()})
    write_json(args.runtime, runtime)

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        trial_id = candidate["trial_id"]
        replay_root = args.replay_root.resolve() / candidate_id / trial_id
        merge_root = replay_root / "merged"
        merge_log = replay_root / "merge.log"
        merge_command = [
            str(args.python_replay.resolve()), str(args.repo.resolve() / "tools" / "v12_merge_mgf_replay.py"),
            "--shard-root", str(replay_root), "--full-cache", str(args.full_cache.resolve()),
            "--output-root", str(merge_root), "--shard-count", str(int(args.shard_count)),
            "--trial-id", trial_id,
        ]
        with merge_log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                merge_command, cwd=str(args.repo.resolve()), env=environment,
                stdout=handle, stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise RuntimeError(f"merge failed for {candidate_id}; see {merge_log}")
        batch.setdefault("merges", []).append(
            {"candidate_id": candidate_id, "root": str(merge_root), "command": merge_command}
        )
    batch.update({"status": "MERGE_PASS", "merge_completed_at_unix": time.time()})
    write_json(args.runtime, runtime)

    post_processes: list[tuple[str, subprocess.Popen[Any]]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        trial_id = candidate["trial_id"]
        replay_root = args.replay_root.resolve() / candidate_id / trial_id
        log = replay_root / "test_tuned_post.log"
        command = [
            str(args.python_post.resolve()), "-u", str(args.repo.resolve() / "tools" / "v12_post_mgf_exploration_replay.py"),
            "--replay-root", str(replay_root), "--full-cache", str(args.full_cache.resolve()),
            "--annotation", str(args.annotation.resolve()), "--repo", str(args.repo.resolve()),
            "--python", str(args.python_post.resolve()), "--shard-count", str(int(args.shard_count)),
            "--poll-seconds", "5", "--evaluation-cores", str(int(args.evaluation_cores)),
            "--card-id", candidate_id,
        ]
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command, cwd=str(args.repo.resolve()), env=environment,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
        post_processes.append((candidate_id, process))
        batch.setdefault("post", []).append(
            {"candidate_id": candidate_id, "pid": int(process.pid), "log": str(log), "status": "RUNNING", "argv": command}
        )
    write_json(args.runtime, runtime)
    for candidate_id, process in post_processes:
        returncode = int(process.wait())
        for item in batch["post"]:
            if item["candidate_id"] == candidate_id:
                item.update({"returncode": returncode, "status": "PASS" if returncode == 0 else "FAILED"})
                break
        write_json(args.runtime, runtime)
        if returncode != 0:
            raise RuntimeError(f"Test-tuned postprocess failed for {candidate_id}")
    batch.update({"status": "PASS", "completed_at_unix": time.time()})
    write_json(args.runtime, runtime)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--frontend-root", type=Path, required=True)
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python-replay", type=Path, required=True)
    parser.add_argument("--python-post", type=Path, required=True)
    parser.add_argument("--cov-source", type=Path, required=True)
    parser.add_argument("--cov-config", type=Path, required=True)
    parser.add_argument("--cov-checkpoint", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--evaluation-cores", type=int, default=8)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    if int(args.batch_size) not in {1, 2}:
        raise ValueError("refinement launcher supports batch-size 1 or 2")
    args.replay_root.mkdir(parents=True, exist_ok=True)
    plan = validate_plan(args.plan_json.resolve())
    environment = capture_environment(args.capture.resolve(), args.repo.resolve(), args.cov_source.resolve())
    runtime: dict[str, Any] = {
        "status": "RUNNING",
        "artifact": "v12_mgf_test_tuned_refinement_replay_controller",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "selection_scope": "VAL_TEST_TUNED_EXPLORATION",
        "test_used_for_selection": True,
        "plan_json": str(args.plan_json.resolve()),
        "plan_sha256": sha256_file(args.plan_json.resolve()),
        "candidate_ids": [item["candidate_id"] for item in plan["candidates"]],
        "batch_size": int(args.batch_size),
        "started_at_unix": time.time(),
    }
    write_json(args.runtime, runtime)
    try:
        candidates = plan["candidates"]
        for start in range(0, len(candidates), int(args.batch_size)):
            _start_batch(candidates[start : start + int(args.batch_size)], args=args, runtime=runtime, environment=environment)
        runtime.update({"status": "PASS", "completed_at_unix": time.time()})
        write_json(args.runtime, runtime)
        print(json.dumps(runtime, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        runtime.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}", "ended_at_unix": time.time()})
        write_json(args.runtime, runtime)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
