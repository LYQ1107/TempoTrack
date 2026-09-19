#!/usr/bin/env python3
"""Run causal Official-Val replays for already selected V12 Test-tuned cards.

The Val cache is shared and is never replaced by GT inputs.  Workers select
disjoint videos from that cache by round-robin index; annotations are passed
only to the postprocessor after all replay shards have completed.  This tool
does not select a model or threshold and every receipt remains explicitly
``TEST_TUNED_EXPLORATION`` because Test was used upstream for selection.
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
        raise RuntimeError("Official-Val plan must be a JSON object")
    if plan.get("status") != "PASS" or plan.get("paper_status") != "TEST_TUNED_EXPLORATION":
        raise RuntimeError("Official-Val plan is not TEST_TUNED_EXPLORATION")
    if plan.get("paper_valid") is not False or plan.get("diagnostic_only") is not True:
        raise RuntimeError("Official-Val plan paper-status guard failed")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Official-Val plan has no candidates")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RuntimeError("Official-Val candidate is not an object")
        candidate_id = _safe_name(candidate.get("candidate_id"), "candidate_id")
        card_id = _safe_name(candidate.get("card_id"), "card_id")
        trial_id = _safe_name(candidate.get("trial_id", "s00_m00"), "trial_id")
        config = Path(str(candidate.get("config", ""))).resolve()
        if candidate_id in seen:
            raise RuntimeError(f"duplicate Official-Val candidate: {candidate_id}")
        if not config.is_file():
            raise FileNotFoundError(f"Official-Val config missing: {config}")
        seen.add(candidate_id)
        normalized.append(
            {
                **candidate,
                "candidate_id": candidate_id,
                "card_id": card_id,
                "trial_id": trial_id,
                "config": str(config),
                "config_sha256": sha256_file(config),
            }
        )
    plan["candidates"] = normalized
    return plan


def _run_logged(command: list[str], *, cwd: Path, environment: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), env=environment, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _start_batch(
    candidates: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    environment: dict[str, str],
) -> None:
    gpus = [value.strip() for value in str(args.gpus).split(",") if value.strip()]
    if len(gpus) != int(args.shard_count):
        raise ValueError("one GPU is required per Official-Val video shard")
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
            shard = f"shard_{index:02d}"
            output = args.replay_root.resolve() / candidate_id / trial_id / shard
            log = args.replay_root.resolve() / candidate_id / "logs" / f"{trial_id}_{shard}.log"
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"refusing to overwrite Official-Val replay: {output}")
            child_environment = dict(environment)
            child_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                str(args.python_replay.resolve()),
                "-u",
                "-m",
                "tools.v12_mgf_overlay_replay",
                "--cache",
                str(args.full_cache.resolve()),
                "--tempo-config",
                str(config),
                "--output-root",
                str(output.resolve()),
                "--trial-id",
                trial_id,
                "--repo",
                str(args.repo.resolve()),
                "--cov-source",
                str(args.cov_source.resolve()),
                "--cov-config",
                str(args.cov_config.resolve()),
                "--cov-checkpoint",
                str(args.cov_checkpoint.resolve()),
                "--device",
                "cuda:0",
                "--track-offset-scope",
                "global",
                "--video-shard-index",
                str(index),
                "--video-shard-count",
                str(int(args.shard_count)),
            ]
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(args.repo.resolve()),
                    env=child_environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            batch["workers"].append(
                {
                    "candidate_id": candidate_id,
                    "card_id": candidate["card_id"],
                    "trial_id": trial_id,
                    "shard": shard,
                    "gpu": str(gpu),
                    "pid": int(process.pid),
                    "cache": str(args.full_cache.resolve()),
                    "video_shard_index": index,
                    "video_shard_count": int(args.shard_count),
                    "output": str(output.resolve()),
                    "log": str(log.resolve()),
                    "argv": command,
                    "status": "RUNNING",
                    "started_at_unix": time.time(),
                }
            )
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
                if worker["candidate_id"] == candidate_id and worker["shard"] == shard:
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
        raise RuntimeError(f"Official-Val replay worker failure: {failures}")
    batch.update({"status": "REPLAY_PASS", "replay_completed_at_unix": time.time()})
    write_json(args.runtime, runtime)

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        trial_id = candidate["trial_id"]
        replay_root = args.replay_root.resolve() / candidate_id / trial_id
        merge_root = replay_root / "merged"
        _run_logged(
            [
                str(args.python_post.resolve()),
                str(args.repo.resolve() / "tools" / "v12_merge_mgf_replay.py"),
                "--shard-root",
                str(replay_root),
                "--full-cache",
                str(args.full_cache.resolve()),
                "--output-root",
                str(merge_root),
                "--shard-count",
                str(int(args.shard_count)),
                "--trial-id",
                trial_id,
            ],
            cwd=args.repo.resolve(),
            environment=environment,
            log=replay_root / "merge.log",
        )
        _run_logged(
            [
                str(args.python_post.resolve()),
                "-u",
                str(args.repo.resolve() / "tools" / "v12_post_mgf_exploration_replay.py"),
                "--replay-root",
                str(replay_root),
                "--full-cache",
                str(args.full_cache.resolve()),
                "--annotation",
                str(args.annotation.resolve()),
                "--repo",
                str(args.repo.resolve()),
                "--python",
                str(args.python_post.resolve()),
                "--shard-count",
                str(int(args.shard_count)),
                "--poll-seconds",
                "5",
                "--evaluation-cores",
                str(int(args.evaluation_cores)),
                "--card-id",
                candidate_id,
                "--split",
                "official_val_tuned",
            ],
            cwd=args.repo.resolve(),
            environment=environment,
            log=replay_root / "official_val_tuned_post.log",
        )
        batch.setdefault("completed_candidates", []).append(
            {
                "candidate_id": candidate_id,
                "replay_root": str(replay_root),
                "metrics": str((merge_root / "val_metrics.json").resolve()),
            }
        )
        write_json(args.runtime, runtime)
    batch.update({"status": "PASS", "completed_at_unix": time.time()})
    write_json(args.runtime, runtime)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python-replay", type=Path, required=True)
    parser.add_argument("--python-post", type=Path, required=True)
    parser.add_argument("--cov-source", type=Path, required=True)
    parser.add_argument("--cov-config", type=Path, required=True)
    parser.add_argument("--cov-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--evaluation-cores", type=int, default=8)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    if int(args.batch_size) not in {1, 2}:
        raise ValueError("Official-Val launcher supports batch-size 1 or 2")
    args.replay_root.mkdir(parents=True, exist_ok=True)
    plan = validate_plan(args.plan_json.resolve())
    if args.runtime.exists():
        current = read_json(args.runtime)
        if current.get("status") == "PASS":
            print(json.dumps(current, ensure_ascii=False, indent=2))
            return 0
        raise RuntimeError(f"refusing to overwrite existing nonterminal runtime: {args.runtime}")
    environment = capture_environment(args.capture.resolve(), args.repo.resolve(), args.cov_source.resolve())
    runtime: dict[str, Any] = {
        "status": "RUNNING",
        "artifact": "v12_mgf_test_tuned_official_val_replay_controller",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "test_used_for_selection": True,
        "selection_scope": "VAL_TEST_TUNED_EXPLORATION",
        "split": "Official-Val",
        "plan_json": str(args.plan_json.resolve()),
        "plan_sha256": sha256_file(args.plan_json.resolve()),
        "full_cache": str(args.full_cache.resolve()),
        "full_cache_manifest_sha256": sha256_file(args.full_cache.resolve() / "manifest.json"),
        "annotation": str(args.annotation.resolve()),
        "annotation_sha256": sha256_file(args.annotation.resolve()),
        "candidate_ids": [item["candidate_id"] for item in plan["candidates"]],
        "batch_size": int(args.batch_size),
        "shard_count": int(args.shard_count),
        "started_at_unix": time.time(),
    }
    write_json(args.runtime, runtime)
    try:
        candidates = plan["candidates"]
        for start in range(0, len(candidates), int(args.batch_size)):
            _start_batch(candidates[start : start + int(args.batch_size)], args=args, runtime=runtime, environment=environment)
        runtime.update({"status": "PASS", "completed_at_unix": time.time()})
        write_json(args.runtime, runtime)
        print(json.dumps(runtime, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        runtime.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}", "ended_at_unix": time.time()})
        write_json(args.runtime, runtime)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
