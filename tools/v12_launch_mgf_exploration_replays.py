#!/usr/bin/env python3
"""Run the selected Top-4 Test-tuned exploration cards in two-card batches."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
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


def _wait_for_ranking(path: Path, poll_seconds: float) -> dict[str, Any]:
    while not path.is_file():
        time.sleep(max(1.0, poll_seconds))
    ranking = read_json(path)
    if ranking.get("status") != "PASS" or ranking.get("paper_status") != "TEST_TUNED_EXPLORATION":
        raise RuntimeError("ranking receipt is not a completed Test-tuned exploration result")
    top4 = ranking.get("top4")
    if not isinstance(top4, list) or len(top4) != 4 or not all(
        len(str(card)) == 3 and str(card).startswith("E") and str(card)[1:].isdigit()
        for card in top4
    ):
        raise RuntimeError(f"ranking top4 must contain four exploratory cards: {top4}")
    return ranking


def _start_card_batch(
    cards: list[str],
    *,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    environment: dict[str, str],
) -> None:
    gpus = [value.strip() for value in str(args.gpus).split(",") if value.strip()]
    if len(gpus) != int(args.shard_count):
        raise ValueError("one GPU is required per frontend shard")
    processes: list[tuple[str, str, subprocess.Popen[Any]]] = []
    batch_record: dict[str, Any] = {"cards": cards, "status": "RUNNING", "started_at_unix": time.time(), "workers": []}
    runtime.setdefault("batches", []).append(batch_record)
    write_json(args.runtime, runtime)
    for card in cards:
        config = args.config_root.resolve() / f"{card}.yaml"
        if not config.is_file():
            raise FileNotFoundError(f"exploration replay config missing: {config}")
        for index, gpu in enumerate(gpus):
            shard = f"{index:02d}"
            cache = args.frontend_root.resolve() / f"shard_{shard}" / "frontend_cache"
            output = args.replay_root.resolve() / card / "s00_m00" / f"shard_{shard}"
            log = args.replay_root.resolve() / card / "logs" / f"s00_m00_shard_{shard}.log"
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"refusing to overwrite existing exploration replay: {output}")
            child_environment = dict(environment)
            child_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                str(args.python_replay.resolve()), "-u", "-m", "tools.v12_mgf_overlay_replay",
                "--cache", str(cache.resolve()), "--tempo-config", str(config.resolve()),
                "--output-root", str(output.resolve()), "--trial-id", "s00_m00",
                "--repo", str(args.repo.resolve()), "--cov-source", str(args.cov_source.resolve()),
                "--cov-config", str(args.cov_config.resolve()),
                "--cov-checkpoint", str(args.cov_checkpoint.resolve()),
                "--device", "cuda:0", "--track-offset-scope", "global",
            ]
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                # The audited COV config contains relative prompt/checkpoint
                # paths.  Running from the COV root is part of the reference
                # replay environment; running from the TempoTrack repo makes
                # prompt-path resolution fall back to a slow/non-parity path.
                cwd=str(args.cov_source.resolve()),
                env=child_environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            handle.close()
            worker = {
                "card": card, "shard": f"shard_{shard}", "gpu": str(gpu),
                "pid": int(process.pid), "cache": str(cache.resolve()),
                "output": str(output.resolve()), "log": str(log.resolve()),
                "argv": command, "status": "RUNNING", "started_at_unix": time.time(),
            }
            batch_record["workers"].append(worker)
            processes.append((card, shard, process))
    write_json(args.runtime, runtime)

    remaining = {(card, shard): process for card, shard, process in processes}
    while remaining:
        for key, process in list(remaining.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            card, shard = key
            for worker in batch_record["workers"]:
                if worker["card"] == card and worker["shard"] == f"shard_{shard}":
                    worker.update({"returncode": int(returncode), "status": "PASS" if returncode == 0 else "FAILED", "ended_at_unix": time.time()})
                    break
            del remaining[key]
            write_json(args.runtime, runtime)
        if remaining:
            time.sleep(5.0)
    failures = [worker for worker in batch_record["workers"] if worker.get("status") != "PASS"]
    if failures:
        batch_record.update({"status": "FAILED", "failures": failures, "ended_at_unix": time.time()})
        write_json(args.runtime, runtime)
        raise RuntimeError(f"exploration replay worker failure: {failures}")
    batch_record.update({"status": "REPLAY_PASS", "replay_completed_at_unix": time.time()})
    write_json(args.runtime, runtime)

    for card in cards:
        replay_root = args.replay_root.resolve() / card / "s00_m00"
        merge_root = replay_root / "merged"
        merge_log = replay_root / "merge.log"
        merge_command = [
            str(args.python_replay.resolve()), str(args.repo.resolve() / "tools" / "v12_merge_mgf_replay.py"),
            "--shard-root", str(replay_root), "--full-cache", str(args.full_cache.resolve()),
            "--output-root", str(merge_root), "--shard-count", str(int(args.shard_count)), "--trial-id", card,
        ]
        with merge_log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(merge_command, cwd=str(args.repo.resolve()), env=environment, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"merge failed for {card}; see {merge_log}")
        batch_record.setdefault("merges", []).append({"card": card, "root": str(merge_root), "command": merge_command})
    batch_record.update({"status": "MERGE_PASS", "merge_completed_at_unix": time.time()})
    write_json(args.runtime, runtime)

    post_processes: list[tuple[str, subprocess.Popen[Any]]] = []
    for card in cards:
        replay_root = args.replay_root.resolve() / card / "s00_m00"
        log = replay_root / "test_tuned_post.log"
        command = [
            str(args.python_post.resolve()), "-u", str(args.repo.resolve() / "tools" / "v12_post_mgf_exploration_replay.py"),
            "--replay-root", str(replay_root), "--full-cache", str(args.full_cache.resolve()),
            "--annotation", str(args.annotation.resolve()), "--repo", str(args.repo.resolve()),
            "--python", str(args.python_post.resolve()), "--shard-count", str(int(args.shard_count)),
            "--poll-seconds", "5", "--evaluation-cores", str(int(args.evaluation_cores)), "--card-id", card,
        ]
        handle = log.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=str(args.repo.resolve()), env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        handle.close()
        post_processes.append((card, process))
        batch_record.setdefault("post", []).append({"card": card, "pid": int(process.pid), "log": str(log), "status": "RUNNING", "argv": command})
    write_json(args.runtime, runtime)
    for card, process in post_processes:
        returncode = int(process.wait())
        for item in batch_record["post"]:
            if item["card"] == card:
                item.update({"returncode": returncode, "status": "PASS" if returncode == 0 else "FAILED"})
                break
        if returncode != 0:
            raise RuntimeError(f"Test-tuned postprocess failed for {card}")
        write_json(args.runtime, runtime)
    batch_record.update({"status": "PASS", "completed_at_unix": time.time()})
    write_json(args.runtime, runtime)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-json", type=Path, required=True)
    parser.add_argument("--frontend-root", type=Path, required=True)
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
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
    if int(args.batch_size) != 2:
        raise ValueError("this controller is registered for two exploratory cards per GPU batch")
    args.replay_root.mkdir(parents=True, exist_ok=True)
    ranking = _wait_for_ranking(args.ranking_json.resolve(), args.poll_seconds)
    top4 = [str(card) for card in ranking["top4"]]
    environment = capture_environment(args.capture.resolve(), args.repo.resolve(), args.cov_source.resolve())
    runtime: dict[str, Any] = {
        "status": "RUNNING",
        "artifact": "v12_mgf_test_tuned_exploration_replay_controller",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "selection_scope": "VAL_TEST_TUNED_EXPLORATION",
        "test_used_for_selection": True,
        "ranking_json": str(args.ranking_json.resolve()),
        "ranking_sha256": sha256_file(args.ranking_json.resolve()),
        "top4": top4,
        "batch_size": 2,
        "started_at_unix": time.time(),
    }
    write_json(args.runtime, runtime)
    try:
        for start in range(0, len(top4), 2):
            _start_card_batch(top4[start:start + 2], args=args, runtime=runtime, environment=environment)
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
