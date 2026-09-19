#!/usr/bin/env python3
"""Recover the V12 Test-tuned exploration controller after a postprocess bug.

This tool is intentionally narrower than the original controller: it accepts
only the known failed first batch, validates the immutable E05/E10 replay and
merge artifacts, completes their postprocess, and then starts exactly the
remaining E07/E06 batch.  It never reruns a completed replay shard.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

# Support both ``python -m tools...`` and the direct-file invocation used by
# the detached recovery command.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.v12_launch_mgf_exploration_replays import (
    _start_card_batch,
    capture_environment,
    read_json,
    sha256_file,
    write_json,
)
from tools.v12_post_mgf_exploration_replay import (
    _validate_existing_merge,
    _validate_manifests,
)


FIRST_BATCH = ["E05", "E10"]
SECOND_BATCH = ["E07", "E06"]


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo.resolve()), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _post_command(args: argparse.Namespace, card: str) -> list[str]:
    replay_root = args.replay_root.resolve() / card / "s00_m00"
    return [
        str(args.python_post.resolve()), "-u",
        str(args.repo.resolve() / "tools" / "v12_post_mgf_exploration_replay.py"),
        "--replay-root", str(replay_root),
        "--full-cache", str(args.full_cache.resolve()),
        "--annotation", str(args.annotation.resolve()),
        "--repo", str(args.repo.resolve()),
        "--python", str(args.python_post.resolve()),
        "--shard-count", str(int(args.shard_count)),
        "--poll-seconds", "5",
        "--evaluation-cores", str(int(args.evaluation_cores)),
        "--card-id", card,
    ]


def _validate_failed_controller(runtime: dict[str, Any], args: argparse.Namespace) -> None:
    if runtime.get("status") != "FAILED":
        raise RuntimeError(f"recovery requires FAILED controller runtime, got {runtime.get('status')!r}")
    if runtime.get("top4") != FIRST_BATCH + SECOND_BATCH:
        raise RuntimeError(f"unexpected controller top4: {runtime.get('top4')!r}")
    if int(runtime.get("batch_size", -1)) != 2:
        raise RuntimeError("recovery requires the registered two-card batch size")
    batches = runtime.get("batches")
    if not isinstance(batches, list) or len(batches) != 1:
        raise RuntimeError(f"recovery requires exactly one recorded batch: {batches!r}")
    batch = batches[0]
    if batch.get("cards") != FIRST_BATCH or batch.get("status") != "MERGE_PASS":
        raise RuntimeError(
            f"recovery requires first batch MERGE_PASS for {FIRST_BATCH}: "
            f"{batch.get('cards')!r} / {batch.get('status')!r}"
        )
    if "Test-tuned postprocess failed for E05" not in str(runtime.get("error", "")):
        raise RuntimeError("controller failure is not the audited E05 postprocess failure")
    for item in batch.get("post", []):
        pid = item.get("pid")
        if pid and Path(f"/proc/{int(pid)}").exists():
            raise RuntimeError(f"refusing duplicate postprocess while PID {pid} is alive")
    for card in FIRST_BATCH:
        replay_root = args.replay_root.resolve() / card / "s00_m00"
        manifests = _validate_manifests(replay_root, int(args.shard_count))
        if len(manifests) != int(args.shard_count):
            raise RuntimeError(f"{card} does not have all shard manifests")
        _validate_existing_merge(
            replay_root / "merged", card_id=card, full_cache=args.full_cache.resolve()
        )
    for card in SECOND_BATCH:
        output_root = args.replay_root.resolve() / card / "s00_m00"
        if output_root.exists() and any(output_root.iterdir()):
            raise RuntimeError(f"refusing to overwrite pre-existing second-batch output: {output_root}")


def _run_first_batch_postprocess(args: argparse.Namespace, environment: dict[str, str], receipt: dict[str, Any]) -> None:
    recovery_logs = args.recovery_log_root.resolve()
    recovery_logs.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen[Any], Path, list[str]]] = []
    for card in FIRST_BATCH:
        command = _post_command(args, card)
        log = recovery_logs / f"{card}_post.log"
        handle = log.open("w", encoding="utf-8")
        handle.write(json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(args.repo.resolve()),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle.close()
        processes.append((card, process, log, command))
        receipt.setdefault("post_processes", []).append(
            {"card": card, "pid": int(process.pid), "log": str(log), "argv": command, "status": "RUNNING"}
        )
    write_json(args.recovery_receipt, receipt)

    for card, process, log, command in processes:
        returncode = int(process.wait())
        entry = next(item for item in receipt["post_processes"] if item["card"] == card)
        entry.update({"returncode": returncode, "status": "PASS" if returncode == 0 else "FAILED", "ended_at_unix": time.time()})
        write_json(args.recovery_receipt, receipt)
        if returncode != 0:
            raise RuntimeError(f"recovered postprocess failed for {card}; see {log}")
        runtime_manifest = args.replay_root.resolve() / card / f"{card}_test_tuned_runtime_manifest.json"
        if not runtime_manifest.is_file():
            raise RuntimeError(f"postprocess runtime manifest missing for {card}: {runtime_manifest}")
        value = read_json(runtime_manifest)
        if value.get("status") != "PASS" or value.get("merge_reused") is not True:
            raise RuntimeError(f"postprocess did not PASS with merge reuse for {card}: {value}")
        metrics = value.get("metrics")
        if not metrics or not Path(str(metrics)).is_file():
            raise RuntimeError(f"postprocess metrics missing for {card}: {value}")
        entry.update({"runtime_manifest": str(runtime_manifest), "runtime_manifest_sha256": sha256_file(runtime_manifest)})
        write_json(args.recovery_receipt, receipt)


def _mark_first_batch_recovered(runtime: dict[str, Any], args: argparse.Namespace, receipt: dict[str, Any]) -> None:
    batch = runtime["batches"][0]
    recovered_by_card = {item["card"]: item for item in receipt["post_processes"]}
    for item in batch.get("post", []):
        recovered = recovered_by_card.get(item.get("card"))
        if recovered is None or recovered.get("status") != "PASS":
            raise RuntimeError(f"missing recovered post record for {item.get('card')}")
        item.update(
            {
                "status": "PASS",
                "returncode": 0,
                "recovered": True,
                "recovery_log": recovered["log"],
                "recovered_at_unix": recovered["ended_at_unix"],
            }
        )
    batch.update(
        {
            "status": "PASS",
            "completed_at_unix": time.time(),
            "recovered_postprocess": True,
        }
    )
    runtime.update({"status": "RUNNING", "recovery": receipt, "updated_at_unix": time.time()})
    write_json(args.controller_runtime, runtime)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--controller-runtime", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--frontend-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--cov-source", type=Path, required=True)
    parser.add_argument("--cov-config", type=Path, required=True)
    parser.add_argument("--cov-checkpoint", type=Path, required=True)
    parser.add_argument("--python-replay", type=Path, required=True)
    parser.add_argument("--python-post", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--evaluation-cores", type=int, default=8)
    parser.add_argument("--recovery-log-root", type=Path, required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    args = parser.parse_args()
    args.runtime = args.controller_runtime

    runtime = read_json(args.controller_runtime.resolve())
    _validate_failed_controller(runtime, args)
    source_commit = _git_head(args.repo)
    receipt: dict[str, Any] = {
        "status": "RUNNING",
        "artifact": "v12_mgf_test_tuned_controller_recovery",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "reason": "recover_postprocess_after_existing_merge_root_contract_failure",
        "source_commit": source_commit,
        "controller_runtime": str(args.controller_runtime.resolve()),
        "started_at_unix": time.time(),
        "recovered_cards": FIRST_BATCH,
        "next_cards": SECOND_BATCH,
        "post_processes": [],
    }
    write_json(args.recovery_receipt, receipt)
    environment = capture_environment(args.capture.resolve(), args.repo.resolve(), args.cov_source.resolve())
    try:
        _run_first_batch_postprocess(args, environment, receipt)
        receipt.update({"first_batch_post_status": "PASS", "first_batch_post_completed_at_unix": time.time()})
        _mark_first_batch_recovered(runtime, args, receipt)
        _start_card_batch(SECOND_BATCH, args=args, runtime=runtime, environment=environment)
        runtime = read_json(args.controller_runtime.resolve())
        runtime.update({"status": "PASS", "completed_at_unix": time.time(), "controller_pid": os.getpid()})
        write_json(args.controller_runtime, runtime)
        receipt.update({"status": "PASS", "completed_at_unix": time.time(), "controller_runtime_sha256": sha256_file(args.controller_runtime.resolve())})
        write_json(args.recovery_receipt, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        receipt.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}", "ended_at_unix": time.time()})
        write_json(args.recovery_receipt, receipt)
        runtime = read_json(args.controller_runtime.resolve())
        runtime.update({"status": "FAILED", "error": receipt["error"], "recovery": receipt, "ended_at_unix": time.time()})
        write_json(args.controller_runtime, runtime)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
