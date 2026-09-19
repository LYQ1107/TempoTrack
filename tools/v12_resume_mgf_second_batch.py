#!/usr/bin/env python3
"""Retry only the failed E07/E06 Test-tuned replay batch.

This is a fail-closed recovery tool for one audited failure mode: the first
E05/E10 batch is complete, while the second E07/E06 batch exited before replay
because an invalid checkpoint path was supplied.  It preserves the failed
attempt in the controller runtime and uses a distinct retry log namespace.
No completed replay is rerun.
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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.v12_launch_mgf_exploration_replays import (  # noqa: E402
    _start_card_batch,
    capture_environment,
    read_json,
    sha256_file,
    write_json,
)
from tools.v12_post_mgf_exploration_replay import (  # noqa: E402
    _validate_existing_merge,
    _validate_manifests,
)


FIRST_BATCH = ["E05", "E10"]
SECOND_BATCH = ["E07", "E06"]
BAD_CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/"
    "saved_models/ctao_public.pth"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c"
)


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo.resolve()), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _argv_has(argv: Any, value: str) -> bool:
    return isinstance(argv, list) and value in [str(item) for item in argv]


def _validate_successful_initial_batch(args: argparse.Namespace, runtime: dict[str, Any]) -> None:
    batches = runtime.get("batches")
    if not isinstance(batches, list) or len(batches) != 2:
        raise RuntimeError(f"expected exactly two historical batches, got {batches!r}")
    first = batches[0]
    if first.get("cards") != FIRST_BATCH or first.get("status") != "PASS":
        raise RuntimeError("E05/E10 batch is not immutable PASS")
    if len(first.get("workers", [])) != 20 or any(
        item.get("status") != "PASS" for item in first.get("workers", [])
    ):
        raise RuntimeError("E05/E10 does not have 20 PASS workers")
    for card in FIRST_BATCH:
        root = args.replay_root.resolve() / card / "s00_m00"
        manifests = _validate_manifests(root, int(args.shard_count))
        if len(manifests) != int(args.shard_count):
            raise RuntimeError(f"{card} is missing shard manifests")
        _validate_existing_merge(
            root / "merged", card_id=card, full_cache=args.full_cache.resolve()
        )
        runtime_manifest = args.replay_root.resolve() / card / f"{card}_test_tuned_runtime_manifest.json"
        if not runtime_manifest.is_file() or read_json(runtime_manifest).get("status") != "PASS":
            raise RuntimeError(f"{card} runtime manifest is not PASS: {runtime_manifest}")


def _validate_failed_second_batch(args: argparse.Namespace, runtime: dict[str, Any]) -> None:
    if runtime.get("status") != "FAILED":
        raise RuntimeError(f"retry requires FAILED controller runtime, got {runtime.get('status')!r}")
    if runtime.get("top4") != FIRST_BATCH + SECOND_BATCH:
        raise RuntimeError(f"unexpected top4: {runtime.get('top4')!r}")
    batches = runtime.get("batches")
    second = batches[1]
    if second.get("cards") != SECOND_BATCH or second.get("status") != "FAILED":
        raise RuntimeError("E07/E06 historical batch is not FAILED")
    workers = second.get("workers", [])
    if len(workers) != 20 or any(item.get("status") != "FAILED" for item in workers):
        raise RuntimeError("E07/E06 does not have exactly 20 failed workers")
    for item in workers:
        if item.get("returncode") != 1:
            raise RuntimeError(f"unexpected E07/E06 return code: {item}")
        if not _argv_has(item.get("argv"), str(BAD_CHECKPOINT)):
            raise RuntimeError("historical failure is not the audited bad checkpoint invocation")
        pid = item.get("pid")
        if pid and Path(f"/proc/{int(pid)}").exists():
            raise RuntimeError(f"refusing retry while historical worker is alive: {pid}")
        log = Path(str(item.get("log", "")))
        if not log.is_file():
            raise RuntimeError(f"historical worker log missing: {log}")
        content = log.read_text(encoding="utf-8", errors="replace")
        if str(BAD_CHECKPOINT) not in content or "is not a checkpoint file" not in content:
            raise RuntimeError(f"historical worker log lacks exact checkpoint failure: {log}")
    for card in SECOND_BATCH:
        output_root = args.replay_root.resolve() / card / "s00_m00"
        if not output_root.is_dir():
            raise RuntimeError(f"expected failed output root: {output_root}")
        # The launcher creates one empty shard directory before the worker
        # reaches model initialization.  Those directories are safe to reuse;
        # any file or nested artifact inside one is not.
        expected_shards = {f"shard_{index:02d}" for index in range(int(args.shard_count))}
        actual_shards = {item.name for item in output_root.iterdir() if item.is_dir()}
        if actual_shards != expected_shards:
            raise RuntimeError(
                f"unexpected failed output shard layout for {output_root}: "
                f"{sorted(actual_shards)} != {sorted(expected_shards)}"
            )
        unexpected = [
            item for item in output_root.iterdir()
            if not item.is_dir() or any(item.iterdir())
        ]
        if unexpected:
            raise RuntimeError(f"refusing to reuse non-empty failed shard outputs: {unexpected}")


def _make_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    args.runtime = args.controller_runtime
    args.worker_log_tag = "s00_m00_retry01"
    runtime = read_json(args.controller_runtime.resolve())
    _validate_successful_initial_batch(args, runtime)
    _validate_failed_second_batch(args, runtime)
    checkpoint = args.cov_checkpoint.resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"corrected checkpoint missing: {checkpoint}")
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"corrected checkpoint hash mismatch: {actual_sha256} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    if str(checkpoint) == str(BAD_CHECKPOINT):
        raise RuntimeError("corrected checkpoint still equals audited bad path")

    source_commit = _git_head(args.repo)
    receipt: dict[str, Any] = {
        "status": "RUNNING",
        "artifact": "v12_mgf_test_tuned_second_batch_retry",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "reason": "retry_e07_e06_after_pre_replay_bad_checkpoint_path",
        "source_commit": source_commit,
        "controller_runtime": str(args.controller_runtime.resolve()),
        "failed_batch_index": 1,
        "retry_batch_index": 2,
        "retry_cards": SECOND_BATCH,
        "failed_checkpoint": str(BAD_CHECKPOINT),
        "corrected_checkpoint": str(checkpoint),
        "corrected_checkpoint_sha256": actual_sha256,
        "worker_log_tag": args.worker_log_tag,
        "started_at_unix": time.time(),
    }
    write_json(args.recovery_receipt, receipt)
    environment = capture_environment(args.capture.resolve(), args.repo.resolve(), args.cov_source.resolve())
    try:
        _start_card_batch(SECOND_BATCH, args=args, runtime=runtime, environment=environment)
        runtime = read_json(args.controller_runtime.resolve())
        runtime.pop("error", None)
        runtime.update({
            "status": "PASS",
            "completed_at_unix": time.time(),
            "second_batch_retry": receipt,
        })
        write_json(args.controller_runtime, runtime)
        receipt.update({
            "status": "PASS",
            "completed_at_unix": time.time(),
            "controller_runtime_sha256": sha256_file(args.controller_runtime.resolve()),
        })
        write_json(args.recovery_receipt, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        receipt.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}", "ended_at_unix": time.time()})
        write_json(args.recovery_receipt, receipt)
        runtime = read_json(args.controller_runtime.resolve())
        runtime.update({"status": "FAILED", "error": receipt["error"], "second_batch_retry": receipt, "ended_at_unix": time.time()})
        write_json(args.controller_runtime, runtime)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
