#!/usr/bin/env python3
"""Recover a B0 controller receipt after a postprocess-only failure.

The V12 replay launcher writes the merged prediction before invoking the
postprocessor.  Older runs therefore can have a complete, contract-valid
merge while the controller receipt is FAILED because the postprocessor tried
to merge the same output a second time.  This utility is deliberately
fail-closed: it only promotes that controller receipt after validating the
original failure, all shard contracts, the existing merge, the postprocess
runtime, the evaluation receipt, and the parsed metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--post-runtime", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    runtime_path = args.runtime.resolve()
    post_runtime_path = args.post_runtime.resolve()
    replay_root = args.replay_root.resolve()
    receipt_path = args.receipt.resolve()
    old_runtime = read_json(runtime_path)
    if old_runtime.get("status") != "FAILED":
        raise RuntimeError(f"refusing recovery from status {old_runtime.get('status')!r}")
    old_error = str(old_runtime.get("error", ""))
    if "postprocess" not in old_error.lower() and "merge root" not in old_error.lower():
        raise RuntimeError(f"failure is not an acknowledged postprocess-only failure: {old_error}")
    if old_runtime.get("paper_status") != "TEST_TUNED_EXPLORATION" or old_runtime.get("paper_valid") is not False:
        raise RuntimeError("old B0 runtime is missing Test-tuned diagnostic guards")

    post_runtime = read_json(post_runtime_path)
    required_post = {
        "status": "PASS",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "card_id": "B0_INITIAL",
        "split": "Current-Test",
        "merge_reused": True,
    }
    for key, expected in required_post.items():
        if post_runtime.get(key) != expected:
            raise RuntimeError(f"postprocess runtime guard failed for {key}: {post_runtime.get(key)!r}")

    shard_root = replay_root
    shard_manifests: list[Path] = []
    thresholds: Any = None
    for index in range(10):
        path = shard_root / f"shard_{index:02d}" / "manifest.json"
        require_file(path, f"shard {index:02d} manifest")
        manifest = read_json(path)
        if manifest.get("status") != "PASS":
            raise RuntimeError(f"shard is not PASS: {path}")
        if manifest.get("artifact") != "v12_qdic_b0_test_tuned_comparison_replay":
            raise RuntimeError(f"B0 shard artifact mismatch: {path}")
        if manifest.get("paper_status") != "TEST_TUNED_EXPLORATION" or manifest.get("paper_valid") is not False:
            raise RuntimeError(f"B0 shard paper-status guard failed: {path}")
        if manifest.get("diagnostic_only") is not True:
            raise RuntimeError(f"B0 shard diagnostic guard failed: {path}")
        if int(manifest.get("detector_forward_calls", -1)) != 0:
            raise RuntimeError(f"B0 detector contract failed: {path}")
        if manifest.get("gt_loaded_during_replay") is not False:
            raise RuntimeError(f"B0 GT replay contract failed: {path}")
        current_thresholds = manifest.get("thresholds")
        if thresholds is None:
            thresholds = current_thresholds
        elif current_thresholds != thresholds:
            raise RuntimeError(f"B0 threshold mismatch across shards: {path}")
        shard_manifests.append(path)

    merge_root = replay_root / "merged"
    merge_manifest_path = merge_root / "manifest.json"
    prediction_path = merge_root / "tao_track.json"
    require_file(merge_manifest_path, "B0 merge manifest")
    require_file(prediction_path, "B0 merged prediction")
    merge_manifest = read_json(merge_manifest_path)
    for key, expected in {
        "status": "PASS",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "shard_count": 10,
        "detector_forward_calls": 0,
        "gt_loaded_during_replay": False,
    }.items():
        if merge_manifest.get(key) != expected:
            raise RuntimeError(f"B0 merge guard failed for {key}: {merge_manifest.get(key)!r}")
    expected_prediction_hash = merge_manifest.get("prediction_sha256")
    actual_prediction_hash = sha256_file(prediction_path)
    if expected_prediction_hash and expected_prediction_hash != actual_prediction_hash:
        raise RuntimeError("B0 merged prediction hash mismatch")

    evaluation_path = Path(str(post_runtime.get("evaluation", ""))).resolve()
    metrics_path = Path(str(post_runtime.get("metrics", ""))).resolve()
    require_file(evaluation_path, "B0 evaluation receipt")
    require_file(metrics_path, "B0 parsed metrics")
    evaluation = read_json(evaluation_path)
    if evaluation.get("status") != "COMPLETED" or evaluation.get("paper_status") != "TEST_TUNED_EXPLORATION":
        raise RuntimeError("B0 evaluation receipt is not a completed Test-tuned diagnostic")
    if evaluation.get("paper_valid") is not False or evaluation.get("diagnostic_only") is not True:
        raise RuntimeError("B0 evaluation diagnostic guard failed")
    metrics = read_json(metrics_path)
    if metrics.get("status") != "PARSED" or not all(isinstance(metrics.get(split), dict) for split in ("overall", "base", "novel")):
        raise RuntimeError("B0 parsed metrics are incomplete")

    source_commit = git_head(repo)
    failed_backup = runtime_path.with_name(runtime_path.stem + "_failed_before_recovery.json")
    if not failed_backup.exists():
        shutil.copy2(runtime_path, failed_backup)

    corrected = dict(old_runtime)
    corrected.update(
        {
            "status": "PASS",
            "completed_at_unix": time.time(),
            "recovered_after_postprocess_fix": True,
            "recovery_reason": "complete merge existed; old postprocess rejected existing merge root",
            "recovery_source_commit": source_commit,
            "recovery_postprocess_runtime": str(post_runtime_path),
            "recovery_postprocess_runtime_sha256": sha256_file(post_runtime_path),
            "recovery_failed_runtime_backup": str(failed_backup),
            "recovery_failed_runtime_backup_sha256": sha256_file(failed_backup),
            "recovery_merge_manifest": str(merge_manifest_path),
            "recovery_merge_manifest_sha256": sha256_file(merge_manifest_path),
            "recovery_metrics": str(metrics_path),
            "recovery_metrics_sha256": sha256_file(metrics_path),
            "recovery_evaluation": str(evaluation_path),
            "recovery_evaluation_sha256": sha256_file(evaluation_path),
        }
    )
    write_json(runtime_path, corrected)

    receipt = {
        "status": "PASS",
        "artifact": "v12_mgf_b0_postprocess_only_recovery",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "test_used_for_selection": True,
        "recovery_reason": "complete causal replay and merge were valid; postprocess failed only because it attempted to overwrite the existing merge root",
        "source_commit": source_commit,
        "previous_runtime": {
            "path": str(failed_backup),
            "sha256": sha256_file(failed_backup),
            "status": old_runtime.get("status"),
            "error": old_error,
        },
        "corrected_runtime": {"path": str(runtime_path), "sha256": sha256_file(runtime_path)},
        "postprocess_runtime": {"path": str(post_runtime_path), "sha256": sha256_file(post_runtime_path)},
        "replay_root": str(replay_root),
        "shard_count": 10,
        "shard_manifests": [{"path": str(path), "sha256": sha256_file(path)} for path in shard_manifests],
        "merge_manifest": {"path": str(merge_manifest_path), "sha256": sha256_file(merge_manifest_path)},
        "merged_prediction": {"path": str(prediction_path), "sha256": actual_prediction_hash},
        "evaluation": {"path": str(evaluation_path), "sha256": sha256_file(evaluation_path)},
        "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        "thresholds": thresholds,
        "detector_forward_calls": 0,
        "gt_loaded_during_replay": False,
        "created_at_unix": time.time(),
    }
    write_json(receipt_path, receipt)
    # Bind the corrected runtime to the receipt after the receipt exists.
    corrected = read_json(runtime_path)
    corrected["recovery_receipt"] = str(receipt_path)
    corrected["recovery_receipt_sha256"] = sha256_file(receipt_path)
    write_json(runtime_path, corrected)
    receipt = read_json(receipt_path)
    receipt["corrected_runtime"]["sha256"] = sha256_file(runtime_path)
    write_json(receipt_path, receipt)
    print(json.dumps({"status": "PASS", "runtime": str(runtime_path), "receipt": str(receipt_path), "source_commit": source_commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
