#!/usr/bin/env python3
"""Close one TEST_TUNED_EXPLORATION Full-Test replay.

This postprocessor is intentionally separate from the formal Official-Val
supervisor.  It accepts only the diagnostic exploration replay artifact and
labels every evaluation as Test-tuned/non-paper-valid.
"""

from __future__ import annotations

import argparse
import csv
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _manifest_state(shard_root: Path, shard_count: int) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for index in range(int(shard_count)):
        shard = f"shard_{index:02d}"
        path = shard_root / shard / "manifest.json"
        if not path.is_file():
            missing.append(shard)
            continue
        try:
            manifest = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            records.append({"shard": shard, "status": "INVALID", "error": str(error), "manifest": str(path)})
            continue
        records.append(
            {
                "shard": shard,
                "status": manifest.get("status"),
                "artifact": manifest.get("artifact"),
                "frames": manifest.get("frames"),
                "videos": manifest.get("videos"),
                "rows": manifest.get("rows"),
                "detector_forward_calls": manifest.get("detector_forward_calls"),
                "gt_loaded_during_replay": manifest.get("gt_loaded_during_replay"),
                "manifest": str(path.resolve()),
            }
        )
    return records, missing


def _validate_manifests(shard_root: Path, shard_count: int) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    for index in range(int(shard_count)):
        path = shard_root / f"shard_{index:02d}" / "manifest.json"
        manifest = read_json(path)
        if manifest.get("status") != "PASS":
            raise RuntimeError(f"exploration replay shard is not PASS: {path}")
        artifact = manifest.get("artifact")
        if artifact not in {
            "v12_qdic_mgf_test_tuned_exploration_replay",
            "v12_qdic_b0_test_tuned_comparison_replay",
        }:
            raise RuntimeError(f"exploration replay artifact mismatch: {path}")
        if artifact == "v12_qdic_b0_test_tuned_comparison_replay":
            if manifest.get("paper_status") != "TEST_TUNED_EXPLORATION":
                raise RuntimeError(f"B0 Test-tuned paper status missing: {path}")
            if manifest.get("paper_valid") is not False or manifest.get("diagnostic_only") is not True:
                raise RuntimeError(f"B0 Test-tuned diagnostic guard failed: {path}")
        elif "paper_status" in manifest:
            if manifest.get("paper_status") != "TEST_TUNED_EXPLORATION":
                raise RuntimeError(f"Test-tuned comparison paper status invalid: {path}")
            if manifest.get("paper_valid") is not False or manifest.get("diagnostic_only") is not True:
                raise RuntimeError(f"Test-tuned diagnostic guard failed: {path}")
        if int(manifest.get("detector_forward_calls", -1)) != 0:
            raise RuntimeError(f"detector_forward_calls is nonzero: {path}")
        if manifest.get("gt_loaded_during_replay") is not False:
            raise RuntimeError(f"GT was loaded during replay: {path}")
        provenance = manifest.get("mgf_provenance")
        if not isinstance(provenance, dict):
            raise RuntimeError(f"exploration provenance guard failed: {path}")
        if artifact == "v12_qdic_mgf_test_tuned_exploration_replay":
            if provenance.get("paper_status") != "TEST_TUNED_EXPLORATION":
                raise RuntimeError(f"exploration provenance guard failed: {path}")
        else:
            if (
                provenance.get("status") != "QDIC_V11_MODEL_CODE_AND_WEIGHTS"
                or provenance.get("paper_status") != "BASE_TRAIN"
                or provenance.get("comparison_role") != "B0_OFFICIAL_V11_TEST_TUNED_COMPARATOR"
            ):
                raise RuntimeError(f"B0 comparator provenance guard failed: {path}")
        if reference is not None:
            if manifest.get("thresholds") != reference.get("thresholds"):
                raise RuntimeError("exploration replay threshold mismatch across shards")
            if provenance.get("checkpoint_sha256") != reference.get("mgf_provenance", {}).get("checkpoint_sha256"):
                raise RuntimeError("exploration replay checkpoint mismatch across shards")
        reference = manifest
        manifests.append(manifest)
    return manifests


def _validate_existing_merge(merge_root: Path, shard_count: int) -> tuple[Path, dict[str, Any]]:
    """Validate a merge already produced by the replay launcher.

    The launcher performs the merge before it starts this postprocessor.  The
    postprocessor is also callable on its own for recovery, so an existing
    complete merge must be reused rather than treated as an overwrite.  An
    incomplete or contract-incompatible directory remains fail-closed.
    """
    merge_manifest = merge_root / "manifest.json"
    merged_prediction = merge_root / "tao_track.json"
    if not merge_manifest.is_file() or not merged_prediction.is_file():
        raise RuntimeError(
            f"existing merge root is incomplete; refusing to overwrite: {merge_root}"
        )
    manifest = read_json(merge_manifest)
    if manifest.get("status") != "PASS":
        raise RuntimeError(f"existing merge manifest is not PASS: {merge_manifest}")
    if int(manifest.get("shard_count", -1)) != int(shard_count):
        raise RuntimeError(f"existing merge shard-count mismatch: {merge_manifest}")
    if int(manifest.get("detector_forward_calls", -1)) != 0:
        raise RuntimeError(f"existing merge detector contract failed: {merge_manifest}")
    if manifest.get("gt_loaded_during_replay") is not False:
        raise RuntimeError(f"existing merge GT contract failed: {merge_manifest}")
    expected_hash = manifest.get("prediction_sha256")
    if expected_hash and expected_hash != sha256_file(merged_prediction):
        raise RuntimeError(f"existing merge prediction hash mismatch: {merged_prediction}")
    if manifest.get("paper_status") != "TEST_TUNED_EXPLORATION":
        raise RuntimeError(f"existing merge paper status invalid: {merge_manifest}")
    return merged_prediction, manifest


def _run(command: list[str], *, cwd: Path, environment: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), env=environment, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _load_annotation_protocol(annotation: Path) -> Any:
    from tools.v11_evaluate_b0_calibration import AnnotationCategoryProtocol

    payload = read_json(annotation)
    categories = payload.get("categories", [])
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"annotation has no category metadata: {annotation}")
    return AnnotationCategoryProtocol(categories)


def _teta_metrics(summary: Path, *, category_protocol: Any, evaluation_manifest: Path) -> dict[str, Any]:
    from tempotrack_research.evaluation.teta_parser import inspect_installed_teta, parse_teta_summary

    return parse_teta_summary(
        summary,
        category_protocol=category_protocol,
        teta_schema=inspect_installed_teta(),
        evaluation_manifest=read_json(evaluation_manifest),
    )


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = []
    for split in ("overall", "base", "novel"):
        value = metrics.get(split)
        if isinstance(value, dict):
            rows.append({"split": split, **value})
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--evaluation-cores", type=int, default=8)
    parser.add_argument("--card-id", required=True)
    parser.add_argument(
        "--split",
        choices=("test_tuned", "official_val_tuned"),
        default="test_tuned",
        help="evaluation split label; official_val_tuned writes val_metrics.json",
    )
    args = parser.parse_args()

    replay_root = args.replay_root.resolve()
    full_cache = args.full_cache.resolve()
    annotation = args.annotation.resolve()
    repo = args.repo.resolve()
    is_val = args.split == "official_val_tuned"
    runtime_suffix = "official_val_tuned" if is_val else "test_tuned"
    metric_stem = "val_metrics" if is_val else "test_tuned_metrics"
    evaluation_name = f"{args.card_id}_OFFICIAL_VAL_TUNED" if is_val else f"{args.card_id}_TEST_TUNED"
    runtime_path = replay_root.parent / f"{args.card_id}_{runtime_suffix}_runtime_manifest.json"
    runtime: dict[str, Any] = {
        "status": "WAITING_FOR_SHARDS",
        "artifact": (
            "v12_mgf_test_tuned_exploration_official_val_postprocess"
            if is_val
            else "v12_mgf_test_tuned_exploration_replay_postprocess"
        ),
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "selection_scope": "VAL_TEST_TUNED_EXPLORATION",
        "test_used_for_selection": True,
        "split": "Official-Val" if is_val else "Current-Test",
        "card_id": args.card_id,
        "replay_root": str(replay_root),
        "full_cache": str(full_cache),
        "annotation": str(annotation),
        "annotation_sha256": sha256_file(annotation),
        "shard_count": int(args.shard_count),
        "started_at_unix": time.time(),
    }
    write_json(runtime_path, runtime)
    try:
        while True:
            records, missing = _manifest_state(replay_root, args.shard_count)
            runtime.update({"shard_records": records, "missing_shards": missing, "updated_at_unix": time.time()})
            write_json(runtime_path, runtime)
            invalid = [item for item in records if item.get("status") in {"FAILED", "INVALID"}]
            if invalid:
                raise RuntimeError(f"exploration replay shard failure: {invalid}")
            if not missing and len(records) == int(args.shard_count):
                break
            time.sleep(max(1.0, float(args.poll_seconds)))

        manifests = _validate_manifests(replay_root, args.shard_count)
        merge_root = replay_root / "merged"
        environment = os.environ.copy()
        pythonpath = [str(repo)]
        if environment.get("PYTHONPATH"):
            pythonpath.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
        if merge_root.exists():
            merged_prediction, _merge_manifest = _validate_existing_merge(
                merge_root, int(args.shard_count)
            )
            merge_manifest = merge_root / "manifest.json"
            runtime["merge_reused"] = True
            runtime["merge_manifest_sha256"] = sha256_file(merge_manifest)
        else:
            merge_log = replay_root / "merge.log"
            _run(
                [
                    str(args.python.resolve()), str(repo / "tools" / "v12_merge_mgf_replay.py"),
                    "--shard-root", str(replay_root), "--full-cache", str(full_cache),
                    "--output-root", str(merge_root), "--shard-count", str(int(args.shard_count)),
                    "--trial-id", args.card_id,
                ], cwd=repo, environment=environment, log=merge_log,
            )
            merge_manifest = merge_root / "manifest.json"
            merged_prediction = merge_root / "tao_track.json"
            if not merge_manifest.is_file() or not merged_prediction.is_file():
                raise RuntimeError("exploration merge did not produce a complete prediction")

        evaluation_root = merge_root / "evaluation"
        evaluation_dir = evaluation_root / evaluation_name
        evaluation_manifest = evaluation_dir / "evaluation.json"
        summary = evaluation_dir / "teta_summary_results.pth"
        evaluation_environment = dict(environment)
        evaluation_environment["CUDA_VISIBLE_DEVICES"] = ""
        _run(
            [
                str(args.python.resolve()), str(repo / "tools" / "eval_ovmot_teta.py"),
                "--gt", str(annotation), "--pred", str(merged_prediction),
                "--out", str(evaluation_root), "--name", evaluation_name,
                "--cores", str(int(args.evaluation_cores)),
            ], cwd=repo, environment=evaluation_environment, log=evaluation_dir / "official.log",
        )
        if not summary.is_file():
            raise RuntimeError(f"official TETA summary missing: {summary}")
        category_protocol = _load_annotation_protocol(annotation)
        evaluation_receipt = {
            "status": "COMPLETED",
            "artifact": (
                "v12_mgf_test_tuned_exploration_official_val_teta_evaluation"
                if is_val
                else "v12_mgf_test_tuned_exploration_teta_evaluation"
            ),
            "paper_status": "TEST_TUNED_EXPLORATION",
            "paper_valid": False,
            "diagnostic_only": True,
            "test_used_for_selection": True,
            "selection_scope": "VAL_TEST_TUNED_EXPLORATION",
            "card_id": args.card_id,
            "prediction": str(merged_prediction.resolve()),
            "prediction_sha256": sha256_file(merged_prediction),
            "annotation": str(annotation),
            "annotation_sha256": sha256_file(annotation),
            "summary": str(summary.resolve()),
            "summary_sha256": sha256_file(summary),
            "category_protocol_hash": category_protocol.content_hash(),
        }
        write_json(evaluation_manifest, evaluation_receipt)
        metrics = _teta_metrics(summary, category_protocol=category_protocol, evaluation_manifest=evaluation_manifest)
        metrics_path = merge_root / f"{metric_stem}.json"
        write_json(metrics_path, metrics)
        _write_metrics_csv(merge_root / f"{metric_stem}.csv", metrics)
        runtime.update(
            {
                "status": "PASS",
                "completed_at_unix": time.time(),
                "validated_shards": manifests,
                "merge_manifest": str(merge_manifest.resolve()),
                "merge_manifest_sha256": sha256_file(merge_manifest),
                "merged_prediction": str(merged_prediction.resolve()),
                "merged_prediction_sha256": sha256_file(merged_prediction),
                "evaluation": str(evaluation_manifest.resolve()),
                "metrics": str(metrics_path.resolve()),
            }
        )
        write_json(runtime_path, runtime)
        print(json.dumps(runtime, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        runtime.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}", "ended_at_unix": time.time()})
        write_json(runtime_path, runtime)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
