#!/usr/bin/env python3
"""Close the frozen V12 MGF Official-Val replay.

The replay workers never read annotations.  This supervisor waits for their
causal manifests, validates the no-detector/no-GT contract, merges the ten
video-shard predictions, and only then invokes the official TETA evaluator.
It is intentionally a Val-only postprocessor; it cannot select a card or
change beta/thresholds after the Train freeze.
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
    for index in range(int(shard_count)):
        path = shard_root / f"shard_{index:02d}" / "manifest.json"
        manifest = read_json(path)
        if manifest.get("status") != "PASS":
            raise RuntimeError(f"V12 shard is not PASS: {path}")
        if manifest.get("artifact") != "v12_qdic_mgf_causal_replay":
            raise RuntimeError(f"V12 shard artifact mismatch: {path}")
        if int(manifest.get("detector_forward_calls", -1)) != 0:
            raise RuntimeError(f"detector_forward_calls is nonzero: {path}")
        if manifest.get("gt_loaded_during_replay") is not False:
            raise RuntimeError(f"GT was loaded during replay: {path}")
        thresholds = manifest.get("thresholds")
        if not isinstance(thresholds, dict) or float(thresholds.get("score_threshold", float("nan"))) != 0.0:
            raise RuntimeError(f"V12 replay threshold contract failed: {path}")
        manifests.append(manifest)
    return manifests


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


def _teta_metrics(
    summary: Path,
    *,
    category_protocol: Any,
    evaluation_manifest: Path,
) -> dict[str, Any]:
    from tempotrack_research.evaluation.teta_parser import inspect_installed_teta, parse_teta_summary

    parsed = parse_teta_summary(
        summary,
        category_protocol=category_protocol,
        teta_schema=inspect_installed_teta(),
        evaluation_manifest=read_json(evaluation_manifest),
    )
    return parsed


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
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
    parser.add_argument("--replay-root", type=Path, required=True, help="s00_m00 root containing shard_00..09")
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--evaluation-cores", type=int, default=8)
    args = parser.parse_args()

    replay_root = args.replay_root.resolve()
    full_cache = args.full_cache.resolve()
    annotation = args.annotation.resolve()
    repo = args.repo.resolve()
    runtime_path = replay_root.parent / "v12_mgf_replay_runtime_manifest.json"
    runtime: dict[str, Any] = {
        "status": "WAITING_FOR_SHARDS",
        "artifact": "v12_mgf_official_val_replay_postprocess",
        "replay_root": str(replay_root),
        "full_cache": str(full_cache),
        "annotation": str(annotation),
        "annotation_sha256": sha256_file(annotation),
        "shard_count": int(args.shard_count),
        "test_used_for_selection": False,
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
                raise RuntimeError(f"V12 replay shard failure: {invalid}")
            if not missing and len(records) == int(args.shard_count):
                break
            time.sleep(max(1.0, float(args.poll_seconds)))

        manifests = _validate_manifests(replay_root, args.shard_count)
        merge_root = replay_root / "merged"
        if merge_root.exists() and any(merge_root.iterdir()):
            raise RuntimeError(f"refusing to overwrite existing merge root: {merge_root}")
        merge_script = repo / "tools" / "v12_merge_mgf_replay.py"
        environment = os.environ.copy()
        pythonpath = [str(repo)]
        if environment.get("PYTHONPATH"):
            pythonpath.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
        merge_log = replay_root / "merge.log"
        _run(
            [
                str(args.python.resolve()),
                str(merge_script),
                "--shard-root",
                str(replay_root),
                "--full-cache",
                str(full_cache),
                "--output-root",
                str(merge_root),
                "--shard-count",
                str(int(args.shard_count)),
                "--trial-id",
                "s00_m00",
            ],
            cwd=repo,
            environment=environment,
            log=merge_log,
        )
        merge_manifest = merge_root / "manifest.json"
        merged_prediction = merge_root / "tao_track.json"
        if not merge_manifest.is_file() or not merged_prediction.is_file():
            raise RuntimeError("V12 merge did not produce a complete prediction")

        evaluation_root = merge_root / "evaluation"
        evaluation_name = "V12_MGF_VAL"
        evaluation_dir = evaluation_root / evaluation_name
        evaluation_manifest = evaluation_dir / "evaluation.json"
        summary = evaluation_dir / "teta_summary_results.pth"
        evaluation_environment = dict(environment)
        evaluation_environment["CUDA_VISIBLE_DEVICES"] = ""
        eval_log = evaluation_dir / "official.log"
        _run(
            [
                str(args.python.resolve()),
                str(repo / "tools" / "eval_ovmot_teta.py"),
                "--gt",
                str(annotation),
                "--pred",
                str(merged_prediction),
                "--out",
                str(evaluation_root),
                "--name",
                evaluation_name,
                "--cores",
                str(int(args.evaluation_cores)),
            ],
            cwd=repo,
            environment=evaluation_environment,
            log=eval_log,
        )
        if not summary.is_file():
            raise RuntimeError(f"official TETA summary missing: {summary}")
        category_protocol = _load_annotation_protocol(annotation)
        evaluation_receipt = {
            "status": "COMPLETED",
            "artifact": "v12_mgf_official_val_teta_evaluation",
            "prediction": str(merged_prediction.resolve()),
            "prediction_sha256": sha256_file(merged_prediction),
            "annotation": str(annotation),
            "annotation_sha256": sha256_file(annotation),
            "summary": str(summary.resolve()),
            "summary_sha256": sha256_file(summary),
            "category_protocol_hash": category_protocol.content_hash(),
            "test_used_for_selection": False,
            "selection_scope": "OFFICIAL_TRAIN_INTERNAL_HOLDOUT_ONLY",
        }
        write_json(evaluation_manifest, evaluation_receipt)
        metrics = _teta_metrics(
            summary,
            category_protocol=category_protocol,
            evaluation_manifest=evaluation_manifest,
        )
        metrics_path = merge_root / "val_metrics.json"
        write_json(metrics_path, metrics)
        _write_metrics_csv(merge_root / "val_metrics.csv", metrics)
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
                "test_status": "TEST_MGF_CACHE_UNAVAILABLE_UNLESS_SEPARATE_LEGAL_CACHE_IS_FOUND",
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
