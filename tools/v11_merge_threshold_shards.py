#!/usr/bin/env python3
"""Merge sharded causal threshold trials into evaluator-ready trials.

The reduced B0 calibration replay is intentionally laid out as::

    calibration_root/shard_XX/s00_mYY/{manifest.json,tao_track.json}

The historical aggregate validator consumes a flat trial root instead.  This
adapter bridges those layouts without changing any prediction row or replay
parameter.  It validates every shard first, then delegates the actual
complete-video merge to ``v10_merge_complete_video_tao.py``.  No GT is read
until the merge utility validates annotation coverage; the replay manifests
remain the source of all causal guards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable


DEFAULT_TRIALS = ("s00_m00", "s00_m01", "s00_m02", "s00_m03", "s00_m04")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def trial_path(root: Path, shard_index: int, trial_id: str) -> Path:
    return root / f"shard_{shard_index:02d}" / trial_id


def formal_pass(manifest: dict[str, Any], trial_id: str) -> bool:
    return bool(
        manifest.get("status") == "PASS"
        and manifest.get("trial_id") == trial_id
        and int(manifest.get("detector_forward_calls", -1)) == 0
        and manifest.get("gt_loaded_during_replay") is False
    )


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _require_file(value: Any, label: str) -> Path:
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _event_bundle(manifests: Iterable[dict[str, Any]]) -> list[str]:
    first: list[str] | None = None
    for manifest in manifests:
        events = [str(value) for value in manifest.get("events", ())]
        if not events:
            raise RuntimeError(f"trial manifest has no event sources: {manifest.get('trial_id')}")
        for event in events:
            if not Path(event).is_file():
                raise FileNotFoundError(f"event source is missing: {event}")
        if first is None:
            first = events
        elif events != first:
            raise RuntimeError("event source list mismatch across calibration shards")
    return first or []


def validate_trial(
    *, root: Path, shard_annotation_root: Path, trial_id: str, shard_count: int
) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    for index in range(shard_count):
        trial_root = trial_path(root, index, trial_id)
        manifest_path = trial_root / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"{trial_id}: missing shard manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        if not formal_pass(manifest, trial_id):
            raise RuntimeError(f"{trial_id}: shard_{index:02d} is not a formal PASS")
        prediction = _require_file(manifest.get("prediction"), f"{trial_id} prediction")
        expected_hash = str(manifest.get("prediction_sha256", ""))
        if not expected_hash or sha256_file(prediction) != expected_hash:
            raise RuntimeError(f"{trial_id}: prediction hash mismatch: {prediction}")
        annotation = shard_annotation_root / f"shard_{index:02d}" / "annotation.json"
        if not annotation.is_file():
            raise FileNotFoundError(f"missing shard annotation: {annotation}")
        thresholds = manifest.get("thresholds")
        if not isinstance(thresholds, dict):
            raise RuntimeError(f"{trial_id}: missing thresholds in {manifest_path}")
        cache = Path(str(manifest.get("cache", ""))).resolve()
        cache_manifest = cache / "manifest.json"
        cache_hash = str(manifest.get("cache_manifest_sha256", ""))
        if not cache_manifest.is_file() or not cache_hash:
            raise RuntimeError(f"{trial_id}: missing cache provenance in {manifest_path}")
        if sha256_file(cache_manifest) != cache_hash:
            raise RuntimeError(f"{trial_id}: cache manifest hash mismatch: {cache_manifest}")
        manifests.append(manifest)
        shard_records.append(
            {
                "index": index,
                "shard": f"shard_{index:02d}",
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "annotation": str(annotation.resolve()),
                "annotation_sha256": sha256_file(annotation),
                "prediction": str(prediction),
                "prediction_sha256": expected_hash,
                "cache": str(cache),
                "cache_manifest_sha256": cache_hash,
                "frames": int(manifest.get("frames", -1)),
                "videos": int(manifest.get("videos", -1)),
                "rows": int(manifest.get("rows", -1)),
            }
        )

    reference = manifests[0]
    for manifest in manifests[1:]:
        if manifest.get("trial_id") != reference.get("trial_id"):
            raise RuntimeError(f"{trial_id}: trial id mismatch across shards")
        if not _same_json(manifest.get("thresholds"), reference.get("thresholds")):
            raise RuntimeError(f"{trial_id}: threshold mismatch across shards")
        for field in ("score_distribution", "margin_distribution", "score_grid", "margin_grid"):
            if field in reference and not _same_json(manifest.get(field), reference.get(field)):
                raise RuntimeError(f"{trial_id}: {field} mismatch across shards")

    events = _event_bundle(manifests)
    cache_bundle = [
        {
            "shard": record["shard"],
            "cache": record["cache"],
            "cache_manifest_sha256": record["cache_manifest_sha256"],
        }
        for record in shard_records
    ]
    return {
        "trial_id": trial_id,
        "thresholds": reference["thresholds"],
        "cache_bundle": cache_bundle,
        "cache_bundle_sha256": canonical_hash(cache_bundle),
        "events": events,
        "score_distribution": reference.get("score_distribution"),
        "margin_distribution": reference.get("margin_distribution"),
        "score_grid": reference.get("score_grid"),
        "margin_grid": reference.get("margin_grid"),
        "grid_trial_count": reference.get("grid_trial_count"),
        "selected_trial_indices": reference.get("selected_trial_indices"),
        "shards": shard_records,
        "detector_forward_calls": sum(int(item.get("detector_forward_calls", -1)) for item in manifests),
        "gt_loaded_during_replay": any(bool(item.get("gt_loaded_during_replay", True)) for item in manifests),
        "frames": sum(int(item.get("frames", 0)) for item in manifests),
        "videos": sum(int(item.get("videos", 0)) for item in manifests),
        "rows": sum(int(item.get("rows", 0)) for item in manifests),
    }


def _existing_aggregate_is_valid(path: Path, trial_id: str) -> bool:
    try:
        value = read_json(path)
        prediction = _require_file(value.get("prediction"), "aggregate prediction")
        return (
            value.get("status") == "PASS"
            and value.get("trial_id") == trial_id
            and value.get("detector_forward_calls") == 0
            and value.get("gt_loaded_during_replay") is False
            and value.get("prediction_sha256") == sha256_file(prediction)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def merge_trial(
    *,
    root: Path,
    output_root: Path,
    annotation: Path,
    repo: Path,
    python: Path,
    merge_script: Path,
    trial: dict[str, Any],
) -> dict[str, Any]:
    trial_id = str(trial["trial_id"])
    flat_root = output_root / trial_id
    aggregate_path = flat_root / "manifest.json"
    if aggregate_path.is_file():
        if not _existing_aggregate_is_valid(aggregate_path, trial_id):
            raise RuntimeError(f"refusing to overwrite invalid aggregate: {aggregate_path}")
        return read_json(aggregate_path)

    merged_root = flat_root / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)
    prediction = merged_root / "tao_track.json"
    merge_manifest = merged_root / "merge_manifest.json"
    merge_log = merged_root / "merge.log"
    command = [
        str(python),
        str(merge_script),
        "--annotation",
        str(annotation),
        "--output",
        str(prediction),
        "--manifest",
        str(merge_manifest),
    ]
    for record in trial["shards"]:
        command.extend(["--shard", str(record["annotation"]), str(record["prediction"])])
    with merge_log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=str(repo),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{trial_id}: complete-video merge failed; see {merge_log}")
    merged = read_json(merge_manifest)
    if merged.get("status") != "PASS" or not prediction.is_file():
        raise RuntimeError(f"{trial_id}: merge contract failed: {merge_manifest}")
    if merged.get("output_sha256") != sha256_file(prediction):
        raise RuntimeError(f"{trial_id}: merged prediction hash mismatch")

    aggregate = {
        "status": "PASS",
        "artifact": "v11_b0_reduced_margin_calibration_trial",
        "trial_id": trial_id,
        "source_calibration_root": str(root),
        "full_annotation": str(annotation),
        "full_annotation_sha256": sha256_file(annotation),
        "cache": str(root.parent.parent / "annotations"),
        "cache_bundle": trial["cache_bundle"],
        "cache_manifest_sha256": trial["cache_bundle_sha256"],
        "events": trial["events"],
        "score_distribution": trial.get("score_distribution"),
        "margin_distribution": trial.get("margin_distribution"),
        "score_grid": trial.get("score_grid"),
        "margin_grid": trial.get("margin_grid"),
        "grid_trial_count": trial.get("grid_trial_count"),
        "selected_trial_indices": trial.get("selected_trial_indices"),
        "thresholds": trial["thresholds"],
        "detector_forward_calls": trial["detector_forward_calls"],
        "gt_loaded_during_replay": trial["gt_loaded_during_replay"],
        "frames": int(merged["images"]),
        "videos": trial["videos"],
        "rows": int(merged["rows"]),
        "prediction": str(prediction.resolve()),
        "prediction_sha256": sha256_file(prediction),
        "merge_manifest": str(merge_manifest.resolve()),
        "merge_manifest_sha256": sha256_file(merge_manifest),
        "merge_command": command,
        "merge_log": str(merge_log.resolve()),
        "shards": trial["shards"],
        "repository_head": git_head(repo),
        "created_at_unix": time.time(),
    }
    atomic_write_json(aggregate_path, aggregate)
    return aggregate


def validate_flat_aggregate(
    *, output_root: Path, repo: Path, python: Path, expected_count: int
) -> Path:
    """Run the existing aggregate validator on the newly flat trial layout."""

    aggregate_script = repo / "tools" / "v11_aggregate_threshold_trials.py"
    if not aggregate_script.is_file():
        raise FileNotFoundError(f"aggregate validator is missing: {aggregate_script}")
    output = output_root / "search_manifest.json"
    if output.exists():
        existing = read_json(output)
        if existing.get("status") == "PASS" and int(existing.get("trial_count", -1)) == expected_count:
            return output
        raise RuntimeError(f"refusing to overwrite invalid aggregate manifest: {output}")
    log = output_root / "aggregate.log"
    command = [
        str(python),
        str(aggregate_script),
        "--output-root",
        str(output_root),
        "--output",
        str(output),
        "--expected-count",
        str(expected_count),
    ]
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=str(repo),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(f"flat threshold aggregate failed; see {log}")
    return output


def wait_for_trials(root: Path, trial_ids: list[str], shard_count: int, poll_seconds: float) -> None:
    while True:
        complete = 0
        for trial_id in trial_ids:
            if all(
                (trial_path(root, index, trial_id) / "manifest.json").is_file()
                for index in range(shard_count)
            ):
                try:
                    if all(
                        formal_pass(read_json(trial_path(root, index, trial_id) / "manifest.json"), trial_id)
                        for index in range(shard_count)
                    ):
                        complete += 1
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
        if complete == len(trial_ids):
            return
        time.sleep(max(1.0, poll_seconds))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--shard-annotation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--merge-script", type=Path)
    parser.add_argument("--trial-id", action="append", dest="trial_ids")
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--wait-for-complete", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    annotation = args.annotation.resolve()
    shard_annotation_root = args.shard_annotation_root.resolve()
    output_root = args.output_root.resolve()
    repo = args.repo.resolve()
    python = args.python.resolve()
    merge_script = (args.merge_script or repo / "tools" / "v10_merge_complete_video_tao.py").resolve()
    trial_ids = list(args.trial_ids or DEFAULT_TRIALS)
    if not annotation.is_file() or not merge_script.is_file():
        raise FileNotFoundError("annotation or merge script is missing")
    if args.shard_count < 1 or not trial_ids:
        raise ValueError("shard-count and trial-id list must be non-empty")

    runtime_manifest = output_root.parent / "threshold_merge_runtime_manifest.json"
    runtime = {
        "status": "WAITING" if args.wait_for_complete else "RUNNING",
        "artifact": "v11_b0_reduced_margin_calibration_merge",
        "root": str(root),
        "annotation": str(annotation),
        "annotation_sha256": sha256_file(annotation),
        "shard_annotation_root": str(shard_annotation_root),
        "output_root": str(output_root),
        "trial_ids": trial_ids,
        "shard_count": args.shard_count,
        "repository_head": git_head(repo),
        "started_at_unix": time.time(),
    }
    atomic_write_json(runtime_manifest, runtime)
    try:
        if args.wait_for_complete:
            wait_for_trials(root, trial_ids, args.shard_count, args.poll_seconds)
        aggregates = []
        for trial_id in trial_ids:
            trial = validate_trial(
                root=root,
                shard_annotation_root=shard_annotation_root,
                trial_id=trial_id,
                shard_count=args.shard_count,
            )
            aggregates.append(
                merge_trial(
                    root=root,
                    output_root=output_root,
                    annotation=annotation,
                    repo=repo,
                    python=python,
                    merge_script=merge_script,
                    trial=trial,
                )
            )
        aggregate_manifest = validate_flat_aggregate(
            output_root=output_root,
            repo=repo,
            python=python,
            expected_count=len(trial_ids),
        )
        runtime.update(
            {
                "status": "PASS",
                "trial_count": len(aggregates),
                "trial_manifests": [str(output_root / item["trial_id"] / "manifest.json") for item in aggregates],
                "aggregate_manifest": str(aggregate_manifest),
                "aggregate_manifest_sha256": sha256_file(aggregate_manifest),
                "ended_at_unix": time.time(),
            }
        )
        atomic_write_json(runtime_manifest, runtime)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "trial_count": len(aggregates),
                    "output_root": str(output_root),
                    "trial_ids": trial_ids,
                    "aggregate_manifest": str(aggregate_manifest),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as error:
        runtime.update(
            {
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
                "ended_at_unix": time.time(),
            }
        )
        atomic_write_json(runtime_manifest, runtime)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
