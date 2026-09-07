"""Run one baseline control as a real, independently tracked CPU worker.

The V4 coordinator must not execute a full official evaluator inline: doing so
would prevent unrelated GPU branches from acquiring a lease.  This worker
still calls the production inference and evaluation entry points through the
compatibility pipeline; it is only the process boundary that is new.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config import load_yaml
from .executor import JobExecutor
from .v4_pipeline import _V4CompatPipeline, _atomic


def _compact_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"status": "UNKNOWN_RESULT"}
    keys = (
        "status", "scheme", "profile", "seed", "split", "prediction",
        "prediction_path", "evaluation", "summary", "summary_path",
        "artifact_hashes", "source_manifest", "source_payload_hash",
        "tracklet_manifest", "tracklet_manifest_hash", "duration_seconds",
        "metric_units", "calibration", "gt_path", "gt_split_hash",
    )
    return {key: value[key] for key in keys if key in value}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TempoTrack V4 baseline control worker")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--local", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-internal", required=True)
    parser.add_argument("--source-official", required=True)
    parser.add_argument("--replay-internal", required=True)
    parser.add_argument("--replay-official", required=True)
    parser.add_argument("--annotation-internal", required=True)
    parser.add_argument("--annotation-official", required=True)
    parser.add_argument("--memory-checkpoint")
    parser.add_argument("--resume", default="auto")
    return parser.parse_args()


def main() -> int:
    args = _parse()
    repo = Path(args.repo).resolve()
    suite_path = Path(args.suite).resolve()
    local_path = Path(args.local).resolve()
    reference = Path(args.reference_root).resolve()
    run_root = Path(args.run_root).resolve()
    report_root = repo / "reports/v4"
    run_dir = run_root / "artifacts" / args.job_id.replace("/", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    executor = JobExecutor(
        repo,
        run_root,
        report_root,
        python=str(load_yaml(local_path).get("research_python", "python")),
        heartbeat_seconds=30,
    )
    compat = _V4CompatPipeline(
        repo,
        suite_path,
        local_path,
        reference,
        run_root,
        executor,
        resume=str(args.resume),
    )
    with Path(args.prepared).open("r", encoding="utf-8") as handle:
        prepared: dict[str, Any] = json.load(handle)

    memory_checkpoint = Path(args.memory_checkpoint).resolve() if args.memory_checkpoint else None
    values = [
        compat._infer_eval(
            args.scheme,
            args.profile,
            int(args.seed),
            "val_base_internal",
            Path(args.source_internal).resolve(),
            Path(args.replay_internal).resolve(),
            None,
            memory_checkpoint,
            prepared,
            annotation=Path(args.annotation_internal).resolve(),
            tag=args.tag,
        ),
        compat._infer_eval(
            args.scheme,
            args.profile,
            int(args.seed),
            "official_validation",
            Path(args.source_official).resolve(),
            Path(args.replay_official).resolve(),
            None,
            memory_checkpoint,
            prepared,
            annotation=Path(args.annotation_official).resolve(),
            tag=args.tag,
        ),
    ]
    payload = {
        "schema_version": 4,
        "status": "COMPLETED",
        "job_id": args.job_id,
        "scheme": args.scheme,
        "profile": args.profile,
        "seed": int(args.seed),
        "values": [_compact_result(value) for value in values],
        "memory_checkpoint": str(memory_checkpoint) if memory_checkpoint else None,
    }
    _atomic(run_dir / "baseline_result.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
