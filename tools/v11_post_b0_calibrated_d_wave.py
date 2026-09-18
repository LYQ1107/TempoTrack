#!/usr/bin/env python3
"""Run the calibrated D-wave after a completed B0 Val-Base selection.

The supervisor is intentionally dormant until the B0 calibration receipt is
``PASS``.  It then runs at most two complete cards at a time using the
independent calibrated-card launcher, waits for formal causal shard PASS
manifests, merges each card, and runs the official Val evaluator on that card's
single frozen operating point.  It never starts C/H/Test; those decisions are
made from the D-wave reports by the selection gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable


DEFAULT_CARDS = ("D1_LS010", "D2_LS025", "D3_LS050", "D4_LS100")
SHARD_COUNT = 10


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def formal_pass(path: Path, trial_id: str) -> bool:
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        value.get("status") == "PASS"
        and value.get("trial_id") == trial_id
        and int(value.get("detector_forward_calls", -1)) == 0
        and value.get("gt_loaded_during_replay") is False
    )


def wait_for_b0(report: Path, poll_seconds: float) -> dict[str, Any]:
    while True:
        if report.is_file():
            try:
                value = read_json(report)
                if value.get("status") == "PASS":
                    selection = value.get("selection")
                    if isinstance(selection, dict) and selection.get("selected_trial_id"):
                        return value
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        time.sleep(max(1.0, float(poll_seconds)))


def wait_for_card(root: Path, card: str, trial_id: str, poll_seconds: float) -> None:
    while True:
        manifests = [
            root / card / f"shard_{shard:02d}" / trial_id / "manifest.json"
            for shard in range(SHARD_COUNT)
        ]
        if all(formal_pass(path, trial_id) for path in manifests):
            return
        time.sleep(max(1.0, float(poll_seconds)))


def run_logged(command: list[str], *, cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed with {result.returncode}; see {log}")


def child_environment(args: argparse.Namespace, *, disable_cuda: bool = False) -> dict[str, str]:
    """Build a detached-child environment with the repository importable.

    The supervisor is commonly launched by ``nohup``/``setsid`` without the
    interactive shell's ``PYTHONPATH``.  The replay workers can still start
    because their entry points are absolute paths, but the post-replay
    evaluator imports ``tempotrack_research`` as a package.  Injecting the
    audited repository root here keeps planner, merge, and evaluation children
    consistent without changing any scientific runtime argument.
    """

    environment = dict(os.environ)
    repo_path = str(args.repo.resolve())
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (repo_path, existing) if value
    )
    if disable_cuda:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def planner_command(args: argparse.Namespace, cards: Iterable[str], batch_root: Path, plan_path: Path) -> list[str]:
    command = [
        str(args.python.resolve()),
        str(args.planner.resolve()),
        "--b0-report",
        str(args.b0_report.resolve()),
        "--cache-root",
        str(args.cache_root.resolve()),
        "--events-root",
        str(args.events_root.resolve()),
        "--config-dir",
        str(args.config_dir.resolve()),
        "--output-root",
        str(batch_root.resolve()),
        "--repo",
        str(args.repo.resolve()),
        "--python",
        str(args.python.resolve()),
        "--search-script",
        str(args.search_script.resolve()),
        "--cov-source",
        str(args.cov_source.resolve()),
        "--cov-config",
        str(args.cov_config.resolve()),
        "--cov-checkpoint",
        str(args.cov_checkpoint.resolve()),
        "--plan-output",
        str(plan_path.resolve()),
        "--launch",
    ]
    for card in cards:
        command.extend(["--card", card])
    command.extend(["--gpus", *[str(index) for index in args.gpus]])
    return command


def merge_command(args: argparse.Namespace, card_root: Path, output_root: Path, trial_id: str) -> list[str]:
    return [
        str(args.python.resolve()),
        str(args.merge_script.resolve()),
        "--root",
        str(card_root.resolve()),
        "--annotation",
        str(args.annotation.resolve()),
        "--shard-annotation-root",
        str(args.shard_annotation_root.resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--repo",
        str(args.repo.resolve()),
        "--python",
        str(args.python.resolve()),
        "--trial-id",
        trial_id,
        "--shard-count",
        str(SHARD_COUNT),
        "--once",
    ]


def evaluate_command(args: argparse.Namespace, aggregate_root: Path, trial_id: str) -> list[str]:
    return [
        str(args.python.resolve()),
        str(args.evaluator.resolve()),
        "--aggregated-root",
        str(aggregate_root.resolve()),
        "--annotation",
        str(args.annotation.resolve()),
        "--repo",
        str(args.repo.resolve()),
        "--python",
        str(args.python.resolve()),
        "--trial-id",
        trial_id,
        "--evaluation-cores",
        str(int(args.evaluation_cores)),
        "--once",
    ]


def _status_pass(path: Path) -> bool:
    try:
        return read_json(path).get("status") == "PASS"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _card_replay_complete(batch_root: Path, card: str, trial_id: str) -> bool:
    return all(
        formal_pass(
            batch_root / card / f"shard_{shard:02d}" / trial_id / "manifest.json",
            trial_id,
        )
        for shard in range(SHARD_COUNT)
    )


def _completed_card_ids(runtime: dict[str, Any]) -> set[str]:
    return {
        str(record.get("card_id"))
        for record in runtime.get("completed_cards", [])
        if isinstance(record, dict) and record.get("card_id")
    }


def run_batch(args: argparse.Namespace, cards: list[str], trial_id: str, runtime: dict[str, Any]) -> None:
    batch_name = "_".join(cards)
    batch_root = args.output_root.resolve() / batch_name
    batch_root.mkdir(parents=True, exist_ok=True)
    plan_path = batch_root / "launch_plan.json"
    launch_log = batch_root / "supervisor.log"
    completed = _completed_card_ids(runtime)
    cards_to_launch = [
        card
        for card in cards
        if card not in completed and not _card_replay_complete(batch_root, card, trial_id)
    ]
    if cards_to_launch:
        run_logged(
            planner_command(args, cards_to_launch, batch_root, plan_path),
            cwd=args.repo.resolve(),
            log=launch_log,
            env=child_environment(args),
        )
        batches = runtime.setdefault("batches", [])
        if not any(
            isinstance(record, dict) and record.get("batch") == batch_name
            for record in batches
        ):
            batches.append(
                {
                    "batch": batch_name,
                    "cards": cards,
                    "launched_cards": cards_to_launch,
                    "trial_id": trial_id,
                    "plan": str(plan_path),
                }
            )
        atomic_write_json(args.runtime_manifest.resolve(), runtime)
    for card in cards:
        if card in completed:
            continue
        wait_for_card(batch_root, card, trial_id, args.poll_seconds)
        card_root = args.output_root.resolve() / batch_name / card
        aggregate_root = card_root / "aggregated"
        if not _status_pass(aggregate_root / "search_manifest.json"):
            run_logged(
                merge_command(args, card_root, aggregate_root, trial_id),
                cwd=args.repo.resolve(),
                log=card_root / "merge.log",
                env=child_environment(args, disable_cuda=True),
            )
        evaluation_manifest = card_root / "b0_calibration_evaluation_runtime_manifest.json"
        report = card_root / "b0_calibration_report" / "b0_calibration_metrics.json"
        if not (_status_pass(evaluation_manifest) and report.is_file()):
            run_logged(
                evaluate_command(args, aggregate_root, trial_id),
                cwd=args.repo.resolve(),
                log=card_root / "evaluation.log",
                env=child_environment(args, disable_cuda=True),
            )
        runtime.setdefault("completed_cards", []).append(
            {
                "card_id": card,
                "batch": batch_name,
                "trial_id": trial_id,
                "report": str(report),
            }
        )
        atomic_write_json(args.runtime_manifest.resolve(), runtime)
        completed.add(card)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-report", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--events-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--shard-annotation-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--planner", type=Path, default=Path(__file__).resolve().with_name("v11_calibrated_card_replay.py"))
    parser.add_argument("--search-script", type=Path, required=True)
    parser.add_argument("--merge-script", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--cov-source", type=Path, required=True)
    parser.add_argument("--cov-config", type=Path, required=True)
    parser.add_argument("--cov-checkpoint", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--evaluation-cores", type=int, default=8)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing D-wave runtime without replaying PASS shards",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.gpus) != SHARD_COUNT:
        raise ValueError("exactly ten GPU assignments are required")
    for path in (
        args.cache_root,
        args.events_root,
        args.config_dir,
        args.annotation,
        args.shard_annotation_root,
        args.repo,
        args.python,
        args.planner,
        args.search_script,
        args.merge_script,
        args.evaluator,
        args.cov_source,
        args.cov_config,
        args.cov_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    runtime_path = args.runtime_manifest.resolve()
    if args.resume:
        if not runtime_path.is_file():
            raise FileNotFoundError(f"cannot resume missing runtime manifest: {runtime_path}")
        runtime = read_json(runtime_path)
        if runtime.get("test_started") is True:
            raise ValueError("refusing to resume after Current Test has started")
        if runtime.get("status") in {
            "D_WAVE_COMPLETE",
            "DSSL_GATE_PASS",
            "DSSL_NEGATIVE_GATE",
        }:
            return 0
    else:
        runtime = {
            "status": "WAITING_FOR_B0",
            "artifact": "v11_post_b0_calibrated_d_wave_supervisor",
            "b0_report": str(args.b0_report.resolve()),
            "output_root": str(args.output_root.resolve()),
            "cards": list(DEFAULT_CARDS),
            "test_started": False,
            "created_at_unix": time.time(),
        }
        atomic_write_json(runtime_path, runtime)
    b0 = wait_for_b0(args.b0_report.resolve(), args.poll_seconds)
    trial_id = str(b0["selection"]["selected_trial_id"])
    previous_trial_id = runtime.get("selected_trial_id")
    if previous_trial_id is not None and str(previous_trial_id) != trial_id:
        raise ValueError(
            f"resume trial mismatch: runtime={previous_trial_id!r}, B0={trial_id!r}"
        )
    runtime.update({"status": "B0_PASS", "selected_trial_id": trial_id, "b0_selection": b0["selection"]})
    atomic_write_json(runtime_path, runtime)
    for cards in (list(DEFAULT_CARDS[:2]), list(DEFAULT_CARDS[2:])):
        runtime["status"] = "RUNNING_D_WAVE"
        atomic_write_json(runtime_path, runtime)
        run_batch(args, cards, trial_id, runtime)
    runtime.update({"status": "D_WAVE_COMPLETE", "ended_at_unix": time.time()})
    atomic_write_json(runtime_path, runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
