#!/usr/bin/env python3
"""Plan or launch calibrated Official-Val card replays.

This module is deliberately separate from ``v11_qdic_threshold_search.py`` and
from the reduced B0 calibration supervisor.  It consumes a completed B0
selection receipt, derives the already-selected reduced-grid trial index, and
builds the same causal replay command for one or two downstream cards.  By
default it only writes a plan; ``--launch`` is an explicit opt-in.

The driver does not load annotations or GT.  Official evaluation and card
selection remain separate post-replay steps.  A card replay may use the same
ten video shards as B0, but it must use that card's checkpoint/config and the
single frozen B0 operating point.
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


DEFAULT_CARDS = (
    "D1_LS010",
    "D2_LS025",
    "D3_LS050",
    "D4_LS100",
)
SHARD_COUNT = 10
GRID_WIDTH = 5
SEARCH_SCRIPT_NAME = "v11_qdic_threshold_search.py"


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


def trial_index_from_id(trial_id: str) -> int:
    """Map the reduced score=0 trial id to the search driver's global index."""

    parts = str(trial_id).split("_")
    if len(parts) != 2 or not parts[0].startswith("s") or not parts[1].startswith("m"):
        raise ValueError(f"invalid B0 trial id: {trial_id!r}")
    score_index = int(parts[0][1:])
    margin_index = int(parts[1][1:])
    if score_index != 0 or not 0 <= margin_index < GRID_WIDTH:
        raise ValueError(
            "downstream replay requires a reduced B0 trial s00_m00..s00_m04"
        )
    return score_index * GRID_WIDTH + margin_index


def selected_b0_operating_point(report_path: Path) -> dict[str, Any]:
    report = read_json(report_path)
    if report.get("status") != "PASS":
        raise ValueError(f"B0 calibration report is not PASS: {report_path}")
    if report.get("test_status") != "UNTOUCHED":
        raise ValueError("B0 report must declare Current Test UNTOUCHED")
    if report.get("novel_used_for_selection") is not False:
        raise ValueError("B0 report must explicitly exclude Novel from selection")
    selection = report.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("B0 report has no selection receipt")
    trial_id = str(selection.get("selected_trial_id", ""))
    trial_index = trial_index_from_id(trial_id)
    if selection.get("novel_used_for_selection") is not False:
        raise ValueError("B0 selection must explicitly exclude Novel")
    if selection.get("overall_used_for_selection") is not False:
        raise ValueError("B0 selection must explicitly exclude Overall")
    metric = str(selection.get("selection_metric", ""))
    if "Base AssocA" not in metric:
        raise ValueError(f"unexpected B0 selection metric: {metric}")
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("B0 report has no metric rows")
    matches = [row for row in rows if isinstance(row, dict) and row.get("trial_id") == trial_id]
    if len(matches) != 1:
        raise ValueError(f"B0 report must contain exactly one selected row: {trial_id}")
    row = matches[0]
    thresholds = row.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"selected B0 row has no thresholds: {trial_id}")
    if float(thresholds.get("score_threshold", float("nan"))) != 0.0:
        raise ValueError("reduced B0 selection must keep score_threshold=0")
    return {
        "trial_id": trial_id,
        "trial_index": trial_index,
        "score_threshold": float(thresholds["score_threshold"]),
        "margin_threshold": float(thresholds["margin_threshold"]),
        "selection_metric": metric,
        "tie_break_rule": str(selection.get("tie_break_rule", "")),
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
    }


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label}: {path}")


def _config_checkpoint(config_path: Path) -> tuple[Path, str]:
    """Read only the checkpoint fields needed for provenance validation."""

    payload = read_yaml_minimal(config_path)
    checkpoint = payload.get("qdic_checkpoint")
    if not checkpoint:
        raise ValueError(f"config has no qdic_checkpoint: {config_path}")
    checkpoint_path = Path(str(checkpoint)).resolve()
    _require_file(checkpoint_path, "qdic checkpoint")
    expected_hash = str(payload.get("qdic_checkpoint_sha256", ""))
    if expected_hash and sha256_file(checkpoint_path) != expected_hash:
        raise ValueError(f"qdic checkpoint hash mismatch: {config_path}")
    return checkpoint_path, sha256_file(checkpoint_path)


def _config_record(config_path: Path, expected_card: str) -> dict[str, str]:
    payload = read_yaml_minimal(config_path)
    if str(payload.get("card_id", "")) != expected_card:
        raise ValueError(
            f"card config identity mismatch: expected {expected_card}, "
            f"got {payload.get('card_id')!r} in {config_path}"
        )
    checkpoint, checkpoint_hash = _config_checkpoint(config_path)
    return {
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
    }


def read_yaml_minimal(path: Path) -> dict[str, Any]:
    """Read the small scalar config subset without adding a YAML dependency."""

    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key in {"qdic_checkpoint", "qdic_checkpoint_sha256", "card_id", "score_threshold", "margin_threshold"}:
            result[key] = value
    return result


def build_command(
    *,
    args: argparse.Namespace,
    card: str,
    shard: int,
    trial_index: int,
    gpu: int,
) -> list[str]:
    command = [
        str(args.python.resolve()),
        "-u",
        str(args.search_script.resolve()),
        "--cache",
        str((args.cache_root.resolve() / f"shard_{shard:02d}" / "frontend_cache")),
    ]
    for index in range(SHARD_COUNT):
        command.extend(
            ["--events", str(args.events_root.resolve() / f"shard_{index:02d}.jsonl")]
        )
    command.extend(
        [
            "--tempo-config",
            str((args.config_dir.resolve() / f"{card}.yaml")),
            "--output-root",
            str(args.output_root.resolve() / card / f"shard_{shard:02d}"),
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
            "--trial-index",
            str(trial_index),
            "--allow-existing-output-root",
        ]
    )
    return command


def validate_inputs(args: argparse.Namespace, cards: Iterable[str]) -> dict[str, str]:
    _require_dir(args.cache_root.resolve(), "cache root")
    _require_dir(args.events_root.resolve(), "events root")
    _require_dir(args.config_dir.resolve(), "config directory")
    for path, label in (
        (args.repo.resolve(), "repository"),
        (args.python.resolve(), "python"),
        (args.search_script.resolve(), "search script"),
        (args.cov_source.resolve(), "COV source"),
        (args.cov_config.resolve(), "COV config"),
        (args.cov_checkpoint.resolve(), "COV checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")
    events: dict[str, str] = {}
    for index in range(SHARD_COUNT):
        path = args.events_root.resolve() / f"shard_{index:02d}.jsonl"
        _require_file(path, "event source")
        events[str(index)] = sha256_file(path)
    configs: dict[str, dict[str, str]] = {}
    for card in cards:
        config = args.config_dir.resolve() / f"{card}.yaml"
        _require_file(config, "card config")
        configs[card] = _config_record(config, card)
    return {"events": json.dumps(events, sort_keys=True), "configs": json.dumps(configs, sort_keys=True)}


def _repo_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_environment(
    repo: Path,
    *,
    gpu: int | None = None,
    cov_source: Path | None = None,
) -> dict[str, str]:
    """Return the child environment used for a replay worker.

    The search entry point is invoked by absolute path.  Python then puts its
    ``tools`` directory, rather than the repository root, on ``sys.path``;
    relying on the caller's inherited ``PYTHONPATH`` therefore made the
    launcher work only from some shells and fail closed before replay in
    others.  Keep the repository root first while preserving any caller
    entries needed by the COV/TempoTrack runtime.
    """

    environment = os.environ.copy()
    import_roots = [str(repo.resolve())]
    if cov_source is not None:
        import_roots.append(str(cov_source.resolve()))
    inherited = environment.get("PYTHONPATH", "")
    if inherited:
        import_roots.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(import_roots)
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def make_plan(args: argparse.Namespace) -> dict[str, Any]:
    cards = list(dict.fromkeys(args.card or DEFAULT_CARDS))
    if not cards:
        raise ValueError("at least one card is required")
    if len(cards) > int(args.max_cards):
        raise ValueError(
            f"at most {args.max_cards} downstream full cards may be launched together"
        )
    op = selected_b0_operating_point(args.b0_report.resolve())
    input_hashes = validate_inputs(args, cards)
    card_records: list[dict[str, Any]] = []
    for card in cards:
        commands = []
        for shard in range(SHARD_COUNT):
            commands.append(
                {
                    "shard": f"shard_{shard:02d}",
                    "gpu": int(args.gpus[shard % len(args.gpus)]),
                    "log": str(args.output_root.resolve() / "logs" / f"{card}_shard_{shard:02d}.log"),
                    "command": build_command(
                        args=args,
                        card=card,
                        shard=shard,
                        trial_index=int(op["trial_index"]),
                        gpu=int(args.gpus[shard % len(args.gpus)]),
                    ),
                }
            )
        card_records.append(
            {
                "card_id": card,
                "config": json.loads(input_hashes["configs"])[card]["config"],
                "config_sha256": json.loads(input_hashes["configs"])[card]["config_sha256"],
                "checkpoint": json.loads(input_hashes["configs"])[card]["checkpoint"],
                "checkpoint_sha256": json.loads(input_hashes["configs"])[card]["checkpoint_sha256"],
                "shards": commands,
            }
        )
    return {
        "status": "PLAN_ONLY" if not args.launch else "LAUNCHING",
        "artifact": "v11_calibrated_official_val_card_replay_plan",
        "repository": str(args.repo.resolve()),
        "repository_head": _repo_head(args.repo.resolve()),
        "b0_operating_point": op,
        "cards": card_records,
        "shard_count": SHARD_COUNT,
        "gpus": list(args.gpus),
        "max_cards": int(args.max_cards),
        "input_hashes": {
            "event_sources": json.loads(input_hashes["events"]),
            "cov_source": sha256_file(args.cov_source.resolve()) if args.cov_source.is_file() else None,
            "cov_config": sha256_file(args.cov_config.resolve()),
            "cov_checkpoint": sha256_file(args.cov_checkpoint.resolve()),
        },
        "gt_loaded_during_replay": False,
        "detector_forward_calls": 0,
        "created_at_unix": time.time(),
    }


def launch_plan(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    launched: list[dict[str, Any]] = []
    for card in plan["cards"]:
        for item in card["shards"]:
            log_path = Path(item["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            output_root = Path(item["command"][item["command"].index("--output-root") + 1])
            output_root.mkdir(parents=True, exist_ok=True)
            environment = runtime_environment(
                args.repo,
                gpu=int(item["gpu"]),
                cov_source=args.cov_source,
            )
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    item["command"],
                    cwd=str(args.repo.resolve()),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            launched.append(
                {
                    "card_id": card["card_id"],
                    "shard": item["shard"],
                    "gpu": item["gpu"],
                    "pid": process.pid,
                    "log": str(log_path),
                }
            )
    plan["status"] = "LAUNCHED"
    plan["launched"] = launched
    plan["launched_at_unix"] = time.time()
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-report", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--events-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--search-script", type=Path, default=Path(__file__).resolve().with_name(SEARCH_SCRIPT_NAME))
    parser.add_argument("--cov-source", type=Path, required=True)
    parser.add_argument("--cov-config", type=Path, required=True)
    parser.add_argument("--cov-checkpoint", type=Path, required=True)
    parser.add_argument("--card", action="append")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(SHARD_COUNT)))
    parser.add_argument("--max-cards", type=int, default=2)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--plan-output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.gpus) != SHARD_COUNT:
        raise ValueError("exactly ten GPU assignments are required")
    plan = make_plan(args)
    if args.launch:
        plan = launch_plan(plan, args)
    atomic_write_json(args.plan_output.resolve(), plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
