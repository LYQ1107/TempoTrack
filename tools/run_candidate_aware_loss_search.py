#!/usr/bin/env python3
"""Sequentially search loss weights for the architecture-comparison winner.

The architecture comparison is a hard prerequisite.  The two searches are
also deliberately sequential: lambda_dist is selected with lambda_hard fixed,
then lambda_hard is selected with lambda_dist fixed.  No architecture or
Full-Test candidate is launched by this tool.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tempotrack_v10.candidate_aware_qdic_loader import load_candidate_aware_checkpoint  # noqa: E402
from tempotrack_v10.candidate_aware_qdic_trainer import train_architecture  # noqa: E402
from tempotrack_v10.qdic_trainer import sha256  # noqa: E402
from run_candidate_aware_qdic_fulltest import _validate_structure_gate  # noqa: E402


NEW_ARCHITECTURES = ("A1-DS-QDIC", "A2-DGSA-QDIC")


def _canonical_architecture(value: str) -> str:
    name = str(value).strip().upper()
    if name in {"A1", "A1-DS-QDIC", "DS-QDIC"}:
        return "A1-DS-QDIC"
    if name in {"A2", "A2-DGSA-QDIC", "DGSA-QDIC"}:
        return "A2-DGSA-QDIC"
    return name


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_summary(result: dict[str, Any], *, search_axis: str, value: float) -> dict[str, Any]:
    history = list(result.get("history", []))
    best = min(
        (item for item in history if item.get("holdout_loss") is not None),
        key=lambda item: float(item["holdout_loss"]),
        default=None,
    )
    return {
        "search_axis": search_axis,
        "search_value": float(value),
        "architecture_name": result.get("architecture_name"),
        "status": result.get("status"),
        "lambda_dist": result.get("lambda_dist"),
        "lambda_hard": result.get("lambda_hard"),
        "best_epoch": result.get("best_epoch"),
        "best_holdout_loss": result.get("best_holdout_loss"),
        "best_holdout_final_top1": None if best is None else best.get("holdout_final_top1"),
        "best_holdout_final_mrr": None if best is None else best.get("holdout_final_mrr"),
        "best_holdout_distribution_top1": None if best is None else best.get("holdout_distribution_top1"),
        "best_holdout_distribution_mrr": None if best is None else best.get("holdout_distribution_mrr"),
        "checkpoint": result.get("checkpoint"),
        "checkpoint_hash": result.get("checkpoint_hash"),
        "training_json": str(Path(str(result.get("checkpoint"))).parent / "training.json"),
    }


def _best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "COMPLETED" and row.get("best_holdout_loss") is not None]
    if not valid:
        raise RuntimeError("loss search produced no completed candidate")
    return dict(
        min(
            valid,
            key=lambda row: (
                float(row["best_holdout_loss"]),
                -float(row.get("best_holdout_final_top1") or float("-inf")),
                -float(row.get("best_holdout_final_mrr") or float("-inf")),
                float(row.get("search_value", 0.0)),
            ),
        )
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    structure_root = Path(args.structure_search_root).resolve()
    gate = _validate_structure_gate(structure_root)
    comparison_path = Path(args.architecture_comparison).resolve()
    comparison = _read_json(comparison_path)
    if comparison.get("status") != "COMPLETED":
        raise RuntimeError("architecture comparison is not COMPLETED")
    architectures = list(comparison.get("architectures", []))
    if not architectures:
        raise RuntimeError("architecture comparison has no architecture rows")
    # A0 has no distributional branch, so its total objective omits L_dist
    # and is not numerically comparable to the A0-D/A1/A2 totals.  The
    # protocol explicitly selects the best *new* architecture first; only
    # that winner is eligible for the subsequent lambda search.
    eligible = [
        row
        for row in architectures
        if row.get("status") == "COMPLETED"
        and _canonical_architecture(row.get("architecture_name", ""))
        in NEW_ARCHITECTURES
    ]
    if not eligible:
        raise RuntimeError(
            "architecture comparison has no completed A1/A2 candidate-aware architecture"
        )
    winner = min(
        eligible,
        key=lambda row: (
            float(row.get("best_holdout_loss", float("inf"))),
            -float(row.get("best_holdout_final_top1") or float("-inf")),
            -float(row.get("best_holdout_final_mrr") or float("-inf")),
            str(row.get("architecture_name")),
        ),
    )
    architecture = _canonical_architecture(str(winner["architecture_name"]))
    features_dir = Path(args.features_dir).resolve()
    parent_checkpoint = Path(args.parent_checkpoint).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not (features_dir / "features.json").is_file():
        raise FileNotFoundError(features_dir / "features.json")
    parent_hash = sha256(parent_checkpoint)
    expected_feature_hash = sha256(features_dir / "features.json")
    comparison_contract = comparison.get("contract", {})
    if comparison_contract.get("parent_v11_checkpoint_hash") != parent_hash:
        raise RuntimeError("architecture comparison parent hash does not match loss-search parent")
    if comparison_contract.get("features_hash") != expected_feature_hash:
        raise RuntimeError("architecture comparison feature hash does not match loss-search cache")

    lambda_dist_values = tuple(dict.fromkeys(float(value) for value in args.lambda_dist_values))
    lambda_hard_values = tuple(dict.fromkeys(float(value) for value in args.lambda_hard_values))
    if any(value < 0.0 for value in lambda_dist_values + lambda_hard_values):
        raise ValueError("loss weights must be non-negative")
    started = time.time()
    dist_rows: list[dict[str, Any]] = []
    for value in lambda_dist_values:
        output = output_root / "lambda_dist" / f"value_{value:g}"
        if output.exists():
            raise FileExistsError(output)
        result = train_architecture(
            features_dir,
            parent_checkpoint,
            output,
            architecture=architecture,
            device=args.device,
            epochs=int(args.epochs),
            seed=int(args.seed),
            lambda_dist=value,
            lambda_hard=float(args.lambda_hard_fixed),
            hard_margin=float(args.hard_margin),
            mixed_k_values=args.mixed_k,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        dist_rows.append(_candidate_summary(result, search_axis="lambda_dist", value=value))
    best_dist = _best(dist_rows)

    hard_rows: list[dict[str, Any]] = []
    for value in lambda_hard_values:
        output = output_root / "lambda_hard" / f"value_{value:g}"
        if output.exists():
            raise FileExistsError(output)
        result = train_architecture(
            features_dir,
            parent_checkpoint,
            output,
            architecture=architecture,
            device=args.device,
            epochs=int(args.epochs),
            seed=int(args.seed),
            lambda_dist=float(best_dist["lambda_dist"]),
            lambda_hard=value,
            hard_margin=float(args.hard_margin),
            mixed_k_values=args.mixed_k,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        hard_rows.append(_candidate_summary(result, search_axis="lambda_hard", value=value))
    best_hard = _best(hard_rows)
    selected_checkpoint = Path(str(best_hard["checkpoint"])).resolve()
    artifact = load_candidate_aware_checkpoint(selected_checkpoint, device="cpu")
    result = {
        "schema_version": 1,
        "artifact": "candidate_aware_qdic_loss_search",
        "status": "COMPLETED",
        "protocol": "QDIC_V11_BASE_ONLY_WINNER_ONLY_SEQUENTIAL_LOSS_SEARCH",
        "unbiased_test": False,
        "structure_gate": gate,
        "architecture_comparison": str(comparison_path),
        "architecture_comparison_sha256": sha256(comparison_path),
        "winner_architecture": architecture,
        "search_contract": {
            "lambda_dist_values": list(lambda_dist_values),
            "lambda_dist_fixed_lambda_hard": float(args.lambda_hard_fixed),
            "lambda_hard_values": list(lambda_hard_values),
            "lambda_hard_fixed_lambda_dist": float(best_dist["lambda_dist"]),
            "selection_metric": (
                "minimum video-disjoint holdout total loss among completed A1/A2 "
                "new architectures; final top1/mrr tie-break; A0/A0-D excluded "
                "from architecture selection"
            ),
            "architecture_selection_eligible": list(NEW_ARCHITECTURES),
            "architecture_selection_excluded": ["A0", "A0-D"],
            "no_other_architecture_searched": True,
            "no_full_test_started": True,
        },
        "best_lambda_dist": best_dist,
        "best_lambda_hard": best_hard,
        "selected": {
            "architecture_name": architecture,
            "lambda_dist": float(best_hard["lambda_dist"]),
            "lambda_hard": float(best_hard["lambda_hard"]),
            "checkpoint": str(selected_checkpoint),
            "checkpoint_hash": sha256(selected_checkpoint),
            "loader_status": artifact.provenance.get("status"),
        },
        "lambda_dist_rows": dist_rows,
        "lambda_hard_rows": hard_rows,
        "features_dir": str(features_dir),
        "features_hash": expected_feature_hash,
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_hash": parent_hash,
        "started_at_unix": started,
        "ended_at_unix": time.time(),
    }
    result["duration_seconds"] = result["ended_at_unix"] - started
    _write_json(output_root / "loss_search.json", result)
    _write_csv(output_root / "lambda_dist_search.csv", dist_rows)
    _write_csv(output_root / "lambda_hard_search.csv", hard_rows)
    _write_json(output_root / "final_champion.json", result["selected"])
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-search-root", required=True)
    parser.add_argument("--architecture-comparison", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-dist-values", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--lambda-hard-fixed", type=float, default=0.2)
    parser.add_argument("--lambda-hard-values", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    parser.add_argument("--hard-margin", type=float, default=0.2)
    parser.add_argument("--mixed-k", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
