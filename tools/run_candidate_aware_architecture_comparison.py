#!/usr/bin/env python3
"""Train and compare A0/A0-D/A1/A2 under one frozen Base-only contract.

The command is gated on completion of the six structural Full-Test trials and
therefore cannot accidentally start model training while that search is live.
It only writes its own output root; it never edits the structural search root.
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

from tempotrack_v10.candidate_aware_qdic_trainer import (  # noqa: E402
    DEFAULT_HARD_MARGIN,
    DEFAULT_LAMBDA_HARD,
    DEFAULT_LAMBDA_DIST,
    DEFAULT_MIXED_K,
    train_architecture,
)
from tempotrack_v10.qdic_loader import load_qdic_checkpoint  # noqa: E402
from tempotrack_v10.qdic_trainer import sha256  # noqa: E402
from run_candidate_aware_qdic_fulltest import _validate_structure_gate  # noqa: E402


ARCHITECTURES = ("A0", "A0-D", "A1", "A2")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _canonical_architecture(value: str) -> str:
    name = str(value).strip().upper()
    if name == "A0":
        return "A0"
    if name in {"A0-D", "A0_D", "A0DIST", "A0-DISTLOSS"}:
        return "A0-D"
    if name in {"A1", "A1-DS-QDIC", "DS-QDIC"}:
        return "A1-DS-QDIC"
    if name in {"A2", "A2-DGSA-QDIC", "DGSA-QDIC"}:
        return "A2-DGSA-QDIC"
    raise ValueError(f"unsupported architecture: {value!r}")


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    history = list(result.get("history", []))
    best = min(
        (item for item in history if item.get("holdout_loss") is not None),
        key=lambda item: float(item["holdout_loss"]),
        default=None,
    )
    return {
        "architecture_name": result.get("architecture_name"),
        "architecture_config": result.get("architecture_config"),
        "status": result.get("status"),
        "training_split": result.get("training_split"),
        "paper_status": result.get("paper_status"),
        "paper_valid": result.get("paper_valid"),
        "features": result.get("features"),
        "features_hash": result.get("features_hash"),
        "parent_v11_checkpoint": result.get("parent_v11_checkpoint"),
        "parent_v11_checkpoint_hash": result.get("parent_v11_checkpoint_hash"),
        "base_only_supervision": result.get("base_only_supervision"),
        "novel_gt_used": result.get("novel_gt_used"),
        "test_gt_used_for_optimizer": result.get("test_gt_used_for_optimizer"),
        "test_weights_used": result.get("test_weights_used"),
        "lambda_dist": result.get("lambda_dist"),
        "lambda_hard": result.get("lambda_hard"),
        "seed": result.get("seed"),
        "epochs": result.get("epochs"),
        "optimizer": result.get("optimizer"),
        "optimizer_steps": result.get("optimizer_steps"),
        "trainable_parameter_count": result.get("trainable_parameter_count"),
        "parent_frozen": result.get("parent_frozen"),
        "best_epoch": result.get("best_epoch"),
        "best_holdout_loss": result.get("best_holdout_loss"),
        "best_holdout_final_top1": None if best is None else best.get("holdout_final_top1"),
        "best_holdout_final_mrr": None if best is None else best.get("holdout_final_mrr"),
        "best_holdout_distribution_top1": (
            None if best is None else best.get("holdout_distribution_top1")
        ),
        "best_holdout_distribution_mrr": (
            None if best is None else best.get("holdout_distribution_mrr")
        ),
        "train_video_count": len(result.get("train_video_ids", [])),
        "holdout_video_count": len(result.get("holdout_video_ids", [])),
        "history": history,
    }


def run(args: argparse.Namespace) -> int:
    structure_root = Path(args.structure_search_root).resolve()
    gate = _validate_structure_gate(structure_root)
    features_dir = Path(args.features_dir).resolve()
    parent_checkpoint = Path(args.parent_checkpoint).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not (features_dir / "features.json").is_file():
        raise FileNotFoundError(features_dir / "features.json")
    parent_artifact = load_qdic_checkpoint(parent_checkpoint, device="cpu")
    parent_hash = sha256(parent_checkpoint)
    architectures = tuple(dict.fromkeys(_canonical_architecture(item) for item in args.architectures))
    missing = [name for name in architectures if name not in ARCHITECTURES]
    if missing:
        raise ValueError(f"unsupported architecture list: {missing}")

    started = time.time()
    results: list[dict[str, Any]] = []
    for architecture in architectures:
        architecture_dir = output_root / architecture.replace("-", "_")
        if architecture_dir.exists():
            raise FileExistsError(
                f"refusing to reuse existing architecture output: {architecture_dir}"
            )
        lambda_dist = 0.0 if architecture == "A0" else float(args.lambda_dist)
        result = train_architecture(
            features_dir,
            parent_checkpoint,
            architecture_dir,
            architecture=architecture,
            device=args.device,
            epochs=int(args.epochs),
            seed=int(args.seed),
            lambda_dist=lambda_dist,
            lambda_hard=float(args.lambda_hard),
            hard_margin=float(args.hard_margin),
            mixed_k_values=args.mixed_k,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        summary = _summary(result)
        if summary["status"] != "COMPLETED":
            raise RuntimeError(f"architecture training did not complete: {architecture}")
        if summary["parent_v11_checkpoint_hash"] != parent_hash:
            raise RuntimeError(f"parent hash changed in architecture receipt: {architecture}")
        results.append(summary)

    comparison = {
        "schema_version": 1,
        "artifact": "candidate_aware_qdic_architecture_comparison",
        "status": "COMPLETED",
        "protocol": "QDIC_V11_BASE_ONLY_CANDIDATE_AWARE_ARCHITECTURE_COMPARISON",
        "paper_status": (
            results[0].get("paper_status")
            if len({row.get("paper_status") for row in results}) == 1
            else "MIXED"
        ),
        "paper_valid": bool(results) and all(bool(row.get("paper_valid")) for row in results),
        "structure_gate": gate,
        "contract": {
            "features_dir": str(features_dir),
            "features_hash": sha256(features_dir / "features.json"),
            "parent_v11_checkpoint": str(parent_checkpoint),
            "parent_v11_checkpoint_hash": parent_hash,
            "parent_loader_status": parent_artifact.provenance.get("status"),
            "base_only_supervision": True,
            "novel_gt_used": False,
            "test_gt_used_for_optimizer": False,
            "test_weights_used": False,
            "video_disjoint_train_holdout": True,
            "mixed_k_values": [int(value) for value in args.mixed_k],
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "optimizer": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "lambda_dist": float(args.lambda_dist),
            "lambda_hard": float(args.lambda_hard),
            "hard_margin": float(args.hard_margin),
        },
        "architectures": results,
        "started_at_unix": started,
        "ended_at_unix": time.time(),
    }
    comparison["duration_seconds"] = comparison["ended_at_unix"] - started
    _write_json(output_root / "architecture_comparison.json", comparison)
    fields = [
        "architecture_name",
        "status",
        "training_split",
        "paper_status",
        "paper_valid",
        "lambda_dist",
        "lambda_hard",
        "seed",
        "epochs",
        "optimizer_steps",
        "best_epoch",
        "best_holdout_loss",
        "best_holdout_final_top1",
        "best_holdout_final_mrr",
        "best_holdout_distribution_top1",
        "best_holdout_distribution_mrr",
        "train_video_count",
        "holdout_video_count",
    ]
    with (output_root / "architecture_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in results)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-search-root", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-dist", type=float, default=DEFAULT_LAMBDA_DIST)
    parser.add_argument("--lambda-hard", type=float, default=DEFAULT_LAMBDA_HARD)
    parser.add_argument("--hard-margin", type=float, default=DEFAULT_HARD_MARGIN)
    parser.add_argument("--mixed-k", type=int, nargs="+", default=list(DEFAULT_MIXED_K))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
