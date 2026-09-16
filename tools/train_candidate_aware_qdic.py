#!/usr/bin/env python3
"""Train one auditable A0/A0-D/A1/A2 candidate-aware QDIC model."""

from __future__ import annotations

import argparse
import json

from tempotrack_v10.candidate_aware_qdic_trainer import (
    DEFAULT_HARD_MARGIN,
    DEFAULT_LAMBDA_HARD,
    DEFAULT_MIXED_K,
    train_architecture,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", choices=("A0", "A0-D", "A1", "A2"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-dist", type=float)
    parser.add_argument("--lambda-hard", type=float, default=DEFAULT_LAMBDA_HARD)
    parser.add_argument("--hard-margin", type=float, default=DEFAULT_HARD_MARGIN)
    parser.add_argument("--mixed-k", type=int, nargs="+", default=list(DEFAULT_MIXED_K))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = train_architecture(
        args.features_dir,
        args.parent_checkpoint,
        args.output,
        architecture=args.architecture,
        device=args.device,
        epochs=args.epochs,
        seed=args.seed,
        lambda_dist=args.lambda_dist,
        lambda_hard=args.lambda_hard,
        hard_margin=args.hard_margin,
        mixed_k_values=args.mixed_k,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
