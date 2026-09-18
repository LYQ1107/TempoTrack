#!/usr/bin/env python3
"""Train one pre-registered exploratory fixed-beta or adaptive card."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tempotrack_v10.qdic_features import QDIC_MGF_EXPLORATION_BETAS
from tempotrack_v10.qdic_mgf_exploration_trainer import train_exploration_card


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--card-id", required=True)
    parser.add_argument("--card-type", choices=("fixed", "adaptive"), required=True)
    parser.add_argument("--mgf-mode", choices=("core", "fused"), required=True)
    parser.add_argument("--beta-index", type=int)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.card_type == "fixed":
        if args.beta_index is None or not (0 <= args.beta_index < len(QDIC_MGF_EXPLORATION_BETAS)):
            parser.error("fixed cards require --beta-index in the pre-registered beta bank")
    elif args.beta_index is not None:
        parser.error("adaptive cards must not specify --beta-index")
    output = args.output_root / args.card_id
    result = train_exploration_card(
        args.features,
        output,
        card_id=args.card_id,
        card_type=args.card_type,
        mgf_mode=args.mgf_mode,
        beta_index=args.beta_index,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps({"status": result["status"], "card_id": result["card_id"], "output": str(output), "best_epoch": result["best_epoch"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
