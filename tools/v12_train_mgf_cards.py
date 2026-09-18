#!/usr/bin/env python3
"""Train one pre-registered V12 exact-log-MGF card."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tempotrack_v10.qdic_mgf_trainer import train_official_mgf_card


CARD_MODES = {
    "M1_MGF_CORE": "core",
    "M2_MGF_FUSED": "fused",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--card", choices=sorted(CARD_MODES), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = args.output_root / args.card
    result = train_official_mgf_card(
        args.features,
        output,
        card_id=args.card,
        mgf_mode=CARD_MODES[args.card],
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps({"status": result["status"], "card_id": result["card_id"], "output": str(output), "best_epoch": result["best_epoch"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
