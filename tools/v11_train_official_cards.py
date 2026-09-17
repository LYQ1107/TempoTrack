"""Run one Official-Train DSSL card with an immutable experiment receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from tempotrack_v10.qdic_dssl_trainer import train_official_card


CARDS: dict[str, dict[str, Any]] = {
    "B0_OFFICIAL_V11": {
        "structured_branch_mode": "legacy",
        "lambda_struct": 0.0,
        "lambda_cons": 0.0,
        "lambda_hard": 0.0,
        "temporal_conflict": False,
    },
    "B1_DSSL_ARCH_ONLY": {
        "structured_branch_mode": "dssl",
        "lambda_struct": 0.0,
        "lambda_cons": 0.0,
        "lambda_hard": 0.2,
        "temporal_conflict": False,
    },
    "D1_LS010": {
        "structured_branch_mode": "dssl",
        "lambda_struct": 0.10,
        "lambda_cons": 0.0,
        "lambda_hard": 0.2,
        "temporal_conflict": False,
    },
    "D2_LS025": {
        "structured_branch_mode": "dssl",
        "lambda_struct": 0.25,
        "lambda_cons": 0.0,
        "lambda_hard": 0.2,
        "temporal_conflict": False,
    },
    "D3_LS050": {
        "structured_branch_mode": "dssl",
        "lambda_struct": 0.50,
        "lambda_cons": 0.0,
        "lambda_hard": 0.2,
        "temporal_conflict": False,
    },
    "D4_LS100": {
        "structured_branch_mode": "dssl",
        "lambda_struct": 1.00,
        "lambda_cons": 0.0,
        "lambda_hard": 0.2,
        "temporal_conflict": False,
    },
    "C1_LC005": {
        "structured_branch_mode": "dssl",
        "lambda_struct": 0.25,
        "lambda_cons": 0.05,
        "lambda_hard": 0.2,
        "temporal_conflict": False,
    },
    "C2_LC010": {
        "structured_branch_mode": "dssl",
        "lambda_struct": 0.25,
        "lambda_cons": 0.10,
        "lambda_hard": 0.2,
        "temporal_conflict": False,
    },
    "C3_LC025": {
        "structured_branch_mode": "dssl",
        "lambda_struct": 0.25,
        "lambda_cons": 0.25,
        "lambda_hard": 0.2,
        "temporal_conflict": False,
    },
    "H1_TEMPORAL_CONFLICT": {
        "structured_branch_mode": "dssl",
        "lambda_struct": 0.25,
        "lambda_cons": 0.10,
        "lambda_hard": 0.2,
        "temporal_conflict": True,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run_card(
    *,
    card_id: str,
    features: Path,
    output_root: Path,
    device: str,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    if card_id not in CARDS:
        raise ValueError(f"unknown card: {card_id}")
    features = features.resolve()
    output_root = output_root.resolve()
    output = output_root / card_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite card output: {output}")
    output_root.mkdir(parents=True, exist_ok=True)
    feature_metadata = json.loads((features / "features.json").read_text(encoding="utf-8"))
    config = {
        "artifact": "qdic_v11_dssl_official_card_config",
        "card_id": card_id,
        "card": dict(CARDS[card_id]),
        "features": str(features / "features.json"),
        "features_hash": _sha256(features / "features.json"),
        "exact_split_name": feature_metadata.get("exact_split_name"),
        "source_role": feature_metadata.get("source_role"),
        "input_source": feature_metadata.get("input_source"),
        "supervision_source": feature_metadata.get("supervision_source"),
        "feature_dim": feature_metadata.get("feature_dim"),
        "feature_config": feature_metadata.get("feature_config"),
        "fixed_runtime": {
            "candidate_top_k": 8,
            "context_candidate_top_k": 64,
            "memory_capacity": 64,
            "recent_k": 8,
            "max_gap": 360,
            "query_observations": 1,
            "mixed_candidate_k": [8, 16, 32, 64],
            "normalization_fit": "internal_train_base_only_after_video_split",
        },
        "epochs": int(epochs),
        "seed": int(seed),
        "device": str(device),
        "repo_head": _repo_head(),
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "OMP_NUM_THREADS")
        },
    }
    result = train_official_card(
        features,
        output,
        card_id=card_id,
        device=device,
        epochs=int(epochs),
        seed=int(seed),
        **CARDS[card_id],
    )
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "status": result["status"],
        "card_id": card_id,
        "output": str(output),
        "checkpoint": result["checkpoint"],
        "checkpoint_hash": result["checkpoint_hash"],
        "best_epoch": result["best_epoch"],
        "repo_head": config["repo_head"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--card", required=True, choices=sorted(CARDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_card(
        card_id=args.card,
        features=args.features,
        output_root=args.output_root,
        device=args.device,
        epochs=args.epochs,
        seed=args.seed,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
