#!/usr/bin/env python3
"""Freeze M1/M2 using only their Official-Train internal holdout receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparator(receipt: dict[str, Any]) -> tuple[float, float, float, float]:
    history = receipt.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("training receipt has no internal holdout history")
    best_epoch = int(receipt.get("best_epoch", -1))
    rows = [row for row in history if int(row.get("epoch", -1)) == best_epoch]
    if len(rows) != 1:
        raise ValueError("training receipt best_epoch is not uniquely recorded")
    row = rows[0]
    return (
        float(row["final_mrr"]),
        float(row["final_top1"]),
        float(row["net_correction"]),
        -float(row["final_listwise_loss"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for card_id in ("M1_MGF_CORE", "M2_MGF_FUSED"):
        card = args.train_root / card_id
        receipt_path = card / "training.json"
        checkpoint = card / "best.pt"
        if not receipt_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"incomplete MGF card: {card}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("protocol") != "QDIC_V12_MGF_BASE_ONLY_TRAINING":
            raise ValueError(f"invalid MGF protocol in {receipt_path}")
        if receipt.get("paper_valid") is not True or receipt.get("diagnostic_only") is not False:
            raise ValueError(f"invalid paper role in {receipt_path}")
        comparator = _comparator(receipt)
        records.append(
            {
                "card_id": card_id,
                "mgf_mode": receipt["mgf_mode"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "training_receipt": str(receipt_path.resolve()),
                "training_receipt_sha256": _sha256(receipt_path),
                "best_epoch": int(receipt["best_epoch"]),
                "comparator": list(comparator),
                "comparator_source": "Official Train internal video-disjoint holdout only",
                "mgf_beta": float(receipt["mgf_beta"]),
                "feature_dim": int(receipt["feature_dim"]),
                "repo_head": receipt.get("repo_head"),
                "train_cache": receipt.get("features"),
                "train_cache_hash": receipt.get("features_hash"),
            }
        )
    selected = max(records, key=lambda item: tuple(item["comparator"]))
    result = {
        "status": "FROZEN",
        "artifact": "qdic_v12_mgf_method_freeze",
        "selection_scope": "OFFICIAL_TRAIN_INTERNAL_VIDEO_DISJOINT_HOLDOUT_ONLY",
        "val_used_for_selection": False,
        "novel_used_for_selection": False,
        "test_used_for_selection": False,
        "selection_metric": "(final_mrr, final_top1, net_correction, -final_listwise_loss)",
        "selected_card": selected["card_id"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_mgf_mode": selected["mgf_mode"],
        "mgf_beta": 1.0,
        "feature_schema_version": 12,
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected_card": result["selected_card"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
