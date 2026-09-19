#!/usr/bin/env python3
"""Create replay configs for registered TEST_TUNED_EXPLORATION cards.

The generated configs are deliberately boring: the causal runtime contract is
fixed, and only the audited card checkpoint changes.  They are not operating
point search configs and do not alter the formal V12 branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quote(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _write_config(
    card_dir: Path,
    output: Path,
    *,
    source_role: str,
    split: str,
    score_threshold: float = 0.0,
    margin_threshold: float = 0.0,
) -> dict[str, Any]:
    receipt_path = card_dir / "training.json"
    checkpoint = (card_dir / "best.pt").resolve()
    if not receipt_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"missing audited card artifacts: {card_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "COMPLETED" or receipt.get("paper_status") != "TEST_TUNED_EXPLORATION":
        raise ValueError(f"card is not TEST_TUNED_EXPLORATION: {card_dir}")
    if receipt.get("paper_valid") is not False or receipt.get("diagnostic_only") is not True:
        raise ValueError(f"card paper-status guard failed: {card_dir}")
    card = str(receipt.get("card_id") or card_dir.name)
    text = "\n".join(
        (
            "frontend: covtrack",
            "protocol:",
            "  name: TEST_TUNED_EXPLORATION",
            f"  source_role: {_quote(source_role)}",
            f"  exact_split_name: {_quote(split)}",
            "  optimizer_source_allowed: false",
            "  unbiased_test: false",
            f"  card_id: {_quote(card)}",
            f"  card_type: {_quote(receipt.get('card_type'))}",
            "  selection_scope: VAL_TEST_TUNED_EXPLORATION",
            "  test_used_for_selection: true",
            f"  train_checkpoint: {_quote(str(checkpoint))}",
            f"  train_checkpoint_sha256: {_quote(_sha256(checkpoint))}",
            "tempo:",
            "  enabled: true",
            "  alpha_fast: 0.70",
            "  alpha_slow: 0.15",
            "  min_gap: 0",
            "  max_gap: 360",
            "  candidate_top_k: 8",
            "  top_r: 3",
            "  memory_capacity: 64",
            "  qdic_weight: 1.0",
            "  qdic_recent_k: 8",
            "  qdic_context_top_k: 64",
            f"  qdic_checkpoint: {_quote(str(checkpoint))}",
            "  qdic_device: cpu",
            f"  score_threshold: {_quote(float(score_threshold))}",
            f"  margin_threshold: {_quote(float(margin_threshold))}",
            "search_fields:",
            "- score_threshold",
            "- margin_threshold",
            "",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return {
        "card_id": card,
        "card_type": receipt.get("card_type"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "config": str(output.resolve()),
        "config_sha256": _sha256(output),
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "score_threshold": float(score_threshold),
        "margin_threshold": float(margin_threshold),
    }


def _write_b0_config(
    checkpoint: Path,
    output: Path,
    *,
    source_role: str,
    split: str,
    score_threshold: float = 0.0,
    margin_threshold: float = 0.0,
) -> dict[str, Any]:
    """Create the explicitly diagnostic B0 comparator config.

    The checkpoint's source receipt remains BASE_TRAIN.  Only the replay
    protocol is Test-tuned; this function never rewrites the source receipt or
    relabels B0 as an exploration-trained card.
    """
    checkpoint = checkpoint.resolve()
    receipt_path = checkpoint.parent / "training.json"
    if not checkpoint.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing audited B0 artifacts: {checkpoint}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "status": "COMPLETED",
        "artifact": "qdic_v11_dssl_official_training",
        "card_id": "B0_OFFICIAL_V11",
        "protocol": "QDIC_V11_BASE_ONLY_TRAINING",
        "training_split": "train_base_official",
        "paper_status": "BASE_TRAIN",
        "paper_valid": True,
        "diagnostic_only": False,
        "base_only_supervision": True,
        "novel_gt_used": False,
        "test_weights_used": False,
        "feature_dim": 33,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"B0 receipt guard failed for {key}: {checkpoint}")
    checkpoint_sha256 = _sha256(checkpoint)
    if receipt.get("checkpoint_hash") != checkpoint_sha256:
        raise ValueError(f"B0 checkpoint hash mismatch: {checkpoint}")
    card = "B0_OFFICIAL_V11"
    text = "\n".join(
        (
            "frontend: covtrack",
            "protocol:",
            "  name: TEST_TUNED_EXPLORATION",
            f"  source_role: {_quote(source_role)}",
            f"  exact_split_name: {_quote(split)}",
            "  optimizer_source_allowed: false",
            "  unbiased_test: false",
            f"  card_id: {_quote(card)}",
            "  card_type: b0_comparator",
            "  comparison_role: B0_OFFICIAL_V11_TEST_TUNED_COMPARATOR",
            "  selection_scope: VAL_TEST_TUNED_EXPLORATION",
            "  test_used_for_selection: true",
            f"  train_checkpoint: {_quote(str(checkpoint))}",
            f"  train_checkpoint_sha256: {_quote(checkpoint_sha256)}",
            "tempo:",
            "  enabled: true",
            "  alpha_fast: 0.70",
            "  alpha_slow: 0.15",
            "  min_gap: 0",
            "  max_gap: 360",
            "  candidate_top_k: 8",
            "  top_r: 3",
            "  memory_capacity: 64",
            "  qdic_weight: 1.0",
            "  qdic_recent_k: 8",
            "  qdic_context_top_k: 64",
            f"  qdic_checkpoint: {_quote(str(checkpoint))}",
            "  qdic_device: cpu",
            f"  score_threshold: {_quote(float(score_threshold))}",
            f"  margin_threshold: {_quote(float(margin_threshold))}",
            "search_fields:",
            "- score_threshold",
            "- margin_threshold",
            "",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return {
        "card_id": card,
        "card_type": "b0_comparator",
        "comparison_role": "B0_OFFICIAL_V11_TEST_TUNED_COMPARATOR",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "config": str(output.resolve()),
        "config_sha256": _sha256(output),
        "source_paper_status": "BASE_TRAIN",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "score_threshold": float(score_threshold),
        "margin_threshold": float(margin_threshold),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cards", nargs="*", default=[])
    parser.add_argument("--b0-checkpoint", type=Path)
    parser.add_argument("--b0-output-name", default="B0_OFFICIAL_V11")
    parser.add_argument("--source-role", default="CURRENT_TEST")
    parser.add_argument("--split", default="test")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    args = parser.parse_args()
    if not args.cards and args.b0_checkpoint is None:
        parser.error("provide --cards and/or --b0-checkpoint")
    cards = []
    for card in args.cards:
        cards.append(
            _write_config(
                args.card_root / card,
                args.output_root / f"{card}.yaml",
                source_role=args.source_role,
                split=args.split,
                score_threshold=args.score_threshold,
                margin_threshold=args.margin_threshold,
            )
        )
    if args.b0_checkpoint is not None:
        cards.append(
            _write_b0_config(
                args.b0_checkpoint,
                args.output_root / f"{args.b0_output_name}.yaml",
                source_role=args.source_role,
                split=args.split,
                score_threshold=args.score_threshold,
                margin_threshold=args.margin_threshold,
            )
        )
    receipt = {
        "status": "PASS",
        "artifact": "qdic_v12_mgf_test_tuned_exploration_replay_configs",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "selection_scope": "VAL_TEST_TUNED_EXPLORATION",
        "cards": cards,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "config_manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
