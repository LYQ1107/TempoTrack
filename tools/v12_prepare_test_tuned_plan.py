#!/usr/bin/env python3
"""Prepare explicit B0 or Top-2 margin-refinement replay plans.

This tool does not run tracking.  It only consumes completed Test-tuned
artifacts and writes configs plus a hash-bound plan for the refinement
launcher.  The selection order is frozen as:
Test Novel AssocA, Test Overall AssocA, Test Overall TETA, Test Base AssocA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from tools.v12_make_mgf_exploration_configs import _write_b0_config, _write_config
except ModuleNotFoundError:  # direct ``python tools/...`` invocation
    from v12_make_mgf_exploration_configs import _write_b0_config, _write_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe plan name: {value!r}")
    return value


def _metric(metrics: dict[str, Any], split: str, name: str) -> float:
    value = metrics.get(split, {}).get(name)
    if value is None:
        raise RuntimeError(f"missing {split}.{name} in {metrics}")
    return float(value)


def _load_initial_candidate(replay_root: Path, card: str) -> dict[str, Any]:
    metrics_path = replay_root / card / "s00_m00" / "merged" / "test_tuned_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"completed initial Full-Test metrics missing: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or metrics.get("overall", {}).get("TETA") is None:
        raise RuntimeError(f"invalid initial Full-Test metrics: {metrics_path}")
    return {
        "card_id": card,
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "test_novel_assoc_a": _metric(metrics, "novel", "AssocA"),
        "test_overall_assoc_a": _metric(metrics, "overall", "AssocA"),
        "test_overall_teta": _metric(metrics, "overall", "TETA"),
        "test_base_assoc_a": _metric(metrics, "base", "AssocA"),
    }


def _margin_label(value: float) -> str:
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _candidate(
    *,
    candidate_id: str,
    card_id: str,
    trial_id: str,
    config: Path,
    margin: float,
) -> dict[str, Any]:
    return {
        "candidate_id": _safe_name(candidate_id),
        "card_id": _safe_name(card_id),
        "trial_id": _safe_name(trial_id),
        "config": str(config.resolve()),
        "config_sha256": sha256_file(config),
        "score_threshold": 0.0,
        "margin_threshold": float(margin),
    }


def prepare_b0_initial(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    config = output_root / "configs" / "B0_OFFICIAL_V11.yaml"
    row = _write_b0_config(
        args.b0_checkpoint,
        config,
        source_role="CURRENT_TEST",
        split="test",
        score_threshold=0.0,
        margin_threshold=0.0,
    )
    plan = {
        "status": "PASS",
        "artifact": "v12_mgf_test_tuned_initial_b0_plan",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "selection_scope": "VAL_TEST_TUNED_EXPLORATION",
        "test_used_for_selection": True,
        "protocol": "initial score=0 margin=0 B0 comparator Full-Test",
        "candidates": [
            _candidate(
                candidate_id="B0_INITIAL",
                card_id="B0_OFFICIAL_V11",
                trial_id="s00_m00",
                config=config,
                margin=0.0,
            )
        ],
        "config_receipt": row,
    }
    write_json(args.plan_path.resolve(), plan)
    return plan


def prepare_refinement(args: argparse.Namespace) -> dict[str, Any]:
    margins = [float(value) for value in args.margins]
    if len(margins) != 4 or any(value < 0.0 for value in margins):
        raise ValueError("--margins requires four nonnegative margin values")
    cards = [str(value) for value in args.initial_cards]
    if len(cards) < 2:
        raise ValueError("at least two completed initial cards are required")
    initial = [_load_initial_candidate(args.initial_replay_root.resolve(), card) for card in cards]
    ranked = sorted(
        initial,
        key=lambda row: (
            -row["test_novel_assoc_a"],
            -row["test_overall_assoc_a"],
            -row["test_overall_teta"],
            -row["test_base_assoc_a"],
            row["card_id"],
        ),
    )
    top2 = ranked[:2]
    output_root = args.output_root.resolve()
    config_root = output_root / "configs"
    candidates: list[dict[str, Any]] = []
    config_receipts: list[dict[str, Any]] = []
    for card_index, row in enumerate(top2):
        card = row["card_id"]
        for point_index, margin in enumerate(margins[card_index * 2 : card_index * 2 + 2], start=1):
            label = _margin_label(margin)
            candidate_id = f"{card}_REF_M{label}_{point_index}"
            config = config_root / f"{candidate_id}.yaml"
            receipt = _write_config(
                args.card_root.resolve() / card,
                config,
                source_role="CURRENT_TEST",
                split="test",
                score_threshold=0.0,
                margin_threshold=margin,
            )
            config_receipts.append(receipt)
            candidates.append(
                _candidate(
                    candidate_id=candidate_id,
                    card_id=card,
                    trial_id=f"ref_m{label}_{point_index}",
                    config=config,
                    margin=margin,
                )
            )
    # B0 receives the same total number of refinement trials as the two MGF
    # cards combined.  Its points are the exact four Test-tuned points used by
    # the selected MGF cards, so the comparison changes only the checkpoint.
    for point_index, margin in enumerate(margins, start=1):
        label = _margin_label(margin)
        candidate_id = f"B0_REF_M{label}_{point_index}"
        config = config_root / f"{candidate_id}.yaml"
        receipt = _write_b0_config(
            args.b0_checkpoint,
            config,
            source_role="CURRENT_TEST",
            split="test",
            score_threshold=0.0,
            margin_threshold=margin,
        )
        config_receipts.append(receipt)
        candidates.append(
            _candidate(
                candidate_id=candidate_id,
                card_id="B0_OFFICIAL_V11",
                trial_id=f"ref_m{label}_{point_index}",
                config=config,
                margin=margin,
            )
        )
    plan = {
        "status": "PASS",
        "artifact": "v12_mgf_test_tuned_margin_refinement_plan",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "selection_scope": "VAL_TEST_TUNED_EXPLORATION",
        "test_used_for_selection": True,
        "selection_order": [
            "Test Novel AssocA",
            "Test Overall AssocA",
            "Test Overall TETA",
            "Test Base AssocA",
        ],
        "initial_cards": initial,
        "top2": top2,
        "margin_points": margins,
        "b0_fairness": {
            "rule": "B0 receives the same four refinement trial count as Top-2 combined",
            "trial_count": 4,
            "points": margins,
        },
        "candidates": candidates,
        "config_receipts": config_receipts,
    }
    write_json(args.plan_path.resolve(), plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("initial_b0", "refinement"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path, required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-replay-root", type=Path)
    parser.add_argument("--initial-cards", nargs="+", default=[])
    parser.add_argument("--card-root", type=Path)
    parser.add_argument("--margins", nargs="+", default=[])
    args = parser.parse_args()
    if args.mode == "initial_b0":
        result = prepare_b0_initial(args)
    else:
        if args.initial_replay_root is None or args.card_root is None:
            parser.error("refinement requires --initial-replay-root and --card-root")
        result = prepare_refinement(args)
    print(json.dumps({"status": result["status"], "artifact": result["artifact"], "candidates": [item["candidate_id"] for item in result["candidates"]]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
