#!/usr/bin/env python3
"""Prepare an auditable Official-Val replay plan from the final Test choice.

The Test-tuned leaderboard is the source of the selected operating points, but
Official-Val is replayed independently from the shared Val COV cache.  This
tool only writes protocol/config receipts; it never runs tracking or TETA.
The generated configs preserve the checkpoint and thresholds from the selected
Test candidates while changing only the split/provenance label to the exact
Official-Val split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

try:
    from tools.v12_make_mgf_exploration_configs import _write_b0_config, _write_config
except ModuleNotFoundError:  # direct ``python tools/...`` invocation
    from v12_make_mgf_exploration_configs import _write_b0_config, _write_config


VAL_SPLIT = "validation_ours_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _safe_name(value: Any, field: str) -> str:
    text = str(value)
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"unsafe {field}: {text!r}")
    return text


def _selected_rows(final_json: Path) -> list[dict[str, Any]]:
    result = read_json(final_json)
    if not isinstance(result, dict):
        raise RuntimeError("final leaderboard is not a JSON object")
    if result.get("status") != "PASS" or result.get("paper_status") != "TEST_TUNED_EXPLORATION":
        raise RuntimeError("final leaderboard is not TEST_TUNED_EXPLORATION")
    best = result.get("best_mgf")
    b0 = result.get("b0_reference")
    if not isinstance(best, dict) or not isinstance(b0, dict):
        raise RuntimeError("final leaderboard lacks best_mgf or b0_reference")
    rows = [best, b0]
    seen: set[str] = set()
    for row in rows:
        candidate_id = _safe_name(row.get("candidate_id"), "candidate_id")
        method = str(row.get("method", ""))
        if candidate_id in seen:
            raise RuntimeError(f"duplicate selected candidate: {candidate_id}")
        if row.get("test_tuned") is not True or row.get("paper_valid") is not False:
            raise RuntimeError(f"selected candidate is not diagnostic Test-tuned: {candidate_id}")
        if method != "B0" and not method.startswith("E"):
            raise RuntimeError(f"best MGF method is not a registered exploration card: {method}")
        if row.get("score_threshold") is None or row.get("margin_threshold") is None:
            raise RuntimeError(f"selected candidate has incomplete thresholds: {candidate_id}")
        seen.add(candidate_id)
    return rows


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    final_json = args.final_json.resolve()
    rows = _selected_rows(final_json)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Preserve the exact Test-only selection receipt.  The later Val-aware
    # finalizer rewrites final_20h_test_tuned_leaderboard.json, so hashing that
    # mutable path directly would make the plan's provenance stale.
    test_snapshot = output_root / "source_test_final_leaderboard.json"
    if test_snapshot.exists():
        raise RuntimeError(f"refusing to overwrite Test selection snapshot: {test_snapshot}")
    shutil.copy2(final_json, test_snapshot)
    config_root = output_root / "configs"
    candidates: list[dict[str, Any]] = []
    config_receipts: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = _safe_name(row["candidate_id"], "candidate_id")
        method = str(row["method"])
        score = float(row["score_threshold"])
        margin = float(row["margin_threshold"])
        config = config_root / f"{candidate_id}.yaml"
        if method == "B0":
            receipt = _write_b0_config(
                args.b0_checkpoint,
                config,
                source_role="OFFICIAL_VAL",
                split=VAL_SPLIT,
                score_threshold=score,
                margin_threshold=margin,
            )
            card_id = "B0_OFFICIAL_V11"
        else:
            card_id = _safe_name(method, "card_id")
            receipt = _write_config(
                args.card_root / card_id,
                config,
                source_role="OFFICIAL_VAL",
                split=VAL_SPLIT,
                score_threshold=score,
                margin_threshold=margin,
            )
        config_receipts.append(receipt)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "card_id": card_id,
                "trial_id": "s00_m00",
                "config": str(config.resolve()),
                "config_sha256": sha256_file(config),
                "score_threshold": score,
                "margin_threshold": margin,
                "source_test_candidate_id": candidate_id,
            }
        )
    plan = {
        "status": "PASS",
        "artifact": "v12_mgf_test_tuned_official_val_replay_plan",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "selection_scope": "VAL_TEST_TUNED_EXPLORATION",
        "test_used_for_selection": True,
        "val_used_for_selection": False,
        "split": "Official-Val",
        "exact_split_name": VAL_SPLIT,
        "protocol": "Replay the final Test-selected MGF and B0 operating points on an independent Official-Val COV cache; Val does not select the operating point.",
        "source_test_final_json": str(test_snapshot),
        "source_test_final_json_sha256": sha256_file(test_snapshot),
        "candidates": candidates,
        "config_receipts": config_receipts,
    }
    write_json(args.plan_path.resolve(), plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path, required=True)
    parser.add_argument(
        "--card-root",
        type=Path,
        default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/03_train/exploration_cards"),
    )
    parser.add_argument(
        "--b0-checkpoint",
        type=Path,
        default=Path("/data2/usr_for_deadline/tempotrack_v11_dssl_official_20260917_full/03_train_rerun_c4f1bab/B0_OFFICIAL_V11/best.pt"),
    )
    args = parser.parse_args()
    plan = prepare(args)
    print(json.dumps({"status": plan["status"], "candidates": plan["candidates"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
