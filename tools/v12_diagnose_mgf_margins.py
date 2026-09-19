#!/usr/bin/env python3
"""Offline Test proposal-margin diagnostic for selected V12 cards.

This consumes the frozen 49-D Test event feature cache and checkpoints only.
It does not instantiate COVTrack, read images, load GT, or run tracking replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tempotrack_v10.qdic_features import QDIC_RAW_DIM
from tempotrack_v10.qdic_loader import load_qdic_checkpoint
from tempotrack_v10.qdic_mgf_exploration_loader import load_qdic_mgf_exploration_checkpoint
from tools.v12_evaluate_mgf_exploration_ranking import _fixed_indices, _load_feature_cache, _predict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_card(card_root: Path, card: str, features: np.ndarray, device: str) -> tuple[np.ndarray, dict[str, Any]]:
    artifact = load_qdic_mgf_exploration_checkpoint(card_root / card / "best.pt", device=device)
    if artifact.receipt.get("card_type") == "fixed":
        values = features[:, _fixed_indices(int(artifact.receipt["beta_index"]))]
    else:
        values = features
    return _predict(artifact.model, values, device=device), artifact.provenance


def _score_b0(checkpoint: Path, features: np.ndarray, device: str) -> tuple[np.ndarray, dict[str, Any]]:
    artifact = load_qdic_checkpoint(checkpoint, device=device)
    return _predict(artifact.model, np.asarray(features[:, :QDIC_RAW_DIM], dtype=np.float32), device=device), artifact.provenance


def _margins(scores: np.ndarray, features: np.ndarray, offsets: np.ndarray, candidate_top_k: int) -> np.ndarray:
    values: list[float] = []
    ranks = np.asarray(features[:, 17], dtype=np.float32)
    for start_value, end_value in zip(offsets[:-1], offsets[1:]):
        start, end = int(start_value), int(end_value)
        valid = np.isfinite(scores[start:end]) & np.isfinite(ranks[start:end]) & (ranks[start:end] <= candidate_top_k)
        selected = np.asarray(scores[start:end])[valid]
        if len(selected) < 2:
            continue
        ordered = np.sort(selected.astype(np.float64))[::-1]
        margin = float(ordered[0] - ordered[1])
        if np.isfinite(margin) and margin >= 0.0:
            values.append(margin)
    return np.asarray(values, dtype=np.float64)


def diagnose(*, feature_root: Path, card_root: Path, b0_checkpoint: Path, cards: list[str], output: Path, device: str, candidate_top_k: int) -> dict[str, Any]:
    metadata, arrays = _load_feature_cache(feature_root.resolve())
    features = np.asarray(arrays["features"])
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != 49:
        raise RuntimeError("margin diagnostic requires the audited 49-D Test exploration cache")
    scores: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    b0_scores, b0_provenance = _score_b0(b0_checkpoint.resolve(), features, device)
    scores["B0"] = (b0_scores, b0_provenance)
    provenance: dict[str, Any] = {"B0": b0_provenance}
    for card in cards:
        card_scores, card_provenance = _score_card(card_root.resolve(), card, features, device)
        scores[card] = (card_scores, card_provenance)
        provenance[card] = card_provenance
    result: dict[str, Any] = {
        "status": "PASS",
        "artifact": "qdic_v12_mgf_test_tuned_offline_margin_diagnostic",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "test_used_for_operating_point_selection": True,
        "feature_cache": str(feature_root.resolve()),
        "feature_cache_metadata_sha256": sha256_file(feature_root.resolve() / "features.json"),
        "candidate_top_k": int(candidate_top_k),
        "cards": cards,
        "provenance": provenance,
        "distributions": {},
    }
    for method, (values, _method_provenance) in scores.items():
        margins = _margins(values, features, offsets, int(candidate_top_k))
        if not len(margins):
            raise RuntimeError(f"no finite proposal margins for {method}")
        result["distributions"][method] = {
            "count": int(len(margins)),
            "winner_score_min": float(np.min(margins)),
            "p10": float(np.quantile(margins, 0.10)),
            "p25": float(np.quantile(margins, 0.25)),
            "p40": float(np.quantile(margins, 0.40)),
            "p50": float(np.quantile(margins, 0.50)),
            "p60": float(np.quantile(margins, 0.60)),
            "p95": float(np.quantile(margins, 0.95)),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "methods": list(result["distributions"])}, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--card-root", type=Path, required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--cards", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-top-k", type=int, default=8)
    args = parser.parse_args()
    diagnose(
        feature_root=args.feature_root,
        card_root=args.card_root,
        b0_checkpoint=args.b0_checkpoint,
        cards=[str(card) for card in args.cards],
        output=args.output,
        device=args.device,
        candidate_top_k=args.candidate_top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
