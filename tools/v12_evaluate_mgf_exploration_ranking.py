#!/usr/bin/env python3
"""Rank all exploratory cards on Train holdout, Val, and Test.

This evaluator intentionally uses Test labels for the explicit exploratory
ranking objective.  Every output is marked ``TEST_TUNED_EXPLORATION`` and is
not an unbiased/paper-valid result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from tempotrack_v10.qdic_features import (
    QDIC_MGF_EXPLORATION_BETAS,
    QDIC_RAW_DIM,
    _load_cache,
)
from tempotrack_v10.qdic_mgf_exploration_loader import load_qdic_mgf_exploration_checkpoint
from tempotrack_v10.query_distributional_calibrator import QueryDistributionalCalibrator
from tempotrack_v10.qdic_mgf_exploration_trainer import build_official_train_groups


METHOD_ORDER = (
    "Q1_SUPPORT",
    "MEAN",
    "MAX",
    "MO2",
    "EXACT_LMGF_BETA_M2",
    "EXACT_LMGF_BETA_M1",
    "EXACT_LMGF_BETA_M05",
    "EXACT_LMGF_BETA_P025",
    "EXACT_LMGF_BETA_P05",
    "EXACT_LMGF_BETA_P1",
    "EXACT_LMGF_BETA_P2",
    "EXACT_LMGF_BETA_P4",
    "B0",
) + tuple(f"E{index:02d}" for index in range(1, 19))
SPLITS = ("TRAIN_HOLDOUT", "VAL", "TEST")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_feature_cache(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((root / "features.json").read_text(encoding="utf-8"))
    if metadata.get("artifact") != "qdic_v12_mgf_exploration_feature_cache":
        raise ValueError(f"not an exploration feature cache: {root}")
    if int(metadata.get("feature_dim", -1)) != 49:
        raise ValueError("exploration ranking requires 49-D features")
    paths = dict(metadata.get("arrays", {}))
    hashes = dict(metadata.get("array_hashes", {}))
    required = {"features", "labels", "target_base", "offsets", "videos", "group_ids"}
    if not required.issubset(paths):
        raise ValueError(f"exploration cache missing arrays: {sorted(required - set(paths))}")
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        path = Path(str(paths[name]))
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or _sha256(path) != hashes.get(name):
            raise ValueError(f"exploration array hash mismatch: {path}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    count = len(arrays["features"])
    if arrays["features"].shape != (count, 49):
        raise ValueError("exploration feature array must be [N,49]")
    if len(arrays["offsets"]) != len(arrays["videos"]) + 1 or int(arrays["offsets"][-1]) != count:
        raise ValueError("exploration offsets/videos are inconsistent")
    return metadata, arrays


def _aligned_raw_history_values(event_cache: Path, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    _metadata, arrays, _inline, _rows = _load_cache(event_cache)
    cosine = np.asarray(arrays["cosine"])
    mem_len = np.asarray(arrays["mem_len"], dtype=np.int64).reshape(-1)
    rank = np.asarray(arrays["prefilter_rank_b1"], dtype=np.int64).reshape(-1)
    selected = rank <= 64
    if cosine.ndim != 3 or len(mem_len) != len(cosine):
        raise ValueError("raw event cache cosine/mem_len mismatch")
    width = int(cosine.shape[2])
    positions = np.arange(width, dtype=np.int64)[None, :]
    max_scores: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    for start in range(0, len(cosine), 100_000):
        end = min(start + 100_000, len(cosine))
        value = np.asarray(cosine[start:end, 0, :], dtype=np.float32)
        valid = positions < mem_len[start:end, None]
        max_scores.append(np.max(np.where(valid, value, -np.inf), axis=1)[selected[start:end]])
        lengths.append(mem_len[start:end][selected[start:end]])
    result_scores = np.concatenate(max_scores).astype(np.float32, copy=False)
    result_lengths = np.concatenate(lengths).astype(np.int64, copy=False)
    if len(result_scores) != expected_rows:
        raise ValueError(f"raw event alignment mismatch: {len(result_scores)} vs {expected_rows}")
    return result_scores, result_lengths


def _manual_load_b0(checkpoint: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    receipt = json.loads((checkpoint.parent / "training.json").read_text(encoding="utf-8"))
    if receipt.get("paper_status") != "BASE_TRAIN" or receipt.get("paper_valid") is not True:
        raise ValueError("B0 checkpoint is not the official Base-trained baseline")
    state = torch.load(checkpoint, map_location=device)
    model = QueryDistributionalCalibrator(
        structured_branch_mode=str(receipt.get("structured_branch_mode", "legacy"))
    ).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    return model, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "training_json_sha256": _sha256(checkpoint.parent / "training.json"),
        "training_split": receipt.get("training_split"),
        "paper_status": receipt.get("paper_status"),
        "paper_valid": receipt.get("paper_valid"),
        "source_repo_head": receipt.get("repo_head"),
    }


def _predict(model: torch.nn.Module, features: np.ndarray, *, device: str) -> np.ndarray:
    model.eval()
    result = np.empty(len(features), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(features), 65_536):
            end = min(start + 65_536, len(features))
            values = torch.from_numpy(np.asarray(features[start:end], dtype=np.float32)).to(device)
            result[start:end] = model(values).detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise FloatingPointError("ranking scores are non-finite")
    return result


def _top_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    positives = np.flatnonzero(labels[order] == 1)
    if len(positives) == 0:
        raise ValueError("event has no positive label")
    rank = int(positives[0]) + 1
    return {"top1": float(rank == 1), "mrr": 1.0 / float(rank), "true_candidate_rank": float(rank)}


def _event_is_split(
    labels: np.ndarray,
    target_base: np.ndarray,
    *,
    split: str,
) -> bool:
    positives = np.flatnonzero(labels == 1)
    if not len(positives):
        return False
    positive_base = np.asarray(target_base[positives], dtype=bool)
    if not np.all(positive_base == positive_base[0]):
        return False
    if split == "Overall":
        return True
    return bool(positive_base[0]) == (split == "Base")


def _evaluate_method(
    scores: np.ndarray,
    raw_scores: np.ndarray,
    labels: np.ndarray,
    target_base: np.ndarray,
    offsets: np.ndarray,
    *,
    split: str,
    group_filter: set[int] | None,
) -> dict[str, Any]:
    values: list[dict[str, float]] = []
    corrections = 0
    regressions = 0
    for group, (start_value, end_value) in enumerate(zip(offsets[:-1], offsets[1:])):
        if group_filter is not None and group not in group_filter:
            continue
        start, end = int(start_value), int(end_value)
        row_labels = np.asarray(labels[start:end])
        valid = np.isin(row_labels, (0, 1))
        local = np.flatnonzero(valid)
        if not len(local) or not _event_is_split(row_labels[local], np.asarray(target_base[start:end])[local], split=split):
            continue
        local_labels = row_labels[local]
        if not np.any(local_labels == 1) or not np.any(local_labels == 0):
            continue
        method_values = _top_metrics(scores[start:end][local], local_labels)
        raw_values = _top_metrics(raw_scores[start:end][local], local_labels)
        if raw_values["top1"] == 0.0 and method_values["top1"] == 1.0:
            corrections += 1
        elif raw_values["top1"] == 1.0 and method_values["top1"] == 0.0:
            regressions += 1
        values.append(method_values)
    if not values:
        return {"status": "NO_ELIGIBLE_EVENTS", "events": 0}
    result: dict[str, Any] = {"status": "PASS", "events": len(values)}
    for key in values[0]:
        array = np.asarray([item[key] for item in values], dtype=np.float64)
        result[key] = float(array.mean())
        if key == "true_candidate_rank":
            result["true_candidate_rank_median"] = float(np.median(array))
    result.update(
        {
            "raw_wrong_to_method_correct": int(corrections),
            "raw_correct_to_method_wrong": int(regressions),
            "net_correction": int(corrections - regressions),
            "correction_rate": float(corrections / len(values)),
            "regression_rate": float(regressions / len(values)),
        }
    )
    return result


def _fixed_indices(beta_index: int) -> np.ndarray:
    return np.asarray(list(range(QDIC_RAW_DIM)) + [33 + beta_index, 41 + beta_index], dtype=np.int64)


def _predict_card(
    card: str,
    card_root: Path,
    features: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    artifact = load_qdic_mgf_exploration_checkpoint(card_root / "best.pt", device=device)
    if artifact.receipt.get("card_type") == "fixed":
        beta_index = int(artifact.receipt["beta_index"])
        values = features[:, _fixed_indices(beta_index)]
    else:
        values = features
    scores = _predict(artifact.model, values, device=device)
    return scores, artifact.provenance


def evaluate(
    *,
    split_inputs: Mapping[str, tuple[Path, Path]],
    train_features: Path,
    card_root: Path,
    b0_checkpoint: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    loaded: dict[str, tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray]] = {}
    for split, (feature_root, event_root) in split_inputs.items():
        metadata, arrays = _load_feature_cache(feature_root)
        raw_max, history = _aligned_raw_history_values(event_root, len(arrays["features"]))
        loaded[split] = (metadata, arrays, raw_max, history)

    train_meta, train_arrays, _train_max, _train_history = loaded["TRAIN_HOLDOUT"]
    train_groups, holdout_groups = build_official_train_groups(train_arrays)
    holdout_set = set(holdout_groups)
    b0_model, b0_provenance = _manual_load_b0(b0_checkpoint.resolve(), device)

    scores_by_split: dict[str, dict[str, np.ndarray]] = {}
    card_provenance: dict[str, Any] = {"B0": b0_provenance}
    for split, (metadata, arrays, raw_max, _history) in loaded.items():
        features = arrays["features"]
        scores: dict[str, np.ndarray] = {
            "Q1_SUPPORT": np.asarray(features[:, 5], dtype=np.float32),
            "MEAN": np.asarray(features[:, 29], dtype=np.float32),
            "MAX": raw_max,
            "MO2": np.asarray(features[:, 32], dtype=np.float32),
        }
        for index, beta in enumerate(QDIC_MGF_EXPLORATION_BETAS):
            sign = "M" if beta < 0 else "P"
            value = _fixed_name(beta)
            scores[value] = np.asarray(features[:, 41 + index], dtype=np.float32)
        scores["B0"] = _predict(b0_model, np.asarray(features[:, :QDIC_RAW_DIM], dtype=np.float32), device=device)
        for card_index in range(1, 19):
            card = f"E{card_index:02d}"
            scores[card], card_provenance[card] = _predict_card(
                card, card_root / card, features, device=device
            )
        scores_by_split[split] = scores

    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for split_key, (metadata, arrays, _raw_max, _history) in loaded.items():
        labels = np.asarray(arrays["labels"], dtype=np.int8)
        target_base = np.asarray(arrays["target_base"], dtype=bool)
        offsets = np.asarray(arrays["offsets"], dtype=np.int64)
        group_filter = holdout_set if split_key == "TRAIN_HOLDOUT" else None
        metrics[split_key] = {}
        for method in METHOD_ORDER:
            score_name = method
            if method.startswith("EXACT_LMGF_"):
                score_name = method
            metrics[split_key][method] = {}
            for split in ("Overall", "Base", "Novel"):
                metrics[split_key][method][split] = _evaluate_method(
                    scores_by_split[split_key][score_name],
                    scores_by_split[split_key]["Q1_SUPPORT"],
                    labels,
                    target_base,
                    offsets,
                    split=split,
                    group_filter=group_filter,
                )

    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        row: dict[str, Any] = {"method": method, "paper_status": "TEST_TUNED_EXPLORATION", "paper_valid": False}
        for split_key in SPLITS:
            for class_split in ("Overall", "Base", "Novel"):
                values = metrics[split_key][method][class_split]
                prefix = f"{split_key.lower()}_{class_split.lower()}"
                for key in ("events", "top1", "mrr", "net_correction", "raw_wrong_to_method_correct", "raw_correct_to_method_wrong"):
                    row[f"{prefix}_{key}"] = values.get(key)
        row["test_objective"] = _objective_tuple(metrics["TEST"][method])
        rows.append(row)
    ranked = sorted(rows, key=lambda row: tuple(-float(value) for value in row["test_objective"]))
    for rank, row in enumerate(ranked, start=1):
        row["test_rank"] = rank
        row["test_objective"] = ",".join(f"{float(value):.9g}" for value in row["test_objective"])
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "ranking_leaderboard_test_tuned.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in ranked for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranked)
    exploration_ranked = [
        row["method"] for row in ranked if row["method"].startswith("E")
    ]
    result = {
        "status": "PASS",
        "artifact": "qdic_v12_mgf_test_tuned_exploration_ranking",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "test_used_for_selection": True,
        "selection_objective": ["Novel Top1", "Novel MRR", "Overall Top1", "Overall MRR", "net correction", "Base Top1"],
        "method_order": list(METHOD_ORDER),
        "metrics": metrics,
        "ranked_methods": [row["method"] for row in ranked],
        # Full-Test replay is reserved for trained exploratory cards.  The
        # analytic methods and B0/Q1 rows remain diagnostic comparators and
        # must not consume the four replay slots.
        "top4": exploration_ranked[:4],
        "exploration_ranked_methods": exploration_ranked,
        "card_provenance": card_provenance,
        "feature_caches": {split: str(pair[0].resolve()) for split, pair in split_inputs.items()},
        "event_caches": {split: str(pair[1].resolve()) for split, pair in split_inputs.items()},
        "b0_checkpoint": str(b0_checkpoint.resolve()),
    }
    (output_root / "ranking_leaderboard_test_tuned.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "top4": result["top4"], "output": str(output_root)}, indent=2))
    return result


def _fixed_name(beta: float) -> str:
    labels = {-2.0: "EXACT_LMGF_BETA_M2", -1.0: "EXACT_LMGF_BETA_M1", -0.5: "EXACT_LMGF_BETA_M05", 0.25: "EXACT_LMGF_BETA_P025", 0.5: "EXACT_LMGF_BETA_P05", 1.0: "EXACT_LMGF_BETA_P1", 2.0: "EXACT_LMGF_BETA_P2", 4.0: "EXACT_LMGF_BETA_P4"}
    return labels[float(beta)]


def _objective_tuple(metrics: Mapping[str, Mapping[str, Any]]) -> tuple[float, float, float, float, float, float]:
    novel = metrics["Novel"]
    overall = metrics["Overall"]
    base = metrics["Base"]
    return (
        float(novel.get("top1", -np.inf)),
        float(novel.get("mrr", -np.inf)),
        float(overall.get("top1", -np.inf)),
        float(overall.get("mrr", -np.inf)),
        float(overall.get("net_correction", -np.inf)),
        float(base.get("top1", -np.inf)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--train-events", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--val-events", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--test-events", type=Path, required=True)
    parser.add_argument("--card-root", type=Path, required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(
        split_inputs={
            "TRAIN_HOLDOUT": (args.train_features, args.train_events),
            "VAL": (args.val_features, args.val_events),
            "TEST": (args.test_features, args.test_events),
        },
        train_features=args.train_features,
        card_root=args.card_root,
        b0_checkpoint=args.b0_checkpoint,
        output_root=args.output_root,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
