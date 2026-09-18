#!/usr/bin/env python3
"""Evaluate non-learned temporal scores, B0, and frozen B0-MGF rankings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from tempotrack_v10.qdic_features import _load_cache
from tempotrack_v10.qdic_loader import load_qdic_checkpoint
from tempotrack_v10.qdic_mgf_loader import load_qdic_mgf_checkpoint


METHOD_ORDER = ("Raw", "Mean", "Max", "MO2", "Exact-LMGF", "B0", "B0-MGF")
SPLIT_ORDER = ("Overall", "Base", "Novel")
HISTORY_BINS = (
    ("L=1", 1, 1),
    ("2-3", 2, 3),
    ("4-7", 4, 7),
    ("8", 8, 8),
    ("9-15", 9, 15),
    ("16-31", 16, 31),
    ("32+", 32, None),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_feature_cache(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((root / "features.json").read_text(encoding="utf-8"))
    if metadata.get("artifact") != "qdic_v12_mgf_feature_cache":
        raise ValueError("evaluator requires qdic_v12_mgf_feature_cache")
    if int(metadata.get("schema_version", -1)) != 12 or int(metadata.get("feature_dim", -1)) != 35:
        raise ValueError("MGF feature schema mismatch")
    paths = dict(metadata.get("arrays", {}))
    hashes = dict(metadata.get("array_hashes", {}))
    required = {"features", "labels", "target_base", "offsets", "videos", "group_ids"}
    if not required.issubset(paths):
        raise ValueError(f"MGF cache missing arrays: {sorted(required - set(paths))}")
    arrays = {}
    for name in required:
        path = Path(str(paths[name]))
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or _sha256(path) != hashes.get(name):
            raise ValueError(f"MGF array hash mismatch: {path}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    count = len(arrays["features"])
    if arrays["features"].shape != (count, 35):
        raise ValueError("MGF features must be [N,35]")
    for name in ("labels", "target_base", "group_ids"):
        if len(arrays[name]) != count:
            raise ValueError("MGF evaluation arrays are not row aligned")
    if len(arrays["offsets"]) != len(arrays["videos"]) + 1 or int(arrays["offsets"][-1]) != count:
        raise ValueError("MGF offsets/videos are inconsistent")
    return metadata, arrays


def _aligned_raw_history_values(event_cache: Path, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    _metadata, arrays, _inline, _rows = _load_cache(event_cache)
    cosine = np.asarray(arrays["cosine"])
    mem_len = np.asarray(arrays["mem_len"], dtype=np.int64).reshape(-1)
    rank = np.asarray(arrays["prefilter_rank_b1"], dtype=np.int64).reshape(-1)
    selected = rank <= 64
    if cosine.ndim != 3 or len(mem_len) != len(cosine):
        raise ValueError("raw event cache cosine/mem_len mismatch")
    max_scores: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    width = int(cosine.shape[2])
    positions = np.arange(width, dtype=np.int64)[None, :]
    for start in range(0, len(cosine), 100_000):
        end = min(start + 100_000, len(cosine))
        value = np.asarray(cosine[start:end, 0, :], dtype=np.float32)
        valid = positions < mem_len[start:end, None]
        max_scores.append(np.max(np.where(valid, value, -np.inf), axis=1)[selected[start:end]])
        lengths.append(mem_len[start:end][selected[start:end]])
    result_scores = np.concatenate(max_scores).astype(np.float32, copy=False)
    result_lengths = np.concatenate(lengths).astype(np.int64, copy=False)
    if len(result_scores) != expected_rows or len(result_lengths) != expected_rows:
        raise ValueError(
            f"raw event alignment mismatch: {len(result_scores)} vs MGF rows {expected_rows}"
        )
    return result_scores, result_lengths


def _predict(model: Any, features: np.ndarray, *, device: str, width: int) -> np.ndarray:
    model.eval()
    result = np.empty(len(features), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(features), 65_536):
            end = min(start + 65_536, len(features))
            values = torch.from_numpy(np.asarray(features[start:end, :width], dtype=np.float32)).to(device)
            result[start:end] = model(values).detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise FloatingPointError("model ranking scores are non-finite")
    return result


def _top_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    positives = np.flatnonzero(labels[order] == 1)
    if len(positives) == 0:
        raise ValueError("event has no positive label")
    rank = int(positives[0]) + 1
    return {
        "top1": float(rank == 1),
        "mrr": 1.0 / float(rank),
        "true_candidate_rank": float(rank),
        "r_at_1": float(np.any(labels[order[:1]] == 1)),
        "r_at_8": float(np.any(labels[order[:8]] == 1)),
        "r_at_16": float(np.any(labels[order[:16]] == 1)),
        "r_at_32": float(np.any(labels[order[:32]] == 1)),
        "r_at_64": float(np.any(labels[order[:64]] == 1)),
    }


def _evaluate_method(
    scores: np.ndarray,
    raw_scores: np.ndarray,
    labels: np.ndarray,
    target_base: np.ndarray,
    offsets: np.ndarray,
    *,
    split: str,
) -> dict[str, Any]:
    values: list[dict[str, float]] = []
    corrections = 0
    regressions = 0
    event_ids: list[int] = []
    for group, (start_value, end_value) in enumerate(zip(offsets[:-1], offsets[1:])):
        start, end = int(start_value), int(end_value)
        row_labels = np.asarray(labels[start:end])
        valid = np.isin(row_labels, (0, 1))
        if split == "Base" and not bool(np.all(np.asarray(target_base[start:end], dtype=bool))):
            continue
        if split == "Novel" and not bool(np.all(~np.asarray(target_base[start:end], dtype=bool))):
            continue
        if split == "Overall" and not bool(np.all(np.isin(target_base[start:end], (False, True)))):
            continue
        if not bool(np.any(valid & (row_labels == 1))) or not bool(np.any(valid & (row_labels == 0))):
            continue
        local = np.flatnonzero(valid)
        local_labels = row_labels[local]
        method_values = _top_metrics(scores[start:end][local], local_labels)
        raw_values = _top_metrics(raw_scores[start:end][local], local_labels)
        method_pred = int(np.argmax(scores[start:end][local]))
        raw_pred = int(np.argmax(raw_scores[start:end][local]))
        if raw_values["top1"] == 0.0 and method_values["top1"] == 1.0:
            corrections += 1
        elif raw_values["top1"] == 1.0 and method_values["top1"] == 0.0:
            regressions += 1
        values.append(method_values)
        event_ids.append(group)
    if not values:
        return {"split": split, "events": 0, "status": "NO_ELIGIBLE_EVENTS"}
    keys = tuple(values[0])
    result: dict[str, Any] = {"split": split, "events": len(values), "status": "PASS"}
    for key in keys:
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
            "event_ids": event_ids,
        }
    )
    return result


def _history_breakdown(
    method_scores: np.ndarray,
    b0_scores: np.ndarray,
    labels: np.ndarray,
    target_base: np.ndarray,
    offsets: np.ndarray,
    history_lengths: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, low, high in HISTORY_BINS:
        b0_values: list[float] = []
        mgf_values: list[float] = []
        for start_value, end_value in zip(offsets[:-1], offsets[1:]):
            start, end = int(start_value), int(end_value)
            candidate_lengths = history_lengths[start:end]
            positive = np.flatnonzero(np.asarray(labels[start:end]) == 1)
            if not len(positive):
                continue
            length = int(candidate_lengths[int(positive[0])])
            if length < low or (high is not None and length > high):
                continue
            b0_values.append(_top_metrics(b0_scores[start:end], labels[start:end])["top1"])
            mgf_values.append(_top_metrics(method_scores[start:end], labels[start:end])["top1"])
        rows.append(
            {
                "history_bin": name,
                "n": len(b0_values),
                "b0_top1": None if not b0_values else float(np.mean(b0_values)),
                "mgf_top1": None if not mgf_values else float(np.mean(mgf_values)),
                "delta": None if not mgf_values else float(np.mean(mgf_values) - np.mean(b0_values)),
            }
        )
    return rows


def evaluate(
    *,
    feature_cache: Path,
    event_cache: Path,
    b0_checkpoint: Path,
    mgf_checkpoint: Path,
    output_root: Path,
    split_name: str,
    device: str = "cpu",
) -> dict[str, Any]:
    metadata, arrays = _load_feature_cache(feature_cache)
    features = arrays["features"]
    labels = np.asarray(arrays["labels"], dtype=np.int8)
    target_base = np.asarray(arrays["target_base"], dtype=bool)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    raw_max, history_lengths = _aligned_raw_history_values(event_cache, len(features))
    raw_local = np.asarray(features[:, 5], dtype=np.float32)
    scores: dict[str, np.ndarray] = {
        "Raw": raw_local,
        "Mean": np.asarray(features[:, 29], dtype=np.float32),
        "Max": raw_max,
        "MO2": np.asarray(features[:, 32], dtype=np.float32),
        "Exact-LMGF": np.asarray(features[:, 34], dtype=np.float32),
    }
    b0 = load_qdic_checkpoint(b0_checkpoint, device=device)
    mgf = load_qdic_mgf_checkpoint(mgf_checkpoint, device=device)
    scores["B0"] = _predict(b0.model, features, device=device, width=33)
    scores["B0-MGF"] = _predict(mgf.model, features, device=device, width=35)
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for method in METHOD_ORDER:
        metrics[method] = {}
        for split in SPLIT_ORDER:
            metrics[method][split] = _evaluate_method(
                scores[method], raw_local, labels, target_base, offsets, split=split
            )
    selected_method = scores["B0-MGF"]
    b0_method = scores["B0"]
    higher_order = np.asarray(scores["Exact-LMGF"] - scores["MO2"], dtype=np.float32)
    corrected_gain: list[float] = []
    uncorrected_gain: list[float] = []
    for start_value, end_value in zip(offsets[:-1], offsets[1:]):
        start, end = int(start_value), int(end_value)
        positive = np.flatnonzero(labels[start:end] == 1)
        if not len(positive):
            continue
        pos = start + int(positive[0])
        b0_top = int(np.argmax(b0_method[start:end]))
        mgf_top = int(np.argmax(selected_method[start:end]))
        gain = float(higher_order[pos])
        if labels[start + b0_top] == 0 and labels[start + mgf_top] == 1:
            corrected_gain.append(gain)
        else:
            uncorrected_gain.append(gain)
    result = {
        "status": "PASS",
        "artifact": "qdic_v12_mgf_vs_b0_ranking_evaluation",
        "split": split_name,
        "feature_cache": str(feature_cache.resolve()),
        "feature_cache_metadata_sha256": _sha256(feature_cache / "features.json"),
        "event_cache": str(event_cache.resolve()),
        "b0_checkpoint": str(b0_checkpoint.resolve()),
        "b0_checkpoint_sha256": _sha256(b0_checkpoint),
        "mgf_checkpoint": str(mgf_checkpoint.resolve()),
        "mgf_checkpoint_sha256": _sha256(mgf_checkpoint),
        "mgf_beta": 1.0,
        "mapping_source": metadata.get("source_annotation"),
        "target_base_source": "official cache target_base cross-checked against source annotation metadata",
        "methods": metrics,
        "history_length_breakdown": {
            "B0_vs_B0-MGF": _history_breakdown(
                scores["B0-MGF"], scores["B0"], labels, target_base, offsets, history_lengths
            )
        },
        "higher_order_gain": {
            "definition": "Exact-LMGF - MO2 on the first positive candidate in each event",
            "corrected_by_B0_MGF_n": len(corrected_gain),
            "corrected_by_B0_MGF_mean": None if not corrected_gain else float(np.mean(corrected_gain)),
            "uncorrected_n": len(uncorrected_gain),
            "uncorrected_mean": None if not uncorrected_gain else float(np.mean(uncorrected_gain)),
        },
        "provenance": {
            "b0": b0.provenance,
            "mgf": mgf.provenance,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{split_name.lower()}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    rows = []
    for method in METHOD_ORDER:
        for split in SPLIT_ORDER:
            row = {"method": method, **metrics[method][split]}
            row.pop("event_ids", None)
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (output_root / f"{split_name.lower()}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": result["status"], "split": split_name, "output": str(output_root)}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--event-cache", type=Path, required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--mgf-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--device", default="cpu")
    main_args = parser.parse_args()
    evaluate(
        feature_cache=main_args.feature_cache,
        event_cache=main_args.event_cache,
        b0_checkpoint=main_args.b0_checkpoint,
        mgf_checkpoint=main_args.mgf_checkpoint,
        output_root=main_args.output_root,
        split_name=main_args.split_name,
        device=main_args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
