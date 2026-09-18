#!/usr/bin/env python3
"""Build one compact 49-D exploratory beta-bank feature cache.

The first 33 columns are copied byte-for-byte from the audited V11 prefix in
the formal beta=1 cache.  The 16 exploratory columns are recomputed from the
same causal cosine/memory arrays; no labels or GT fields enter the feature
calculation.  Metadata arrays remain shared by absolute path to avoid making a
second copy of the large supervision/cache arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tempotrack_v10.qdic_features import (
    QDIC_MGF_BETA,
    QDIC_MGF_EXPLORATION_BETAS,
    QDIC_MGF_EXPLORATION_FEATURE_NAMES,
    QDIC_MGF_EXPLORATION_FEATURE_SCHEMA_VERSION,
    QDIC_MGF_EXPLORATION_RAW_DIM,
    QDIC_RAW_DIM,
    empirical_log_mgf,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_array(metadata: dict[str, Any], root: Path, name: str) -> np.ndarray:
    value = Path(str(metadata["arrays"][name]))
    if not value.is_absolute():
        value = root / value
    if not value.is_file():
        raise FileNotFoundError(value)
    expected = dict(metadata.get("array_hashes", {})).get(name)
    if expected is not None and _sha256(value) != str(expected):
        raise ValueError(f"source feature array hash mismatch: {value}")
    return np.load(value, mmap_mode="r", allow_pickle=False)


def _event_array_paths(event_root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = event_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact") != "v9_1_psmr_event_cache" or int(metadata.get("schema_version", -1)) != 10:
        raise ValueError("exploration cache requires the audited V9.1 event cache")
    arrays: dict[str, np.ndarray] = {}
    for name in ("cosine", "mem_len", "prefilter_rank_b1"):
        value = Path(str(metadata["arrays"][name]))
        if not value.is_absolute():
            value = event_root / value
        expected = dict(metadata.get("array_hashes", {})).get(name)
        if not value.is_file() or expected is not None and _sha256(value) != str(expected):
            raise ValueError(f"event array missing/hash mismatch: {value}")
        arrays[name] = np.load(value, mmap_mode="r", allow_pickle=False)
    return metadata, arrays


def _chunk_beta_bank(cosine: np.ndarray, mem_len: np.ndarray) -> np.ndarray:
    values = np.asarray(cosine, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 1:
        raise ValueError(f"cosine must be [N,Q,L], got {values.shape}")
    values = values[:, 0, :]
    lengths = np.asarray(mem_len, dtype=np.int64).reshape(-1)
    if len(lengths) != len(values) or np.any(lengths < 1) or np.any(lengths > values.shape[1]):
        raise ValueError("invalid event memory lengths")
    positions = np.arange(values.shape[1], dtype=np.int64)[None, :]
    slow_mask = positions < lengths[:, None]
    fast_start = np.maximum(lengths - 8, 0)
    fast_mask = slow_mask & (positions >= fast_start[:, None])

    def reduce_log_mgf(mask: np.ndarray, beta: float) -> np.ndarray:
        scaled = float(beta) * values
        masked = np.where(mask, scaled, -np.inf)
        maximum = np.max(masked, axis=1)
        shifted = np.where(mask, np.exp(masked - maximum[:, None]), 0.0)
        count = mask.sum(axis=1, dtype=np.float64)
        return (maximum + np.log(shifted.sum(axis=1) / count)) / float(beta)

    fast = np.stack([reduce_log_mgf(fast_mask, beta) for beta in QDIC_MGF_EXPLORATION_BETAS], axis=1)
    slow = np.stack([reduce_log_mgf(slow_mask, beta) for beta in QDIC_MGF_EXPLORATION_BETAS], axis=1)
    result = np.concatenate((fast, slow), axis=1).astype(np.float32, copy=False)
    if result.shape != (len(values), 16) or not np.isfinite(result).all():
        raise FloatingPointError("non-finite beta-bank values")
    return result


def build_exploration_cache(
    source_features: str | Path,
    source_event_cache: str | Path,
    output: str | Path,
    *,
    chunk_rows: int = 50000,
) -> dict[str, Any]:
    source_root = Path(source_features).resolve()
    event_root = Path(source_event_cache).resolve()
    output_root = Path(output).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite exploratory cache: {output_root}")
    source_metadata = json.loads((source_root / "features.json").read_text(encoding="utf-8"))
    if source_metadata.get("artifact") not in {"qdic_v12_mgf_feature_cache", "qdic_v11_feature_cache"}:
        raise ValueError("source must be an audited V11/V12 causal feature cache")
    source_feature_dim = int(source_metadata.get("feature_dim", -1))
    if source_feature_dim not in {33, 35}:
        raise ValueError("source feature cache must be 33-D or formal 35-D")
    source_features_array = _load_array(source_metadata, source_root, "features")
    if source_features_array.ndim != 2 or source_features_array.shape[1] != source_feature_dim:
        raise ValueError(f"source features must be [N,{source_feature_dim}]")
    event_metadata, event_arrays = _event_array_paths(event_root)
    event_count = len(event_arrays["mem_len"])
    if event_count != len(source_features_array):
        raise ValueError(f"event/source row mismatch: {event_count} vs {len(source_features_array)}")

    output_root.mkdir(parents=True, exist_ok=False)
    feature_path = output_root / "features.npy"
    feature_map = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(event_count, QDIC_MGF_EXPLORATION_RAW_DIM),
    )
    max_beta_one_error = 0.0
    chunk = max(int(chunk_rows), 1)
    for start in range(0, event_count, chunk):
        end = min(start + chunk, event_count)
        bank = _chunk_beta_bank(
            event_arrays["cosine"][start:end],
            event_arrays["mem_len"][start:end],
        )
        prefix = np.asarray(source_features_array[start:end, :QDIC_RAW_DIM], dtype=np.float32)
        feature_map[start:end, :QDIC_RAW_DIM] = prefix
        feature_map[start:end, QDIC_RAW_DIM:] = bank
        if source_feature_dim == 35:
            formal_beta_one = np.asarray(source_features_array[start:end, 33:35], dtype=np.float32)
            max_beta_one_error = max(
                max_beta_one_error,
                float(np.max(np.abs(formal_beta_one - bank[:, [5, 13]]))),
            )
    feature_map.flush()
    del feature_map
    if max_beta_one_error > 2e-5:
        raise AssertionError(f"beta=1 parity failed: max_abs={max_beta_one_error}")

    array_paths: dict[str, str] = {"features": str(feature_path)}
    array_hashes: dict[str, str] = {"features": _sha256(feature_path)}
    for name, value in dict(source_metadata["arrays"]).items():
        if name == "features":
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = source_root / path
        array_paths[name] = str(path.resolve())
        expected = dict(source_metadata.get("array_hashes", {})).get(name)
        array_hashes[name] = str(expected) if expected else _sha256(path)

    feature_config = dict(source_metadata.get("feature_config", {}))
    feature_config.update(
        {
            "exploration": True,
            "mgf_betas": list(QDIC_MGF_EXPLORATION_BETAS),
            "mgf_bank_layout": "fast_beta_bank_then_full_history_beta_bank",
            "mgf_beta": None,
        }
    )
    metadata = {
        "status": "COMPLETED",
        "artifact": "qdic_v12_mgf_exploration_feature_cache",
        "schema_version": QDIC_MGF_EXPLORATION_FEATURE_SCHEMA_VERSION,
        "feature_names": list(QDIC_MGF_EXPLORATION_FEATURE_NAMES),
        "feature_dim": QDIC_MGF_EXPLORATION_RAW_DIM,
        "qdic_v11_prefix_dim": QDIC_RAW_DIM,
        "prefix_parity_max_abs": 0.0,
        "mgf_betas": list(QDIC_MGF_EXPLORATION_BETAS),
        "mgf_bank_layout": "fast_beta_bank_then_full_history_beta_bank",
        "formal_beta_one_parity_max_abs": max_beta_one_error,
        "feature_config": feature_config,
        "source_feature_cache": str((source_root / "features.json").resolve()),
        "source_feature_cache_sha256": _sha256(source_root / "features.json"),
        "source_feature_dim": source_feature_dim,
        "source_event_cache": str(event_root),
        "source_event_cache_metadata_sha256": _sha256(event_root / "metadata.json"),
        "source_frontend_cache": source_metadata.get("source_frontend_cache"),
        "source_annotation": source_metadata.get("source_annotation"),
        "source_role": source_metadata.get("source_role"),
        "exact_split_name": source_metadata.get("exact_split_name"),
        "input_source": source_metadata.get("input_source", "COVTRACK_FRONTEND"),
        "supervision_source": source_metadata.get("supervision_source"),
        "oracle_features_used": False,
        "gt_boxes_used_as_model_input": False,
        "gt_tracks_used_as_memory": False,
        "gt_used_only_for_supervision": True,
        "official_train_annotation_sha256": source_metadata.get("official_train_annotation_sha256"),
        "base_only_supervision": source_metadata.get("base_only_supervision", True),
        "novel_gt_used_for_optimizer": False,
        "test_gt_used_for_optimizer": False,
        "optimizer_source_allowed": source_metadata.get("optimizer_source_allowed", False),
        "video_disjoint_split": source_metadata.get("video_disjoint_split", False),
        "normalization_fit": source_metadata.get("normalization_fit"),
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "test_used_for_selection": True,
        "array_paths_are_shared_source_artifacts": True,
        "arrays": array_paths,
        "array_hashes": array_hashes,
        "rows": int(event_count),
        "events": int(event_count),
        "status_contract": "exploration_only; formal_v12_loader_must_reject",
    }
    (output_root / "features.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata["metadata_sha256"] = _sha256(output_root / "features.json")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--source-event-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=50000)
    args = parser.parse_args()
    result = build_exploration_cache(
        args.source_features,
        args.source_event_cache,
        args.output,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "rows": result["rows"], "beta_one_parity_max_abs": result["formal_beta_one_parity_max_abs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
