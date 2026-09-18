"""Official-Train-only trainer for TEST_TUNED_EXPLORATION cards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .qdic_features import (
    QDIC_FEATURE_NAMES,
    QDIC_MGF_EXPLORATION_BETAS,
    QDIC_MGF_EXPLORATION_FEATURE_NAMES,
    QDIC_MGF_EXPLORATION_FEATURE_SCHEMA_VERSION,
    QDIC_MGF_EXPLORATION_RAW_DIM,
    QDIC_MGF_FEATURE_NAMES,
    QDIC_MGF_FEATURE_SCHEMA_VERSION,
    QDIC_MGF_RAW_DIM,
    QDIC_RAW_DIM,
)
from .query_conditioned_reranker import group_ranking_loss
from .query_mgf_calibrator import QueryMomentGeneratingCalibrator
from .query_mgf_exploration import QueryAdaptiveMGFCalibrator


EXPLORATION_PROTOCOL = "QDIC_V12_MGF_TEST_TUNED_EXPLORATION"
EXPLORATION_STATUS = "QDIC_V12_MGF_EXPLORATION_MODEL_CODE_AND_WEIGHTS"
MIXED_CANDIDATE_K = (8, 16, 32, 64)
CHECKPOINT_COMPARATOR = "(final_mrr, final_top1, net_correction, -final_listwise_loss)"
_SOURCE_FILES = (
    "tempotrack_v10/qdic_features.py",
    "tempotrack_v10/query_mgf_calibrator.py",
    "tempotrack_v10/query_mgf_exploration.py",
    "tempotrack_v10/qdic_mgf_exploration_trainer.py",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def current_exploration_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {str(root / relative): _sha256(root / relative) for relative in _SOURCE_FILES}


def _load_arrays(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = root / "features.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact") != "qdic_v12_mgf_exploration_feature_cache":
        raise ValueError("exploration trainer requires the 49-D exploration cache")
    if int(metadata.get("schema_version", -1)) != QDIC_MGF_EXPLORATION_FEATURE_SCHEMA_VERSION:
        raise ValueError("exploration cache schema mismatch")
    if int(metadata.get("feature_dim", -1)) != QDIC_MGF_EXPLORATION_RAW_DIM:
        raise ValueError("exploration cache feature dimension mismatch")
    if tuple(metadata.get("feature_names", ())) != tuple(QDIC_MGF_EXPLORATION_FEATURE_NAMES):
        raise ValueError("exploration cache feature names mismatch")
    if metadata.get("paper_status") != "TEST_TUNED_EXPLORATION" or metadata.get("paper_valid") is not False:
        raise ValueError("exploration cache must be explicitly diagnostic-only")
    if metadata.get("diagnostic_only") is not True:
        raise ValueError("exploration cache diagnostic_only guard missing")
    if metadata.get("novel_gt_used_for_optimizer") is not False or metadata.get("test_gt_used_for_optimizer") is not False:
        raise ValueError("Novel/Test labels cannot enter optimizer")
    if metadata.get("source_role") != "OFFICIAL_TRAIN" or metadata.get("exact_split_name") != "train":
        raise ValueError("exploration optimizer must use Official Train only")
    if metadata.get("optimizer_source_allowed") is not True:
        raise ValueError("Official Train exploration cache is not optimizer-allowed")
    if metadata.get("base_only_supervision") is not True or metadata.get("gt_used_only_for_supervision") is not True:
        raise ValueError("exploration cache GT guards are invalid")
    paths = dict(metadata.get("arrays", {}))
    required = {"features", "labels", "supervision_allowed", "offsets", "videos", "candidate_base", "target_base", "group_ids"}
    if not required.issubset(paths):
        raise ValueError(f"exploration cache missing arrays: {sorted(required - set(paths))}")
    hashes = dict(metadata.get("array_hashes", {}))
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        path = Path(str(paths[name]))
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or hashes.get(name) != _sha256(path):
            raise ValueError(f"exploration cache array hash mismatch: {path}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    count = len(arrays["features"])
    if arrays["features"].ndim != 2 or arrays["features"].shape[1] != QDIC_MGF_EXPLORATION_RAW_DIM:
        raise ValueError("exploration features must be [N,49]")
    for name in ("labels", "supervision_allowed", "candidate_base", "target_base", "group_ids"):
        if len(arrays[name]) != count:
            raise ValueError("exploration arrays are not row aligned")
    allowed = (
        np.asarray(arrays["supervision_allowed"], dtype=bool)
        & np.asarray(arrays["candidate_base"], dtype=bool)
        & np.asarray(arrays["target_base"], dtype=bool)
        & np.isin(np.asarray(arrays["labels"]), (0, 1))
    )
    if not np.array_equal(allowed, np.asarray(arrays["supervision_allowed"], dtype=bool)):
        raise ValueError("exploration supervision_allowed is inconsistent")
    if len(arrays["offsets"]) != len(arrays["videos"]) + 1:
        raise ValueError("exploration offsets/videos are inconsistent")
    if int(arrays["offsets"][0]) != 0 or int(arrays["offsets"][-1]) != count:
        raise ValueError("exploration offsets do not cover features")
    return metadata, arrays


def build_official_train_groups(
    arrays: Mapping[str, np.ndarray],
    *,
    holdout_modulus: int = 5,
    holdout_bucket: int = 0,
) -> tuple[list[int], list[int]]:
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    videos = np.asarray(arrays["videos"], dtype=np.int64)
    labels = np.asarray(arrays["labels"])
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool)
    train_groups: list[int] = []
    holdout_groups: list[int] = []
    for group, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        row_mask = allowed[int(start):int(end)]
        row_labels = labels[int(start):int(end)]
        if not bool((row_mask & (row_labels == 1)).any()) or not bool((row_mask & (row_labels == 0)).any()):
            continue
        bucket = int(hashlib.sha256(str(int(videos[group])).encode("utf-8")).hexdigest()[:8], 16) % int(holdout_modulus)
        (holdout_groups if bucket == holdout_bucket else train_groups).append(group)
    if not train_groups or not holdout_groups:
        raise ValueError("exploration cache needs nonempty train/holdout groups")
    train_videos = {int(videos[index]) for index in train_groups}
    holdout_videos = {int(videos[index]) for index in holdout_groups}
    if train_videos & holdout_videos:
        raise AssertionError("video-disjoint split invariant failed")
    return train_groups, holdout_groups


def _feature_indices(*, card_type: str, beta_index: int | None) -> np.ndarray | None:
    if card_type == "adaptive":
        return None
    if card_type != "fixed" or beta_index is None or not (0 <= int(beta_index) < len(QDIC_MGF_EXPLORATION_BETAS)):
        raise ValueError("fixed exploration card needs a registered beta index")
    index = int(beta_index)
    return np.asarray(list(range(QDIC_RAW_DIM)) + [QDIC_RAW_DIM + index, QDIC_RAW_DIM + 8 + index], dtype=np.int64)


def _card_features(features: np.ndarray, rows: np.ndarray, indices: np.ndarray | None) -> np.ndarray:
    selected = np.asarray(features[rows], dtype=np.float32)
    if indices is not None:
        selected = selected[:, indices]
    return selected


def _select_rows(
    start: int,
    end: int,
    features: np.ndarray,
    labels: np.ndarray,
    allowed: np.ndarray,
    *,
    candidate_k: int,
    feature_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(allowed[start:end])
    if len(eligible) < 2:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int64)
    selected = eligible[: int(candidate_k)]
    if not bool(np.any(labels[start + selected] == 1)):
        positive = eligible[labels[start + eligible] == 1]
        selected = np.concatenate((selected[:-1], positive[:1]))
    if not bool(np.any(labels[start + selected] == 0)):
        negative = eligible[labels[start + eligible] == 0]
        selected = np.concatenate((selected[:-1], negative[:1]))
    selected = np.unique(selected)
    rows = np.asarray(start + selected, dtype=np.int64)
    return _card_features(features, rows, feature_indices), np.asarray(labels[rows], dtype=np.int64)


def _batch(
    groups: Iterable[int],
    arrays: Mapping[str, np.ndarray],
    *,
    candidate_k: int,
    feature_indices: np.ndarray | None,
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = arrays["features"]
    labels = np.asarray(arrays["labels"])
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    examples: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for group in groups:
        start, end = offsets[group:group + 2]
        value, target = _select_rows(int(start), int(end), features, labels, allowed, candidate_k=candidate_k, feature_indices=feature_indices)
        if len(target) and np.any(target == 1) and np.any(target == 0):
            examples.append(value)
            targets.append(target)
    if not examples:
        raise ValueError("no eligible Official Train Base groups")
    width = max(len(item) for item in targets)
    batch_features = np.zeros((len(examples), width, feature_dim), dtype=np.float32)
    batch_labels = np.full((len(examples), width), -1, dtype=np.int64)
    for index, (value, target) in enumerate(zip(examples, targets)):
        batch_features[index, :len(target)] = value
        batch_labels[index, :len(target)] = target
    return torch.from_numpy(batch_features), torch.from_numpy(batch_labels)


def _metrics(
    model: torch.nn.Module,
    groups: Iterable[int],
    arrays: Mapping[str, np.ndarray],
    *,
    device: str,
    feature_indices: np.ndarray | None,
    feature_dim: int,
) -> dict[str, float]:
    model.eval()
    features = arrays["features"]
    labels = np.asarray(arrays["labels"])
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    final_mrr: list[float] = []
    final_top1: list[float] = []
    raw_top1: list[float] = []
    losses: list[float] = []
    with torch.inference_mode():
        for group in groups:
            start, end = offsets[group:group + 2]
            value, target = _select_rows(int(start), int(end), features, labels, allowed, candidate_k=64, feature_indices=feature_indices)
            if len(target) == 0 or not np.any(target == 1) or not np.any(target == 0):
                continue
            logits = model(torch.from_numpy(value).to(device))
            losses.append(float(group_ranking_loss(logits.unsqueeze(0), torch.from_numpy(target).to(device).unsqueeze(0))))
            order = torch.argsort(logits, descending=True).cpu().numpy()
            positive_positions = np.flatnonzero(target[order] == 1)
            if len(positive_positions):
                final_mrr.append(1.0 / float(positive_positions[0] + 1))
            final_top1.append(float(target[order[0]] == 1))
            raw_order = np.argsort(-value[:, 5], kind="stable")
            raw_top1.append(float(target[raw_order[0]] == 1))
    top1 = float(np.mean(final_top1)) if final_top1 else 0.0
    raw = float(np.mean(raw_top1)) if raw_top1 else 0.0
    return {
        "final_mrr": float(np.mean(final_mrr)) if final_mrr else 0.0,
        "final_top1": top1,
        "raw_top1": raw,
        "net_correction": top1 - raw,
        "final_listwise_loss": float(np.mean(losses)) if losses else float("inf"),
        "events": float(len(final_top1)),
    }


def _comparator(metrics: Mapping[str, float]) -> tuple[float, float, float, float]:
    return (float(metrics["final_mrr"]), float(metrics["final_top1"]), float(metrics["net_correction"]), -float(metrics["final_listwise_loss"]))


def _normalization(
    groups: Iterable[int],
    arrays: Mapping[str, np.ndarray],
    *,
    feature_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    features = arrays["features"]
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    values: list[np.ndarray] = []
    for group in groups:
        start, end = offsets[group:group + 2]
        rows = np.flatnonzero(allowed[start:end]).astype(np.int64) + int(start)
        if len(rows):
            values.append(_card_features(features, rows, feature_indices))
    if not values:
        raise ValueError("no Base rows for exploration normalization")
    combined = np.concatenate(values, axis=0).astype(np.float64)
    mean = combined.mean(axis=0)
    scale = np.sqrt(np.maximum(combined.var(axis=0), 1e-6))
    return mean.astype(np.float32), scale.astype(np.float32), len(combined)


def train_exploration_card(
    features_dir: str | Path,
    output: str | Path,
    *,
    card_id: str,
    card_type: str,
    mgf_mode: str,
    beta_index: int | None = None,
    epochs: int = 12,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    if card_type not in {"fixed", "adaptive"}:
        raise ValueError("card_type must be fixed or adaptive")
    if mgf_mode not in {"core", "fused"}:
        raise ValueError("mgf_mode must be core or fused")
    root = Path(features_dir).resolve()
    out = Path(output).resolve()
    metadata, arrays = _load_arrays(root)
    feature_indices = _feature_indices(card_type=card_type, beta_index=beta_index)
    feature_dim = QDIC_MGF_RAW_DIM if card_type == "fixed" else QDIC_MGF_EXPLORATION_RAW_DIM
    train_groups, holdout_groups = build_official_train_groups(arrays)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))
    if card_type == "fixed":
        model: torch.nn.Module = QueryMomentGeneratingCalibrator(mgf_mode=mgf_mode).to(device)
    else:
        model = QueryAdaptiveMGFCalibrator(mgf_mode=mgf_mode).to(device)
    mean, scale, normalization_rows = _normalization(train_groups, arrays, feature_indices=feature_indices)
    model.set_normalization(mean, scale)  # type: ignore[attr-defined]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    source_hashes = current_exploration_source_hashes()
    best_key: tuple[float, float, float, float] | None = None
    best_epoch = 0
    optimizer_steps = 0
    history: list[dict[str, Any]] = []
    with (out / "metrics.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(int(epochs)):
            model.train()
            shuffled = rng.permutation(train_groups)
            epoch_losses: list[float] = []
            for batch_start in range(0, len(shuffled), 32):
                group_batch = shuffled[batch_start:batch_start + 32]
                candidate_k = MIXED_CANDIDATE_K[(epoch + batch_start // 32) % len(MIXED_CANDIDATE_K)]
                batch_features, batch_labels = _batch(group_batch, arrays, candidate_k=candidate_k, feature_indices=feature_indices, feature_dim=feature_dim)
                diagnostics = model(batch_features.to(device), return_diagnostics=True)
                loss = group_ranking_loss(diagnostics["logit"], batch_labels.to(device))  # type: ignore[index]
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite exploration loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise FloatingPointError("non-finite exploration gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
                epoch_losses.append(float(loss.detach()))
            validation = _metrics(model, holdout_groups, arrays, device=device, feature_indices=feature_indices, feature_dim=feature_dim)
            row = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("inf"), **validation, "candidate_k_schedule": list(MIXED_CANDIDATE_K)}
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            key = _comparator(validation)
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch + 1
                state = {
                    "status": EXPLORATION_STATUS,
                    "model_state": model.state_dict(),
                    "card_id": str(card_id),
                    "card_type": card_type,
                    "mgf_mode": mgf_mode,
                    "mgf_beta": None if card_type == "adaptive" else float(QDIC_MGF_EXPLORATION_BETAS[int(beta_index)]),
                    "mgf_betas": list(QDIC_MGF_EXPLORATION_BETAS),
                    "beta_index": beta_index,
                    "schema_version": QDIC_MGF_FEATURE_SCHEMA_VERSION if card_type == "fixed" else QDIC_MGF_EXPLORATION_FEATURE_SCHEMA_VERSION,
                    "feature_names": list(QDIC_MGF_FEATURE_NAMES if card_type == "fixed" else QDIC_MGF_EXPLORATION_FEATURE_NAMES),
                    "feature_dim": feature_dim,
                    "source_feature_schema_version": QDIC_MGF_EXPLORATION_FEATURE_SCHEMA_VERSION,
                    "training_split": "train_base_official",
                    "protocol": EXPLORATION_PROTOCOL,
                    "base_only_supervision": True,
                    "novel_gt_used": False,
                    "test_weights_used": False,
                    "paper_status": "TEST_TUNED_EXPLORATION",
                    "paper_valid": False,
                    "diagnostic_only": True,
                    "selection_scope": "VAL_TEST_TUNED_EXPLORATION",
                    "source_hashes": source_hashes,
                    "features_hash": _sha256(root / "features.json"),
                    "feature_mean": mean.tolist(),
                    "feature_scale": scale.tolist(),
                    "checkpoint_comparator": CHECKPOINT_COMPARATOR,
                    "epoch": epoch + 1,
                    "optimizer_steps": optimizer_steps,
                    "seed": int(seed),
                }
                torch.save(state, out / "best.pt")
    if not (out / "best.pt").is_file():
        raise RuntimeError("exploration card did not produce best.pt")
    videos = np.asarray(arrays["videos"], dtype=np.int64)
    result = {
        "status": "COMPLETED",
        "artifact": "qdic_v12_mgf_exploration_training",
        "card_id": str(card_id),
        "card_type": card_type,
        "mgf_mode": mgf_mode,
        "mgf_beta": None if card_type == "adaptive" else float(QDIC_MGF_EXPLORATION_BETAS[int(beta_index)]),
        "mgf_betas": list(QDIC_MGF_EXPLORATION_BETAS),
        "beta_index": beta_index,
        "schema_version": QDIC_MGF_FEATURE_SCHEMA_VERSION if card_type == "fixed" else QDIC_MGF_EXPLORATION_FEATURE_SCHEMA_VERSION,
        "feature_names": list(QDIC_MGF_FEATURE_NAMES if card_type == "fixed" else QDIC_MGF_EXPLORATION_FEATURE_NAMES),
        "feature_dim": feature_dim,
        "training_split": "train_base_official",
        "exact_split_name": "train",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "selection_scope": "VAL_TEST_TUNED_EXPLORATION",
        "test_used_for_selection": True,
        "base_only_supervision": True,
        "novel_gt_used": False,
        "test_weights_used": False,
        "features": str(root / "features.json"),
        "features_hash": _sha256(root / "features.json"),
        "train_video_ids": sorted({int(videos[index]) for index in train_groups}),
        "holdout_video_ids": sorted({int(videos[index]) for index in holdout_groups}),
        "video_disjoint": True,
        "normalization_fit": "internal_train_base_only_after_video_split",
        "normalization_row_count": int(normalization_rows),
        "normalization_mean": mean.tolist(),
        "normalization_scale": scale.tolist(),
        "checkpoint_comparator": CHECKPOINT_COMPARATOR,
        "best_epoch": int(best_epoch),
        "optimizer_steps": int(optimizer_steps),
        "checkpoint": str(out / "best.pt"),
        "checkpoint_hash": _sha256(out / "best.pt"),
        "history": history,
        "source_hashes": source_hashes,
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "variables": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        },
        "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
    }
    _write_json(out / "training.json", result)
    return result


__all__ = [
    "CHECKPOINT_COMPARATOR",
    "EXPLORATION_PROTOCOL",
    "EXPLORATION_STATUS",
    "build_official_train_groups",
    "current_exploration_source_hashes",
    "train_exploration_card",
]
