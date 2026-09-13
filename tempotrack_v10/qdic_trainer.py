"""Base-only CPU trainer for the QDIC V11 candidate calibrator."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .qdic_features import QDIC_FEATURE_NAMES, QDIC_RAW_DIM
from .query_conditioned_reranker import group_ranking_loss
from .query_distributional_calibrator import QueryDistributionalCalibrator
from .qdic_loader import QDIC_STATUS, validate_qdic_feature_config


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _memory_guard(spare_gib: int = 12) -> None:
    """Keep the same conservative reserve used by the V9 trainer."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return
    values = dict(line.split(":", 1) for line in meminfo.read_text().splitlines())
    available = int(values["MemAvailable"].split()[0]) * 1024
    if available < int(spare_gib) * 2**30:
        raise MemoryError(f"Available RAM {available / 2**30:.2f} GiB below {spare_gib} GiB reserve")


def _training_role(split: Any) -> dict[str, object]:
    """Return paper-validity metadata for an allowed QDIC training split."""
    value = str(split or "").strip().lower()
    if (
        not value
        or value.startswith("test")
        or "novel" in value
        or value in {"full", "all"}
    ):
        raise ValueError("QDIC Test/Novel/full features are forbidden for optimizer")
    if value.startswith("train"):
        return {
            "paper_status": "BASE_TRAIN",
            "paper_valid": True,
            "diagnostic_only": False,
        }
    if value.startswith("val") or value.startswith("dev"):
        return {
            "paper_status": "VAL_BASE_PILOT",
            "paper_valid": False,
            "diagnostic_only": True,
        }
    raise ValueError(f"unsupported QDIC training split: {split!r}")


def _load_feature_arrays(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = root / "features.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"QDIC feature metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact") != "qdic_v11_feature_cache":
        raise ValueError("optimizer requires a qdic_v11_feature_cache")
    if tuple(metadata.get("feature_names", ())) != tuple(QDIC_FEATURE_NAMES):
        raise ValueError("QDIC feature schema mismatch")
    if int(metadata.get("feature_dim", -1)) != QDIC_RAW_DIM:
        raise ValueError("QDIC feature dimension mismatch")
    _training_role(metadata.get("split"))
    if not metadata.get("base_only_supervision") or metadata.get("novel_gt_used_for_optimizer"):
        raise ValueError("QDIC feature cache is not Base-only")
    validate_qdic_feature_config(metadata.get("feature_config"))
    arrays: dict[str, np.ndarray] = {}
    paths = dict(metadata.get("arrays", {}))
    required = {
        "features",
        "labels",
        "supervision_allowed",
        "offsets",
        "videos",
        "candidate_base",
        "target_base",
        "group_ids",
    }
    if not required.issubset(paths):
        raise ValueError(f"QDIC feature cache missing arrays: {sorted(required - set(paths))}")
    hashes = dict(metadata.get("array_hashes", {}))
    for name in required:
        path = Path(paths[name])
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or name not in hashes or sha256(path) != hashes[name]:
            raise ValueError(f"QDIC feature array hash mismatch: {path}")
        arrays[name] = np.load(path, mmap_mode="r" if name in {"features", "labels", "supervision_allowed", "candidate_base", "target_base", "group_ids"} else None, allow_pickle=False)
    if arrays["features"].ndim != 2 or arrays["features"].shape[1] != QDIC_RAW_DIM:
        raise ValueError("QDIC features must be [N,33]")
    count = len(arrays["features"])
    if any(len(arrays[name]) != count for name in ("labels", "supervision_allowed", "candidate_base", "target_base", "group_ids")):
        raise ValueError("QDIC feature arrays are not row aligned")
    expected_allowed = (
        np.asarray(arrays["candidate_base"], dtype=bool)
        & np.asarray(arrays["target_base"], dtype=bool)
        & np.isin(np.asarray(arrays["labels"]), (0, 1))
    )
    if not np.array_equal(np.asarray(arrays["supervision_allowed"], dtype=bool), expected_allowed):
        raise ValueError("QDIC supervision_allowed is inconsistent with Base-only labels")
    if len(arrays["offsets"]) != len(arrays["videos"]) + 1 or int(arrays["offsets"][0]) != 0 or int(arrays["offsets"][-1]) != count:
        raise ValueError("QDIC offsets/videos are inconsistent")
    return metadata, arrays


def build_training_groups(
    metadata: Mapping[str, Any],
    arrays: dict[str, np.ndarray],
) -> tuple[list[int], list[int]]:
    """Return video-disjoint train/holdout event groups.

    The split is deterministic by video ID and the eligibility test is
    explicitly Base-positive/Base-negative, so unknown and Novel rows cannot
    influence the optimizer or its normalization statistics.
    """
    del metadata
    offsets = np.asarray(arrays["offsets"])
    labels = np.asarray(arrays["labels"])
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool).copy()
    if "candidate_base" in arrays:
        allowed &= np.asarray(arrays["candidate_base"], dtype=bool)
    if "target_base" in arrays:
        allowed &= np.asarray(arrays["target_base"], dtype=bool)
    allowed &= np.isin(labels, (0, 1))
    videos = np.asarray(arrays["videos"], dtype=np.int64)
    train_groups: list[int] = []
    holdout_groups: list[int] = []
    for group, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        mask = allowed[int(start) : int(end)]
        values = labels[int(start) : int(end)]
        if not bool((mask & (values == 1)).any()) or not bool((mask & (values == 0)).any()):
            continue
        # Keep the old V9 deterministic video hash partition, not a row-level
        # split that could leak one video's identity statistics across folds.
        bucket = int(hashlib.sha256(str(int(videos[group])).encode()).hexdigest()[:8], 16) % 5
        (holdout_groups if bucket == 0 else train_groups).append(group)
    if not train_groups or not holdout_groups:
        raise ValueError("need disjoint nonempty Base train/holdout videos")
    return train_groups, holdout_groups


def _source_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    paths = (
        base / "query_distributional_calibrator.py",
        base / "qdic_features.py",
        base / "qdic_trainer.py",
        base / "qdic_loader.py",
    )
    return {str(path): sha256(path) for path in paths}


def train(
    features_dir: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    epochs: int = 12,
    seed: int = 0,
) -> dict[str, Any]:
    """Train only on eligible Base rows from a non-Test QDIC cache."""
    root = Path(features_dir).resolve()
    out = Path(output).resolve()
    metadata, arrays = _load_feature_arrays(root)
    training_role = _training_role(metadata.get("split"))
    feature_config = validate_qdic_feature_config(metadata["feature_config"])
    train_groups, holdout_groups = build_training_groups(metadata, arrays)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = QueryDistributionalCalibrator().to(device)
    features = arrays["features"]
    labels = np.asarray(arrays["labels"])
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    videos = np.asarray(arrays["videos"], dtype=np.int64)

    # Fit normalization from supervised rows in training videos only.
    total = np.zeros(QDIC_RAW_DIM, dtype=np.float64)
    squared = np.zeros(QDIC_RAW_DIM, dtype=np.float64)
    count = 0
    for group in train_groups:
        start, end = offsets[group : group + 2]
        mask = allowed[start:end]
        values = np.asarray(features[start:end][mask], dtype=np.float64)
        total += values.sum(axis=0)
        squared += (values * values).sum(axis=0)
        count += len(values)
    if count < 1:
        raise ValueError("no supervised training rows for QDIC normalization")
    mean = total / count
    scale = np.sqrt(np.maximum(squared / count - mean * mean, 1e-6))
    model.set_normalization(mean.astype(np.float32), scale.astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    source_hashes = _source_hashes()

    def batch(groups: Iterable[int], *, hard: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        examples: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for group in groups:
            start, end = offsets[group : group + 2]
            eligible = np.flatnonzero(allowed[start:end])
            values = np.asarray(labels[start:end])
            if hard:
                positive = eligible[values[eligible] == 1]
                negative = eligible[values[eligible] == 0]
                by_rank = sorted(negative.tolist(), key=lambda i: (float(features[start + i, 17]), int(i)))
                by_raw = sorted(negative.tolist(), key=lambda i: (-float(features[start + i, 5]), int(i)))
                selected: list[int] = []
                for first, second in zip(by_rank, by_raw):
                    for item in (first, second):
                        if item not in selected:
                            selected.append(item)
                    if len(selected) >= 16:
                        break
                eligible = np.concatenate((positive, np.asarray(selected[:16], dtype=np.int64)))
            if len(eligible) < 1:
                continue
            examples.append(np.asarray(features[start:end][eligible], dtype=np.float32))
            targets.append(values[eligible].astype(np.int64, copy=False))
        if not examples:
            raise ValueError("no eligible QDIC groups")
        width = max(len(item) for item in targets)
        batch_features = np.zeros((len(examples), width, QDIC_RAW_DIM), dtype=np.float32)
        batch_labels = np.full((len(examples), width), -1, dtype=np.int64)
        for index, (value, target) in enumerate(zip(examples, targets)):
            batch_features[index, : len(target)] = value
            batch_labels[index, : len(target)] = target
        return torch.from_numpy(batch_features).to(device), torch.from_numpy(batch_labels).to(device)

    def validation() -> tuple[float, float]:
        model.eval()
        losses: list[float] = []
        correct = 0
        total_groups = 0
        with torch.inference_mode():
            for start in range(0, len(holdout_groups), 32):
                batch_features, batch_labels = batch(holdout_groups[start : start + 32])
                logits = model(batch_features)
                losses.append(float(group_ranking_loss(logits, batch_labels)))
                winner = logits.masked_fill(batch_labels < 0, -torch.inf).argmax(dim=1)
                correct += int((batch_labels.gather(1, winner[:, None]) == 1).sum())
                total_groups += len(batch_labels)
        return float(np.mean(losses)), correct / max(total_groups, 1)

    best = float("inf")
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    best_step = 0
    with (out / "metrics.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(int(epochs)):
            _memory_guard()
            model.train()
            shuffled = rng.permutation(train_groups)
            epoch_losses: list[float] = []
            for start in range(0, len(shuffled), 32):
                batch_features, batch_labels = batch(shuffled[start : start + 32], hard=True)
                optimizer.zero_grad(set_to_none=True)
                loss = group_ranking_loss(model(batch_features), batch_labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite QDIC loss")
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("non-finite QDIC gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
                epoch_losses.append(float(loss.detach()))
            holdout_loss, holdout_top1 = validation()
            row = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(epoch_losses)),
                "holdout_loss": holdout_loss,
                "holdout_top1": holdout_top1,
            }
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if holdout_loss < best:
                best = holdout_loss
                best_step = optimizer_steps
                torch.save(
                    {
                        "status": QDIC_STATUS,
                        "model_state": model.state_dict(),
                        "feature_names": list(QDIC_FEATURE_NAMES),
                        "feature_dim": QDIC_RAW_DIM,
                        "feature_config": feature_config,
                        "features_hash": sha256(root / "features.json"),
                        "training_split": metadata["split"],
                        "protocol": "QDIC_V11_BASE_ONLY_TRAINING",
                        "base_only_supervision": True,
                        "novel_gt_used": False,
                        "test_weights_used": False,
                        **training_role,
                        "source_hashes": source_hashes,
                        "epoch": epoch + 1,
                        "optimizer_steps": optimizer_steps,
                        "seed": int(seed),
                    },
                    out / "best.pt",
                )

    checkpoint = out / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError("QDIC trainer did not produce best.pt")
    result: dict[str, Any] = {
        "status": "COMPLETED",
        "artifact": "qdic_v11_training",
        "protocol": "QDIC_V11_BASE_ONLY_TRAINING",
        "training_split": metadata["split"],
        **training_role,
        "training_groups": len(train_groups),
        "holdout_groups": len(holdout_groups),
        "train_video_ids": sorted({int(videos[group]) for group in train_groups}),
        "holdout_video_ids": sorted({int(videos[group]) for group in holdout_groups}),
        "base_only_supervision": True,
        "novel_gt_used": False,
        "test_weights_used": False,
        "normalization_fit": "training-video supervised Base rows only",
        "feature_names": list(QDIC_FEATURE_NAMES),
        "feature_dim": QDIC_RAW_DIM,
        "feature_config": feature_config,
        "features": str(root / "features.json"),
        "features_hash": sha256(root / "features.json"),
        "checkpoint": str(checkpoint),
        "checkpoint_hash": sha256(checkpoint),
        "history": history,
        "device": str(device),
        "epochs": int(epochs),
        "seed": int(seed),
        "optimizer_steps": optimizer_steps,
        "checkpoint_optimizer_steps": best_step,
        "source_hashes": source_hashes,
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "variables": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "LD_PRELOAD",
                )
            },
        },
        "repo_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
    }
    _write_json(out / "training.json", result)
    return result


__all__ = ["_training_role", "build_training_groups", "sha256", "train"]
