"""Official-Train trainer for the two pre-registered V12 MGF cards."""

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
    QDIC_MGF_BETA,
    QDIC_MGF_FEATURE_NAMES,
    QDIC_MGF_FEATURE_SCHEMA_VERSION,
    QDIC_MGF_RAW_DIM,
)
from .qdic_loader import validate_qdic_feature_config
from .qdic_mgf_loader import MGF_PROTOCOL, QDIC_MGF_STATUS, current_mgf_source_hashes, validate_mgf_feature_config
from .query_conditioned_reranker import group_ranking_loss
from .query_mgf_calibrator import QueryMomentGeneratingCalibrator


MIXED_CANDIDATE_K = (8, 16, 32, 64)
CHECKPOINT_COMPARATOR = "(final_mrr, final_top1, net_correction, -final_listwise_loss)"
OFFICIAL_TRAIN_SPLIT = "train"
FORBIDDEN_SOURCE_MARKERS = ("val", "test", "novel", "pilot")


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


def _memory_guard(spare_gib: int = 12) -> None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return
    values = dict(line.split(":", 1) for line in meminfo.read_text().splitlines())
    available = int(values["MemAvailable"].split()[0]) * 1024
    if available < int(spare_gib) * 2**30:
        raise MemoryError(f"Available RAM {available / 2**30:.2f} GiB below {spare_gib} GiB reserve")


def _official_train_guard(metadata: Mapping[str, Any]) -> None:
    if str(metadata.get("artifact")) != "qdic_v12_mgf_feature_cache":
        raise ValueError("MGF optimizer requires qdic_v12_mgf_feature_cache")
    if str(metadata.get("source_role")) != "OFFICIAL_TRAIN":
        raise ValueError("MGF optimizer source_role must be OFFICIAL_TRAIN")
    if str(metadata.get("exact_split_name")) != OFFICIAL_TRAIN_SPLIT:
        raise ValueError("MGF optimizer exact split must be train")
    if int(metadata.get("schema_version", -1)) != QDIC_MGF_FEATURE_SCHEMA_VERSION:
        raise ValueError("MGF feature schema version mismatch")
    if int(metadata.get("feature_dim", -1)) != QDIC_MGF_RAW_DIM:
        raise ValueError("MGF feature dimension mismatch")
    if tuple(metadata.get("feature_names", ())) != tuple(QDIC_MGF_FEATURE_NAMES):
        raise ValueError("MGF feature names mismatch")
    if metadata.get("base_only_supervision") is not True:
        raise ValueError("MGF cache must be Base-only")
    if metadata.get("novel_gt_used_for_optimizer") is not False:
        raise ValueError("Novel GT is forbidden for MGF optimizer")
    if metadata.get("test_gt_used_for_optimizer") is not False:
        raise ValueError("Test GT is forbidden for MGF optimizer")
    if metadata.get("optimizer_source_allowed") is not True:
        raise ValueError("MGF cache is not explicitly optimizer-allowed")
    if metadata.get("video_disjoint_split") is not True:
        raise ValueError("MGF cache must record video-disjoint split readiness")
    if metadata.get("normalization_fit") != "internal_train_base_only_after_video_split":
        raise ValueError("MGF normalization provenance is not train-only")
    if not metadata.get("official_train_annotation_sha256"):
        raise ValueError("MGF cache is missing Official Train annotation hash")
    for key in ("source_event_cache", "source_frontend_cache", "source_annotation"):
        value = str(metadata.get(key, "")).lower()
        if any(marker in value for marker in FORBIDDEN_SOURCE_MARKERS):
            raise ValueError(f"forbidden optimizer source in {key}: {value}")
    if not np.isclose(float(metadata.get("mgf_beta", np.nan)), QDIC_MGF_BETA, rtol=0.0, atol=1e-8):
        raise ValueError("MGF beta must be exactly 1.0")
    validate_mgf_feature_config(metadata.get("feature_config"))


def _load_arrays(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = root / "features.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _official_train_guard(metadata)
    paths = dict(metadata.get("arrays", {}))
    required = {
        "features", "labels", "supervision_allowed", "offsets", "videos",
        "candidate_base", "target_base", "group_ids",
    }
    if not required.issubset(paths):
        raise ValueError(f"MGF cache missing arrays: {sorted(required - set(paths))}")
    hashes = dict(metadata.get("array_hashes", {}))
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        path = Path(str(paths[name]))
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or _sha256(path) != hashes.get(name):
            raise ValueError(f"MGF feature array hash mismatch: {path}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    count = len(arrays["features"])
    if arrays["features"].ndim != 2 or arrays["features"].shape[1] != QDIC_MGF_RAW_DIM:
        raise ValueError("MGF features must be [N,35]")
    for name in ("labels", "supervision_allowed", "candidate_base", "target_base", "group_ids"):
        if len(arrays[name]) != count:
            raise ValueError("MGF arrays are not row aligned")
    allowed = (
        np.asarray(arrays["supervision_allowed"], dtype=bool)
        & np.asarray(arrays["candidate_base"], dtype=bool)
        & np.asarray(arrays["target_base"], dtype=bool)
        & np.isin(np.asarray(arrays["labels"]), (0, 1))
    )
    if not np.array_equal(allowed, np.asarray(arrays["supervision_allowed"], dtype=bool)):
        raise ValueError("MGF supervision_allowed is inconsistent")
    if len(arrays["offsets"]) != len(arrays["videos"]) + 1:
        raise ValueError("MGF offsets/videos are inconsistent")
    if int(arrays["offsets"][0]) != 0 or int(arrays["offsets"][-1]) != count:
        raise ValueError("MGF offsets do not cover all rows")
    return metadata, arrays


def build_official_train_groups(
    arrays: Mapping[str, np.ndarray],
    *,
    holdout_modulus: int = 5,
    holdout_bucket: int = 0,
) -> tuple[list[int], list[int]]:
    """Deterministically split complete videos without row leakage."""
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
        raise ValueError("Official Train MGF cache needs nonempty train and holdout groups")
    train_videos = {int(videos[index]) for index in train_groups}
    holdout_videos = {int(videos[index]) for index in holdout_groups}
    if train_videos & holdout_videos:
        raise AssertionError("MGF video-disjoint split invariant failed")
    return train_groups, holdout_groups


def _select_rows(
    start: int,
    end: int,
    features: np.ndarray,
    labels: np.ndarray,
    allowed: np.ndarray,
    *,
    candidate_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(allowed[start:end])
    if len(eligible) < 2:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    selected = eligible[: int(candidate_k)]
    if not bool(np.any(labels[start + selected] == 1)):
        positive = eligible[labels[start + eligible] == 1]
        selected = np.concatenate((selected[:-1], positive[:1]))
    if not bool(np.any(labels[start + selected] == 0)):
        negative = eligible[labels[start + eligible] == 0]
        selected = np.concatenate((selected[:-1], negative[:1]))
    selected = np.unique(selected)
    return np.asarray(features[start + selected], dtype=np.float32), np.asarray(labels[start + selected], dtype=np.int64)


def _batch(
    groups: Iterable[int],
    arrays: Mapping[str, np.ndarray],
    *,
    candidate_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = arrays["features"]
    labels = np.asarray(arrays["labels"])
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    examples: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for group in groups:
        start, end = offsets[group:group + 2]
        value, target = _select_rows(int(start), int(end), features, labels, allowed, candidate_k=candidate_k)
        if len(target) and np.any(target == 1) and np.any(target == 0):
            examples.append(value)
            targets.append(target)
    if not examples:
        raise ValueError("no eligible Base-only MGF groups")
    width = max(len(item) for item in targets)
    batch_features = np.zeros((len(examples), width, QDIC_MGF_RAW_DIM), dtype=np.float32)
    batch_labels = np.full((len(examples), width), -1, dtype=np.int64)
    for index, (value, target) in enumerate(zip(examples, targets)):
        batch_features[index, :len(target)] = value
        batch_labels[index, :len(target)] = target
    return torch.from_numpy(batch_features), torch.from_numpy(batch_labels)


def _metrics(
    model: QueryMomentGeneratingCalibrator,
    groups: Iterable[int],
    arrays: Mapping[str, np.ndarray],
    *,
    device: str,
    candidate_k: int = 64,
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
            value, target = _select_rows(int(start), int(end), features, labels, allowed, candidate_k=candidate_k)
            if len(target) == 0 or not np.any(target == 1) or not np.any(target == 0):
                continue
            logits = model(torch.from_numpy(value).to(device))
            row_loss = group_ranking_loss(logits.unsqueeze(0), torch.from_numpy(target).to(device).unsqueeze(0))
            losses.append(float(row_loss))
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
    return (
        float(metrics["final_mrr"]),
        float(metrics["final_top1"]),
        float(metrics["net_correction"]),
        -float(metrics["final_listwise_loss"]),
    )


def train_official_mgf_card(
    features_dir: str | Path,
    output: str | Path,
    *,
    card_id: str,
    mgf_mode: str,
    epochs: int = 12,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train one frozen M1/M2 card using only the Official Train cache."""
    if mgf_mode not in {"core", "fused"}:
        raise ValueError("MGF card mode must be core or fused")
    root = Path(features_dir).resolve()
    out = Path(output).resolve()
    metadata, arrays = _load_arrays(root)
    train_groups, holdout_groups = build_official_train_groups(arrays)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = QueryMomentGeneratingCalibrator(mgf_mode=mgf_mode).to(device)
    features = arrays["features"]
    labels = np.asarray(arrays["labels"])
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    normalization_rows: list[np.ndarray] = []
    for group in train_groups:
        start, end = offsets[group:group + 2]
        mask = allowed[start:end]
        if np.any(mask):
            normalization_rows.append(np.asarray(features[start:end][mask], dtype=np.float32))
    if not normalization_rows:
        raise ValueError("no Base rows for Official Train MGF normalization")
    normalization_values = np.concatenate(normalization_rows, axis=0).astype(np.float64)
    mean = normalization_values.mean(axis=0)
    scale = np.sqrt(np.maximum(normalization_values.var(axis=0), 1e-6))
    model.set_normalization(mean.astype(np.float32), scale.astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    source_hashes = current_mgf_source_hashes()
    best_key: tuple[float, float, float, float] | None = None
    best_epoch = 0
    optimizer_steps = 0
    history: list[dict[str, Any]] = []
    with (out / "metrics.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(int(epochs)):
            _memory_guard()
            model.train()
            shuffled = rng.permutation(train_groups)
            epoch_losses: list[float] = []
            for batch_start in range(0, len(shuffled), 32):
                group_batch = shuffled[batch_start:batch_start + 32]
                candidate_k = MIXED_CANDIDATE_K[(epoch + batch_start // 32) % len(MIXED_CANDIDATE_K)]
                batch_features, batch_labels = _batch(group_batch, arrays, candidate_k=candidate_k)
                diagnostics = model(batch_features.to(device), return_diagnostics=True)
                loss = group_ranking_loss(diagnostics["logit"], batch_labels.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite MGF loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise FloatingPointError("non-finite MGF gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
                epoch_losses.append(float(loss.detach()))
            validation = _metrics(model, holdout_groups, arrays, device=device)
            row = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("inf"),
                **validation,
                "candidate_k_schedule": list(MIXED_CANDIDATE_K),
            }
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            key = _comparator(validation)
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch + 1
                torch.save(
                    {
                        "status": QDIC_MGF_STATUS,
                        "model_state": model.state_dict(),
                        "mgf_mode": mgf_mode,
                        "mgf_beta": QDIC_MGF_BETA,
                        "schema_version": QDIC_MGF_FEATURE_SCHEMA_VERSION,
                        "feature_names": list(QDIC_MGF_FEATURE_NAMES),
                        "feature_dim": QDIC_MGF_RAW_DIM,
                        "feature_config": metadata["feature_config"],
                        "features_hash": _sha256(root / "features.json"),
                        "training_split": "train_base_official",
                        "protocol": MGF_PROTOCOL,
                        "base_only_supervision": True,
                        "novel_gt_used": False,
                        "test_weights_used": False,
                        "paper_status": "BASE_TRAIN",
                        "paper_valid": True,
                        "diagnostic_only": False,
                        "source_hashes": source_hashes,
                        "checkpoint_comparator": CHECKPOINT_COMPARATOR,
                        "epoch": epoch + 1,
                        "optimizer_steps": optimizer_steps,
                        "seed": int(seed),
                    },
                    out / "best.pt",
                )
    if not (out / "best.pt").is_file():
        raise RuntimeError("Official Train MGF card did not produce best.pt")
    videos = np.asarray(arrays["videos"], dtype=np.int64)
    result = {
        "status": "COMPLETED",
        "artifact": "qdic_v12_mgf_official_training",
        "card_id": str(card_id),
        "protocol": MGF_PROTOCOL,
        "mgf_mode": mgf_mode,
        "mgf_beta": QDIC_MGF_BETA,
        "schema_version": QDIC_MGF_FEATURE_SCHEMA_VERSION,
        "feature_names": list(QDIC_MGF_FEATURE_NAMES),
        "feature_dim": QDIC_MGF_RAW_DIM,
        "training_split": "train_base_official",
        "exact_split_name": OFFICIAL_TRAIN_SPLIT,
        "paper_status": "BASE_TRAIN",
        "paper_valid": True,
        "diagnostic_only": False,
        "base_only_supervision": True,
        "novel_gt_used": False,
        "test_weights_used": False,
        "features": str(root / "features.json"),
        "features_hash": _sha256(root / "features.json"),
        "train_video_ids": sorted({int(videos[index]) for index in train_groups}),
        "holdout_video_ids": sorted({int(videos[index]) for index in holdout_groups}),
        "video_disjoint": True,
        "normalization_fit": "internal_train_base_only_after_video_split",
        "normalization_row_count": int(len(normalization_values)),
        "normalization_mean": mean.tolist(),
        "normalization_scale": scale.tolist(),
        "feature_config": metadata["feature_config"],
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
            "variables": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "OMP_NUM_THREADS")},
        },
        "repo_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
    }
    _write_json(out / "training.json", result)
    return result


__all__ = [
    "CHECKPOINT_COMPARATOR",
    "MIXED_CANDIDATE_K",
    "build_official_train_groups",
    "train_official_mgf_card",
]
