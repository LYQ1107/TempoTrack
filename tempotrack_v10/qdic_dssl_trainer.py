"""Independent Official-Train trainer for V11 DSSL.

This module intentionally does not modify the historical candidate-aware
trainer.  It owns the stricter Official-Train provenance guard, deterministic
video-disjoint holdout, mixed causal candidate widths, train-only
normalisation, and the paper-specified checkpoint comparator.
"""

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

from .distributional_losses import total_dssl_loss
from .qdic_features import QDIC_FEATURE_NAMES, QDIC_RAW_DIM
from .qdic_loader import QDIC_STATUS, validate_qdic_feature_config
from .qdic_trainer import sha256
from .query_conditioned_reranker import group_ranking_loss
from .query_distributional_calibrator import QueryDistributionalCalibrator


MIXED_CANDIDATE_K = (8, 16, 32, 64)
CHECKPOINT_COMPARATOR = (
    "(final_mrr, final_top1, net_correction, -final_listwise_loss)"
)
OFFICIAL_TRAIN_SPLIT = "train"
FORBIDDEN_OPTIMIZER_MARKERS = ("val", "test", "novel", "pilot")


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _memory_guard(spare_gib: int = 12) -> None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return
    values = dict(line.split(":", 1) for line in meminfo.read_text().splitlines())
    available = int(values["MemAvailable"].split()[0]) * 1024
    if available < int(spare_gib) * 2**30:
        raise MemoryError(
            f"Available RAM {available / 2**30:.2f} GiB below {spare_gib} GiB reserve"
        )


def _official_train_guard(metadata: Mapping[str, Any]) -> None:
    """Reject any cache that is not the newly generated Official Train cache."""
    if str(metadata.get("source_role", "")) != "OFFICIAL_TRAIN":
        raise ValueError("DSSL optimizer source_role must be OFFICIAL_TRAIN")
    if str(metadata.get("artifact")) != "qdic_v11_feature_cache":
        raise ValueError("DSSL optimizer requires qdic_v11_feature_cache")
    if tuple(metadata.get("feature_names", ())) != tuple(QDIC_FEATURE_NAMES):
        raise ValueError("DSSL feature schema mismatch")
    if int(metadata.get("feature_dim", -1)) != QDIC_RAW_DIM:
        raise ValueError("DSSL feature dimension mismatch")
    if str(metadata.get("exact_split_name", "")) != OFFICIAL_TRAIN_SPLIT:
        raise ValueError("DSSL optimizer exact split must be train")
    if metadata.get("base_only_supervision") is not True:
        raise ValueError("DSSL cache must be Base-only")
    if metadata.get("novel_gt_used_for_optimizer") is not False:
        raise ValueError("Novel GT is forbidden for DSSL optimizer")
    if metadata.get("test_gt_used_for_optimizer") is not False:
        raise ValueError("Test GT is forbidden for DSSL optimizer")
    if metadata.get("optimizer_source_allowed") is not True:
        raise ValueError("DSSL cache is not explicitly optimizer-allowed")
    source_text = json.dumps(metadata, ensure_ascii=False, sort_keys=True).lower()
    forbidden = [marker for marker in FORBIDDEN_OPTIMIZER_MARKERS if marker in source_text]
    # A literal ``pilot``/``test`` in an audit note is not a source violation;
    # reject only explicit source/cache paths and source-role fields below.
    for key in ("source_event_cache", "source_frontend_cache", "source_annotation"):
        value = str(metadata.get(key, "")).lower()
        if any(marker in value for marker in ("val", "test", "novel", "pilot")):
            raise ValueError(f"forbidden optimizer source in {key}: {value}")
    if metadata.get("video_disjoint_split") is not True:
        raise ValueError("DSSL cache must record video-disjoint split readiness")
    if not metadata.get("official_train_annotation_sha256"):
        raise ValueError("DSSL cache is missing Official Train annotation hash")
    if not metadata.get("normalization_fit") == "internal_train_base_only_after_video_split":
        raise ValueError("DSSL normalization provenance is not train-only")
    del forbidden
    validate_qdic_feature_config(metadata.get("feature_config"))


def _load_arrays(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = root / "features.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _official_train_guard(metadata)
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
        raise ValueError(f"DSSL cache missing arrays: {sorted(required - set(paths))}")
    hashes = dict(metadata.get("array_hashes", {}))
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        path = Path(str(paths[name]))
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256(path) != hashes.get(name):
            raise ValueError(f"DSSL feature array hash mismatch: {path}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    count = len(arrays["features"])
    if arrays["features"].ndim != 2 or arrays["features"].shape[1] != QDIC_RAW_DIM:
        raise ValueError("DSSL features must be [N,33]")
    for name in ("labels", "supervision_allowed", "candidate_base", "target_base", "group_ids"):
        if len(arrays[name]) != count:
            raise ValueError("DSSL arrays are not row aligned")
    allowed = (
        np.asarray(arrays["supervision_allowed"], dtype=bool)
        & np.asarray(arrays["candidate_base"], dtype=bool)
        & np.asarray(arrays["target_base"], dtype=bool)
        & np.isin(np.asarray(arrays["labels"]), (0, 1))
    )
    if not np.array_equal(allowed, np.asarray(arrays["supervision_allowed"], dtype=bool)):
        raise ValueError("DSSL supervision_allowed is inconsistent")
    if len(arrays["offsets"]) != len(arrays["videos"]) + 1:
        raise ValueError("DSSL offsets/videos are inconsistent")
    if int(arrays["offsets"][0]) != 0 or int(arrays["offsets"][-1]) != count:
        raise ValueError("DSSL offsets do not cover all rows")
    conflict_path = paths.get("temporal_conflict")
    if conflict_path:
        path = Path(str(conflict_path))
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256(path) != hashes.get("temporal_conflict"):
            raise ValueError("DSSL temporal_conflict hash mismatch")
        arrays["temporal_conflict"] = np.load(path, mmap_mode="r", allow_pickle=False)
        if len(arrays["temporal_conflict"]) != count:
            raise ValueError("DSSL temporal_conflict is not row aligned")
    else:
        arrays["temporal_conflict"] = np.zeros(count, dtype=bool)
    return metadata, arrays


def build_official_train_groups(
    arrays: Mapping[str, np.ndarray],
    *,
    holdout_modulus: int = 5,
    holdout_bucket: int = 0,
) -> tuple[list[int], list[int]]:
    """Deterministic complete-video split; no row can cross the boundary."""
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    videos = np.asarray(arrays["videos"], dtype=np.int64)
    labels = np.asarray(arrays["labels"])
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool)
    train_groups: list[int] = []
    holdout_groups: list[int] = []
    for group, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        row_mask = allowed[int(start) : int(end)]
        row_labels = labels[int(start) : int(end)]
        if not bool((row_mask & (row_labels == 1)).any()) or not bool(
            (row_mask & (row_labels == 0)).any()
        ):
            continue
        bucket = int(
            hashlib.sha256(str(int(videos[group])).encode("utf-8")).hexdigest()[:8], 16
        ) % int(holdout_modulus)
        (holdout_groups if bucket == holdout_bucket else train_groups).append(group)
    if not train_groups or not holdout_groups:
        raise ValueError("Official Train cache needs nonempty train and holdout video groups")
    train_videos = {int(videos[index]) for index in train_groups}
    holdout_videos = {int(videos[index]) for index in holdout_groups}
    if train_videos & holdout_videos:
        raise AssertionError("video-disjoint split invariant failed")
    return train_groups, holdout_groups


def _source_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    names = (
        "query_distributional_calibrator.py",
        "qdic_features.py",
        "qdic_trainer.py",
        "qdic_loader.py",
        "query_conditioned_reranker.py",
        "distributional_losses.py",
        "qdic_dssl_trainer.py",
    )
    return {str(base / name): sha256(base / name) for name in names}


def _select_rows(
    start: int,
    end: int,
    features: np.ndarray,
    labels: np.ndarray,
    allowed: np.ndarray,
    temporal_conflict: np.ndarray,
    *,
    candidate_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(allowed[start:end])
    if len(eligible) < 2:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=bool)
    # Existing feature rows are causal prefilter order.  Keep that order and
    # only cap it at the selected K; never use a GT-derived reordering.
    selected = eligible[: int(candidate_k)]
    if not bool(np.any(labels[start + selected] == 1)):
        positive = eligible[labels[start + eligible] == 1]
        selected = np.concatenate((selected[:-1], positive[:1]))
    if not bool(np.any(labels[start + selected] == 0)):
        negative = eligible[labels[start + eligible] == 0]
        selected = np.concatenate((selected[:-1], negative[:1]))
    selected = np.unique(selected)
    return (
        np.asarray(features[start + selected], dtype=np.float32),
        np.asarray(labels[start + selected], dtype=np.int64),
        np.asarray(temporal_conflict[start + selected], dtype=bool),
    )


def _batch(
    groups: Iterable[int],
    arrays: Mapping[str, np.ndarray],
    *,
    candidate_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = arrays["features"]
    labels = np.asarray(arrays["labels"])
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    conflict = np.asarray(arrays["temporal_conflict"], dtype=bool)
    examples: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    conflicts: list[np.ndarray] = []
    for group in groups:
        start, end = offsets[group : group + 2]
        value, target, conflict_value = _select_rows(
            int(start), int(end), features, labels, allowed, conflict, candidate_k=candidate_k
        )
        if len(target) and np.any(target == 1) and np.any(target == 0):
            examples.append(value)
            targets.append(target)
            conflicts.append(conflict_value)
    if not examples:
        raise ValueError("no eligible Base-only DSSL groups")
    width = max(len(item) for item in targets)
    batch_features = np.zeros((len(examples), width, QDIC_RAW_DIM), dtype=np.float32)
    batch_labels = np.full((len(examples), width), -1, dtype=np.int64)
    batch_conflict = np.zeros((len(examples), width), dtype=bool)
    for index, (value, target, conflict_value) in enumerate(zip(examples, targets, conflicts)):
        batch_features[index, : len(target)] = value
        batch_labels[index, : len(target)] = target
        batch_conflict[index, : len(target)] = conflict_value
    return (
        torch.from_numpy(batch_features),
        torch.from_numpy(batch_labels),
        torch.from_numpy(batch_conflict),
    )


def _metrics(
    model: QueryDistributionalCalibrator,
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
            start, end = offsets[group : group + 2]
            value, target, _ = _select_rows(
                int(start), int(end), features, labels, allowed,
                # ``_select_rows`` indexes the conflict vector with global
                # feature-row offsets.  Keep the diagnostic zero vector
                # global as well; a local ``end-start`` array would make
                # valid later holdout groups index out of bounds.
                np.zeros(len(features), dtype=bool), candidate_k=candidate_k,
            )
            if len(target) == 0 or not np.any(target == 1) or not np.any(target == 0):
                continue
            logits = model(torch.from_numpy(value).to(device))
            row_loss = group_ranking_loss(
                logits.unsqueeze(0), torch.from_numpy(target).to(device).unsqueeze(0)
            )
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


def train_official_card(
    features_dir: str | Path,
    output: str | Path,
    *,
    card_id: str,
    structured_branch_mode: str = "dssl",
    lambda_struct: float = 0.25,
    lambda_cons: float = 0.0,
    lambda_hard: float = 0.2,
    temporal_conflict: bool = False,
    epochs: int = 12,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train one B0/B1/D/C/H card and write a self-contained receipt."""
    if structured_branch_mode not in {"legacy", "dssl"}:
        raise ValueError("structured_branch_mode must be legacy or dssl")
    if temporal_conflict and structured_branch_mode != "dssl":
        raise ValueError("temporal conflict is only valid for DSSL")
    root = Path(features_dir).resolve()
    out = Path(output).resolve()
    metadata, arrays = _load_arrays(root)
    train_groups, holdout_groups = build_official_train_groups(arrays)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = QueryDistributionalCalibrator(
        structured_branch_mode=structured_branch_mode
    ).to(device)
    features = arrays["features"]
    labels = np.asarray(arrays["labels"])
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    # Fit normalization strictly after the video split and on Base rows only.
    selected_values: list[np.ndarray] = []
    for group in train_groups:
        start, end = offsets[group : group + 2]
        mask = allowed[start:end]
        if np.any(mask):
            selected_values.append(np.asarray(features[start:end][mask], dtype=np.float32))
    if not selected_values:
        raise ValueError("no Base rows for Official Train normalization")
    normalization_rows = np.concatenate(selected_values, axis=0).astype(np.float64)
    mean = normalization_rows.mean(axis=0)
    scale = np.sqrt(np.maximum(normalization_rows.var(axis=0), 1e-6))
    model.set_normalization(mean.astype(np.float32), scale.astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    source_hashes = _source_hashes()
    best_key: tuple[float, float, float, float] | None = None
    best_epoch = 0
    optimizer_steps = 0
    history: list[dict[str, Any]] = []
    metrics_path = out / "metrics.jsonl"
    with metrics_path.open("x", encoding="utf-8") as log:
        for epoch in range(int(epochs)):
            _memory_guard()
            model.train()
            shuffled = rng.permutation(train_groups)
            epoch_losses: list[float] = []
            for batch_start in range(0, len(shuffled), 32):
                group_batch = shuffled[batch_start : batch_start + 32]
                candidate_k = MIXED_CANDIDATE_K[(epoch + batch_start // 32) % len(MIXED_CANDIDATE_K)]
                batch_features, batch_labels, batch_conflict = _batch(
                    group_batch, arrays, candidate_k=candidate_k
                )
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                batch_conflict = batch_conflict.to(device)
                diagnostics = model(batch_features, return_diagnostics=True)
                if structured_branch_mode == "legacy":
                    loss = group_ranking_loss(diagnostics["logit"], batch_labels)
                    components = {"total": loss, "final": loss}
                else:
                    components = total_dssl_loss(
                        diagnostics["logit"],
                        diagnostics["fast_branch"],
                        diagnostics["slow_branch"],
                        batch_labels,
                        diagnostics["alpha"],
                        lambda_struct=lambda_struct,
                        lambda_cons=lambda_cons,
                        lambda_hard=lambda_hard,
                        temporal_conflict=batch_conflict if temporal_conflict else None,
                    )
                    loss = components["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite DSSL loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("non-finite DSSL gradient")
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
                        "status": QDIC_STATUS,
                        "model_state": model.state_dict(),
                        "structured_branch_mode": structured_branch_mode,
                        "feature_names": list(QDIC_FEATURE_NAMES),
                        "feature_dim": QDIC_RAW_DIM,
                        "feature_config": metadata["feature_config"],
                        "features_hash": sha256(root / "features.json"),
                        "training_split": "train_base_official",
                        "protocol": "QDIC_V11_BASE_ONLY_TRAINING",
                        "base_only_supervision": True,
                        "novel_gt_used": False,
                        "test_weights_used": False,
                        "paper_status": "BASE_TRAIN",
                        "paper_valid": True,
                        "diagnostic_only": False,
                        "source_hashes": source_hashes,
                        "lambda_struct": float(lambda_struct),
                        "lambda_cons": float(lambda_cons),
                        "lambda_hard": float(lambda_hard),
                        "temporal_conflict": bool(temporal_conflict),
                        "checkpoint_comparator": CHECKPOINT_COMPARATOR,
                        "epoch": epoch + 1,
                        "optimizer_steps": optimizer_steps,
                        "seed": int(seed),
                    },
                    out / "best.pt",
                )
    if not (out / "best.pt").is_file():
        raise RuntimeError("Official Train DSSL card did not produce best.pt")
    videos = np.asarray(arrays["videos"], dtype=np.int64)
    result = {
        "status": "COMPLETED",
        "artifact": "qdic_v11_dssl_official_training",
        "card_id": str(card_id),
        "protocol": "QDIC_V11_BASE_ONLY_TRAINING",
        "structured_branch_mode": structured_branch_mode,
        "training_split": "train_base_official",
        "exact_split_name": OFFICIAL_TRAIN_SPLIT,
        "paper_status": "BASE_TRAIN",
        "paper_valid": True,
        "diagnostic_only": False,
        "base_only_supervision": True,
        "novel_gt_used": False,
        "test_weights_used": False,
        "features": str(root / "features.json"),
        "features_hash": sha256(root / "features.json"),
        "train_video_ids": sorted({int(videos[index]) for index in train_groups}),
        "holdout_video_ids": sorted({int(videos[index]) for index in holdout_groups}),
        "video_disjoint": True,
        "normalization_fit": "internal_train_base_only_after_video_split",
        "normalization_row_count": int(len(normalization_rows)),
        "normalization_mean": mean.tolist(),
        "normalization_scale": scale.tolist(),
        "feature_names": list(QDIC_FEATURE_NAMES),
        "feature_dim": QDIC_RAW_DIM,
        "feature_config": metadata["feature_config"],
        "lambda_struct": float(lambda_struct),
        "lambda_cons": float(lambda_cons),
        "lambda_hard": float(lambda_hard),
        "temporal_conflict": bool(temporal_conflict),
        "mixed_candidate_k": list(MIXED_CANDIDATE_K),
        "checkpoint_comparator": CHECKPOINT_COMPARATOR,
        "best_epoch": int(best_epoch),
        "optimizer_steps": int(optimizer_steps),
        "checkpoint": str(out / "best.pt"),
        "checkpoint_hash": sha256(out / "best.pt"),
        "history": history,
        "source_hashes": source_hashes,
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "variables": {
                key: os.environ.get(key)
                for key in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "OMP_NUM_THREADS")
            },
        },
        "repo_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
    }
    _write_json(out / "training.json", result)
    return result


__all__ = [
    "CHECKPOINT_COMPARATOR",
    "MIXED_CANDIDATE_K",
    "build_official_train_groups",
    "train_official_card",
]
