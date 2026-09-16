"""Base-only trainer for the A0/A0-D/A1/A2 QDIC comparison.

This trainer intentionally owns the comparison protocol instead of modifying
the frozen V11 trainer.  Every architecture sees the same video-disjoint
groups, first-K causal views, normalization buffers, optimizer, seed, and
hard-negative policy.  Novel and unknown rows are masked before either the
final or distribution-only loss is evaluated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .candidate_aware_qdic import DistributionalAuxiliaryQDIC
from .deepset_qdic import DeepSetQDIC
from .dgsa_qdic import DistributionGuidedSetAttentionQDIC
from .distributional_losses import (
    distributional_ranking_loss,
    hard_negative_margin_loss,
    listwise_group_loss,
)
from .qdic_features import QDIC_FEATURE_NAMES, QDIC_RAW_DIM
from .qdic_loader import load_qdic_checkpoint, validate_qdic_feature_config
from .qdic_trainer import (
    _load_feature_arrays,
    _memory_guard,
    _training_role,
    build_training_groups,
    sha256,
)
from .query_distributional_calibrator import QueryDistributionalCalibrator


CANDIDATE_AWARE_STATUS = "CANDIDATE_AWARE_QDIC_MODEL_CODE_AND_WEIGHTS"
DEFAULT_MIXED_K = (8, 16, 32, 64)
DEFAULT_HARD_MARGIN = 0.2
DEFAULT_LAMBDA_DIST = 1.0
DEFAULT_LAMBDA_HARD = 0.2


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    paths = (
        root / "candidate_aware_qdic.py",
        root / "candidate_aware_qdic_trainer.py",
        root / "deepset_qdic.py",
        root / "dgsa_qdic.py",
        root / "distributional_identity.py",
        root / "distributional_losses.py",
        root / "qdic_features.py",
        root / "qdic_loader.py",
        root / "qdic_trainer.py",
        root / "query_distributional_calibrator.py",
    )
    return {str(path): sha256(path) for path in paths}


def _normalise_k_values(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in values}))
    if not result or any(value < 1 or value > 64 for value in result):
        raise ValueError("mixed_k_values must contain integers in [1,64]")
    return result


def _allowed_arrays(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    labels = np.asarray(arrays["labels"])
    allowed = np.asarray(arrays["supervision_allowed"], dtype=bool).copy()
    allowed &= np.asarray(arrays["candidate_base"], dtype=bool)
    allowed &= np.asarray(arrays["target_base"], dtype=bool)
    allowed &= np.isin(labels, (0, 1))
    return allowed


def _hard_negative_mask(
    features: np.ndarray,
    labels: np.ndarray,
    allowed: np.ndarray,
) -> np.ndarray:
    """Select the union of closest-prefilter and strongest-raw negatives."""
    mask = np.zeros(len(labels), dtype=bool)
    positive = np.flatnonzero(allowed & (labels == 1))
    negative = np.flatnonzero(allowed & (labels == 0))
    mask[positive] = True
    by_prefilter = sorted(negative.tolist(), key=lambda i: (float(features[i, 17]), int(i)))
    by_raw = sorted(negative.tolist(), key=lambda i: (-float(features[i, 5]), int(i)))
    selected: list[int] = []
    for first, second in zip(by_prefilter, by_raw):
        for item in (first, second):
            if item not in selected:
                selected.append(item)
        if len(selected) >= 16:
            break
    mask[np.asarray(selected[:16], dtype=np.int64)] = True
    return mask


def make_first_k_view(
    arrays: Mapping[str, np.ndarray],
    group: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return a causal first-K view, or skip it when no pos/neg survives."""
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    features = arrays["features"]
    labels = np.asarray(arrays["labels"])
    allowed_all = _allowed_arrays(arrays)
    start, end = (int(value) for value in offsets[group : group + 2])
    width = min(int(k), end - start)
    if width < 1:
        return None
    values = np.asarray(features[start : start + width], dtype=np.float32)
    target = np.asarray(labels[start : start + width], dtype=np.int64)
    allowed = allowed_all[start : start + width]
    if not bool((allowed & (target == 1)).any()) or not bool((allowed & (target == 0)).any()):
        # Crucially, the GT is never inserted into a K view after truncation.
        return None
    hard = _hard_negative_mask(values, target, allowed)
    return values, target, allowed, hard


def pad_views(
    views: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    device: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    rows = list(views)
    if not rows:
        raise ValueError("cannot pad an empty view batch")
    width = max(len(row[1]) for row in rows)
    features = np.zeros((len(rows), width, QDIC_RAW_DIM), dtype=np.float32)
    labels = np.full((len(rows), width), -1, dtype=np.int64)
    rank_mask = np.zeros((len(rows), width), dtype=bool)
    hard_mask = np.zeros((len(rows), width), dtype=bool)
    for index, (values, target, allowed, hard) in enumerate(rows):
        length = len(target)
        features[index, :length] = values
        labels[index, :length] = target
        rank_mask[index, :length] = allowed
        hard_mask[index, :length] = hard
    return (
        torch.from_numpy(features).to(device),
        torch.from_numpy(labels).to(device),
        torch.from_numpy(rank_mask).to(device),
        torch.from_numpy(hard_mask).to(device),
    )


def _model_details(model: nn.Module, features: Tensor, mask: Tensor) -> dict[str, Tensor]:
    try:
        output = model(features, mask=mask, return_diagnostics=True)
    except TypeError as exc:
        # The frozen A0 parent intentionally has no candidate-mask argument;
        # its padded rows are already excluded by the loss mask.
        if not isinstance(model, QueryDistributionalCalibrator):
            raise
        if "mask" not in str(exc):
            raise
        output = model(features, return_diagnostics=True)
    if isinstance(output, Mapping):
        return {str(key): value for key, value in output.items() if isinstance(value, Tensor)}
    return {"logit": output}


def _loss_components(
    details: Mapping[str, Tensor],
    labels: Tensor,
    rank_mask: Tensor,
    hard_mask: Tensor,
    *,
    lambda_dist: float,
    lambda_hard: float,
    hard_margin: float,
) -> dict[str, Tensor]:
    final = listwise_group_loss(details["logit"], labels, rank_mask)
    hard = hard_negative_margin_loss(
        details["logit"], labels, hard_mask, margin=hard_margin
    )
    distributional = details.get("distribution_logit")
    if distributional is None:
        distributional_value = final.new_zeros(())
    else:
        distributional_value = distributional_ranking_loss(
            distributional, labels, rank_mask
        )
    total = final + float(lambda_dist) * distributional_value + float(lambda_hard) * hard
    return {
        "total": total,
        "final": final,
        "distributional": distributional_value,
        "hard": hard,
    }


def _rank_summary(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    parent_logits: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    top1 = 0
    reciprocal = 0.0
    events = 0
    correction = 0
    regression = 0
    for row_index, (row, target, valid) in enumerate(zip(logits, labels, mask)):
        indices = np.flatnonzero(valid & np.isin(target, (0, 1)))
        positives = np.flatnonzero(valid & (target == 1))
        if len(indices) == 0 or len(positives) == 0:
            continue
        order = indices[np.argsort(-row[indices], kind="stable")]
        positive_ranks = [position + 1 for position, item in enumerate(order) if target[item] == 1]
        if not positive_ranks:
            continue
        events += 1
        top1 += int(target[order[0]] == 1)
        reciprocal += 1.0 / min(positive_ranks)
        if parent_logits is not None:
            parent_row = np.asarray(
                parent_logits[row_index] if parent_logits.ndim == 2 else parent_logits
            )
            parent_order = indices[np.argsort(-parent_row[indices], kind="stable")]
            parent_correct = bool(target[parent_order[0]] == 1)
            final_correct = bool(target[order[0]] == 1)
            correction += int((not parent_correct) and final_correct)
            regression += int(parent_correct and (not final_correct))
    result: dict[str, float | int | None] = {
        "events": events,
        "top1": top1 / max(events, 1),
        "mrr": reciprocal / max(events, 1),
        "correction": correction,
        "regression": regression,
        "net": correction - regression,
    }
    return result


def _architecture_model(parent: nn.Module, architecture: str) -> nn.Module:
    name = str(architecture).strip().upper()
    if name == "A0":
        if not isinstance(parent, QueryDistributionalCalibrator):
            raise TypeError("A0 requires the frozen V11 QueryDistributionalCalibrator")
        return parent
    if name in {"A0-D", "A0_D", "A0DIST", "A0-DISTLOSS"}:
        return DistributionalAuxiliaryQDIC(parent)
    if name in {"A1", "A1-DS-QDIC", "DS-QDIC"}:
        return DeepSetQDIC(parent)
    if name in {"A2", "A2-DGSA-QDIC", "DGSA-QDIC"}:
        return DistributionGuidedSetAttentionQDIC(parent)
    raise ValueError(f"unsupported candidate-aware architecture: {architecture!r}")


def _canonical_architecture_name(architecture: str) -> str:
    value = str(architecture).strip().upper()
    if value == "A0":
        return "A0"
    if value in {"A0-D", "A0_D", "A0DIST", "A0-DISTLOSS"}:
        return "A0-D"
    if value in {"A1", "A1-DS-QDIC", "DS-QDIC"}:
        return "A1-DS-QDIC"
    if value in {"A2", "A2-DGSA-QDIC", "DGSA-QDIC"}:
        return "A2-DGSA-QDIC"
    raise ValueError(f"unsupported candidate-aware architecture: {architecture!r}")


def _architecture_config(
    architecture_name: str,
    k_values: tuple[int, ...],
    hard_margin: float,
) -> dict[str, Any]:
    """Return the auditable layer contract for one candidate architecture."""
    result: dict[str, Any] = {
        "distribution_input_dim": 9,
        "distribution_token_dim": 32,
        "distribution_encoder": "9->32->ReLU->32->ReLU",
        "distribution_head": "34->32->ReLU->1",
        "legacy_input_dim": 24,
        "candidate_token_dim": 64,
        "mixed_k_values": list(k_values),
        "hard_margin": float(hard_margin),
        "parent_qdic": "frozen_v11_parent",
    }
    if architecture_name == "A0":
        result.update(
            {
                "mode": "current_parent_inference_only",
                "final_score": "parent_logit",
                "residual_head_zero_initialized": False,
            }
        )
    elif architecture_name == "A0-D":
        result.update(
            {
                "mode": "parent_final_plus_distributional_auxiliary_loss",
                "final_score": "parent_logit_unchanged",
                "residual_head_zero_initialized": False,
            }
        )
    elif architecture_name == "A1-DS-QDIC":
        result.update(
            {
                "mode": "candidate_aware_deepset_residual",
                "candidate_fusion": "67->64->ReLU->64",
                "set_pooling": ["masked_mean", "masked_max"],
                "residual_representation_dim": 320,
                "residual_head": "320->64->ReLU->32->ReLU->1",
                "residual_head_zero_initialized": True,
                "final_score": "parent_logit_plus_delta_set",
                "positional_encoding": False,
            }
        )
    elif architecture_name == "A2-DGSA-QDIC":
        result.update(
            {
                "mode": "distribution_guided_set_attention_residual",
                "candidate_fusion": "67->64->ReLU->64",
                "query_projection": "32->64",
                "key_projection": "32->64",
                "value_dim": 64,
                "attention": "MultiheadAttention(embed_dim=64,num_heads=4)",
                "attention_residual_norm": True,
                "feed_forward": "64->128->GELU->64",
                "residual_representation_dim": 192,
                "residual_head": "192->64->ReLU->32->ReLU->1",
                "residual_head_zero_initialized": True,
                "final_score": "parent_logit_plus_delta_attention",
                "positional_encoding": False,
            }
        )
    return result


def _make_training_batch(
    arrays: Mapping[str, np.ndarray],
    groups: list[int],
    *,
    k_values: tuple[int, ...],
    rng: np.random.Generator,
    batch_size: int = 32,
) -> tuple[list[tuple[Tensor, Tensor, Tensor, Tensor]], int]:
    batches: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    skipped = 0
    for start in range(0, len(groups), int(batch_size)):
        views = []
        for group in groups[start : start + int(batch_size)]:
            k = int(rng.choice(k_values))
            view = make_first_k_view(arrays, group, k)
            if view is not None:
                views.append(view)
            else:
                skipped += 1
        if views:
            # Device placement is applied by the caller so this helper can be
            # used by CPU protocol tests without CUDA initialization.
            batches.append(pad_views(views, device="cpu"))
    return batches, skipped


def _move_batch(batch: tuple[Tensor, Tensor, Tensor, Tensor], device: str):
    return tuple(value.to(device) for value in batch)


def _evaluate(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    groups: list[int],
    *,
    k_values: tuple[int, ...],
    device: str,
    lambda_dist: float,
    lambda_hard: float,
    hard_margin: float,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    component_values = {name: [] for name in ("final", "distributional", "hard")}
    final_top1: list[float] = []
    final_mrr: list[float] = []
    dist_top1: list[float] = []
    dist_mrr: list[float] = []
    corrections = 0
    regressions = 0
    skipped = 0
    view_count = 0
    with torch.inference_mode():
        for group in groups:
            for k in k_values:
                view = make_first_k_view(arrays, group, k)
                if view is None:
                    skipped += 1
                    continue
                batch = _move_batch(pad_views([view], device="cpu"), device)
                features, labels, rank_mask, hard_mask = batch
                details = _model_details(model, features, rank_mask)
                components = _loss_components(
                    details,
                    labels,
                    rank_mask,
                    hard_mask,
                    lambda_dist=lambda_dist,
                    lambda_hard=lambda_hard,
                    hard_margin=hard_margin,
                )
                losses.append(float(components["total"]))
                for name in component_values:
                    component_values[name].append(float(components[name]))
                final = details["logit"].detach().cpu().numpy()
                target = labels.detach().cpu().numpy()
                valid = rank_mask.detach().cpu().numpy()
                parent_output = details.get("parent_logit", details["logit"])
                parent_array = parent_output.detach().cpu().numpy()
                summary = _rank_summary(final, target, valid, parent_array)
                final_top1.append(float(summary["top1"]))
                final_mrr.append(float(summary["mrr"]))
                corrections += int(summary["correction"] or 0)
                regressions += int(summary["regression"] or 0)
                if "distribution_logit" in details:
                    dist = details["distribution_logit"].detach().cpu().numpy()
                    dist_summary = _rank_summary(dist, target, valid)
                    dist_top1.append(float(dist_summary["top1"]))
                    dist_mrr.append(float(dist_summary["mrr"]))
                view_count += 1
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "final_loss": float(np.mean(component_values["final"])) if losses else None,
        "distributional_loss": float(np.mean(component_values["distributional"])) if losses else None,
        "hard_loss": float(np.mean(component_values["hard"])) if losses else None,
        "final_top1": float(np.mean(final_top1)) if final_top1 else None,
        "final_mrr": float(np.mean(final_mrr)) if final_mrr else None,
        "correction": corrections,
        "regression": regressions,
        "net": corrections - regressions,
        "distribution_top1": float(np.mean(dist_top1)) if dist_top1 else None,
        "distribution_mrr": float(np.mean(dist_mrr)) if dist_mrr else None,
        "views": view_count,
        "skipped_views": skipped,
    }


def train_architecture(
    features_dir: str | Path,
    parent_checkpoint: str | Path,
    output: str | Path,
    *,
    architecture: str,
    device: str = "cpu",
    epochs: int = 12,
    seed: int = 0,
    lambda_dist: float | None = None,
    lambda_hard: float = DEFAULT_LAMBDA_HARD,
    hard_margin: float = DEFAULT_HARD_MARGIN,
    mixed_k_values: Iterable[int] = DEFAULT_MIXED_K,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    """Train one architecture under the frozen Base-only comparison contract."""
    architecture_name = _canonical_architecture_name(architecture)
    k_values = _normalise_k_values(mixed_k_values)
    if lambda_dist is None:
        lambda_dist = 0.0 if architecture_name == "A0" else DEFAULT_LAMBDA_DIST
    if float(lambda_dist) < 0.0 or float(lambda_hard) < 0.0:
        raise ValueError("loss weights must be non-negative")
    architecture_config = _architecture_config(
        architecture_name, k_values, float(hard_margin)
    )
    root = Path(features_dir).resolve()
    out = Path(output).resolve()
    parent_path = Path(parent_checkpoint).resolve()
    metadata, arrays = _load_feature_arrays(root)
    training_role = _training_role(metadata.get("split"))
    feature_config = validate_qdic_feature_config(metadata["feature_config"])
    train_groups, holdout_groups = build_training_groups(metadata, arrays)
    parent_artifact = load_qdic_checkpoint(parent_path, device=device)
    parent = parent_artifact.model
    # The V11 parent is the frozen reference scorer for every candidate-aware
    # architecture.  In particular, A0 is an inference-only current-parent
    # baseline; it must not silently become a second fine-tuned model.
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    parent.eval()
    model = _architecture_model(parent, architecture_name).to(device)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = (
        torch.optim.AdamW(
            trainable_parameters,
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        if trainable_parameters
        else None
    )
    rng = np.random.default_rng(int(seed))
    source_hashes = _source_hashes()
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    optimizer_steps = 0
    skipped_train_views_total = 0

    with (out / "metrics.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(int(epochs)):
            _memory_guard()
            model.train()
            # ``Module.train()`` recurses into the parent submodule.  Restore
            # its frozen/eval contract after changing the candidate branch.
            parent.eval()
            shuffled = rng.permutation(train_groups).tolist()
            # A fresh deterministic generator makes the mixed-K sequence
            # independent of model architecture while keeping the seed in the
            # receipt sufficient for exact reproduction.
            batch_list, skipped_train_views = _make_training_batch(
                arrays,
                shuffled,
                k_values=k_values,
                rng=np.random.default_rng(int(seed) + 1009 * (epoch + 1)),
            )
            skipped_train_views_total += int(skipped_train_views)
            train_components = {name: [] for name in ("total", "final", "distributional", "hard")}
            for batch in batch_list:
                features, labels, rank_mask, hard_mask = _move_batch(batch, device)
                details = _model_details(model, features, rank_mask)
                components = _loss_components(
                    details,
                    labels,
                    rank_mask,
                    hard_mask,
                    lambda_dist=float(lambda_dist),
                    lambda_hard=float(lambda_hard),
                    hard_margin=float(hard_margin),
                )
                loss = components["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite candidate-aware loss")
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if any(
                        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    ):
                        raise FloatingPointError("non-finite candidate-aware gradient")
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
                    optimizer.step()
                    optimizer_steps += 1
                for name in train_components:
                    train_components[name].append(float(components[name].detach()))
            train_metrics = {
                f"train_{name}": float(np.mean(values)) if values else None
                for name, values in train_components.items()
            }
            validation = _evaluate(
                model,
                arrays,
                holdout_groups,
                k_values=k_values,
                device=device,
                lambda_dist=float(lambda_dist),
                lambda_hard=float(lambda_hard),
                hard_margin=float(hard_margin),
            )
            row = {
                "epoch": epoch + 1,
                "optimizer_steps": optimizer_steps,
                "mixed_k_values": list(k_values),
                **train_metrics,
                **{f"holdout_{key}": value for key, value in validation.items()},
            }
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            current = validation["loss"]
            if current is not None and float(current) < best_loss:
                best_loss = float(current)
                best_epoch = epoch + 1
                torch.save(
                    {
                        "status": CANDIDATE_AWARE_STATUS,
                        "model_state": model.state_dict(),
                        "architecture_name": architecture_name,
                        "architecture_config": architecture_config,
                        "parent_v11_checkpoint": str(parent_path),
                        "parent_v11_checkpoint_hash": parent_artifact.checkpoint_hash,
                        "feature_names": list(QDIC_FEATURE_NAMES),
                        "feature_dim": QDIC_RAW_DIM,
                        "feature_config": feature_config,
                        "training_split": metadata["split"],
                        "protocol": "QDIC_V11_BASE_ONLY_CANDIDATE_AWARE_TRAINING",
                        "base_only_supervision": True,
                        "novel_gt_used": False,
                        "test_gt_used_for_optimizer": False,
                        "test_weights_used": False,
                        **training_role,
                        "train_video_ids": sorted({int(arrays["videos"][group]) for group in train_groups}),
                        "holdout_video_ids": sorted({int(arrays["videos"][group]) for group in holdout_groups}),
                        "lambda_dist": float(lambda_dist),
                        "lambda_hard": float(lambda_hard),
                        "seed": int(seed),
                        "epochs": int(epochs),
                        "optimizer": "AdamW",
                        "lr": float(lr),
                        "weight_decay": float(weight_decay),
                        "source_hashes": source_hashes,
                        "normalization_source": "parent_v11_checkpoint",
                        "tsa_reference": "conceptual_inspiration_only_no_runtime_dependency",
                    },
                    out / "best.pt",
                )

    checkpoint = out / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError("candidate-aware trainer did not produce best.pt")
    result: dict[str, Any] = {
        "status": "COMPLETED",
        "artifact": "candidate_aware_qdic_training",
        "architecture_name": architecture_name,
        "architecture_config": architecture_config,
        "parent_v11_checkpoint": str(parent_path),
        "parent_v11_checkpoint_hash": parent_artifact.checkpoint_hash,
        "protocol": "QDIC_V11_BASE_ONLY_CANDIDATE_AWARE_TRAINING",
        "training_split": metadata["split"],
        **training_role,
        "train_video_ids": sorted({int(arrays["videos"][group]) for group in train_groups}),
        "holdout_video_ids": sorted({int(arrays["videos"][group]) for group in holdout_groups}),
        "base_only_supervision": True,
        "novel_gt_used": False,
        "test_gt_used_for_optimizer": False,
        "test_weights_used": False,
        "feature_names": list(QDIC_FEATURE_NAMES),
        "feature_dim": QDIC_RAW_DIM,
        "feature_config": feature_config,
        "features": str(root / "features.json"),
        "features_hash": sha256(root / "features.json"),
        "checkpoint": str(checkpoint),
        "checkpoint_hash": sha256(checkpoint),
        "history": history,
        "best_epoch": best_epoch,
        "best_holdout_loss": best_loss,
        "device": str(device),
        "epochs": int(epochs),
        "seed": int(seed),
        "optimizer": "AdamW",
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "lambda_dist": float(lambda_dist),
        "lambda_hard": float(lambda_hard),
        "hard_margin": float(hard_margin),
        "mixed_k_values": list(k_values),
        "optimizer_steps": optimizer_steps,
        "trainable_parameter_count": sum(
            int(parameter.numel()) for parameter in trainable_parameters
        ),
        "parent_frozen": True,
        "skipped_train_views_total": skipped_train_views_total,
        "normalization_source": "parent_v11_checkpoint",
        "source_hashes": source_hashes,
        "parent_v11_provenance": parent_artifact.provenance,
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
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
    }
    _write_json(out / "training.json", result)
    return result


__all__ = [
    "CANDIDATE_AWARE_STATUS",
    "DEFAULT_HARD_MARGIN",
    "DEFAULT_LAMBDA_DIST",
    "DEFAULT_LAMBDA_HARD",
    "DEFAULT_MIXED_K",
    "make_first_k_view",
    "pad_views",
    "train_architecture",
]
