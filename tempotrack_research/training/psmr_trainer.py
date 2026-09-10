"""Real PSMR rank/reliability training and checkpointing."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import fields
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..analysis.partial_support import PartialSupportConfig, PartialSupportScorer
from ..models.memory_reliability import MemoryReliabilityCalibrator
from ..streaming.partial_support import build_memory_anchor
from ..streaming.psmr_dataset import VideoData, load_episodes


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _evidence(video: VideoData, rows: Sequence[int], max_gap: int = 60) -> torch.Tensor:
    anchor = build_memory_anchor(
        fragment_id="training", root_id=0, video_id=int(video.video_id), rows=list(rows),
        features=video.features, boxes_xyxy=video.boxes_xyxy, scores=video.scores,
        frames=video.frames, dedup_cos=0.95, capacity=64, max_gap=max_gap,
    )
    return torch.as_tensor(anchor.evidence, dtype=torch.float32)


def _candidate_score(
    scorer: PartialSupportScorer,
    calibrator: MemoryReliabilityCalibrator,
    video: VideoData,
    query_rows: Sequence[int],
    candidate_rows: Sequence[int],
    *,
    learned: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = next(calibrator.parameters()).device
    query = torch.as_tensor(video.features[list(query_rows)], dtype=torch.float32, device=dev)
    anchor = build_memory_anchor(
        fragment_id="training", root_id=0, video_id=int(video.video_id), rows=list(candidate_rows),
        features=video.features, boxes_xyxy=video.boxes_xyxy, scores=video.scores,
        frames=video.frames, dedup_cos=float(scorer.config.dedup_cos), capacity=int(scorer.config.memory_capacity), max_gap=int(scorer.config.max_gap),
    )
    memory = torch.as_tensor(anchor.features, dtype=torch.float32, device=dev)
    evidence = torch.as_tensor(anchor.evidence, dtype=torch.float32, device=dev)
    logits = calibrator(evidence).reshape(-1)
    if learned:
        reliability = calibrator.reliability(evidence)
        result = scorer(query, memory, memory_reliability=reliability, reliability_scale=calibrator.reliability_scale)
    else:
        result = scorer(query, memory)
    return result.score, logits


def train_psmr(
    *,
    episodes_path: str | Path,
    videos: Mapping[int, VideoData],
    run_root: str | Path,
    config: Mapping,
    seed: int = 0,
    device: str = "cuda:0",
    max_steps: int = 20_000,
    resume: str = "auto",
    input_hash: str = "",
) -> dict:
    """Train one real seed and save ``last.pt``/``best.pt`` only in run_root."""
    run_root = Path(run_root); run_root.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(episodes_path)
    if not episodes:
        raise ValueError("PSMR training requires at least one Base-only episode")
    torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))
    requested_device = str(device)
    actual_device = torch.device(requested_device if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    # The V9 YAML keeps frontend lifecycle knobs next to scorer knobs.  Only
    # the fields owned by PartialSupportConfig belong in this constructor;
    # retain the ignored names as provenance instead of silently treating
    # alpha_fast/alpha_slow as scorer parameters.
    partial_mapping = dict(config.get("partial_support", {}))
    scorer_fields = {item.name for item in fields(PartialSupportConfig)}
    ignored_partial_keys = sorted(set(partial_mapping) - scorer_fields)
    pconfig = PartialSupportConfig(**{key: value for key, value in partial_mapping.items() if key in scorer_fields})
    scorer = PartialSupportScorer(pconfig, beta=0.0).to(actual_device)
    calibrator = MemoryReliabilityCalibrator().to(actual_device)
    training_config = dict(config.get("training", {}))
    algorithm_revision = str(training_config.get("algorithm_revision", "per_anchor_v8"))
    checkpoint_artifact = str(training_config.get("checkpoint_artifact", "psmr_v8_checkpoint"))
    checkpoint_steps = {int(value) for value in training_config.get("checkpoint_steps", [])}
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=float(training_config.get("lr", 1e-3)), weight_decay=float(training_config.get("weight_decay", 1e-4)))
    config_hash = _sha(config)
    start_step = 0; best_loss = float("inf")
    last_path = run_root / "last.pt"
    if resume in {"auto", "strict"} and last_path.exists():
        checkpoint = torch.load(last_path, map_location=actual_device)
        if checkpoint.get("algorithm_revision") != algorithm_revision or checkpoint.get("artifact") != checkpoint_artifact:
            raise ValueError(f"refusing to resume an incompatible PSMR checkpoint; expected {algorithm_revision}/{checkpoint_artifact}")
        if checkpoint.get("config_hash") != config_hash or checkpoint.get("input_hash", input_hash) != input_hash:
            if resume == "strict":
                raise ValueError("existing PSMR checkpoint is incompatible with config/input hash")
        else:
            calibrator.load_state_dict(checkpoint["model_state"]); optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_step = int(checkpoint.get("optimizer_steps", checkpoint.get("step", 0))); best_loss = float(checkpoint.get("best_loss", best_loss))
            if start_step >= int(max_steps):
                result = {"status": "COMPLETED", "requested_steps": int(max_steps), "optimizer_steps": start_step, "algorithm_revision": algorithm_revision, "base_only_supervision": True, "official_validation_used": False, "checkpoint": str(last_path), "best_checkpoint": str(run_root / "best.pt"), "step_checkpoints": {str(step): str(run_root / f"step_{step}.pt") for step in sorted(checkpoint_steps) if (run_root / f"step_{step}.pt").exists()}, "step": start_step, "device": str(actual_device), "config_hash": config_hash, "input_hash": input_hash, "ignored_partial_support_keys": ignored_partial_keys}
                (run_root / "train_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                return result
    metrics_path = run_root / "metrics.jsonl"
    log_mode = "a" if start_step else "w"
    handle = metrics_path.open(log_mode, encoding="utf-8")
    try:
        generator = np.random.default_rng(int(seed) + start_step)
        optimizer_steps = int(start_step)
        attempts = 0
        while optimizer_steps < int(max_steps):
            attempts += 1
            if attempts > max(1000, int(max_steps) * 1000):
                raise RuntimeError("unable to sample a valid rank episode with masked anchor supervision")
            episode = episodes[int(generator.integers(0, len(episodes)))]
            video = videos[int(episode["video_id"])]
            scores = []; rel_logits = []; labels = []; anchor_targets = []; anchor_masks = []
            for candidate in episode["candidates"]:
                score, rel_logit = _candidate_score(scorer, calibrator, video, episode["query_rows"], candidate["rows"], learned=True)
                if torch.isfinite(score):
                    scores.append(score); rel_logits.append(rel_logit); labels.append(float(candidate["label"]))
                    labels_for_anchor = candidate.get("anchor_labels")
                    masks_for_anchor = candidate.get("anchor_label_mask")
                    if labels_for_anchor is None or masks_for_anchor is None or len(labels_for_anchor) != len(rel_logit) or len(masks_for_anchor) != len(rel_logit):
                        raise ValueError("V8 episode is missing aligned per-anchor labels/masks")
                    anchor_targets.append(torch.as_tensor(labels_for_anchor, dtype=torch.float32, device=actual_device))
                    anchor_masks.append(torch.as_tensor(masks_for_anchor, dtype=torch.bool, device=actual_device))
            if not scores or sum(labels) < 1 or sum(label == 0 for label in labels) < 1:
                continue
            score_tensor = torch.stack(scores)
            label_tensor = torch.as_tensor(labels, dtype=torch.float32, device=actual_device)
            positive = torch.where(label_tensor > 0.5)[0][0]
            rank_loss = -F.log_softmax(score_tensor / float(config.get("training", {}).get("temperature", 0.07)), dim=0)[positive]
            logits = torch.cat(rel_logits)
            rel_targets = torch.cat(anchor_targets)
            rel_mask = torch.cat(anchor_masks)
            if bool(rel_mask.any()):
                positive_count = rel_targets[rel_mask].sum().clamp_min(1.0)
                negative_count = (rel_mask.sum() - rel_targets[rel_mask].sum()).clamp_min(1.0)
                pos_weight = negative_count / positive_count
                rel_loss = F.binary_cross_entropy_with_logits(logits[rel_mask], rel_targets[rel_mask], pos_weight=pos_weight)
            else:
                rel_loss = logits.sum() * 0.0
            loss = rank_loss + 0.5 * rel_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), float(config.get("training", {}).get("grad_clip", 1.0)))
            optimizer.step()
            optimizer_steps += 1
            record = {"step": optimizer_steps, "loss": float(loss.detach()), "rank_loss": float(rank_loss.detach()), "reliability_loss": float(rel_loss.detach()), "episode_id": int(episode["episode_id"]), "anchor_supervision_count": int(rel_mask.sum())}
            handle.write(json.dumps(record) + "\n")
            if optimizer_steps % int(training_config.get("log_every", 100)) == 0:
                handle.flush()
            current = float(loss.detach())
            checkpoint_payload = {
                "schema_version": 3 if algorithm_revision.endswith("v9") else 2,
                "artifact": checkpoint_artifact,
                "algorithm_revision": algorithm_revision,
                "optimizer_steps": optimizer_steps,
                "step": optimizer_steps,
                "seed": int(seed),
                "model_state": calibrator.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config_hash": config_hash,
                "input_hash": input_hash,
                "partial_support_config": {key: value for key, value in partial_mapping.items() if key in scorer_fields},
                "ignored_partial_support_keys": ignored_partial_keys,
                "loss": record,
                "best_loss": best_loss,
            }
            if current < best_loss:
                best_loss = current
                checkpoint_payload["best_loss"] = best_loss
                torch.save(checkpoint_payload, run_root / "best.pt")
            if optimizer_steps % int(training_config.get("save_every", 500)) == 0 or optimizer_steps == int(max_steps):
                torch.save(checkpoint_payload, last_path)
            if optimizer_steps in checkpoint_steps:
                torch.save(checkpoint_payload, run_root / f"step_{optimizer_steps}.pt")
    finally:
        handle.close()
    result = {"status": "COMPLETED", "requested_steps": int(max_steps), "optimizer_steps": int(optimizer_steps), "algorithm_revision": algorithm_revision, "base_only_supervision": True, "official_validation_used": False, "checkpoint": str(last_path), "best_checkpoint": str(run_root / "best.pt"), "step_checkpoints": {str(step): str(run_root / f"step_{step}.pt") for step in sorted(checkpoint_steps) if (run_root / f"step_{step}.pt").exists()}, "step": int(optimizer_steps), "best_loss": best_loss, "device": str(actual_device), "config_hash": config_hash, "input_hash": input_hash, "ignored_partial_support_keys": ignored_partial_keys}
    (run_root / "train_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = ["train_psmr"]
