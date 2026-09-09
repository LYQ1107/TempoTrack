"""Real PSMR rank/reliability training and checkpointing."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..analysis.partial_support import PartialSupportConfig, PartialSupportScorer
from ..models.memory_reliability import MemoryReliabilityCalibrator
from ..streaming.psmr_dataset import VideoData, load_episodes


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _evidence(video: VideoData, rows: Sequence[int], max_gap: int = 60) -> torch.Tensor:
    rows = list(rows)
    if not rows:
        return torch.zeros((1, 7), dtype=torch.float32)
    first = int(rows[0]); last = int(rows[-1])
    feat = video.features[rows]
    if len(feat) > 1:
        fast = feat[-min(4, len(feat)):].mean(axis=0)
        slow = feat.mean(axis=0)
        norm = lambda a: a / max(float(np.linalg.norm(a)), 1e-6)
        fast_cos = float(np.dot(norm(feat[0]), norm(fast)))
        slow_cos = float(np.dot(norm(feat[0]), norm(slow)))
        agreement = float(np.dot(norm(fast), norm(slow)))
    else:
        fast_cos = slow_cos = 0.0
        agreement = 1.0
    boxes = video.boxes_xyxy[rows]
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6)
    area_change = float(np.clip(np.log(area[-1] / area[0]), -2.0, 2.0) / 2.0)
    return torch.as_tensor([[float(video.scores[first]), fast_cos, slow_cos, agreement, min(1.0, len(rows) / 100.0), min(1.0, max(0, last - first) / max_gap), area_change]], dtype=torch.float32)


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
    memory = torch.as_tensor(video.features[list(candidate_rows)][-64:], dtype=torch.float32, device=dev)
    evidence = _evidence(video, candidate_rows).to(dev)
    logit = calibrator(evidence).reshape(())
    if learned:
        reliability = calibrator.reliability(evidence).expand(memory.shape[0])
        result = scorer(query, memory, memory_reliability=reliability)
    else:
        result = scorer(query, memory)
    return result.score, logit


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
    pconfig = PartialSupportConfig(**dict(config.get("partial_support", {})))
    scorer = PartialSupportScorer(pconfig, beta=float(config.get("training", {}).get("reliability_beta", 0.25))).to(actual_device)
    calibrator = MemoryReliabilityCalibrator().to(actual_device)
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=float(config.get("training", {}).get("lr", 1e-3)), weight_decay=float(config.get("training", {}).get("weight_decay", 1e-4)))
    config_hash = _sha(config)
    start_step = 0; best_loss = float("inf")
    last_path = run_root / "last.pt"
    if resume in {"auto", "strict"} and last_path.exists():
        checkpoint = torch.load(last_path, map_location=actual_device)
        if checkpoint.get("config_hash") != config_hash or checkpoint.get("input_hash", input_hash) != input_hash:
            if resume == "strict":
                raise ValueError("existing PSMR checkpoint is incompatible with config/input hash")
        else:
            calibrator.load_state_dict(checkpoint["model_state"]); optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_step = int(checkpoint["step"]); best_loss = float(checkpoint.get("best_loss", best_loss))
            if start_step >= int(max_steps):
                return {"status": "REUSED", "checkpoint": str(last_path), "step": start_step, "device": str(actual_device)}
    metrics_path = run_root / "metrics.jsonl"
    log_mode = "a" if start_step else "w"
    handle = metrics_path.open(log_mode, encoding="utf-8")
    try:
        generator = np.random.default_rng(int(seed) + start_step)
        for step in range(start_step + 1, int(max_steps) + 1):
            episode = episodes[int(generator.integers(0, len(episodes)))]
            video = videos[int(episode["video_id"])]
            scores = []; rel_logits = []; labels = []
            for candidate in episode["candidates"]:
                score, rel_logit = _candidate_score(scorer, calibrator, video, episode["query_rows"], candidate["rows"], learned=True)
                if torch.isfinite(score):
                    scores.append(score); rel_logits.append(rel_logit); labels.append(float(candidate["label"]))
            if not scores or sum(labels) < 1 or sum(label == 0 for label in labels) < 1:
                continue
            score_tensor = torch.stack(scores)
            label_tensor = torch.as_tensor(labels, dtype=torch.float32, device=actual_device)
            positive = torch.where(label_tensor > 0.5)[0][0]
            rank_loss = -F.log_softmax(score_tensor / float(config.get("training", {}).get("temperature", 0.07)), dim=0)[positive]
            logits = torch.stack(rel_logits)
            pos_weight = torch.where(label_tensor > 0.5, (label_tensor.numel() - label_tensor.sum()).clamp_min(1) / label_tensor.sum().clamp_min(1), torch.ones_like(label_tensor))
            rel_loss = F.binary_cross_entropy_with_logits(logits, label_tensor, weight=pos_weight)
            loss = rank_loss + 0.5 * rel_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), float(config.get("training", {}).get("grad_clip", 1.0)))
            optimizer.step()
            record = {"step": step, "loss": float(loss.detach()), "rank_loss": float(rank_loss.detach()), "reliability_loss": float(rel_loss.detach()), "episode_id": int(episode["episode_id"])}
            handle.write(json.dumps(record) + "\n")
            if step % int(config.get("training", {}).get("log_every", 100)) == 0:
                handle.flush()
            current = float(loss.detach())
            if current < best_loss:
                best_loss = current
                torch.save({"schema_version": 1, "artifact": "psmr_v7_checkpoint", "step": step, "seed": int(seed), "model_state": calibrator.state_dict(), "optimizer_state": optimizer.state_dict(), "config_hash": config_hash, "input_hash": input_hash, "partial_support_config": dict(config.get("partial_support", {})), "loss": record, "best_loss": best_loss}, run_root / "best.pt")
            if step % int(config.get("training", {}).get("save_every", 500)) == 0 or step == int(max_steps):
                torch.save({"schema_version": 1, "artifact": "psmr_v7_checkpoint", "step": step, "seed": int(seed), "model_state": calibrator.state_dict(), "optimizer_state": optimizer.state_dict(), "config_hash": config_hash, "input_hash": input_hash, "partial_support_config": dict(config.get("partial_support", {})), "loss": record, "best_loss": best_loss}, last_path)
    finally:
        handle.close()
    return {"status": "COMPLETED", "checkpoint": str(last_path), "best_checkpoint": str(run_root / "best.pt"), "step": int(max_steps), "best_loss": best_loss, "device": str(actual_device), "config_hash": config_hash, "input_hash": input_hash}


__all__ = ["train_psmr"]
