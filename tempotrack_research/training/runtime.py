"""Method dispatch for real NPZ episode tensors.

The runner refuses to manufacture tensors.  Once a train feature export is
available, this module gives every registered method the same optimizer,
checkpoint, metric, and resume plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from ..config import file_hash
from ..evaluation.result_writer import append_jsonl
from ..losses.predictive import predictive_memory_loss
from ..models.continuation_flow import ContinuationFlowModel
from ..models.edit_policy import EditPolicy
from ..models.graph_diffusion import GraphDiffusionMatcher
from ..models.graph_flow import GraphFlowMatcher
from ..models.identity_predictor import JEPAIdentityLinker, PairMetricLinker
from ..memory.predictive_dual import PredictiveDualMemory
from .checkpoint import AtomicCheckpoint
from .engine import TrainConfig, TrainingEngine, seed_everything
from .rl_trainer import behavior_cloning_step, ppo_step


class DataUnavailable(RuntimeError):
    pass


def _tensor(value: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(np.asarray(value)).to(device)


def _batch_from_npz(path: Path, device: torch.device) -> dict[str, Tensor]:
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise DataUnavailable(f"cannot read safe train episode NPZ: {path}: {exc}") from exc
    with archive:
        return {name: _tensor(archive[name], device) for name in archive.files}


def _batch_path(repo: Path, method: str, frontend: str) -> Path:
    return repo / "outputs" / "research" / "training_inputs" / frontend / f"{method}.npz"


def _make_model(method: str, batch: Mapping[str, Tensor]) -> nn.Module:
    if method == "s1_jepa":
        dim = int(batch["context_appearance"].shape[-1])
        return JEPAIdentityLinker(dim)
    if method == "ordinary_metric":
        dim = int(batch["left_appearance"].shape[-1])
        return PairMetricLinker(dim)
    if method == "s2_state_fm":
        return ContinuationFlowModel(int(batch["condition"].shape[-1]), int(batch["target_state"].shape[-1]))
    if method == "s3_graph_fm":
        return GraphFlowMatcher(int(batch["node_features"].shape[-1]), int(batch["edge_features"].shape[-1]))
    if method == "s4_graph_diffusion":
        return GraphDiffusionMatcher(int(batch["node_features"].shape[-1]), int(batch["edge_features"].shape[-1]))
    if method == "s5_rl_edit":
        return EditPolicy(int(batch["node_features"].shape[-1]), int(batch["edge_features"].shape[-1]))
    if method == "predictive_dual":
        return PredictiveDualMemory(int(batch["history_state"].shape[-1]), int(batch["observation"].shape[-1]), int(batch["causal_evidence"].shape[-1]))
    raise ValueError(f"No runtime trainer registered for {method}")


def _loss_for(method: str, model: nn.Module, batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    if method == "s1_jepa":
        labels = {name: value for name, value in batch.items() if name in {"positive"}}
        return model.compute_loss(batch, labels)
    if method == "ordinary_metric":
        left = model.encoder(batch["left_appearance"], batch["left_geometry"], batch["left_time"], batch.get("left_valid"))
        right = model.encoder(batch["right_appearance"], batch["right_geometry"], batch["right_time"], batch.get("right_valid"))
        score = model.score_pair(left, right)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(score, batch["same_identity"].to(score.dtype))
        return {"total": loss, "metric_bce": loss.detach()}
    if method == "s2_state_fm":
        return model.compute_loss(batch["source_state"], batch["target_state"], batch["condition"], batch.get("exists"))
    if method == "s3_graph_fm":
        return model.compute_loss(batch["target_graph"], batch["node_features"], batch["edge_features"], batch["edge_index"], batch["edge_valid"], batch.get("node_valid"), batch.get("initial_graph"))
    if method == "s4_graph_diffusion":
        return model.compute_loss(batch["target_graph"], batch["node_features"], batch["edge_features"], batch["edge_index"], batch["edge_valid"], batch.get("initial_graph", torch.zeros_like(batch["target_graph"])), batch.get("node_valid"))
    if method == "s5_rl_edit":
        if batch.get("stage", torch.zeros((), device=next(model.parameters()).device)).item() == 0:
            return behavior_cloning_step(model, batch)
        return ppo_step(model, batch)
    if method == "predictive_dual":
        state = model.initialize(batch["prototype"])
        updated, rates = model.update(state, batch["observation"], batch["history_state"], batch["causal_evidence"])
        output = {"fast": updated.fast, "slow": updated.slow, "rates": rates}
        return predictive_memory_loss(output["fast"], output["slow"], batch["future_embedding"], batch["same_identity"], rates, batch.get("reliability_logit"), batch.get("reliability"))
    raise ValueError(f"No loss registered for {method}")


def run_available_training(repo: str | Path, method: str, frontend: str, profile: str, seed: int, resume: str = "auto", device: str | None = None) -> dict[str, Any]:
    repo = Path(repo).resolve()
    path = _batch_path(repo, method, frontend)
    if not path.exists():
        raise DataUnavailable(f"missing train episode tensor artifact: {path}")
    seed_everything(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch = _batch_from_npz(path, selected_device)
    model = _make_model(method, batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)
    steps = {"trial": 3000, "full": 100000}.get(profile, 3000)
    if method in {"s3_graph_fm", "s4_graph_diffusion"}:
        steps = 120000 if profile == "full" else 3000
    if method == "predictive_dual":
        steps = 60000 if profile == "full" else 3000
    if method == "s5_rl_edit":
        steps = 30000 if profile == "full" else 3000
    run_dir = repo / "outputs" / "research" / "runs" / f"{frontend}_{method}_{profile}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = AtomicCheckpoint(run_dir / "last.pt")
    metadata = {"method": method, "frontend": frontend, "profile": profile, "seed": seed, "input_hash": file_hash(path)}
    engine = TrainingEngine(model, optimizer, TrainConfig(max_steps=steps, validate_every=2000, save_every=5000), selected_device)
    if resume == "auto" and checkpoint.path.exists():
        payload = checkpoint.load(model, optimizer, scheduler=engine.scheduler, scaler=engine.scaler, expected={"method": method, "frontend": frontend, "input_hash": metadata["input_hash"]}, map_location=selected_device)
        saved_metadata = payload.get("metadata", {})
        engine.global_step = int(saved_metadata.get("global_step", 0))
        engine.optimizer_steps = int(saved_metadata.get("optimizer_steps", engine.global_step))

    def batches():
        while True:
            yield batch

    def on_step(step: int, metrics: Mapping[str, float]) -> None:
        append_jsonl({"method": method, "frontend": frontend, "profile": profile, "seed": seed, "step": step, **dict(metrics)}, repo / "reports" / "metrics.jsonl")
        if step == 1 or step % 5000 == 0 or step == steps:
            checkpoint.save(model, optimizer, metadata={**metadata, "global_step": step, "optimizer_steps": engine.optimizer_steps})
        if method == "s1_jepa" and metrics.get("optimizer_step_success", 0) > 0:
            model.update_target(True)

    result = engine.run(batches(), lambda current: _loss_for(method, model, current), on_step)
    checkpoint.save(model, optimizer, metadata={**metadata, "global_step": engine.global_step, "optimizer_steps": engine.optimizer_steps})
    return {"status": "COMPLETED", "run_dir": str(run_dir), "checkpoint": str(checkpoint.path), **result}
