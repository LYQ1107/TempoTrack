"""S3/S4 objective dispatch."""

from __future__ import annotations

from typing import Mapping

from torch import Tensor, nn


def compute_graph_loss(model: nn.Module, batch: Mapping[str, Tensor], diffusion: bool = False) -> dict[str, Tensor]:
    if diffusion:
        return model.compute_loss(batch["target_graph"], batch["node_features"], batch["edge_features"], batch["edge_index"], batch["edge_valid"], batch["initial_graph"], batch.get("node_valid"))
    return model.compute_loss(batch["target_graph"], batch["node_features"], batch["edge_features"], batch["edge_index"], batch["edge_valid"], batch.get("node_valid"), batch.get("initial_graph"))
