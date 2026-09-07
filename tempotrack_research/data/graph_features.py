"""Shared graph feature construction for training and deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ..schemas import GraphInputs, SegmentClock, SegmentInputs
from .tensorization import TrajectoryTensorizer, TransformSpec


@dataclass(frozen=True)
class GraphFeatureSpec:
    node_dim_extra: int = 5
    edge_dim: int = 11
    time_scale: float = 1.0


def _last_valid(value: Tensor, valid: Tensor) -> Tensor:
    indexes = torch.nonzero(valid.bool(), as_tuple=False).flatten()
    if indexes.numel() == 0:
        raise ValueError("graph segment has no valid observations")
    return value[indexes[-1]]


class GraphFeaturizer:
    def __init__(self, tensorizer: TrajectoryTensorizer | None = None, spec: GraphFeatureSpec | None = None):
        self.tensorizer = tensorizer or TrajectoryTensorizer()
        self.spec = spec or GraphFeatureSpec(time_scale=self.tensorizer.spec.time_scale)

    def build(self, segments: Sequence[SegmentInputs], clocks: Sequence[SegmentClock], candidate_edges: Tensor, initial_selected: Tensor) -> GraphInputs:
        if len(segments) != len(clocks):
            raise ValueError("segment/clock count mismatch")
        if candidate_edges.ndim == 2 and candidate_edges.shape[0] == 2:
            edges = candidate_edges.long()
        else:
            edges = candidate_edges.long().reshape(2, -1)
        graph_origin = min(float(clock.first_time) for clock in clocks) if clocks else 0.0
        node_values = [self.tensorizer.encode_graph_node(segment, clock, graph_origin, TransformSpec(time_scale=self.spec.time_scale)) for segment, clock in zip(segments, clocks)]
        if node_values:
            node = torch.stack(node_values)
        else:
            node = torch.empty((0, 0), dtype=torch.float32)
        edge_values: list[Tensor] = []
        for source, target in edges.t().tolist():
            if not (0 <= int(source) < len(segments) and 0 <= int(target) < len(segments)):
                raise ValueError("graph edge endpoint out of bounds")
            left = node[int(source), -5:]
            right = node[int(target), -5:]
            gap = (float(clocks[int(target)].first_time) - float(clocks[int(source)].last_time)) / float(self.spec.time_scale)
            edge_values.append(torch.cat((left, right, node.new_tensor([gap]))))
        edge_features = torch.stack(edge_values) if edge_values else node.new_empty((0, self.spec.edge_dim))
        edge_valid = torch.ones((edges.shape[1],), dtype=torch.bool, device=node.device)
        selected = initial_selected.to(node.device).bool().reshape(-1)
        if selected.numel() != edges.shape[1]:
            raise ValueError("initial graph and candidate edge counts disagree")
        node_times = node[:, -1] if node.numel() else node.new_empty((0,))
        return GraphInputs(node, edge_features, edges, torch.ones((node.shape[0],), dtype=torch.bool, device=node.device), edge_valid, selected.float(), node_times)


__all__ = ["GraphFeatureSpec", "GraphFeaturizer"]
