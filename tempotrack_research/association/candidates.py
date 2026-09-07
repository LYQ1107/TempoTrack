"""One detector-independent candidate graph for every backend."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from ..schemas import CandidateGraph


def _get(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _boundary(item: Any, name: str, index: int, default: np.ndarray) -> np.ndarray:
    value = _get(item, name, None)
    if value is None:
        return default
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array[:4]
    if not len(array):
        return default
    return array[index, :4]


def _boundary_app(item: Any, index: int) -> np.ndarray | None:
    value = _get(item, "appearance", _get(item, "embeds", None))
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array
    if not len(array):
        return None
    return array[index]


def build_candidate_graph(tracklets: Iterable[Any], max_gap: int | float = 90, top_k: int | None = 20, with_cats: bool = False) -> CandidateGraph:
    """Build legal temporal candidates without using GT identities.

    ``top_k`` ranks only by deployment-visible boundary appearance/geometry;
    it does not inspect labels or validation outcomes.
    """
    items = list(tracklets)
    if not items:
        return CandidateGraph(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_features=torch.empty((0, 6), dtype=torch.float32),
            valid=torch.empty((0,), dtype=torch.bool),
            metadata={"max_gap": max_gap, "top_k": top_k, "with_cats": with_cats, "candidate_count": 0, "node_count": 0, "pruning_uses_gt": False},
        )

    # Candidate ranking is a deployment-visible operation, but the old
    # implementation normalized and multiplied one pair at a time in Python.
    # TAO videos contain hundreds/thousands of frontend fragments, so that
    # scalar torch call overhead dominated the actual ranking.  Precompute
    # boundary fields and score each source against all legal temporal targets
    # with NumPy; the ordering/tie rule and the emitted six edge features are
    # unchanged.
    count = len(items)
    video_ids = np.asarray([int(_get(item, "video_id", -1)) for item in items], dtype=np.int64)
    first_frames = np.asarray([float(_get(item, "first_frame", _get(item, "frame", 0))) for item in items], dtype=np.float64)
    last_frames = np.asarray([float(_get(item, "last_frame", _get(item, "frame", 0))) for item in items], dtype=np.float64)
    first_boxes = np.asarray([_boundary(item, "bboxes", 0, _boundary(item, "bbox", 0, np.asarray([0, 0, 1, 1], dtype=np.float32))) for item in items], dtype=np.float32)
    last_boxes = np.asarray([_boundary(item, "bboxes", -1, _boundary(item, "bbox", -1, np.asarray([0, 0, 1, 1], dtype=np.float32))) for item in items], dtype=np.float32)
    first_centers = (first_boxes[:, :2] + first_boxes[:, 2:]) / 2.0
    last_centers = (last_boxes[:, :2] + last_boxes[:, 2:]) / 2.0
    first_width = np.maximum(first_boxes[:, 2] - first_boxes[:, 0], 1e-3)
    first_height = np.maximum(first_boxes[:, 3] - first_boxes[:, 1], 1e-3)
    last_width = np.maximum(last_boxes[:, 2] - last_boxes[:, 0], 1e-3)
    last_height = np.maximum(last_boxes[:, 3] - last_boxes[:, 1], 1e-3)
    raw_apps = [_boundary_app(item, -1) for item in items]
    app_dim = next((int(value.size) for value in raw_apps if value is not None and value.size), None)
    app_matrix: np.ndarray | None = None
    if app_dim is not None and all(value is not None and int(value.size) == app_dim for value in raw_apps):
        app_matrix = np.asarray([np.asarray(value, dtype=np.float32).reshape(-1) for value in raw_apps], dtype=np.float32)
        norms = np.linalg.norm(app_matrix, axis=1, keepdims=True)
        app_matrix = app_matrix / np.maximum(norms, 1e-12)
    categories = [_get(item, "category_id", None) for item in items]
    candidates: list[tuple[int, int]] = []
    features: list[list[float]] = []
    all_indices = np.arange(count, dtype=np.int64)
    for i in range(count):
        gap = first_frames - last_frames[i]
        mask = (all_indices != i) & (video_ids == video_ids[i]) & (gap > 0.0) & (gap <= float(max_gap))
        if with_cats:
            mask &= np.asarray([categories[j] == categories[i] for j in range(count)], dtype=bool)
        target_indices = all_indices[mask]
        if target_indices.size == 0:
            continue
        center_distance = np.linalg.norm(first_centers[target_indices] - last_centers[i], axis=1)
        if app_matrix is None:
            app = np.zeros(target_indices.shape[0], dtype=np.float32)
        else:
            # Boundary appearance is already deployment-visible frozen MASA
            # feature content; no labels or validation data enter this score.
            app = app_matrix[target_indices] @ app_matrix[i]
        rank_score = gap[target_indices] + center_distance - 10.0 * app
        # np.lexsort uses the last key as primary, so rank score is primary
        # and the original target index is the deterministic tie breaker.
        order = np.lexsort((target_indices, rank_score))
        if top_k is not None:
            order = order[: int(top_k)]
        for position in order.tolist():
            target = int(target_indices[position])
            target_gap = float(gap[target])
            target_app = float(app[position])
            target_center = float(center_distance[position])
            edge_features = [
                target_gap,
                target_app,
                target_center,
                float(np.log(first_width[target] / last_width[i])),
                float(np.log(first_height[target] / last_height[i])),
                float(np.log((first_width[target] * first_height[target]) / (last_width[i] * last_height[i]))),
            ]
            candidates.append((i, target))
            features.append(edge_features)
    edge_index = torch.tensor(candidates, dtype=torch.long).t().contiguous() if candidates else torch.empty((2, 0), dtype=torch.long)
    edge_features = torch.tensor(features, dtype=torch.float32) if features else torch.empty((0, 6), dtype=torch.float32)
    return CandidateGraph(edge_index=edge_index, edge_features=edge_features, valid=torch.ones(edge_index.shape[1], dtype=torch.bool), metadata={"max_gap": max_gap, "top_k": top_k, "with_cats": with_cats, "candidate_count": int(edge_index.shape[1]), "node_count": int(count), "pruning_uses_gt": False})


def temporal_graph_windows(items: Iterable[Any], *, max_nodes: int = 96, overlap: int = 16) -> list[np.ndarray]:
    """Return deterministic, deployment-visible node windows for large graphs.

    Candidate ranking is still performed once on the complete frontend view.
    This helper only partitions that already-ranked graph for joint message
    passing.  Windows are ordered by first observation time and frontend local
    id, never by GT identity.  The final window is forced to contain the last
    node so every frontend fragment is covered, while neighbouring windows
    share ``overlap`` context nodes.
    """
    values = list(items)
    max_nodes = int(max_nodes)
    overlap = int(overlap)
    if max_nodes < 2:
        raise ValueError("graph window max_nodes must be at least 2")
    if overlap < 0 or overlap >= max_nodes:
        raise ValueError("graph window overlap must satisfy 0 <= overlap < max_nodes")
    if not values:
        return []
    order = sorted(
        range(len(values)),
        key=lambda index: (
            float(_get(values[index], "first_frame", _get(values[index], "frame", 0))),
            int(_get(values[index], "local_id", index)),
            int(index),
        ),
    )
    if len(order) <= max_nodes:
        return [np.asarray(order, dtype=np.int64)]
    stride = max_nodes - overlap
    starts = list(range(0, len(order) - max_nodes + 1, stride))
    final_start = len(order) - max_nodes
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return [np.asarray(order[start : start + max_nodes], dtype=np.int64) for start in starts]


def slice_candidate_graph(graph: CandidateGraph, node_indices: Iterable[int]) -> tuple[CandidateGraph, np.ndarray]:
    """Slice a global candidate graph to one node window.

    The returned second value is the mapping from local edge positions back to
    the global edge positions.  Keeping this mapping explicit lets deployment
    score overlapping windows and run one final global path-cover projection.
    """
    nodes = np.asarray(list(node_indices), dtype=np.int64).reshape(-1)
    if len(set(int(value) for value in nodes.tolist())) != len(nodes):
        raise ValueError("graph window contains duplicate node indices")
    if nodes.size and (int(nodes.min()) < 0 or int(nodes.max()) >= int(graph.metadata.get("node_count", max(nodes) + 1))):
        # ``node_count`` is optional metadata.  The explicit upper-bound check
        # below uses the edge index and therefore also handles old graphs.
        edge_values = graph.edge_index.detach().cpu().numpy()
        max_endpoint = int(edge_values.max(initial=-1))
        if int(nodes.max()) > max_endpoint:
            raise ValueError("graph window node index is outside candidate graph")
    edge_values = graph.edge_index.detach().cpu().numpy().reshape(2, -1)
    node_to_local = {int(value): index for index, value in enumerate(nodes.tolist())}
    global_positions = [
        position
        for position, (source, target) in enumerate(edge_values.T.tolist())
        if int(source) in node_to_local and int(target) in node_to_local
    ]
    positions = np.asarray(global_positions, dtype=np.int64)
    if positions.size:
        local_edges = np.asarray(
            [[node_to_local[int(edge_values[0, position])], node_to_local[int(edge_values[1, position])]] for position in positions.tolist()],
            dtype=np.int64,
        ).T
        edge_features = graph.edge_features.index_select(0, torch.as_tensor(positions, dtype=torch.long))
        valid = graph.valid.index_select(0, torch.as_tensor(positions, dtype=torch.long)).bool()
    else:
        local_edges = np.empty((2, 0), dtype=np.int64)
        edge_features = graph.edge_features.new_empty((0, graph.edge_features.shape[-1]))
        valid = graph.valid.new_empty((0,), dtype=torch.bool)
    local = CandidateGraph(
        edge_index=torch.as_tensor(local_edges, dtype=torch.long),
        edge_features=edge_features,
        valid=valid,
        metadata={
            **dict(graph.metadata),
            "node_count": int(nodes.size),
            "global_node_indices": nodes.tolist(),
            "global_edge_indices": positions.tolist(),
            "candidate_count": int(positions.size),
        },
    )
    return local, positions
