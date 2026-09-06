"""Graph score helpers shared by S3, S4, and S5."""

from __future__ import annotations

from typing import Any

import numpy as np

from .path_cover import solve_path_cover, validate_path_cover


def project_graph_scores(num_nodes: int, edge_index: Any, scores: Any, edge_valid: Any | None = None, threshold: float = 0.0, graph_metadata: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, object]]:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim == 3:
        edges = edges[0]
    if edges.shape[0] == 2:
        edges = edges
    elif edges.shape[-1] == 2:
        edges = edges.T
    selected = solve_path_cover(num_nodes, edges, np.asarray(scores), None if edge_valid is None else np.asarray(edge_valid), threshold, graph_metadata=graph_metadata)
    return selected, validate_path_cover(
        num_nodes,
        edges,
        selected,
        None if graph_metadata is None else np.asarray(graph_metadata.get("first_frames")) if graph_metadata.get("first_frames") is not None else None,
        None if graph_metadata is None else np.asarray(graph_metadata.get("last_frames")) if graph_metadata.get("last_frames") is not None else None,
        None if graph_metadata is None else np.asarray(graph_metadata.get("video_ids")) if graph_metadata.get("video_ids") is not None else None,
    )
