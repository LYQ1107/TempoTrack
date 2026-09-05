"""Graph score helpers shared by S3, S4, and S5."""

from __future__ import annotations

from typing import Any

import numpy as np

from .path_cover import solve_path_cover, validate_path_cover


def project_graph_scores(num_nodes: int, edge_index: Any, scores: Any, edge_valid: Any | None = None, threshold: float = 0.0) -> tuple[np.ndarray, dict[str, object]]:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim == 3:
        edges = edges[0]
    if edges.shape[0] == 2:
        edges = edges
    elif edges.shape[-1] == 2:
        edges = edges.T
    selected = solve_path_cover(num_nodes, edges, np.asarray(scores), None if edge_valid is None else np.asarray(edge_valid), threshold)
    return selected, validate_path_cover(num_nodes, edges, selected)
