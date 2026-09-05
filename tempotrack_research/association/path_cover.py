"""Deterministic legal path-cover projection with birth/end dummies."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def solve_path_cover(num_nodes: int, edge_index: np.ndarray, edge_benefit: np.ndarray, edge_valid: np.ndarray | None = None, threshold: float = 0.0) -> np.ndarray:
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(2, -1)
    edge_benefit = np.asarray(edge_benefit, dtype=np.float64).reshape(-1)
    if edge_index.shape[1] != len(edge_benefit):
        raise ValueError("edge_index and edge_benefit disagree")
    valid = np.ones(len(edge_benefit), dtype=bool) if edge_valid is None else np.asarray(edge_valid, dtype=bool)
    if len(valid) != len(edge_benefit):
        raise ValueError("edge_valid and edge_benefit disagree")
    if num_nodes == 0:
        return np.zeros(len(edge_benefit), dtype=bool)
    size = 2 * num_nodes
    cost = np.zeros((size, size), dtype=np.float64)
    cost[:num_nodes, :num_nodes] = 1e6
    edge_lookup: dict[tuple[int, int], list[int]] = {}
    for index, ((source, target), benefit, allowed) in enumerate(zip(edge_index.T.tolist(), edge_benefit.tolist(), valid.tolist())):
        if allowed and 0 <= source < num_nodes and 0 <= target < num_nodes and source != target:
            cost[source, target] = min(cost[source, target], -float(benefit))
            edge_lookup.setdefault((source, target), []).append(index)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
    except ImportError:
        # Deterministic finite fallback for a minimal environment.
        order = sorted(range(len(edge_benefit)), key=lambda i: (-edge_benefit[i], int(edge_index[0, i]), int(edge_index[1, i])))
        used_left, used_right = set(), set()
        selected = np.zeros(len(edge_benefit), dtype=bool)
        for index in order:
            source, target = edge_index[:, index]
            if valid[index] and edge_benefit[index] > threshold and source not in used_left and target not in used_right:
                selected[index] = True
                used_left.add(int(source)); used_right.add(int(target))
        return selected
    selected = np.zeros(len(edge_benefit), dtype=bool)
    for source, target in zip(rows, cols):
        if source < num_nodes and target < num_nodes and cost[source, target] < -threshold:
            for index in edge_lookup.get((int(source), int(target)), []):
                if valid[index] and edge_benefit[index] > threshold:
                    selected[index] = True
                    break
    return selected


def validate_path_cover(num_nodes: int, edge_index: np.ndarray, selected: np.ndarray, first_frames: np.ndarray | None = None, last_frames: np.ndarray | None = None) -> dict[str, object]:
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(2, -1)
    selected = np.asarray(selected, dtype=bool)
    if len(selected) != edge_index.shape[1]:
        raise ValueError("selected length does not match edges")
    chosen = edge_index[:, selected]
    indegree = np.bincount(chosen[1], minlength=num_nodes) if chosen.size else np.zeros(num_nodes, dtype=int)
    outdegree = np.bincount(chosen[0], minlength=num_nodes) if chosen.size else np.zeros(num_nodes, dtype=int)
    legal_time = True
    if first_frames is not None and last_frames is not None and chosen.size:
        legal_time = bool(np.all(np.asarray(last_frames)[chosen[0]] < np.asarray(first_frames)[chosen[1]]))
    graph = {int(i): [] for i in range(num_nodes)}
    for source, target in chosen.T.tolist():
        graph[int(source)].append(int(target))
    visiting, visited = set(), set()

    def visit(node: int) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        ok = all(visit(child) for child in graph[node])
        visiting.remove(node); visited.add(node)
        return ok

    acyclic = all(visit(node) for node in range(num_nodes))
    return {"valid": bool(np.all(indegree <= 1) and np.all(outdegree <= 1) and legal_time and acyclic), "max_indegree": int(indegree.max(initial=0)), "max_outdegree": int(outdegree.max(initial=0)), "legal_time": legal_time, "acyclic": acyclic, "selected_edges": int(selected.sum())}
