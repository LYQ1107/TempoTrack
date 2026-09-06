"""Exact, deterministic sparse path-cover projection."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from ..errors import DependencyUnavailable


def _components(num_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    graph: dict[int, set[int]] = defaultdict(set)
    for source, target in edges:
        graph[source].add(target)
        graph[target].add(source)
    seen: set[int] = set()
    result: list[list[int]] = []
    for start in sorted(graph):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        values: list[int] = []
        while queue:
            node = queue.popleft()
            values.append(node)
            for child in sorted(graph[node]):
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        result.append(sorted(values))
    return result


def solve_path_cover(num_nodes: int, edge_index: np.ndarray, edge_benefit: np.ndarray, edge_valid: np.ndarray | None = None, threshold: float = 0.0, *, null_cost: float = 0.0, solver: str = "scipy", graph_metadata: dict[str, Any] | None = None) -> np.ndarray:
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
    elif edge_index.shape != (2, edge_index.shape[1]):
        edge_index = edge_index.reshape(2, -1)
    benefit = np.asarray(edge_benefit, dtype=np.float64).reshape(-1)
    if edge_index.shape[1] != len(benefit):
        raise ValueError("edge_index and edge_benefit disagree")
    valid = np.ones(len(benefit), dtype=bool) if edge_valid is None else np.asarray(edge_valid, dtype=bool).reshape(-1)
    if len(valid) != len(benefit):
        raise ValueError("edge_valid and edge_benefit disagree")
    if np.any(~np.isfinite(benefit)):
        raise ValueError("edge benefit contains non-finite values")
    for source, target in edge_index.T.tolist():
        if not (0 <= source < num_nodes and 0 <= target < num_nodes):
            raise ValueError(f"edge endpoint out of bounds: {(source, target)}")
        if source == target:
            raise ValueError(f"self-loop is not a legal path-cover edge: {source}")
    if graph_metadata:
        videos = graph_metadata.get("video_ids")
        first = graph_metadata.get("first_frames")
        last = graph_metadata.get("last_frames")
        if videos is not None and len(videos) != num_nodes:
            raise ValueError("graph_metadata.video_ids length mismatch")
        if (first is not None and len(first) != num_nodes) or (last is not None and len(last) != num_nodes):
            raise ValueError("graph_metadata frame lengths mismatch")
        for source, target in edge_index.T.tolist():
            if videos is not None and int(videos[source]) != int(videos[target]):
                valid[np.where((edge_index[0] == source) & (edge_index[1] == target))[0]] = False
            if first is not None and last is not None and int(last[source]) >= int(first[target]):
                valid[np.where((edge_index[0] == source) & (edge_index[1] == target))[0]] = False
    net = benefit - float(threshold)
    # Duplicates are resolved before solving.  The selected result maps back
    # to the maximum-net-benefit original row, with lowest index as tie-break.
    best: dict[tuple[int, int], int] = {}
    for index, (source, target) in enumerate(edge_index.T.tolist()):
        if not valid[index] or net[index] <= 0:
            continue
        key = (int(source), int(target))
        if key not in best or (net[index], -index) > (net[best[key]], -best[key]):
            best[key] = index
    selected = np.zeros(len(benefit), dtype=bool)
    if not best or num_nodes == 0:
        return selected
    if solver == "greedy":
        used_left: set[int] = set()
        used_right: set[int] = set()
        for key, index in sorted(best.items(), key=lambda item: (-net[item[1]], item[0])):
            source, target = key
            if source not in used_left and target not in used_right:
                selected[index] = True
                used_left.add(source)
                used_right.add(target)
        return selected
    if solver != "scipy":
        raise ValueError(f"unknown path-cover solver: {solver}")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise DependencyUnavailable("exact path-cover solver requires scipy; use solver=greedy only for explicit control") from exc
    components = _components(num_nodes, list(best))
    # Small components use the exact dense Hungarian reduction.  Larger
    # components stay sparse: a TAO video may have thousands of candidate
    # nodes, and materialising a blind (2N)^2 matrix would turn a legal
    # candidate graph into an avoidable OOM.  Both branches optimize the same
    # optional maximum-weight matching objective; neither silently switches
    # to the approximate greedy control.
    # 2,048 nodes means a bounded 4,096 x 4,096 float64 reduction (~128 MiB)
    # at the largest dense component.  Larger components use the sparse
    # integral formulation below.
    max_dense = 2048
    for nodes in components:
        size = len(nodes)
        node_pos = {node: index for index, node in enumerate(nodes)}
        if size > max_dense:
            try:
                from scipy.optimize import Bounds, LinearConstraint, milp
                from scipy.sparse import coo_matrix
            except ImportError as exc:
                raise DependencyUnavailable("large exact path-cover component requires scipy.optimize.milp") from exc
            component_edges = [(key, original) for key, original in best.items() if key[0] in node_pos and key[1] in node_pos]
            if component_edges:
                rows = []
                cols = []
                values = []
                costs = []
                for column, ((source, target), original) in enumerate(component_edges):
                    rows.extend((node_pos[source], size + node_pos[target]))
                    cols.extend((column, column))
                    values.extend((1.0, 1.0))
                    costs.append(-float(net[original]))
                matrix = coo_matrix((values, (rows, cols)), shape=(2 * size, len(component_edges))).tocsr()
                result = milp(
                    c=np.asarray(costs, dtype=np.float64),
                    integrality=np.ones(len(component_edges), dtype=np.int8),
                    bounds=Bounds(0.0, 1.0),
                    constraints=LinearConstraint(matrix, -np.inf, 1.0),
                    options={"presolve": True},
                )
                if not result.success or result.x is None:
                    raise DependencyUnavailable(f"sparse exact path-cover solve failed for component of {size} nodes: {result.message}")
                for value, ((source, target), original) in zip(result.x.tolist(), component_edges):
                    if value > 0.5:
                        selected[original] = True
            continue
        matrix_size = 2 * size
        cost = np.full((matrix_size, matrix_size), 0.0, dtype=np.float64)
        # Real-to-real invalid entries are expensive; every real row/column
        # still has a zero-cost end/birth dummy, so a solution always exists.
        max_abs = max([abs(float(net[index])) for index in best.values()] + [1.0])
        invalid_cost = max_abs + abs(float(null_cost)) + 1.0
        cost[:size, :size] = invalid_cost
        for (source, target), original in best.items():
            if source in node_pos and target in node_pos:
                cost[node_pos[source], node_pos[target]] = -net[original]
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows.tolist(), cols.tolist()):
            if row < size and col < size and cost[row, col] < -float(null_cost):
                original = best[(nodes[row], nodes[col])]
                selected[original] = True
    return selected


def validate_path_cover(num_nodes: int, edge_index: np.ndarray, selected: np.ndarray, first_frames: np.ndarray | None = None, last_frames: np.ndarray | None = None, video_ids: np.ndarray | None = None) -> dict[str, object]:
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(2, -1)
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    if len(selected) != edge_index.shape[1]:
        raise ValueError("selected length does not match edges")
    if np.any(edge_index[:, selected] < 0) or np.any(edge_index[:, selected] >= num_nodes):
        raise ValueError("selected edge endpoint out of bounds")
    chosen = edge_index[:, selected]
    indegree = np.bincount(chosen[1], minlength=num_nodes) if chosen.size else np.zeros(num_nodes, dtype=int)
    outdegree = np.bincount(chosen[0], minlength=num_nodes) if chosen.size else np.zeros(num_nodes, dtype=int)
    legal_time = True
    same_video = True
    if chosen.size and first_frames is not None and last_frames is not None:
        legal_time = bool(np.all(np.asarray(last_frames)[chosen[0]] < np.asarray(first_frames)[chosen[1]]))
    if chosen.size and video_ids is not None:
        same_video = bool(np.all(np.asarray(video_ids)[chosen[0]] == np.asarray(video_ids)[chosen[1]]))
    adjacency: dict[int, int] = {}
    for source, target in chosen.T.tolist():
        if source in adjacency:
            return {"valid": False, "reason": "multiple outgoing edge", "max_indegree": int(indegree.max(initial=0)), "max_outdegree": int(outdegree.max(initial=0)), "legal_time": legal_time, "same_video": same_video, "acyclic": False, "selected_edges": int(selected.sum())}
        adjacency[int(source)] = int(target)
    # Functional graph cycle detection without recursion (long TAO paths can
    # exceed Python's recursion limit).
    acyclic = True
    state = np.zeros(num_nodes, dtype=np.int8)
    for start in range(num_nodes):
        if state[start] == 2:
            continue
        node = start
        path: set[int] = set()
        while node in adjacency and state[node] == 0:
            if node in path:
                acyclic = False
                break
            path.add(node)
            state[node] = 1
            node = adjacency[node]
        if not acyclic:
            break
        for value in path:
            state[value] = 2
    return {"valid": bool(np.all(indegree <= 1) and np.all(outdegree <= 1) and legal_time and same_video and acyclic), "max_indegree": int(indegree.max(initial=0)), "max_outdegree": int(outdegree.max(initial=0)), "legal_time": legal_time, "same_video": same_video, "acyclic": acyclic, "selected_edges": int(selected.sum())}
