"""Shared graph supervision semantics for matched and clean episode builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class GraphTargets:
    successor: np.ndarray
    successor_known: np.ndarray
    same_identity: np.ndarray
    identity_known: np.ndarray
    node_known: np.ndarray
    boundary_censored: np.ndarray
    candidate_missed: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _array(value: Any, dtype: Any, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if result.shape != (length,):
        raise ValueError(f"graph supervision field has shape {result.shape}, expected {(length,)}")
    return result


def build_direct_successor_targets(
    nodes: Sequence[Mapping[str, Any]],
    edge_index: np.ndarray,
    supervision: Mapping[str, Any] | None = None,
    *,
    full_video_context: Mapping[str, Any] | None = None,
) -> GraphTargets:
    """Build direct-successor and compatibility labels on the real candidate set.

    Direct successors are obtained from the complete known node sequence first;
    candidate retrieval and window slicing only mask whether that edge can be
    supervised.  Thus A-B-C receives two successor positives and the A-C
    compatibility edge is not a successor shortcut.
    """
    del full_video_context
    count = len(nodes)
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((2, 0), dtype=np.int64)
    edges = edges.reshape(2, -1)
    edge_count = int(edges.shape[1])
    supervision = dict(supervision or {})
    identities = supervision.get("identity", supervision.get("identities", [node.get("identity", -1) for node in nodes]))
    known = supervision.get("known", supervision.get("node_known", [node.get("known", int(node.get("identity", -1)) >= 0) for node in nodes]))
    identities = _array(identities, np.int64, count)
    known = _array(known, bool, count)
    video_ids = _array(supervision.get("video_ids", [node.get("video_id", -1) for node in nodes]), np.int64, count)
    first = _array(supervision.get("first_frames", [node.get("first_frame", 0.0) for node in nodes]), np.float64, count)
    last = _array(supervision.get("last_frames", [node.get("last_frame", 0.0) for node in nodes]), np.float64, count)
    node_known = known.copy()
    same_identity = np.zeros(edge_count, dtype=bool)
    identity_known = np.zeros(edge_count, dtype=bool)
    successor = np.zeros(edge_count, dtype=bool)
    successor_known = np.zeros(edge_count, dtype=bool)
    edge_lookup = {(int(source), int(target)): index for index, (source, target) in enumerate(edges.T.tolist())}
    direct_pairs: set[tuple[int, int]] = set()
    groups: dict[tuple[int, int], list[int]] = {}
    for index in range(count):
        if known[index] and identities[index] >= 0 and video_ids[index] >= 0:
            groups.setdefault((int(video_ids[index]), int(identities[index])), []).append(index)
    for group in groups.values():
        group.sort(key=lambda index: (float(first[index]), float(last[index]), int(index)))
        for left, right in zip(group, group[1:]):
            # Overlapping or indeterminate fragments do not get a direct edge.
            if float(first[right]) <= float(last[left]):
                continue
            direct_pairs.add((int(left), int(right)))
    for edge_number, (source, target) in enumerate(edges.T.tolist()):
        source = int(source); target = int(target)
        if source < 0 or target < 0 or source >= count or target >= count:
            continue
        identity_known[edge_number] = bool(known[source] and known[target] and identities[source] >= 0 and identities[target] >= 0)
        same_identity[edge_number] = bool(identity_known[edge_number] and identities[source] == identities[target])
        legal = bool(video_ids[source] == video_ids[target] and last[source] < first[target])
        successor_known[edge_number] = bool(identity_known[edge_number] and legal)
        successor[edge_number] = bool(successor_known[edge_number] and (source, target) in direct_pairs)
    candidate_missed = np.zeros(count, dtype=bool)
    for source, target in direct_pairs:
        if (source, target) not in edge_lookup:
            candidate_missed[source] = True
            candidate_missed[target] = True
    diagnostics = {
        "node_count": count,
        "edge_count": edge_count,
        "known_nodes": int(known.sum()),
        "unknown_nodes": int((~known).sum()),
        "direct_successor_edges": int(successor.sum()),
        "successor_known_edges": int(successor_known.sum()),
        "same_identity_compatible_edges": int(same_identity.sum()),
        "candidate_missed_nodes": int(candidate_missed.sum()),
        "unknown_edges": int((~identity_known).sum()),
        "direct_pairs": sorted([list(pair) for pair in direct_pairs]),
    }
    return GraphTargets(successor, successor_known, same_identity, identity_known, node_known, np.zeros(count, dtype=bool), candidate_missed, diagnostics)


def validate_target_paths(nodes: Sequence[Mapping[str, Any]], edge_index: np.ndarray, targets: GraphTargets) -> dict[str, Any]:
    edges = np.asarray(edge_index, dtype=np.int64).reshape(2, -1)
    selected = np.asarray(targets.successor & targets.successor_known, dtype=bool)
    sources = edges[0, selected] if selected.any() else np.empty(0, dtype=np.int64)
    dests = edges[1, selected] if selected.any() else np.empty(0, dtype=np.int64)
    errors: list[str] = []
    if len(np.unique(sources)) != len(sources):
        errors.append("successor target has multiple outgoing edges")
    if len(np.unique(dests)) != len(dests):
        errors.append("successor target has multiple incoming edges")
    for source, target in zip(sources.tolist(), dests.tolist()):
        left, right = nodes[int(source)], nodes[int(target)]
        if int(left.get("video_id", -1)) != int(right.get("video_id", -1)):
            errors.append("successor target crosses videos")
        if float(left.get("last_frame", 0.0)) >= float(right.get("first_frame", 0.0)):
            errors.append("successor target is not time ordered")
    return {"valid": not errors, "errors": errors, "selected_edges": int(selected.sum()), "diagnostics": dict(targets.diagnostics)}


__all__ = ["GraphTargets", "build_direct_successor_targets", "validate_target_paths"]
