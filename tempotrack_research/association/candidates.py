"""One detector-independent candidate graph for every backend."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

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
    candidates: list[tuple[int, int]] = []
    features: list[list[float]] = []
    for i, left in enumerate(items):
        ranked: list[tuple[float, int, list[float]]] = []
        left_last_frame = float(_get(left, "last_frame", _get(left, "frame", 0)))
        left_first = _boundary(left, "bboxes", -1, _boundary(left, "bbox", -1, np.asarray([0, 0, 1, 1], dtype=np.float32)))
        left_app = _boundary_app(left, -1)
        for j, right in enumerate(items):
            if i == j or int(_get(left, "video_id", -1)) != int(_get(right, "video_id", -2)):
                continue
            right_first_frame = float(_get(right, "first_frame", _get(right, "frame", 0)))
            gap = right_first_frame - left_last_frame
            if gap <= 0 or gap > float(max_gap):
                continue
            if with_cats and _get(left, "category_id", None) != _get(right, "category_id", None):
                continue
            right_first = _boundary(right, "bboxes", 0, _boundary(right, "bbox", 0, np.asarray([0, 0, 1, 1], dtype=np.float32)))
            left_center = np.asarray([(left_first[0] + left_first[2]) / 2, (left_first[1] + left_first[3]) / 2], dtype=np.float32)
            right_center = np.asarray([(right_first[0] + right_first[2]) / 2, (right_first[1] + right_first[3]) / 2], dtype=np.float32)
            left_width = max(float(left_first[2] - left_first[0]), 1e-3)
            left_height = max(float(left_first[3] - left_first[1]), 1e-3)
            right_width = max(float(right_first[2] - right_first[0]), 1e-3)
            right_height = max(float(right_first[3] - right_first[1]), 1e-3)
            app = 0.0
            right_app = _boundary_app(right, 0)
            if left_app is not None and right_app is not None:
                a = F.normalize(torch.as_tensor(left_app, dtype=torch.float32).reshape(1, -1), dim=-1)
                b = F.normalize(torch.as_tensor(right_app, dtype=torch.float32).reshape(1, -1), dim=-1)
                app = float((a * b).sum())
            center_distance = float(np.linalg.norm(right_center - left_center))
            edge_features = [float(gap), float(app), center_distance, float(np.log(right_width / left_width)), float(np.log(right_height / left_height)), float(np.log((right_width * right_height) / (left_width * left_height)))]
            # A lower temporal/appearance cost is ranked first.  This ranking
            # is only candidate pruning; all backends use the same resulting
            # graph and may assign signed benefits independently.
            ranked.append((float(gap) + center_distance - 10.0 * app, j, edge_features))
        ranked.sort(key=lambda row: (row[0], row[1]))
        selected = ranked if top_k is None else ranked[: int(top_k)]
        for _, target, edge_features in selected:
            candidates.append((i, int(target)))
            features.append(edge_features)
    edge_index = torch.tensor(candidates, dtype=torch.long).t().contiguous() if candidates else torch.empty((2, 0), dtype=torch.long)
    edge_features = torch.tensor(features, dtype=torch.float32) if features else torch.empty((0, 6), dtype=torch.float32)
    return CandidateGraph(edge_index=edge_index, edge_features=edge_features, valid=torch.ones(edge_index.shape[1], dtype=torch.bool), metadata={"max_gap": max_gap, "top_k": top_k, "with_cats": with_cats, "candidate_count": int(edge_index.shape[1]), "pruning_uses_gt": False})
