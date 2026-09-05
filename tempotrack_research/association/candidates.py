"""One candidate graph contract for every backend."""

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


def build_candidate_graph(tracklets: Iterable[Any], max_gap: int = 90, top_k: int | None = 20, with_cats: bool = False) -> CandidateGraph:
    items = list(tracklets)
    candidates: list[tuple[int, int]] = []
    features: list[list[float]] = []
    for i, left in enumerate(items):
        ranked: list[tuple[float, int, list[float]]] = []
        for j, right in enumerate(items):
            if i == j:
                continue
            if int(_get(left, "video_id", -1)) != int(_get(right, "video_id", -2)):
                continue
            left_last = int(_get(left, "last_frame", _get(left, "frame", 0)))
            right_first = int(_get(right, "first_frame", _get(right, "frame", 0)))
            gap = right_first - left_last
            if gap <= 0 or gap > max_gap:
                continue
            if with_cats and _get(left, "category_id", None) != _get(right, "category_id", None):
                continue
            left_box = np.asarray(_get(left, "bbox", [0, 0, 1, 1]), dtype=np.float32).reshape(-1)[:4]
            right_box = np.asarray(_get(right, "bbox", [0, 0, 1, 1]), dtype=np.float32).reshape(-1)[:4]
            lc = np.asarray([(left_box[0] + left_box[2]) / 2, (left_box[1] + left_box[3]) / 2])
            rc = np.asarray([(right_box[0] + right_box[2]) / 2, (right_box[1] + right_box[3]) / 2])
            lw, lh = max(left_box[2] - left_box[0], 1e-3), max(left_box[3] - left_box[1], 1e-3)
            rw, rh = max(right_box[2] - right_box[0], 1e-3), max(right_box[3] - right_box[1], 1e-3)
            app = 0.0
            left_app, right_app = _get(left, "appearance", None), _get(right, "appearance", None)
            if left_app is not None and right_app is not None:
                a = F.normalize(torch.as_tensor(left_app, dtype=torch.float32).reshape(1, -1), dim=-1)
                b = F.normalize(torch.as_tensor(right_app, dtype=torch.float32).reshape(1, -1), dim=-1)
                app = float((a * b).sum())
            edge_features = [float(gap), float(app), float(np.linalg.norm(rc - lc)), float(np.log(rw / lw)), float(np.log(rh / lh)), float(np.log((rw * rh) / (lw * lh)))]
            ranked.append((float(gap) - 10.0 * app, j, edge_features))
        ranked.sort(key=lambda row: (row[0], row[1]))
        for _, j, edge_features in ranked[:top_k] if top_k else ranked:
            candidates.append((i, j))
            features.append(edge_features)
    if candidates:
        edge_index = torch.tensor(candidates, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_features = torch.empty((0, 6), dtype=torch.float32)
    return CandidateGraph(edge_index=edge_index, edge_features=edge_features, valid=torch.ones(edge_index.shape[1], dtype=torch.bool), metadata={"max_gap": max_gap, "top_k": top_k, "with_cats": with_cats, "candidate_count": int(edge_index.shape[1])})
