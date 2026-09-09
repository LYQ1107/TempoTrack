from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import DormantIdentity, TentativeTrack


@dataclass
class DormantCandidate:
    tentative_id: int
    dormant_id: int
    gap: int
    appearance_score: float
    center_distance_norm: float
    rank_score: float


def _mean_feature(values) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float32)
    value = np.mean(np.stack([np.asarray(item.feature_norm, dtype=np.float32) for item in values]), axis=0)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def build_dormant_candidates(
    tentative_tracks: list[TentativeTrack],
    dormant_identities: list[DormantIdentity],
    *,
    max_gap: int,
    top_k: int = 8,
) -> dict[int, list[DormantCandidate]]:
    output: dict[int, list[DormantCandidate]] = {}
    for tentative in tentative_tracks:
        query = _mean_feature(tentative.observations)
        if query.size == 0:
            output[tentative.frontend_track_id] = []
            continue
        query_obs = tentative.observations[-1]
        qcenter = (np.asarray(query_obs.bbox_xyxy[:2]) + np.asarray(query_obs.bbox_xyxy[2:])) / 2.0
        diagonal = max(float((query_obs.image_width ** 2 + query_obs.image_height ** 2) ** 0.5), 1.0)
        candidates: list[DormantCandidate] = []
        for dormant in dormant_identities:
            gap = int(tentative.first_frame - dormant.last_frame)
            if gap <= 0 or gap > int(max_gap):
                continue
            appearance = float(np.dot(query, np.asarray(dormant.slow_prototype, dtype=np.float32)))
            dcenter = (np.asarray(dormant.last_bbox[:2]) + np.asarray(dormant.last_bbox[2:])) / 2.0
            center_distance = float(np.linalg.norm(qcenter - dcenter) / diagonal)
            rank = gap / max(float(max_gap), 1.0) + center_distance - 2.0 * appearance
            candidates.append(DormantCandidate(int(tentative.frontend_track_id), int(dormant.identity_id), gap, appearance, center_distance, rank))
        candidates.sort(key=lambda value: (value.rank_score, value.dormant_id))
        output[tentative.frontend_track_id] = candidates[: int(top_k)]
    return output


def topk_memory_score(tentative: TentativeTrack, dormant: DormantIdentity, *, top_r: int = 3) -> float:
    if not tentative.observations or not dormant.history.observations:
        return float("-inf")
    query = np.stack([np.asarray(item.feature_norm, dtype=np.float32) for item in tentative.observations])
    memory = np.stack([np.asarray(item.feature_norm, dtype=np.float32) for item in dormant.history.observations])
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    memory = memory / np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), 1e-12)
    scores = query @ memory.T
    r = min(int(top_r), scores.shape[1])
    return float(np.mean(np.sort(scores, axis=1)[:, -r:]))

