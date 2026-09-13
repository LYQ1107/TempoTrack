import numpy as np
import torch

from tempotrack_research.streaming.partial_support import replay_fixed_dual_prototypes
from tempotrack_v10.qdic_features import build_qdic_features
from tempotrack_v10.query_distributional_calibrator import QueryDistributionalCalibrator


def _normalise_rows(value):
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)


def test_offline_feature_cache_and_online_payload_are_numerically_identical(tmp_path):
    history = _normalise_rows([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    query = _normalise_rows([[0.6, 0.8]])
    evidence = np.asarray(
        [[0.9, 0.1, 0.2, 0.95, 0.0, 0.1, 0.0],
         [0.8, 0.2, 0.3, 0.90, 0.1, 0.2, 0.0],
         [0.7, 0.3, 0.4, 0.85, 0.2, 0.3, 0.0]],
        dtype=np.float32,
    )
    fast, slow = replay_fixed_dual_prototypes(history)
    direct = {
        "query_fast_cosine": float(query[0] @ fast),
        "query_slow_cosine": float(query[0] @ slow),
        "fast_slow_cosine": float(fast @ slow),
    }
    cache = {
        "metadata": {"split": "val_base_internal", "max_gap": 360, "min_gap": 0},
        "arrays": {
            "cosine": np.pad((query @ history.T)[:, None, :], ((0, 0), (0, 0), (0, 61))),
            "evidence": np.pad(evidence[None], ((0, 0), (0, 61), (0, 0))),
            "mem_len": np.asarray([3], dtype=np.int16),
            "gap": np.asarray([4], dtype=np.int16),
            "group_id": np.asarray([0], dtype=np.int64),
            "target_base": np.asarray([True], dtype=bool),
            "prefilter_rank_b1": np.asarray([1], dtype=np.int32),
        },
        "rows": [
            {
                "video_id": 7,
                "target_serial": 100,
                "candidate_serial": 42,
                "target_base": 1,
                "candidate_base": 1,
                "label": 1,
                "gap": 4,
                "prefilter_rank_b1": 1,
            }
        ],
    }
    sidecar = {"metadata": {"rows": 1}, **{key: np.asarray([value], dtype=np.float32) for key, value in direct.items()}}
    # The mapping-based cache preserves the V9 array semantics without
    # rebuilding any large event cache.
    build_qdic_features(cache, tmp_path / "features", sidecar=sidecar)
    offline_features = np.load(tmp_path / "features" / "features.npy")

    payload = {
        "cosine": query @ history.T,
        "evidence": evidence,
        "gap": 4,
        "rank": 1,
        **direct,
        "recent_k": 8,
        "top_r": 3,
        "max_gap": 360,
    }
    model = QueryDistributionalCalibrator()
    online_features = model.build_event_features([payload])
    np.testing.assert_allclose(online_features, offline_features, rtol=0.0, atol=1e-6)
    offline_diagnostics = model(torch.from_numpy(offline_features), return_diagnostics=True)
    offline_logits = offline_diagnostics["logit"].detach().numpy()
    online_logits, online_diagnostics = model.score_event(
        [payload], return_diagnostics=True
    )
    np.testing.assert_allclose(online_logits, offline_logits, rtol=0.0, atol=1e-6)
    for key in ("alpha", "structured_score", "fast_branch", "slow_branch", "residual"):
        np.testing.assert_allclose(
            online_diagnostics[key], offline_diagnostics[key].detach().numpy(), rtol=0.0, atol=1e-6
        )
