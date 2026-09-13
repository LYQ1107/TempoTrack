import numpy as np

from tempotrack_v10.qdic_features import (
    QDIC_DISTRIBUTIONAL_FEATURE_NAMES,
    QDIC_FEATURE_NAMES,
    QDIC_RAW_DIM,
    build_qdic_candidate_features,
    build_qdic_event_features,
    build_qdic_features,
    projected_distribution_moments,
)


def _event_cache():
    cosine = np.zeros((3, 1, 64), dtype=np.float32)
    cosine[0, 0, :2] = [0.9, 0.2]
    cosine[1, 0, :2] = [0.3, 0.8]
    cosine[2, 0, :1] = [0.6]
    evidence = np.zeros((3, 64, 7), dtype=np.float32)
    evidence[0, :2] = 1.0
    evidence[1, :2] = 2.0
    evidence[2, :1] = 3.0
    rows = [
        {
            "video_id": 10,
            "target_serial": 100,
            "candidate_serial": 1,
            "target_base": 1,
            "candidate_base": 1,
            "label": 1,
            "gap": 4,
            "prefilter_rank_b1": 1,
        },
        {
            "video_id": 10,
            "target_serial": 100,
            "candidate_serial": 2,
            "target_base": 1,
            "candidate_base": 0,
            "label": 0,
            "gap": 4,
            "prefilter_rank_b1": 2,
        },
        {
            "video_id": 11,
            "target_serial": 200,
            "candidate_serial": 3,
            "target_base": 1,
            "candidate_base": 1,
            "label": 1,
            "gap": 5,
            "prefilter_rank_b1": 1,
        },
    ]
    return {
        "metadata": {"split": "val_base_internal", "max_gap": 360, "min_gap": 0},
        "arrays": {
            "cosine": cosine,
            "evidence": evidence,
            "mem_len": np.asarray([2, 2, 1], dtype=np.int16),
            "gap": np.asarray([4, 4, 5], dtype=np.int16),
            "group_id": np.asarray([0, 0, 1], dtype=np.int64),
            "target_base": np.asarray([True, True, True], dtype=bool),
            "prefilter_rank_b1": np.asarray([1, 2, 1], dtype=np.int32),
        },
        "rows": rows,
    }


def test_qdic_candidate_and_event_schema_is_33d_and_finite():
    independent = build_qdic_candidate_features(
        np.asarray([[0.8, 0.2]], dtype=np.float32),
        np.ones((2, 7), dtype=np.float32),
        5,
        1,
        query_fast_cosine=0.9,
        query_slow_cosine=0.8,
        fast_slow_cosine=0.95,
    )
    assert independent.shape == (19 + len(QDIC_DISTRIBUTIONAL_FEATURE_NAMES),)
    event = build_qdic_event_features(
        [
            {"base_features": independent[:19], "distributional_features": independent[19:]},
            {"base_features": independent[:19], "distributional_features": independent[19:]},
        ]
    )
    assert event.shape == (2, QDIC_RAW_DIM)
    assert tuple(QDIC_FEATURE_NAMES[-9:]) == tuple(QDIC_DISTRIBUTIONAL_FEATURE_NAMES)
    assert np.isfinite(event).all()


def test_qdic_fast_moments_use_last_eight_and_slow_moments_use_full_memory():
    projected = np.arange(10, dtype=np.float32) / 10.0
    fast_mean, fast_var, fast_mo, slow_mean, slow_var, slow_mo = projected_distribution_moments(
        projected, recent_k=8
    )
    np.testing.assert_allclose(fast_mean, projected[-8:].mean(), atol=1e-7)
    np.testing.assert_allclose(fast_var, projected[-8:].var(), atol=1e-7)
    np.testing.assert_allclose(fast_mo, fast_mean + 0.5 * fast_var, atol=1e-7)
    np.testing.assert_allclose(slow_mean, projected.mean(), atol=1e-7)
    np.testing.assert_allclose(slow_var, projected.var(), atol=1e-7)
    np.testing.assert_allclose(slow_mo, slow_mean + 0.5 * slow_var, atol=1e-7)


def test_qdic_feature_cache_filters_to_q1_b1_top64_and_preserves_supervision_flags(tmp_path):
    sidecar = {
        "metadata": {"rows": 3},
        "query_fast_cosine": np.asarray([0.7, 0.6, 0.5], dtype=np.float32),
        "query_slow_cosine": np.asarray([0.4, 0.3, 0.2], dtype=np.float32),
        "fast_slow_cosine": np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
    }
    result = build_qdic_features(_event_cache(), tmp_path / "features", sidecar=sidecar)
    features = np.load(tmp_path / "features" / "features.npy")
    allowed = np.load(tmp_path / "features" / "supervision_allowed.npy")
    offsets = np.load(tmp_path / "features" / "offsets.npy")
    videos = np.load(tmp_path / "features" / "videos.npy")
    assert result["feature_dim"] == QDIC_RAW_DIM
    assert features.shape == (3, QDIC_RAW_DIM)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(offsets, [0, 2, 3])
    np.testing.assert_array_equal(videos, [10, 11])
    np.testing.assert_array_equal(allowed, [True, False, True])
