import json

import numpy as np

from tempotrack_v10.qdic_features import (
    QDIC_FEATURE_NAMES,
    QDIC_MGF_FEATURE_NAMES,
    QDIC_MGF_RAW_DIM,
    QDIC_RAW_DIM,
    build_qdic_candidate_features,
    build_qdic_event_features,
    build_qdic_mgf_candidate_features,
    build_qdic_mgf_event_features,
    build_qdic_features,
    build_qdic_mgf_features,
)


def _candidate(cosine, rank, *, query_fast=0.31, query_slow=0.27):
    evidence = np.full((len(cosine), 7), 0.1, dtype=np.float32)
    return {
        "cosine": np.asarray(cosine, dtype=np.float32)[None, :],
        "evidence": evidence,
        "gap": 3,
        "rank": rank,
        "query_fast_cosine": query_fast,
        "query_slow_cosine": query_slow,
        "fast_slow_cosine": 0.88,
    }


def test_mgf_event_appends_only_two_features_and_preserves_v11_prefix():
    first = _candidate([0.1, 0.2, 0.4, 0.5], 1)
    second = _candidate([0.05, 0.15, 0.3], 2, query_fast=0.21, query_slow=0.19)
    old_rows = []
    new_rows = []
    for item in (first, second):
        old = build_qdic_candidate_features(**item)
        new = build_qdic_mgf_candidate_features(**item)
        old_rows.append({"base_features": old[:19], "distributional_features": old[19:]})
        new_rows.append({"base_features": new[:19], "distributional_features": new[19:]})
    old_event = build_qdic_event_features(old_rows)
    new_event = build_qdic_mgf_event_features(new_rows)
    assert old_event.shape == (2, QDIC_RAW_DIM)
    assert new_event.shape == (2, QDIC_MGF_RAW_DIM)
    np.testing.assert_allclose(new_event[:, :QDIC_RAW_DIM], old_event, rtol=0, atol=1e-7)
    assert len(QDIC_FEATURE_NAMES) == QDIC_RAW_DIM
    assert len(QDIC_MGF_FEATURE_NAMES) == QDIC_MGF_RAW_DIM == 35


def _synthetic_cache():
    rows = []
    count = 4
    max_length = 4
    cosine = np.zeros((count, 1, max_length), dtype=np.float32)
    evidence = np.zeros((count, max_length, 7), dtype=np.float32)
    mem_len = np.asarray([2, 3, 2, 4], dtype=np.int64)
    for index, length in enumerate(mem_len):
        cosine[index, 0, :length] = np.linspace(0.1, 0.7, int(length), dtype=np.float32)
        evidence[index, :length] = 0.1 + index
        rows.append(
            {
                "video_id": 10 + index // 2,
                "target_serial": index // 2,
                "label": 1 if index % 2 == 0 else 0,
                "target_base": True,
                "candidate_base": True,
                "prefilter_rank_b1": index % 2 + 1,
            }
        )
    metadata = {
        "artifact": "qdic_v9_event_cache",
        "split": "train",
        "max_gap": 360,
        "source_role": "OFFICIAL_TRAIN",
        "exact_split_name": "train",
        "source_event_cache": "/synthetic/official_train_events",
        "source_frontend_cache": "/synthetic/official_train_frontend",
        "input_source": "COVTRACK_FRONTEND",
        "supervision_source": "OFFICIAL_TRAIN_GT",
        "optimizer_source_allowed": True,
        "base_only_supervision": True,
        "novel_gt_used_for_optimizer": False,
        "test_gt_used_for_optimizer": False,
        "video_disjoint_split": True,
        "normalization_fit": "internal_train_base_only_after_video_split",
        "official_train_annotation_sha256": "synthetic-train-hash",
    }
    arrays = {
        "cosine": cosine,
        "evidence": evidence,
        "mem_len": mem_len,
        "gap": np.full(count, 3, dtype=np.int64),
        "prefilter_rank_b1": np.asarray([1, 2, 1, 2], dtype=np.int64),
        "group_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "label": np.asarray([1, 0, 1, 0], dtype=np.int8),
        "target_base": np.ones(count, dtype=bool),
    }
    sidecar = {
        "metadata": {
            "artifact": "qdic_v11_projected_prototype_sidecar",
            "rows": count,
            "query_observations": 1,
            "memory_capacity": 64,
            "recent_k": 8,
            "alpha_fast": 0.70,
            "alpha_slow": 0.15,
            "memory_dedup_cos": 0.95,
        },
        "arrays": {
            "query_fast_cosine": np.full(count, 0.3, dtype=np.float32),
            "query_slow_cosine": np.full(count, 0.2, dtype=np.float32),
            "fast_slow_cosine": np.full(count, 0.8, dtype=np.float32),
        },
    }
    return {"metadata": metadata, "arrays": arrays, "rows": rows}, sidecar


def test_mgf_cache_has_exact_v11_prefix_and_aligned_contract(tmp_path):
    event_cache, sidecar = _synthetic_cache()
    old_root = tmp_path / "old33"
    new_root = tmp_path / "new35"
    build_qdic_features(event_cache, old_root, sidecar=sidecar)
    build_qdic_mgf_features(event_cache, new_root, sidecar=sidecar)
    old = np.load(old_root / "features.npy", allow_pickle=False)
    new = np.load(new_root / "features.npy", allow_pickle=False)
    assert old.shape == (4, 33)
    assert new.shape == (4, 35)
    np.testing.assert_allclose(new[:, :33], old, rtol=0, atol=1e-7)
    for name in ("labels", "supervision_allowed", "candidate_base", "target_base", "group_ids", "offsets", "videos"):
        np.testing.assert_array_equal(
            np.load(new_root / f"{name}.npy", allow_pickle=False),
            np.load(old_root / f"{name}.npy", allow_pickle=False),
        )
    metadata = json.loads((new_root / "features.json").read_text())
    assert metadata["artifact"] == "qdic_v12_mgf_feature_cache"
    assert metadata["schema_version"] == 12
    assert metadata["feature_dim"] == 35
    assert metadata["mgf_beta"] == 1.0
    assert metadata["feature_config"]["mgf_type"] == "empirical_log_mean_exp"
