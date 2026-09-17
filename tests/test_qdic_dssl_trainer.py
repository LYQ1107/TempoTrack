import json

import numpy as np

from tempotrack_v10.qdic_dssl_trainer import train_official_card
from tempotrack_v10.qdic_features import QDIC_FEATURE_NAMES, QDIC_RAW_DIM
from tempotrack_v10.qdic_trainer import sha256


def test_official_card_holdout_metrics_use_global_conflict_index(tmp_path):
    rows_per_video = 3
    videos = np.arange(1, 13, dtype=np.int64)
    count = len(videos) * rows_per_video
    features = np.zeros((count, QDIC_RAW_DIM), dtype=np.float32)
    labels = np.tile(np.asarray([1, 0, 0], dtype=np.int8), len(videos))
    features[:, 5] = np.tile(np.asarray([0.9, 0.1, 0.2], dtype=np.float32), len(videos))
    values = {
        "features": features,
        "labels": labels,
        "supervision_allowed": np.ones(count, dtype=bool),
        "offsets": np.arange(0, count + 1, rows_per_video, dtype=np.int64),
        "videos": videos,
        "candidate_base": np.ones(count, dtype=bool),
        "target_base": np.ones(count, dtype=bool),
        "group_ids": np.repeat(np.arange(len(videos), dtype=np.int64), rows_per_video),
    }
    root = tmp_path / "features"
    root.mkdir()
    paths = {}
    for name, value in values.items():
        path = root / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        paths[name] = str(path)
    config = {
        "query_observations": 1,
        "recent_k": 8,
        "alpha_fast": 0.70,
        "alpha_slow": 0.15,
        "memory_capacity": 64,
        "memory_dedup_cos": 0.95,
        "context_candidate_top_k": 64,
        "decision_candidate_top_k": 8,
        "top_r": 3,
        "min_gap": 0,
        "max_gap": 360,
    }
    metadata = {
        "artifact": "qdic_v11_feature_cache",
        "schema_version": 11,
        "split": "train",
        "feature_names": list(QDIC_FEATURE_NAMES),
        "feature_dim": QDIC_RAW_DIM,
        "base_only_supervision": True,
        "novel_gt_used_for_optimizer": False,
        "test_gt_used_for_optimizer": False,
        "source_role": "OFFICIAL_TRAIN",
        "exact_split_name": "train",
        "optimizer_source_allowed": True,
        "video_disjoint_split": True,
        "normalization_fit": "internal_train_base_only_after_video_split",
        "official_train_annotation_sha256": "a" * 64,
        "source_event_cache": "/data2/usr_for_deadline/official_train_qdic_events",
        "source_frontend_cache": "/data2/usr_for_deadline/official_train_cov_frontend",
        "input_source": "COVTRACK_FRONTEND",
        "supervision_source": "OFFICIAL_TRAIN_GT",
        "oracle_features_used": False,
        "gt_boxes_used_as_model_input": False,
        "gt_tracks_used_as_memory": False,
        "gt_used_only_for_supervision": True,
        "feature_config": config,
        "arrays": paths,
        "array_hashes": {name: sha256(path) for name, path in paths.items()},
    }
    (root / "features.json").write_text(json.dumps(metadata), encoding="utf-8")
    result = train_official_card(
        root,
        tmp_path / "card",
        card_id="TEST",
        structured_branch_mode="legacy",
        epochs=1,
        seed=0,
        device="cpu",
    )
    assert result["status"] == "COMPLETED"
    assert (tmp_path / "card" / "best.pt").is_file()
