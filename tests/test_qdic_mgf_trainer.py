import hashlib
import json

import numpy as np

from tempotrack_v10.qdic_features import build_qdic_mgf_features
from tempotrack_v10.qdic_mgf_loader import load_qdic_mgf_checkpoint
from tempotrack_v10.qdic_mgf_trainer import (
    CHECKPOINT_COMPARATOR,
    build_official_train_groups,
    train_official_mgf_card,
)


def _bucket(video_id):
    return int(hashlib.sha256(str(video_id).encode()).hexdigest()[:8], 16) % 5


def _cache(tmp_path):
    video_ids = []
    for value in range(10, 200):
        if value not in video_ids and len({
            _bucket(item) for item in video_ids
        }) < 2:
            video_ids.append(value)
        elif value not in video_ids and (_bucket(value) == 0) != (_bucket(video_ids[0]) == 0):
            video_ids.append(value)
        if len(video_ids) >= 6 and any(_bucket(item) == 0 for item in video_ids) and any(_bucket(item) != 0 for item in video_ids):
            break
    rows = []
    count = len(video_ids) * 2
    cosine = np.zeros((count, 1, 3), dtype=np.float32)
    evidence = np.ones((count, 3, 7), dtype=np.float32)
    mem_len = np.full(count, 3, dtype=np.int64)
    group_ids = []
    labels = []
    target_base = []
    for group, video_id in enumerate(video_ids):
        for local in (0, 1):
            index = group * 2 + local
            cosine[index, 0] = np.asarray([0.2 + 0.01 * local, 0.3, 0.4], dtype=np.float32)
            group_ids.append(group)
            labels.append(1 if local == 0 else 0)
            target_base.append(True)
            rows.append({
                "video_id": video_id,
                "target_serial": group,
                "label": labels[-1],
                "target_base": True,
                "candidate_base": True,
                "prefilter_rank_b1": local + 1,
            })
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
        "prefilter_rank_b1": np.asarray([1, 2] * len(video_ids), dtype=np.int64),
        "group_id": np.asarray(group_ids, dtype=np.int64),
        "label": np.asarray(labels, dtype=np.int8),
        "target_base": np.asarray(target_base, dtype=bool),
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
    raw = {"metadata": metadata, "arrays": arrays, "rows": rows}
    root = tmp_path / "features"
    build_qdic_mgf_features(raw, root, sidecar=sidecar)
    return root


def test_official_train_groups_are_video_disjoint(tmp_path):
    root = _cache(tmp_path)
    metadata = json.loads((root / "features.json").read_text())
    arrays = {
        name: np.load(path, allow_pickle=False)
        for name, path in metadata["arrays"].items()
    }
    train, holdout = build_official_train_groups(arrays)
    train_videos = {int(arrays["videos"][index]) for index in train}
    holdout_videos = {int(arrays["videos"][index]) for index in holdout}
    assert train_videos
    assert holdout_videos
    assert train_videos.isdisjoint(holdout_videos)


def test_mgf_training_receipt_and_loader_are_fail_closed(tmp_path):
    root = _cache(tmp_path)
    output = tmp_path / "train" / "M1_MGF_CORE"
    result = train_official_mgf_card(
        root,
        output,
        card_id="M1_MGF_CORE",
        mgf_mode="core",
        epochs=1,
        seed=0,
        device="cpu",
    )
    assert result["status"] == "COMPLETED"
    assert result["checkpoint_comparator"] == CHECKPOINT_COMPARATOR
    assert result["mgf_beta"] == 1.0
    artifact = load_qdic_mgf_checkpoint(output / "best.pt", device="cpu")
    assert artifact.provenance["status"] == "QDIC_V12_MGF_MODEL_CODE_AND_WEIGHTS"
    assert artifact.provenance["feature_dim"] == 35
    assert artifact.provenance["mgf_beta"] == 1.0
    assert artifact.provenance["base_only_supervision"] is True
    assert artifact.provenance["novel_gt_used"] is False
    assert artifact.provenance["test_weights_used"] is False
