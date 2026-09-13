import json

import numpy as np
import torch

from tempotrack_v10.qdic_features import QDIC_FEATURE_NAMES, QDIC_RAW_DIM
from tempotrack_v10.qdic_loader import QDIC_STATUS, load_qdic_checkpoint
from tempotrack_v10.qdic_trainer import build_training_groups, sha256, train
from tempotrack_v10.query_conditioned_reranker import group_ranking_loss


def _feature_cache(tmp_path):
    videos = np.arange(1, 13, dtype=np.int64)
    rows_per_group = 3
    count = len(videos) * rows_per_group
    features = np.zeros((count, QDIC_RAW_DIM), dtype=np.float32)
    labels = np.tile(np.asarray([1, 0, 0], dtype=np.int8), len(videos))
    for group, video in enumerate(videos):
        start = group * rows_per_group
        features[start : start + 3, 5] = [0.95, 0.10, 0.30]
        features[start : start + 3, 17] = [1.0, 2.0, 3.0]
        features[start : start + 3, 9] = 3.0 / 64.0
        features[start : start + 3, 24] = [0.9, 0.2, 0.4]
        features[start : start + 3, 25] = [0.8, 0.1, 0.3]
        features[start : start + 3, 31] = [0.9, 0.1, 0.3]
        features[start : start + 3, 32] = [0.85, 0.1, 0.3]
    values = {
        "features": features,
        "labels": labels,
        "supervision_allowed": np.ones(count, dtype=bool),
        "offsets": np.arange(0, count + 1, rows_per_group, dtype=np.int64),
        "videos": videos,
        "candidate_base": np.ones(count, dtype=bool),
        "target_base": np.ones(count, dtype=bool),
        "group_ids": np.repeat(np.arange(len(videos), dtype=np.int64), rows_per_group),
    }
    root = tmp_path / "feature_cache"
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
    }
    metadata = {
        "artifact": "qdic_v11_feature_cache",
        "schema_version": 11,
        "split": "val_base_internal",
        "feature_names": list(QDIC_FEATURE_NAMES),
        "feature_dim": QDIC_RAW_DIM,
        "base_only_supervision": True,
        "novel_gt_used_for_optimizer": False,
        "feature_config": config,
        "arrays": paths,
        "array_hashes": {name: sha256(path) for name, path in paths.items()},
    }
    (root / "features.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def test_training_split_is_video_disjoint_and_base_only():
    arrays = {
        "offsets": np.asarray([0, 3, 6, 9], dtype=np.int64),
        "videos": np.asarray([11, 1, 2], dtype=np.int64),
        "labels": np.asarray([1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8),
        "supervision_allowed": np.asarray(
            [True, True, True, False, True, True, True, True, True], dtype=bool
        ),
    }
    train_groups, holdout_groups = build_training_groups({}, arrays)
    assert train_groups == [2]
    assert holdout_groups == [0]
    assert set(train_groups).isdisjoint(holdout_groups)


def test_unknown_label_is_masked_out_of_group_loss_gradient():
    logits = torch.tensor([[1.0, 4.0, 0.0]], requires_grad=True)
    labels = torch.tensor([[1, -1, 0]])
    group_ranking_loss(logits, labels).backward()
    assert logits.grad[0, 1].item() == 0.0


def test_cpu_trainer_writes_loadable_base_only_artifact(tmp_path, monkeypatch):
    feature_root = _feature_cache(tmp_path)
    monkeypatch.setattr("tempotrack_v10.qdic_trainer._memory_guard", lambda: None)
    output = tmp_path / "training"
    receipt = train(feature_root, output, device="cpu", epochs=1, seed=7)
    assert receipt["status"] == "COMPLETED"
    assert receipt["base_only_supervision"] is True
    assert receipt["novel_gt_used"] is False
    assert receipt["test_weights_used"] is False
    assert (output / "best.pt").is_file()
    assert json.loads((output / "training.json").read_text())["checkpoint_hash"] == sha256(
        output / "best.pt"
    )
    artifact = load_qdic_checkpoint(output / "best.pt", device="cpu")
    assert artifact.provenance["status"] == QDIC_STATUS
