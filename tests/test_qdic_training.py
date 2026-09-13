import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tempotrack_v10 import SnapshotContractError
from tempotrack_v10.qdic_features import QDIC_FEATURE_NAMES, QDIC_RAW_DIM
from tempotrack_v10.qdic_loader import QDIC_STATUS, load_qdic_checkpoint
from tempotrack_v10.qdic_trainer import _training_role, build_training_groups, sha256, train
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
        "top_r": 3,
        "min_gap": 0,
        "max_gap": 360,
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


def test_training_role_is_explicit_and_unknown_or_forbidden_splits_are_rejected():
    for split in ("train", "train_base"):
        assert _training_role(split) == {
            "paper_status": "BASE_TRAIN",
            "paper_valid": True,
            "diagnostic_only": False,
        }
    for split in ("val", "val_base_internal", "dev"):
        assert _training_role(split) == {
            "paper_status": "VAL_BASE_PILOT",
            "paper_valid": False,
            "diagnostic_only": True,
        }
    for split in ("", "test", "test_base", "novel", "novel_base", "full", "all", "mystery"):
        with pytest.raises(ValueError):
            _training_role(split)


def test_cpu_trainer_writes_loadable_base_only_artifact(tmp_path, monkeypatch):
    feature_root = _feature_cache(tmp_path)
    monkeypatch.setattr("tempotrack_v10.qdic_trainer._memory_guard", lambda: None)
    output = tmp_path / "training"
    receipt = train(feature_root, output, device="cpu", epochs=1, seed=7)
    assert receipt["status"] == "COMPLETED"
    assert receipt["base_only_supervision"] is True
    assert receipt["paper_status"] == "VAL_BASE_PILOT"
    assert receipt["paper_valid"] is False
    assert receipt["diagnostic_only"] is True
    assert receipt["novel_gt_used"] is False
    assert receipt["test_weights_used"] is False
    assert (output / "best.pt").is_file()
    assert json.loads((output / "training.json").read_text())["checkpoint_hash"] == sha256(
        output / "best.pt"
    )
    receipt_payload = json.loads((output / "training.json").read_text())
    assert any(
        Path(key).name == "query_conditioned_reranker.py"
        for key in receipt_payload["source_hashes"]
    )
    checkpoint_state = torch.load(output / "best.pt", map_location="cpu")
    assert any(
        Path(key).name == "query_conditioned_reranker.py"
        for key in checkpoint_state["source_hashes"]
    )
    artifact = load_qdic_checkpoint(output / "best.pt", device="cpu")
    assert artifact.provenance["status"] == QDIC_STATUS
    assert artifact.provenance["paper_status"] == "VAL_BASE_PILOT"
    assert artifact.provenance["paper_valid"] is False
    assert artifact.provenance["diagnostic_only"] is True
    assert artifact.provenance["shared_q1_source_hash_match"] is True


def test_loader_rejects_tampered_shared_q1_source_hash(tmp_path, monkeypatch):
    feature_root = _feature_cache(tmp_path)
    monkeypatch.setattr("tempotrack_v10.qdic_trainer._memory_guard", lambda: None)
    output = tmp_path / "training"
    train(feature_root, output, device="cpu", epochs=1, seed=11)
    receipt_path = output / "training.json"
    receipt = json.loads(receipt_path.read_text())
    source_key = next(
        key
        for key in receipt["source_hashes"]
        if Path(key).name == "query_conditioned_reranker.py"
    )
    receipt["source_hashes"][source_key] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(SnapshotContractError, match="SOURCE_HASH_MISMATCH"):
        load_qdic_checkpoint(output / "best.pt", device="cpu")
