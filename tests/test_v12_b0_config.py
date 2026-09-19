import json

from tools.v12_make_mgf_exploration_configs import _sha256, _write_b0_config


def test_b0_config_preserves_training_role_and_marks_replay_diagnostic(tmp_path):
    card_dir = tmp_path / "B0_OFFICIAL_V11"
    card_dir.mkdir()
    checkpoint = card_dir / "best.pt"
    checkpoint.write_bytes(b"audited-b0-checkpoint")
    receipt = {
        "status": "COMPLETED",
        "artifact": "qdic_v11_dssl_official_training",
        "card_id": "B0_OFFICIAL_V11",
        "protocol": "QDIC_V11_BASE_ONLY_TRAINING",
        "training_split": "train_base_official",
        "paper_status": "BASE_TRAIN",
        "paper_valid": True,
        "diagnostic_only": False,
        "base_only_supervision": True,
        "novel_gt_used": False,
        "test_weights_used": False,
        "feature_dim": 33,
        "checkpoint_hash": _sha256(checkpoint),
    }
    (card_dir / "training.json").write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "configs" / "B0_OFFICIAL_V11.yaml"

    row = _write_b0_config(checkpoint, output, source_role="CURRENT_TEST", split="test")

    text = output.read_text(encoding="utf-8")
    assert row["source_paper_status"] == "BASE_TRAIN"
    assert row["paper_status"] == "TEST_TUNED_EXPLORATION"
    assert row["diagnostic_only"] is True
    assert "card_type: b0_comparator" in text
    assert "comparison_role: B0_OFFICIAL_V11_TEST_TUNED_COMPARATOR" in text
