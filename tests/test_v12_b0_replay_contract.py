import json
from types import SimpleNamespace

import pytest

from tools.v12_mgf_overlay_replay import (
    B0_COMPARISON_ARTIFACT,
    _read_provenance,
)
from tools.v12_post_mgf_exploration_replay import _validate_manifests


def _b0_model() -> SimpleNamespace:
    provenance = {
        "status": "QDIC_V11_MODEL_CODE_AND_WEIGHTS",
        "checkpoint": "/audit/B0_OFFICIAL_V11/best.pt",
        "checkpoint_sha256": "b0-checkpoint",
        "feature_dim": 33,
        "paper_status": "BASE_TRAIN",
        "paper_valid": True,
        "diagnostic_only": False,
        "base_only_supervision": True,
        "novel_gt_used": False,
        "test_weights_used": False,
    }
    return SimpleNamespace(
        tracker=SimpleNamespace(
            _v10_cov_adapter=SimpleNamespace(
                overlay=SimpleNamespace(_qdic=SimpleNamespace(provenance=provenance))
            )
        )
    )


def test_b0_source_provenance_is_retained_but_marked_comparator():
    provenance = _read_provenance(_b0_model())

    assert provenance["status"] == "QDIC_V11_MODEL_CODE_AND_WEIGHTS"
    assert provenance["feature_dim"] == 33
    assert provenance["paper_status"] == "BASE_TRAIN"
    assert provenance["comparison_role"] == "B0_OFFICIAL_V11_TEST_TUNED_COMPARATOR"


def test_b0_test_tuned_manifest_is_accepted(tmp_path):
    shard = tmp_path / "shard_00"
    shard.mkdir()
    provenance = _read_provenance(_b0_model())
    (shard / "manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "artifact": B0_COMPARISON_ARTIFACT,
                "paper_status": "TEST_TUNED_EXPLORATION",
                "paper_valid": False,
                "diagnostic_only": True,
                "thresholds": {"score_threshold": 0.0, "margin_threshold": 0.25},
                "detector_forward_calls": 0,
                "gt_loaded_during_replay": False,
                "mgf_provenance": provenance,
            }
        ),
        encoding="utf-8",
    )

    manifests = _validate_manifests(tmp_path, 1)
    assert len(manifests) == 1


def test_b0_manifest_cannot_be_relabelled_as_paper_valid(tmp_path):
    shard = tmp_path / "shard_00"
    shard.mkdir()
    provenance = _read_provenance(_b0_model())
    (shard / "manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "artifact": B0_COMPARISON_ARTIFACT,
                "paper_status": "BASE_TRAIN",
                "paper_valid": True,
                "diagnostic_only": False,
                "thresholds": {"score_threshold": 0.0, "margin_threshold": 0.25},
                "detector_forward_calls": 0,
                "gt_loaded_during_replay": False,
                "mgf_provenance": provenance,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="paper status"):
        _validate_manifests(tmp_path, 1)
