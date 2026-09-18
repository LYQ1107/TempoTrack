import json

import numpy as np
import pytest

from tempotrack_v10 import PreAssociationSnapshot, SnapshotContractError
from tempotrack_v10.overlay import TempoTrackConfig, TempoTrackOverlay
from tempotrack_v10.qdic_features import (
    QDIC_MGF_BETA,
    QDIC_MGF_FEATURE_NAMES,
    QDIC_MGF_FEATURE_SCHEMA_VERSION,
    QDIC_MGF_RAW_DIM,
    build_qdic_mgf_candidate_features,
    build_qdic_mgf_event_features,
)
from tempotrack_v10.query_mgf_calibrator import QueryMomentGeneratingCalibrator


class _FakeMGF:
    def __init__(self, *, beta=QDIC_MGF_BETA):
        self.model = QueryMomentGeneratingCalibrator(mgf_mode="core")
        self.provenance = {
            "status": "QDIC_V12_MGF_MODEL_CODE_AND_WEIGHTS",
            "feature_names": list(QDIC_MGF_FEATURE_NAMES),
            "feature_dim": QDIC_MGF_RAW_DIM,
            "schema_version": QDIC_MGF_FEATURE_SCHEMA_VERSION,
            "mgf_beta": beta,
            "mgf_mode": "core",
            "feature_config": {
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
                "mgf_beta": beta,
                "mgf_type": "empirical_log_mean_exp",
                "mgf_normalization": "divide_by_beta",
                "mgf_recent_definition": "last_min_recent_k_L",
                "mgf_full_definition": "all_causal_history",
            },
            "training_protocol": "QDIC_V12_MGF_BASE_ONLY_TRAINING",
            "base_only_supervision": True,
            "novel_gt_used": False,
            "test_weights_used": False,
        }

    def score_event(self, candidates, **kwargs):
        return self.model.score_event(candidates, **kwargs)


def _snapshot():
    histories = {
        42: np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
        43: np.asarray([[0.0, 1.0]], dtype=np.float32),
    }
    return PreAssociationSnapshot(
        video_id=7,
        frame_id=10,
        boxes_xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
        det_scores=np.asarray([0.9], dtype=np.float32),
        labels=np.asarray([3], dtype=np.int64),
        observation_uids=("7:10:0",),
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        native_affinity=np.asarray([[0.9, 0.8]], dtype=np.float32),
        memory_ids=(42, 43),
        memory_embeddings=np.asarray([[0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        memory_last_frame=np.asarray([8, 8], dtype=np.int64),
        metadata={
            "association_stage": "pre_association",
            "memory_embedding_history": histories,
            "memory_evidence": {
                key: np.ones((len(value), 7), dtype=np.float32)
                for key, value in histories.items()
            },
        },
    )


def _config(checkpoint=None):
    return TempoTrackConfig(
        qdic_weight=1.0,
        qdic_checkpoint=checkpoint,
        qdic_device="cpu",
        candidate_top_k=8,
        qdic_context_top_k=64,
        max_gap=360,
        score_threshold=-100.0,
        margin_threshold=-1.0,
    )


def test_mgf_overlay_uses_schema_12_and_reports_frozen_beta(monkeypatch):
    monkeypatch.setenv("V11_COV_REPLAY_EVENT_DIAGNOSTICS", "1")
    overlay = TempoTrackOverlay(_config(), qdic=_FakeMGF())
    proposal = overlay.propose(_snapshot())
    assert proposal.diagnostics["qdic_variant"] == "mgf"
    assert proposal.diagnostics["qdic_feature_schema_version"] == 12
    assert proposal.diagnostics["qdic_mgf_beta"] == 1.0
    assert proposal.diagnostics["full_capability_status"] == "FULL_QDIC_MGF_RUNTIME_ACTIVE"
    assert proposal.accepted == (True,)
    event = proposal.diagnostics["replay_events"][0]
    assert len(event["qdic_projected_fast_log_mgf"]) == 2
    assert len(event["qdic_projected_slow_log_mgf"]) == 2


def test_mgf_overlay_rejects_beta_not_equal_to_one():
    with pytest.raises(SnapshotContractError, match="BETA_NOT_FROZEN_TO_ONE"):
        TempoTrackOverlay(_config(), qdic=_FakeMGF(beta=0.5))


def test_mgf_offline_online_event_feature_parity():
    candidate = {
        "cosine": np.asarray([[0.92, 0.81, 0.70]], dtype=np.float32),
        "evidence": np.asarray(
            [
                [0.92, 0.81, 0.70, 0.50, 0.20, 0.10, 0.05],
                [0.88, 0.77, 0.60, 0.40, 0.15, 0.08, 0.03],
                [0.80, 0.65, 0.55, 0.30, 0.12, 0.06, 0.02],
            ],
            dtype=np.float32,
        ),
        "gap": 12,
        "rank": 2,
        "query_fast_cosine": 0.84,
        "query_slow_cosine": 0.80,
        "fast_slow_cosine": 0.97,
        "recent_k": 8,
        "top_r": 3,
        "max_gap": 360,
        "mgf_beta": 1.0,
    }
    raw = build_qdic_mgf_candidate_features(
        candidate["cosine"],
        candidate["evidence"],
        candidate["gap"],
        candidate["rank"],
        query_fast_cosine=candidate["query_fast_cosine"],
        query_slow_cosine=candidate["query_slow_cosine"],
        fast_slow_cosine=candidate["fast_slow_cosine"],
        recent_k=candidate["recent_k"],
        top_r=candidate["top_r"],
        max_gap=candidate["max_gap"],
        mgf_beta=candidate["mgf_beta"],
    )
    offline = build_qdic_mgf_event_features(
        [{"base_features": raw[:19], "distributional_features": raw[19:]}]
    )
    online = QueryMomentGeneratingCalibrator().build_event_features([candidate])
    assert float(np.max(np.abs(offline - online))) <= 1e-7


def test_mgf_checkpoint_receipt_selects_separate_loader_branch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"placeholder")
    (tmp_path / "training.json").write_text(
        json.dumps({"artifact": "qdic_v12_mgf_official_training"}),
        encoding="utf-8",
    )
    fake = _FakeMGF()

    def fake_loader(path, *, device):
        assert path == checkpoint.resolve()
        assert device == "cpu"
        return fake

    monkeypatch.setattr("tempotrack_v10.overlay.load_qdic_mgf_checkpoint", fake_loader)
    overlay = TempoTrackOverlay(_config(checkpoint), qdic=None)
    assert overlay._qdic is fake
    assert overlay._qdic_variant == "mgf"
