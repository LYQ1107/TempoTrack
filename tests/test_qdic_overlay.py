import numpy as np
import pytest

from tempotrack_research.streaming.partial_support import replay_fixed_dual_prototypes
from tempotrack_v10 import PreAssociationSnapshot, SnapshotContractError
from tempotrack_v10.overlay import TempoTrackConfig, TempoTrackOverlay
from tempotrack_v10.qdic_features import QDIC_FEATURE_NAMES
from tempotrack_v10.query_distributional_calibrator import QueryDistributionalCalibrator


class _FakeQDIC:
    def __init__(self):
        self.model = QueryDistributionalCalibrator()
        self.calls = []

    @property
    def provenance(self):
        return {
            "status": "QDIC_V11_MODEL_CODE_AND_WEIGHTS",
            "feature_names": list(QDIC_FEATURE_NAMES),
            "feature_dim": 33,
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
                "max_gap": 360,
            },
            "training_protocol": "QDIC_V11_BASE_ONLY_TRAINING",
            "base_only_supervision": True,
            "novel_gt_used": False,
            "test_weights_used": False,
        }

    def score_event(self, candidates, **kwargs):
        self.calls.append(list(candidates))
        return self.model.score_event(candidates, **kwargs)


def _config(**updates):
    values = dict(
        qdic_weight=1.0,
        candidate_top_k=8,
        qdic_context_top_k=64,
        score_threshold=-100.0,
        margin_threshold=-1.0,
        max_gap=360,
    )
    values.update(updates)
    return TempoTrackConfig(**values)


def _snapshot(
    *,
    frame=10,
    histories=None,
    native_affinity=None,
    include_history=True,
    include_evidence=True,
):
    histories = tuple(histories or (np.asarray([[1.0, 0.0]], dtype=np.float32),))
    count = len(histories)
    ids = tuple(42 + index for index in range(count))
    last = np.asarray([np.asarray(history)[-1] for history in histories], dtype=np.float32)
    metadata = {"association_stage": "pre_association"}
    if include_history:
        metadata["memory_embedding_history"] = {
            memory_id: np.asarray(history, dtype=np.float32)
            for memory_id, history in zip(ids, histories)
        }
    if include_evidence:
        metadata["memory_evidence"] = {
            memory_id: np.ones((len(history), 7), dtype=np.float32)
            for memory_id, history in zip(ids, histories)
        }
    if native_affinity is None:
        native_affinity = np.full((1, count), 0.9, dtype=np.float32)
    return PreAssociationSnapshot(
        video_id=7,
        frame_id=frame,
        boxes_xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
        det_scores=np.asarray([0.9], dtype=np.float32),
        labels=np.asarray([3], dtype=np.int64),
        observation_uids=(f"7:{frame}:0",),
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        native_affinity=np.asarray(native_affinity, dtype=np.float32),
        memory_ids=ids,
        memory_embeddings=last,
        memory_last_frame=np.full(count, frame - 2, dtype=np.int64),
        metadata=metadata,
    )


def _empty_snapshot(frame):
    return PreAssociationSnapshot(
        video_id=7,
        frame_id=frame,
        boxes_xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
        det_scores=np.asarray([0.9], dtype=np.float32),
        labels=np.asarray([3], dtype=np.int64),
        observation_uids=(f"7:{frame}:0",),
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        native_affinity=np.empty((1, 0), dtype=np.float32),
        memory_ids=(),
        memory_embeddings=np.empty((0, 2), dtype=np.float32),
        memory_last_frame=np.empty((0,), dtype=np.int64),
        metadata={"association_stage": "pre_association"},
    )


def test_qdic_prefilter_uses_last_real_observation_and_replays_state():
    fake = _FakeQDIC()
    histories = (
        np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    )
    overlay = TempoTrackOverlay(
        _config(),
        qdic=fake,
    )
    proposal = overlay.propose(
        _snapshot(
            histories=histories,
            native_affinity=np.asarray([[0.01, 0.99]], dtype=np.float32),
        )
    )
    assert len(fake.calls) == 1
    assert [item["rank"] for item in fake.calls[0]] == [1, 2]
    assert proposal.diagnostics["qdic_context_candidate_top_k"] == 64
    assert proposal.diagnostics["qdic_decision_candidate_top_k"] == 8
    expected_fast, expected_slow = replay_fixed_dual_prototypes(histories[0])
    record = overlay._records[("7", 42)]
    np.testing.assert_allclose(record.state.fast.numpy(), expected_fast, atol=1e-6)
    np.testing.assert_allclose(record.state.slow.numpy(), expected_slow, atol=1e-6)
    # Native affinity intentionally ranks ID 43 first; the QDIC context still
    # follows cosine(query, exact last real observation), as in training.
    assert fake.calls[0][0]["rank"] == 1
    assert proposal.diagnostics["full_capability_status"] == "FULL_QDIC_MO_RUNTIME_ACTIVE"


def test_qdic_scores_context_top64_but_decides_from_top8():
    fake = _FakeQDIC()
    overlay = TempoTrackOverlay(_config(), qdic=fake)
    histories = tuple(
        np.asarray([[np.cos(index), np.sin(index)]], dtype=np.float32)
        for index in np.linspace(0.0, 1.2, 10)
    )
    proposal = overlay.propose(_snapshot(histories=histories))
    assert len(fake.calls) == 1
    assert len(fake.calls[0]) == 10
    assert proposal.diagnostics["legal_candidate_count"] <= 8


def test_qdic_dormant_candidate_reenters_without_native_affinity():
    fake = _FakeQDIC()
    overlay = TempoTrackOverlay(_config(), qdic=fake)
    seed = _empty_snapshot(5)
    overlay.propose(seed)
    overlay.commit(seed, [42])
    proposal = overlay.propose(_empty_snapshot(10))
    assert proposal.assignments == (42,)
    assert proposal.diagnostics["native_candidate_count"] == 0
    assert len(fake.calls) == 1


def test_qdic_native_memo_bootstrap_is_fail_closed():
    fake = _FakeQDIC()
    overlay = TempoTrackOverlay(_config(), qdic=fake)
    with pytest.raises(SnapshotContractError, match="BLOCKED_LEARNED_ASSOC_NATIVE_MEMO_BOOTSTRAP"):
        overlay.propose(
            _snapshot(
                include_history=False,
                include_evidence=False,
            )
        )
    assert overlay._qdic_native_memo_bootstrap_count == 1


def test_qdic_and_legacy_reranker_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TempoTrackConfig(reranker_weight=1.0, qdic_weight=1.0)
