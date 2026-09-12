import numpy as np
import pytest

from tempotrack_v10 import (
    FrameCollisionError,
    PreAssociationSnapshot,
    SnapshotContractError,
    TempoTrackConfig,
    TempoTrackOverlay,
)


def make_snapshot(*, frame=10, affinity=None, metadata=None, memory_last=5):
    boxes = np.asarray([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8], dtype=np.float32)
    labels = np.asarray([3, 3], dtype=np.int64)
    embeddings = np.asarray([[1, 0], [1, 0]], dtype=np.float32)
    if affinity is None:
        affinity = np.asarray([[0.95], [0.85]], dtype=np.float32)
    return PreAssociationSnapshot(
        video_id=7,
        frame_id=frame,
        boxes_xyxy=boxes,
        det_scores=scores,
        labels=labels,
        observation_uids=("7:10:0", "7:10:1"),
        embeddings=embeddings,
        native_affinity=np.asarray(affinity, dtype=np.float32),
        memory_ids=(42,),
        memory_embeddings=np.asarray([[1, 0]], dtype=np.float32),
        memory_last_frame=np.asarray([memory_last], dtype=np.int64),
        metadata={"association_stage": "pre_association", **(metadata or {})},
    )


def make_empty_memory_snapshot(*, frame=10, count=1, metadata=None, embeddings=None):
    if embeddings is None:
        embeddings = np.tile(np.asarray([[1, 0]], dtype=np.float32), (count, 1))
    return PreAssociationSnapshot(
        video_id=7,
        frame_id=frame,
        boxes_xyxy=np.tile(np.asarray([[0, 0, 10, 10]], dtype=np.float32), (count, 1)),
        det_scores=np.full(count, 0.9, dtype=np.float32),
        labels=np.full(count, 3, dtype=np.int64),
        observation_uids=tuple(f"7:{frame}:{index}" for index in range(count)),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        native_affinity=np.empty((count, 0), dtype=np.float32),
        memory_ids=(),
        memory_embeddings=np.empty((0, 2), dtype=np.float32),
        memory_last_frame=np.empty((0,), dtype=np.int64),
        metadata={"association_stage": "pre_association", **(metadata or {})},
    )


def test_snapshot_rejects_gt_and_post_association_fields():
    with pytest.raises(SnapshotContractError, match="BLOCKED_POST_ASSOCIATION_INPUT"):
        make_snapshot(metadata={"gt_identity": [1, 1]})
    with pytest.raises(SnapshotContractError, match="BLOCKED_POST_ASSOCIATION_INPUT"):
        make_snapshot(metadata={"assigned_track_ids": [42, 42]})


def test_snapshot_is_immutable_and_overlay_does_not_mutate_detector_fields():
    snapshot = make_snapshot()
    before = snapshot.immutable_observation_hash()
    boxes = snapshot.boxes_xyxy.copy()
    scores = snapshot.det_scores.copy()
    labels = snapshot.labels.copy()
    overlay = TempoTrackOverlay(TempoTrackConfig(score_threshold=0.1))
    proposal = overlay.propose(snapshot)
    overlay.commit(snapshot, [42, 43])
    assert proposal.observation_uids == snapshot.observation_uids
    assert snapshot.immutable_observation_hash() == before
    np.testing.assert_array_equal(snapshot.boxes_xyxy, boxes)
    np.testing.assert_array_equal(snapshot.det_scores, scores)
    np.testing.assert_array_equal(snapshot.labels, labels)


def test_enabled_false_is_exact_noop_and_reset_clears_state():
    snapshot = make_snapshot()
    overlay = TempoTrackOverlay(TempoTrackConfig(enabled=False))
    proposal = overlay.propose(snapshot)
    assert proposal.assignments == (None, None)
    assert proposal.reasons == ("disabled_noop", "disabled_noop")
    overlay.commit(snapshot, [-1, -1])
    overlay.reset(7)
    assert overlay._records == {}


def test_deterministic_same_snapshot_has_same_proposal():
    config = TempoTrackConfig(score_threshold=0.1, margin_threshold=-1.0)
    first = TempoTrackOverlay(config).propose(make_snapshot())
    second = TempoTrackOverlay(config).propose(make_snapshot())
    assert first == second


def test_event_local_competition_has_loser_without_fallback():
    snapshot = make_snapshot(affinity=[[0.95], [0.90]])
    overlay = TempoTrackOverlay(TempoTrackConfig(score_threshold=0.1, margin_threshold=-1.0))
    proposal = overlay.propose(snapshot)
    assert proposal.assignments == (42, None)
    assert proposal.reasons[1] == "competition_loser"
    assert proposal.diagnostics["competition_losers"] == 1


def test_frame_collision_guard_rejects_occupied_and_duplicate_commit():
    snapshot = make_snapshot(metadata={"occupied_ids": [42]})
    overlay = TempoTrackOverlay(TempoTrackConfig(score_threshold=0.1, margin_threshold=-1.0))
    proposal = overlay.propose(snapshot)
    assert proposal.assignments == (None, None)
    assert proposal.reasons[0] == "frame_collision"

    clean = make_snapshot()
    overlay = TempoTrackOverlay(TempoTrackConfig(score_threshold=0.1))
    overlay.propose(clean)
    with pytest.raises(FrameCollisionError):
        overlay.commit(clean, [42, 42])


def test_causal_gap_and_top_k_are_enforced_independently():
    snapshot = make_snapshot(frame=10, memory_last=9)
    overlay = TempoTrackOverlay(TempoTrackConfig(score_threshold=-10.0, min_gap=2))
    proposal = overlay.propose(snapshot)
    assert proposal.assignments == (None, None)
    assert all(reason == "no_legal_candidate" for reason in proposal.reasons)


def test_candidate_order_tie_break_is_by_memory_id():
    snapshot = PreAssociationSnapshot(
        video_id=7,
        frame_id=10,
        boxes_xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
        det_scores=np.asarray([0.9], dtype=np.float32),
        labels=np.asarray([3], dtype=np.int64),
        observation_uids=("7:10:0",),
        embeddings=np.asarray([[1, 0]], dtype=np.float32),
        native_affinity=np.asarray([[0.9, 0.9]], dtype=np.float32),
        memory_ids=(43, 42),
        memory_embeddings=np.asarray([[1, 0], [1, 0]], dtype=np.float32),
        memory_last_frame=np.asarray([5, 5], dtype=np.int64),
        metadata={"association_stage": "pre_association"},
    )
    proposal = TempoTrackOverlay(TempoTrackConfig(score_threshold=-10.0)).propose(snapshot)
    assert proposal.assignments == (42,)


def test_competition_is_local_to_one_event():
    config = TempoTrackConfig(score_threshold=0.1, margin_threshold=-1.0)
    overlay = TempoTrackOverlay(config)
    first = make_snapshot(frame=10, affinity=[[0.95], [0.90]])
    first_proposal = overlay.propose(first)
    overlay.commit(first, [42, -1])

    later = make_snapshot(frame=20, affinity=[[0.95], [0.90]], memory_last=10)
    later_proposal = overlay.propose(later)
    assert later_proposal.assignments[0] == 42
    assert later_proposal.reasons[1] == "competition_loser"


def test_post_association_stage_is_rejected_even_without_gt_fields():
    with pytest.raises(SnapshotContractError, match="BLOCKED_POST_ASSOCIATION_INPUT"):
        make_snapshot(metadata={"association_stage": "post_association"})


def test_future_memory_frame_is_rejected_by_snapshot_contract():
    with pytest.raises(SnapshotContractError, match="BLOCKED_POST_ASSOCIATION_INPUT"):
        make_snapshot(frame=10, memory_last=10)


def test_dormant_union_reactivates_without_native_affinity():
    overlay = TempoTrackOverlay(TempoTrackConfig(score_threshold=-10.0, margin_threshold=-1.0))
    seed = make_snapshot(frame=5, memory_last=1)
    overlay.propose(seed)
    overlay.commit(seed, [42, -1])
    proposal = overlay.propose(make_empty_memory_snapshot(frame=10))
    assert proposal.assignments == (42,)
    assert proposal.diagnostics["native_candidate_count"] == 0
    assert proposal.diagnostics["dormant_candidate_count"] == 1


def test_dormant_candidate_respects_expired_gap():
    overlay = TempoTrackOverlay(TempoTrackConfig(min_gap=6, max_gap=20, score_threshold=-10.0))
    seed = make_snapshot(frame=5, memory_last=1)
    overlay.propose(seed)
    overlay.commit(seed, [42, -1])
    proposal = overlay.propose(make_empty_memory_snapshot(frame=10))
    assert proposal.assignments == (None,)
    assert proposal.reasons == ("no_legal_candidate",)


def test_dormant_root_lineage_competition_is_event_local():
    overlay = TempoTrackOverlay(TempoTrackConfig(score_threshold=-10.0, margin_threshold=-1.0))
    seed = make_empty_memory_snapshot(frame=5, count=2, metadata={"observation_root_ids": [7, 7]})
    overlay.propose(seed)
    overlay.commit(seed, [42, 43])
    proposal = overlay.propose(make_empty_memory_snapshot(frame=10, count=2))
    assert sum(proposal.accepted) == 1
    assert proposal.reasons.count("competition_loser") == 1
    assert proposal.assignments[1] is None


def test_dormant_frame_collision_rejects_without_fallback():
    overlay = TempoTrackOverlay(TempoTrackConfig(score_threshold=-10.0, margin_threshold=-1.0))
    seed = make_snapshot(frame=5, memory_last=1)
    overlay.propose(seed)
    overlay.commit(seed, [42, -1])
    proposal = overlay.propose(make_empty_memory_snapshot(frame=10, metadata={"occupied_ids": [42]}))
    assert proposal.assignments == (None,)
    assert proposal.reasons == ("frame_collision",)


def test_dormant_candidate_order_is_deterministic_after_union():
    def run_once():
        overlay = TempoTrackOverlay(TempoTrackConfig(score_threshold=-10.0, margin_threshold=-1.0))
        seed = make_empty_memory_snapshot(frame=5, count=2)
        overlay.propose(seed)
        overlay.commit(seed, [43, 42])
        return overlay.propose(make_empty_memory_snapshot(frame=10)).assignments

    assert run_once() == run_once() == (42,)


def test_exact_v9_reranker_is_used_with_controlled_provenance():
    checkpoint = "/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/reranker/covtrack/training_seed0/best.pt"
    with pytest.raises(SnapshotContractError, match="FEATURE_CONFIG"):
        TempoTrackOverlay(
            TempoTrackConfig(
                score_threshold=-100.0,
                margin_threshold=-1.0,
                reranker_weight=1.0,
                reranker_checkpoint=checkpoint,
            )
        )


def test_reranker_missing_evidence_fails_closed_without_heuristic_fallback():
    checkpoint = "/data2/usr_for_deadline/tempotrack_v9_relocated_20260910/v9_3/reranker/covtrack/training_seed0/best.pt"
    with pytest.raises(SnapshotContractError, match="FEATURE_CONFIG"):
        TempoTrackOverlay(
            TempoTrackConfig(
                score_threshold=-100.0,
                margin_threshold=-1.0,
                reranker_weight=1.0,
                reranker_checkpoint=checkpoint,
            ),
        )
