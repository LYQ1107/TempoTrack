import numpy as np
import pytest
from types import SimpleNamespace

from tempotrack_v10 import COVTrackTempoAdapter, SnapshotContractError, TempoTrackConfig


def cov_match_state():
    # This is the state after COVTrack's remove_distractor, confused feature
    # fusion, and final native affinity formation. It is not a detector stub.
    return {
        "video_id": 7,
        "frame_id": 10,
        "bboxes": np.asarray([[0, 0, 10, 10, 0.9], [20, 0, 30, 10, 0.8]], dtype=np.float32),
        "labels": np.asarray([3, 3], dtype=np.int64),
        "embeds": np.asarray([[1, 0], [1, 0]], dtype=np.float32),
        "scores": np.asarray([[0.95], [0.85]], dtype=np.float32),
        "memo_ids": (42,),
        "memo_embeds": np.asarray([[1, 0]], dtype=np.float32),
        "memo_last_frame": np.asarray([5], dtype=np.int64),
        "observation_uids": ("7:10:0", "7:10:1"),
    }


def test_disabled_adapter_is_native_id_and_state_noop():
    native_ids = np.asarray([42, -1], dtype=np.int64)
    adapter = COVTrackTempoAdapter(config=TempoTrackConfig(enabled=False))
    decision = adapter.prepare(**cov_match_state())

    assert np.array_equal(COVTrackTempoAdapter.native_seed(decision), [-1, -1])
    assert decision.proposal.reasons == ("disabled_noop", "disabled_noop")
    adapter.commit_after_native_ids(decision, native_ids)
    assert np.array_equal(native_ids, [42, -1])
    assert adapter.overlay._records == {}


def test_disabled_adapter_preserves_post_mcf_observations_and_affinity():
    state = cov_match_state()
    boxes_before = state["bboxes"].copy()
    embeds_before = state["embeds"].copy()
    affinity_before = state["scores"].copy()
    adapter = COVTrackTempoAdapter(config=TempoTrackConfig(enabled=False))
    decision = adapter.prepare(**state)

    assert decision.snapshot.metadata["association_stage"] == "pre_association"
    assert decision.snapshot.metadata["native_affinity_stage"] == "post_mcf_pre_id_commit"
    np.testing.assert_array_equal(decision.snapshot.boxes_xyxy, boxes_before[:, :4])
    np.testing.assert_array_equal(decision.snapshot.embeddings, embeds_before)
    np.testing.assert_array_equal(decision.snapshot.native_affinity, affinity_before)
    np.testing.assert_array_equal(state["bboxes"], boxes_before)
    np.testing.assert_array_equal(state["embeds"], embeds_before)
    np.testing.assert_array_equal(state["scores"], affinity_before)


def test_adapter_requires_causal_memo_timestamps_and_rejects_post_ids():
    state = cov_match_state()
    state.pop("memo_last_frame")
    with pytest.raises(SnapshotContractError, match="memo_last_frame"):
        COVTrackTempoAdapter().prepare(**state)

    state = cov_match_state()
    state["metadata"] = {"assigned_track_ids": [42, -1]}
    with pytest.raises(SnapshotContractError, match="BLOCKED_POST_ASSOCIATION_INPUT"):
        COVTrackTempoAdapter().prepare(**state)


def test_paper_runtime_gate_requires_explicit_no_gt_visualization():
    tracker = SimpleNamespace(
        match_score_thr=0.37,
        memo_frames=50,
        momentum_embed=0.4,
        confused_features=True,
        vis=False,
    )
    COVTrackTempoAdapter.assert_paper_runtime_gate(
        tracker=tracker,
        rcnn_test_cfg=SimpleNamespace(max_per_img=80),
        fusion_head=SimpleNamespace(max_fusion_ratio=2.0),
    )

    tracker.vis = True
    with pytest.raises(SnapshotContractError, match="COV_RUNTIME_GATE_FAILED"):
        COVTrackTempoAdapter.assert_paper_runtime_gate(
            tracker=tracker,
            rcnn_test_cfg=SimpleNamespace(max_per_img=80),
            fusion_head=SimpleNamespace(max_fusion_ratio=2.0),
        )

    tracker.vis = False
    tracker.filename2ann = {}
    with pytest.raises(SnapshotContractError, match="filename2ann"):
        COVTrackTempoAdapter.assert_paper_runtime_gate(
            tracker=tracker,
            rcnn_test_cfg=SimpleNamespace(max_per_img=80),
            fusion_head=SimpleNamespace(max_fusion_ratio=2.0),
        )
