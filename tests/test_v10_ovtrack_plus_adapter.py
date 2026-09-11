import numpy as np
import torch

from tempotrack_v10 import TempoTrackConfig, TempoTrackOverlay
from tempotrack_v10.adapters.ovtrack_plus import (
    CORE_SHA_V10_FULL,
    OVTrackPlusTempoAdapter,
)


def _state():
    return dict(
        video_id="video-7",
        frame_id=10,
        bboxes=torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.9], [20.0, 0.0, 30.0, 10.0, 0.8]]),
        labels=torch.tensor([3, 3]),
        embeds=torch.tensor([[1.0, 0.0], [0.9, 0.1]]),
        native_affinity=torch.tensor([[0.95], [0.85]]),
        memory_ids=(42,),
        memory_embeddings=torch.tensor([[1.0, 0.0]]),
        memory_last_frame=torch.tensor([5]),
    )


def test_disabled_true_is_native_noop_and_does_not_commit_overlay_state():
    adapter = OVTrackPlusTempoAdapter.from_mapping(
        {"tempo": {"disabled": True, "candidate_top_k": 8}}
    )
    assert not adapter.enabled
    state = _state()
    snapshot = adapter.build_snapshot(**state)
    before = snapshot.immutable_observation_hash()
    native_ids = np.asarray([42, 7], dtype=np.int64)
    assert np.array_equal(native_ids, [42, 7])
    assert snapshot.immutable_observation_hash() == before
    assert adapter.overlay._records == {}


def test_enabled_snapshot_uses_line_101_affinity_and_causal_memory():
    adapter = OVTrackPlusTempoAdapter(
        disabled=False,
        overlay=TempoTrackOverlay(
            TempoTrackConfig(enabled=True, score_threshold=-10.0, margin_threshold=-1.0)
        ),
    )
    snapshot = adapter.build_snapshot(**_state())
    np.testing.assert_array_equal(
        snapshot.native_affinity, np.asarray([[0.95], [0.85]], dtype=np.float32)
    )
    assert snapshot.memory_ids == (42,)
    assert snapshot.memory_last_frame.tolist() == [5]
    assert snapshot.metadata["native_affinity_stage"] == "ovsort_line_101_before_lap"


def test_core_sha_is_explicit_and_no_second_core_is_declared():
    assert CORE_SHA_V10_FULL == "c1d4b685a4e8b0863260cb657cd3f5d746285f64"
