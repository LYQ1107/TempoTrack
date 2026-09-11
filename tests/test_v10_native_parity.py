import numpy as np
import torch

from tempotrack_v10.adapters.ovtrack import OVTrackTempoAdapter, load_ovtrack_tempo_config


class _FakeOVTracker:
    def __init__(self):
        self.tracklets = {42: {"frame_ids": [5]}}

    @property
    def empty(self):
        return False

    @property
    def memo(self):
        return (
            torch.zeros((1, 5), dtype=torch.float32),
            torch.tensor([3], dtype=torch.long),
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            torch.tensor([42], dtype=torch.long),
        )


def _native_inputs():
    return dict(
        video_id="val/example",
        frame_id=10,
        bboxes=torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.9], [20.0, 0.0, 30.0, 10.0, 0.8]]),
        labels=torch.tensor([3, 3]),
        embeddings=torch.tensor([[1.0, 0.0], [0.9, 0.1]]),
        native_affinity=torch.tensor([[0.95], [0.85]]),
        tracker=_FakeOVTracker(),
    )


def test_disabled_adapter_preserves_native_ids_and_observations():
    adapter = OVTrackTempoAdapter.from_config("configs/research/v10/ovtrack.yaml")
    snapshot = adapter.build_snapshot(**_native_inputs())
    observation_hash = snapshot.immutable_observation_hash()
    native_ids = np.asarray([42, 7], dtype=np.int64)
    proposal = adapter.propose(snapshot)

    assert adapter.enabled is False
    assert proposal.assignments == (None, None)
    assert proposal.reasons == ("disabled_noop", "disabled_noop")
    np.testing.assert_array_equal(adapter.commit(snapshot, native_ids), native_ids)
    assert snapshot.immutable_observation_hash() == observation_hash
    np.testing.assert_array_equal(snapshot.boxes_xyxy, np.asarray([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.float32))


def test_enabled_proposal_maps_only_to_existing_native_ids():
    adapter = OVTrackTempoAdapter.from_config("configs/research/v10/ovtrack.yaml")
    adapter.config = adapter.config.__class__(
        enabled=True,
        config_path=adapter.config.config_path,
        overlay=adapter.config.overlay.__class__(enabled=True, score_threshold=0.1, margin_threshold=-1.0),
    )
    adapter.overlay = adapter.overlay.__class__(adapter.config.overlay)
    snapshot = adapter.build_snapshot(**_native_inputs())
    proposal = adapter.propose(snapshot)
    np.testing.assert_array_equal(adapter.preassigned_ids(proposal, 2), np.asarray([42, -1]))
