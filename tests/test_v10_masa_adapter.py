from types import SimpleNamespace

import numpy as np
import torch

from tempotrack_v10 import TempoTrackConfig, TempoTrackOverlay
from tempotrack_v10.adapters.masa import (
    MasaTaoPreAssociationAdapter,
    install_masa_overlay,
)


def memory():
    return {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([3]),
        "embeds": torch.tensor([[1.0, 0.0]]),
        "ids": torch.tensor([42]),
        "frame_ids": torch.tensor([5]),
    }


def inputs():
    return dict(
        video_id=7,
        frame_id=10,
        bboxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        labels=torch.tensor([3]),
        scores=torch.tensor([0.9]),
        embeds=torch.tensor([[1.0, 0.0]]),
        native_affinity=torch.tensor([[0.95]]),
        memory=memory(),
    )


class FakeTracker:
    def __init__(self):
        self.attached = None
        self.native_ids = torch.tensor([42])

    def _assign_matches(self, native_affinity, memo_ids, scores):
        assert torch.equal(native_affinity, torch.tensor([[0.95]]))
        assert torch.equal(memo_ids, torch.tensor([42]))
        trace = SimpleNamespace(
            frame_id=-1,
            ids=self.native_ids.clone(),
            accepted=torch.tensor([True]),
            accepted_score=torch.tensor([0.95]),
            detection_margin=torch.tensor([float("nan")]),
            assigned_memo_index=torch.tensor([0]),
            score_matrix=None,
            fast_score=None,
            slow_score=None,
        )
        return self.native_ids.clone(), trace

    def set_pre_association_adapter(self, adapter):
        self.attached = adapter


def test_snapshot_uses_masa_native_affinity_and_forbids_post_state():
    adapter = MasaTaoPreAssociationAdapter(TempoTrackOverlay())
    snapshot = adapter.build_snapshot(**inputs())
    assert snapshot.metadata["association_stage"] == "pre_association"
    assert snapshot.metadata["pre_association"] is True
    np.testing.assert_array_equal(snapshot.native_affinity, np.asarray([[0.95]], dtype=np.float32))
    assert snapshot.memory_ids == (42,)
    assert not any("track_id" in key for key in snapshot.metadata)


def test_disabled_adapter_delegates_to_native_assignment_exactly():
    tracker = FakeTracker()
    adapter = MasaTaoPreAssociationAdapter(
        TempoTrackOverlay(TempoTrackConfig(enabled=False))
    )
    decision = adapter.decide(tracker=tracker, **inputs())
    assert not adapter.enabled
    assert torch.equal(decision.ids, tracker.native_ids)
    adapter.commit(decision, decision.ids)


def test_enabled_adapter_maps_overlay_assignment_before_commit():
    tracker = FakeTracker()
    adapter = MasaTaoPreAssociationAdapter(
        TempoTrackOverlay(TempoTrackConfig(score_threshold=0.1, margin_threshold=-1.0))
    )
    decision = adapter.decide(tracker=tracker, **inputs())
    assert adapter.enabled
    assert decision.proposal.assignments == (42,)
    assert torch.equal(decision.ids, torch.tensor([42]))
    adapter.commit(decision, decision.ids)


def test_enabled_adapter_reactivates_dormant_identity_without_memo_index():
    overlay = TempoTrackOverlay(
        TempoTrackConfig(score_threshold=-10.0, margin_threshold=-1.0)
    )
    adapter = MasaTaoPreAssociationAdapter(overlay)
    first = adapter.decide(tracker=FakeTracker(), **inputs())
    adapter.commit(first, torch.tensor([42]))

    dormant_inputs = inputs()
    dormant_inputs.update(
        frame_id=20,
        native_affinity=torch.empty((1, 0)),
        memory={
            "bboxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.long),
            "embeds": torch.empty((0, 2)),
            "ids": torch.empty((0,), dtype=torch.long),
            "frame_ids": torch.empty((0,), dtype=torch.long),
        },
    )
    decision = adapter.decide(tracker=FakeTracker(), **dormant_inputs)
    assert decision.proposal.assignments == (42,)
    assert torch.equal(decision.ids, torch.tensor([42]))
    assert torch.equal(decision.trace.assigned_memo_index, torch.tensor([-1]))


def test_install_attaches_only_the_thin_adapter():
    tracker = FakeTracker()
    overlay = TempoTrackOverlay(TempoTrackConfig(enabled=False))
    adapter = install_masa_overlay(tracker, overlay)
    assert tracker.attached is adapter
    assert tracker.attached.overlay is overlay


def test_adapter_reset_delegates_to_shared_overlay():
    overlay = TempoTrackOverlay()
    adapter = MasaTaoPreAssociationAdapter(overlay)
    snapshot = adapter.build_snapshot(**inputs())
    overlay.propose(snapshot)
    assert overlay._pending
    adapter.reset(7)
    assert not overlay._pending


def test_future_memory_is_rejected_before_any_overlay_decision():
    adapter = MasaTaoPreAssociationAdapter(TempoTrackOverlay())
    bad = inputs()
    bad["frame_id"] = 5
    import pytest

    with pytest.raises(ValueError, match="current/future frame"):
        adapter.build_snapshot(**bad)
