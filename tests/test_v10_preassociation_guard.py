import pytest
import torch

from tempotrack_v10.adapters.ovtrack import OVTrackTempoAdapter
from tempotrack_v10.contract import SnapshotContractError


def test_ovtrack_adapter_rejects_final_ids_before_snapshot():
    adapter = OVTrackTempoAdapter()
    with pytest.raises(SnapshotContractError, match="BLOCKED_POST_ASSOCIATION_INPUT"):
        adapter.build_snapshot(
            video_id=1,
            frame_id=4,
            bboxes=torch.tensor([[0.0, 0.0, 4.0, 4.0, 0.9]]),
            labels=torch.tensor([1]),
            embeddings=torch.tensor([[1.0, 0.0]]),
            native_affinity=torch.zeros((1, 0)),
            memory_ids=(),
            final_ids=torch.tensor([9]),
        )


def test_ovtrack_adapter_rejects_committed_id_metadata():
    adapter = OVTrackTempoAdapter()
    with pytest.raises(SnapshotContractError, match="BLOCKED_POST_ASSOCIATION_INPUT"):
        adapter.build_snapshot(
            video_id=1,
            frame_id=4,
            bboxes=torch.tensor([[0.0, 0.0, 4.0, 4.0, 0.9]]),
            labels=torch.tensor([1]),
            embeddings=torch.tensor([[1.0, 0.0]]),
            native_affinity=torch.zeros((1, 0)),
            memory_ids=(),
            metadata={"native_ids": [9]},
        )
