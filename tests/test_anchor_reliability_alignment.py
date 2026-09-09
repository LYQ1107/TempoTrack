import numpy as np
import torch

from tempotrack_research.models.memory_reliability import MemoryReliabilityCalibrator
from tempotrack_research.streaming.partial_support import build_memory_anchor


def test_dedup_keeps_feature_evidence_and_row_alignment():
    features = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    boxes = np.asarray([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 20, 10]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    frames = np.asarray([0, 1, 2], dtype=np.int64)
    anchor = build_memory_anchor(
        fragment_id="1:2:0", root_id=2, video_id=1, rows=[0, 1, 2],
        features=features, boxes_xyxy=boxes, scores=scores, frames=frames,
    )
    assert anchor.features.shape == (2, 2)
    assert anchor.evidence.shape == (2, 7)
    assert len(anchor.row_indices) == len(anchor.features) == len(anchor.evidence)
    assert anchor.row_indices == [0, 2]
    assert anchor.evidence[0, 0] == scores[0]
    assert anchor.evidence[1, 0] == scores[2]

    model = MemoryReliabilityCalibrator()
    with torch.no_grad():
        model.net[-1].weight.fill_(1.0)
        model.net[-1].bias.zero_()
    reliability = model.reliability(torch.as_tensor(anchor.evidence))
    assert reliability.shape == (2,)
    assert float(reliability[0]) != float(reliability[1])
    assert torch.all((reliability > 0) & (reliability < 1))
