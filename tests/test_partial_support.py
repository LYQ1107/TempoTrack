import torch

from tempotrack_research.analysis.partial_support import PartialSupportConfig, PartialSupportScorer
from tempotrack_research.models.memory_reliability import MemoryReliabilityCalibrator


def test_top_r_is_per_query_and_masks_padding():
    scorer = PartialSupportScorer(PartialSupportConfig(top_r=2, memory_capacity=64))
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    memory = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [9.0, 9.0]])
    result = scorer(query, memory, memory_mask=torch.tensor([True, True, True, False]))
    assert torch.isfinite(result.score)
    assert result.top_indices.shape == (2, 2)
    assert not torch.any(result.top_indices == 3)


def test_reliability_calibrator_has_required_shape_and_gradient():
    model = MemoryReliabilityCalibrator()
    x = torch.randn(8, 7)
    y = model(x).sum() + model.reliability(x).sum()
    y.backward()
    assert tuple(model(x).shape) == (8,)
    assert model.net[0].weight.grad is not None
    assert model.log_rel_scale.grad is not None
