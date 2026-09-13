import numpy as np
import torch

from tempotrack_research.memory.fixed_dual import FixedDualMemory
from tempotrack_research.streaming.partial_support import replay_fixed_dual_prototypes


def test_replay_matches_the_shared_fixed_dual_update_for_every_raw_observation():
    values = np.asarray(
        [[1.0, 0.0], [0.99, 0.1], [-1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        dtype=np.float32,
    )
    expected_memory = FixedDualMemory(mode="fixed_dual", alpha_fast=0.70, alpha_slow=0.15)
    expected = expected_memory.initialize(torch.from_numpy(values[0]))
    for value in values[1:]:
        expected, _ = expected_memory.update(expected, torch.from_numpy(value), confidence=1.0)

    fast, slow = replay_fixed_dual_prototypes(values)
    np.testing.assert_allclose(fast, expected.fast.numpy(), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(slow, expected.slow.numpy(), rtol=0.0, atol=1e-6)


def test_replay_uses_v104_slow_rate_instead_of_fixed_dual_class_default():
    values = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    fast, slow = replay_fixed_dual_prototypes(values, alpha_fast=0.70, alpha_slow=0.15)
    default_slow = FixedDualMemory(mode="fixed_dual").initialize(torch.from_numpy(values[0]))
    default_slow, _ = FixedDualMemory(mode="fixed_dual").update(
        default_slow, torch.from_numpy(values[1]), confidence=1.0
    )
    assert not np.allclose(slow, default_slow.slow.numpy(), rtol=0.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(fast), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(slow), 1.0, atol=1e-6)
