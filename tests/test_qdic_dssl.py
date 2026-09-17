import numpy as np
import pytest
import torch

from tempotrack_v10.distributional_losses import (
    gate_aware_temporal_js_consistency,
    masked_candidate_softmax,
    temporal_conflict_hard_negative_mask,
)
from tempotrack_v10.qdic_dssl_trainer import build_official_train_groups
from tempotrack_v10.qdic_features import QDIC_RAW_DIM
from tempotrack_v10.query_distributional_calibrator import QueryDistributionalCalibrator


def test_dssl_has_independent_support_softmax_and_positive_penalties():
    model = QueryDistributionalCalibrator(structured_branch_mode="dssl").eval()
    diagnostics = model(torch.zeros(2, 4, QDIC_RAW_DIM), return_diagnostics=True)
    assert torch.allclose(
        diagnostics["recent_support_weights"].sum(dim=-1), torch.ones(2, 4)
    )
    assert torch.allclose(
        diagnostics["long_support_weights"].sum(dim=-1), torch.ones(2, 4)
    )
    assert bool((diagnostics["recent_variance_penalty"] > 0).all())
    assert bool((diagnostics["long_variance_penalty"] > 0).all())


def test_masked_softmax_and_conflict_mask_are_fail_closed():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    result = masked_candidate_softmax(logits, mask)
    assert torch.isclose(result.sum(), torch.tensor(1.0))
    assert result[0, 1].item() == 0.0
    labels = torch.tensor([[1, 0, 0, -1]])
    conflict = torch.tensor([[False, True, False, True]])
    selected = temporal_conflict_hard_negative_mask(labels, conflict)
    assert selected.tolist() == [[True, True, False, False]]
    value = gate_aware_temporal_js_consistency(logits, logits + 0.1, torch.tensor([0.5]), mask)
    assert torch.isfinite(value)


def test_official_train_split_is_complete_video_disjoint():
    arrays = {
        "offsets": np.asarray([0, 3, 6, 9, 12]),
        "videos": np.asarray([1, 2, 3, 4]),
        "labels": np.asarray([1, 0, 0] * 4),
        "supervision_allowed": np.ones(12, dtype=bool),
    }
    train, holdout = build_official_train_groups(arrays, holdout_modulus=2)
    assert train and holdout
    assert set(train).isdisjoint(holdout)


def test_dssl_rejects_non_official_train_metadata():
    from tempotrack_v10.qdic_dssl_trainer import _official_train_guard

    with pytest.raises(ValueError, match="OFFICIAL_TRAIN"):
        _official_train_guard({
            "artifact": "qdic_v11_feature_cache",
            "feature_names": [],
            "feature_dim": 33,
            "source_role": "OFFICIAL_VAL",
        })
