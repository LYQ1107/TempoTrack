import numpy as np
import pytest
import torch

from tempotrack_v10.candidate_aware_qdic import DistributionalAuxiliaryQDIC
from tempotrack_v10.deepset_qdic import DeepSetQDIC
from tempotrack_v10.dgsa_qdic import DistributionGuidedSetAttentionQDIC
from tempotrack_v10.distributional_identity import (
    DistributionEvidenceEncoder,
    DistributionIdentityHead,
)
from tempotrack_v10.distributional_losses import (
    distributional_ranking_loss,
    hard_negative_margin_loss,
    listwise_group_loss,
)
from tempotrack_v10.qdic_features import QDIC_RAW_DIM
from tempotrack_v10.query_distributional_calibrator import QueryDistributionalCalibrator
from tempotrack_v10.query_conditioned_reranker import group_ranking_loss


def _features(batch=2, candidates=6):
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(batch, candidates, QDIC_RAW_DIM, generator=generator)
    # Keep the fields used by the structured parent in a realistic finite
    # range; the tests concern shape/equivariance, not detector calibration.
    return values.float()


def _parent():
    torch.manual_seed(7)
    model = QueryDistributionalCalibrator().eval()
    model.feature_mean.zero_()
    model.feature_scale.fill_(1.0)
    return model


def test_distributional_moment_loss_is_multi_positive_and_masked():
    logits = torch.tensor([[2.0, 1.0, 0.0, 99.0]], requires_grad=True)
    labels = torch.tensor([[1, 1, 0, -1]])
    mask = torch.tensor([[True, True, True, True]])
    expected = torch.logsumexp(logits[0, :3], dim=0) - torch.logsumexp(logits[0, :2], dim=0)
    actual = distributional_ranking_loss(logits, labels, mask)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad[0, 3].item() == 0.0
    assert listwise_group_loss(logits.detach(), labels, mask) == pytest.approx(float(expected))


def test_distributional_loss_requires_positive_and_negative():
    with pytest.raises(ValueError, match="positive and a labeled negative"):
        distributional_ranking_loss(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[1, -1]])
        )


def test_distributional_encoder_and_head_shapes():
    encoder = DistributionEvidenceEncoder()
    head = DistributionIdentityHead()
    token = encoder(torch.zeros(3, 9))
    assert token.shape == (3, 32)
    score = head(token, torch.full((3,), 0.5), torch.zeros(3))
    assert score.shape == (3,)
    with pytest.raises(ValueError, match="end in \[9\]"):
        encoder(torch.zeros(3, 8))


@pytest.mark.parametrize("factory", [DeepSetQDIC, DistributionGuidedSetAttentionQDIC])
def test_zero_initialized_candidate_residual_matches_parent(factory):
    parent = _parent()
    values = _features(batch=2, candidates=6)
    with torch.inference_mode():
        expected = parent(values)
        model = factory(parent).eval()
        actual = model(values)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_a0_d_keeps_parent_final_logit_and_emits_distribution_score():
    parent = _parent()
    model = DistributionalAuxiliaryQDIC(parent).eval()
    values = _features(batch=2, candidates=5)
    with torch.inference_mode():
        parent_score = parent(values)
        details = model(values, return_diagnostics=True)
    torch.testing.assert_close(details["logit"], parent_score, rtol=0.0, atol=1e-6)
    assert details["distribution_logit"].shape == (2, 5)
    assert details["distribution_token"].shape == (2, 5, 32)


@pytest.mark.parametrize("factory", [DeepSetQDIC, DistributionGuidedSetAttentionQDIC])
def test_candidate_models_are_permutation_equivariant(factory):
    values = _features(batch=2, candidates=7)
    permutation = torch.tensor([4, 1, 6, 0, 3, 5, 2])
    parent = _parent()
    model = factory(parent).eval()
    with torch.inference_mode():
        original = model(values, return_diagnostics=True)
        shuffled = model(values[:, permutation], return_diagnostics=True)
    torch.testing.assert_close(
        shuffled["logit"], original["logit"][:, permutation], rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(
        shuffled["distribution_logit"],
        original["distribution_logit"][:, permutation],
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("factory", [DeepSetQDIC, DistributionGuidedSetAttentionQDIC])
def test_candidate_models_are_padding_invariant(factory):
    values = _features(batch=1, candidates=5)
    padded = torch.cat((values, torch.zeros(1, 3, QDIC_RAW_DIM)), dim=1)
    mask = torch.tensor([[True, True, True, True, True, False, False, False]])
    model = factory(_parent()).eval()
    with torch.inference_mode():
        expected = model(values)
        actual = model(padded, mask=mask)
    torch.testing.assert_close(actual[:, :5], expected, rtol=0.0, atol=1e-6)


def test_dgsa_duplicate_distractor_changes_attention_context_without_position_ids():
    values = _features(batch=1, candidates=4)
    model = DistributionGuidedSetAttentionQDIC(_parent()).eval()
    duplicate = values[:, 1:2].clone()
    augmented = torch.cat((values, duplicate), dim=1)
    with torch.inference_mode():
        first = model(values, return_diagnostics=True)["interaction_representation"]
        second = model(augmented, return_diagnostics=True)["interaction_representation"]
    assert not torch.allclose(first[:, 1], second[:, 1], rtol=0.0, atol=1e-7)
    assert not any("position" in name.lower() for name, _ in model.named_parameters())


def test_hard_loss_is_separate_from_listwise_loss():
    logits = torch.tensor([[1.0, 0.7, 0.2]])
    labels = torch.tensor([[1, 0, 0]])
    hard = hard_negative_margin_loss(logits, labels, margin=0.2)
    assert hard.item() == pytest.approx(0.0)


def test_old_group_ranking_loss_matches_refactored_default_components():
    logits = torch.tensor([[1.2, 0.4, 0.1], [0.8, 0.7, -0.2]], requires_grad=True)
    labels = torch.tensor([[1, 0, 0], [1, 1, 0]])
    old = group_ranking_loss(logits, labels)
    refactored = listwise_group_loss(logits, labels) + 0.2 * hard_negative_margin_loss(
        logits, labels, margin=0.2
    )
    torch.testing.assert_close(old, refactored, rtol=0.0, atol=1e-7)
