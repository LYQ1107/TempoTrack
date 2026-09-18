import numpy as np
import pytest
import torch

from tempotrack_v10.qdic_features import (
    QDIC_MGF_FEATURE_NAMES,
    QDIC_MGF_RAW_DIM,
    build_qdic_mgf_candidate_features,
    build_qdic_mgf_event_features,
)
from tempotrack_v10.query_mgf_calibrator import QueryMomentGeneratingCalibrator


def _features():
    values = np.zeros((3, QDIC_MGF_RAW_DIM), dtype=np.float32)
    values[:, 0] = (0.2, 0.3, 0.4)
    values[:, 24] = (0.2, 0.3, 0.4)
    values[:, 25] = (0.1, 0.2, 0.3)
    values[:, 27] = (0.15, 0.25, 0.35)
    values[:, 29] = (0.1, 0.2, 0.3)
    values[:, 32] = (0.12, 0.22, 0.32)
    values[:, 33] = (0.21, 0.31, 0.41)
    values[:, 34] = (0.18, 0.28, 0.38)
    return values


def test_mgf_model_has_frozen_schema_and_two_modes():
    core = QueryMomentGeneratingCalibrator(mgf_mode="core")
    fused = QueryMomentGeneratingCalibrator(mgf_mode="fused")
    assert core.input_dim == QDIC_MGF_RAW_DIM == 35
    assert tuple(core.feature_names) == tuple(QDIC_MGF_FEATURE_NAMES)
    assert core.mgf_beta == 1.0
    with pytest.raises(ValueError):
        QueryMomentGeneratingCalibrator(mgf_mode="beta_sweep")

    values = torch.from_numpy(_features())
    core.eval()
    fused.eval()
    with torch.inference_mode():
        core_diag = core(values, return_diagnostics=True)
        fused_diag = fused(values, return_diagnostics=True)
    assert torch.allclose(core_diag["fast_branch"], values[:, 33])
    assert torch.allclose(core_diag["slow_branch"], values[:, 34])
    assert torch.allclose(fused_diag["fast_branch"], 0.5 * (values[:, 24] + values[:, 33]))
    assert torch.allclose(fused_diag["slow_branch"], 0.5 * (values[:, 25] + values[:, 34]))
    assert torch.isfinite(core_diag["logit"]).all()


def test_normalization_is_35_dim_and_score_event_returns_diagnostics():
    model = QueryMomentGeneratingCalibrator(mgf_mode="fused")
    model.set_normalization(np.zeros(35, dtype=np.float32), np.ones(35, dtype=np.float32))
    cosine = np.asarray([0.1, 0.2, 0.4, 0.5], dtype=np.float32)
    evidence = np.full((len(cosine), 7), 0.1, dtype=np.float32)
    payload = [
        {
            "cosine": cosine[None, :],
            "evidence": evidence,
            "gap": 3,
            "rank": 1,
            "query_fast_cosine": 0.3,
            "query_slow_cosine": 0.2,
            "fast_slow_cosine": 0.8,
        }
    ]
    expected_row = build_qdic_mgf_candidate_features(
        cosine[None, :], evidence, 3, 1,
        query_fast_cosine=0.3,
        query_slow_cosine=0.2,
        fast_slow_cosine=0.8,
    )
    expected = build_qdic_mgf_event_features(
        [{"base_features": expected_row[:19], "distributional_features": expected_row[19:]}]
    )
    logits, details = model.score_event(payload, return_diagnostics=True)
    assert logits.shape == (1,)
    assert np.isfinite(logits).all()
    assert details["mgf_beta"] == 1.0
    assert details["mgf_mode"] == "fused"
    np.testing.assert_allclose(details["projected_fast_log_mgf"], expected[:, 33], atol=1e-7)
    np.testing.assert_allclose(details["projected_slow_log_mgf"], expected[:, 34], atol=1e-7)


def test_model_rejects_v11_width():
    model = QueryMomentGeneratingCalibrator()
    with pytest.raises(ValueError):
        model(torch.zeros((1, 33)))
