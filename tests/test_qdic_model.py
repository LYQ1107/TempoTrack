import numpy as np
import torch

from tempotrack_v10.qdic_features import build_qdic_candidate_features, build_qdic_event_features
from tempotrack_v10.query_distributional_calibrator import (
    FEATURE_INDEX,
    GATE_FEATURE_INDICES,
    GATE_FEATURE_NAMES,
    QueryDistributionalCalibrator,
)


def _event_features():
    rows = []
    for index in range(3):
        independent = build_qdic_candidate_features(
            np.asarray([[0.9 - index * 0.1, 0.2 + index * 0.1]], dtype=np.float32),
            np.full((2, 7), index + 1, dtype=np.float32),
            4,
            index + 1,
            query_fast_cosine=0.8 - index * 0.1,
            query_slow_cosine=0.7 - index * 0.1,
            fast_slow_cosine=0.95,
        )
        rows.append(
            {"base_features": independent[:19], "distributional_features": independent[19:]}
        )
    return build_qdic_event_features(rows)


def test_qdic_model_has_structured_zero_residual_initialization():
    torch.manual_seed(3)
    model = QueryDistributionalCalibrator()
    features = torch.from_numpy(_event_features())
    diagnostics = model(features, return_diagnostics=True)
    assert model.gate[0].in_features == 9
    assert model.gate[0].out_features == 16
    assert model.gate[2].out_features == 8
    assert model.residual_calibrator[0].in_features == 35
    assert model.residual_calibrator[0].out_features == 64
    assert model.residual_calibrator[3].out_features == 32
    assert torch.allclose(diagnostics["alpha"], torch.full((3,), 0.5))
    assert torch.allclose(diagnostics["residual"], torch.zeros(3))
    assert torch.allclose(
        diagnostics["logit"], diagnostics["structured_score"], atol=1e-6, rtol=0.0
    )
    assert torch.allclose(diagnostics["variance_penalty"], torch.full((3,), 0.1), atol=1e-6)
    assert bool(((diagnostics["alpha"] >= 0.0) & (diagnostics["alpha"] <= 1.0)).all())


def test_qdic_model_batch_and_single_row_paths_match_and_backpropagate():
    torch.manual_seed(4)
    model = QueryDistributionalCalibrator()
    features = torch.from_numpy(_event_features())
    batch = model(features)
    singles = torch.cat([model(features[index : index + 1]) for index in range(len(features))])
    assert batch.shape == (3,)
    torch.testing.assert_close(batch, singles, rtol=0.0, atol=1e-6)
    (batch.square().sum()).backward()
    assert model.gate[-2].weight.grad is not None
    assert model.residual_calibrator[-1].weight.grad is not None
    assert model.structured_scale.grad is not None
    assert model.variance_penalty_raw.grad is not None
    assert bool(torch.any(model.gate[-2].weight.grad.abs() > 0))
    assert bool(torch.any(model.residual_calibrator[-1].weight.grad.abs() > 0))


def test_qdic_gate_schema_is_name_driven_and_excludes_prefilter_rank():
    expected = (
        "query_fast_cosine",
        "query_slow_cosine",
        "fast_slow_cosine",
        "projected_fast_mean",
        "projected_fast_variance",
        "projected_slow_mean",
        "projected_slow_variance",
        "gap_over_max_gap",
        "memory_length_over_64",
    )
    assert GATE_FEATURE_NAMES == expected
    assert tuple(FEATURE_INDEX[name] for name in GATE_FEATURE_NAMES) == GATE_FEATURE_INDICES
    assert tuple(GATE_FEATURE_NAMES[-2:]) == ("gap_over_max_gap", "memory_length_over_64")
    assert "prefilter_rank" not in GATE_FEATURE_NAMES


def test_qdic_uncertainty_penalty_is_monotonic_with_zero_residual_and_half_gate():
    model = QueryDistributionalCalibrator().eval()
    features = np.zeros((2, 33), dtype=np.float32)
    for name, value in (
        ("query_fast_cosine", 0.8),
        ("query_slow_cosine", 0.8),
        ("projected_fast_mean", 0.7),
        ("projected_slow_mean", 0.7),
        ("fast_slow_cosine", 0.9),
        ("gap_over_max_gap", 0.1),
        ("memory_length_over_64", 0.5),
    ):
        features[:, FEATURE_INDEX[name]] = value
    features[1, FEATURE_INDEX["projected_fast_variance"]] = 0.5
    features[1, FEATURE_INDEX["projected_slow_variance"]] = 0.4
    features[0, FEATURE_INDEX["projected_fast_mo"]] = 0.7
    features[0, FEATURE_INDEX["projected_slow_mo"]] = 0.7
    features[1, FEATURE_INDEX["projected_fast_mo"]] = 0.95
    features[1, FEATURE_INDEX["projected_slow_mo"]] = 0.9
    diagnostics = model(torch.from_numpy(features), return_diagnostics=True)
    assert torch.allclose(diagnostics["alpha"], torch.full((2,), 0.5))
    assert torch.allclose(diagnostics["residual"], torch.zeros(2))
    assert diagnostics["variance_penalty"][0].item() > 0.0
    assert diagnostics["structured_score"][1] < diagnostics["structured_score"][0]
    np.testing.assert_allclose(
        diagnostics["structured_score"][0].item() - diagnostics["structured_score"][1].item(),
        0.5 * 0.1 * (0.5 + 0.4),
        atol=1e-6,
    )


def test_qdic_score_event_payload_matches_raw_33d_features():
    model = QueryDistributionalCalibrator()
    candidates = []
    for index in range(2):
        candidates.append(
            {
                "cosine": np.asarray([[0.9 - index * 0.2, 0.3]], dtype=np.float32),
                "evidence": np.ones((2, 7), dtype=np.float32),
                "gap": 4,
                "rank": index + 1,
                "query_fast_cosine": 0.8 - index * 0.1,
                "query_slow_cosine": 0.7 - index * 0.1,
                "fast_slow_cosine": 0.95,
            }
        )
    raw = model.build_event_features(candidates)
    payload_logits, payload_raw = model.score_event(candidates)
    raw_logits, raw_returned = model.score_event(raw)
    np.testing.assert_allclose(payload_raw, raw, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(payload_logits, raw_logits, rtol=0.0, atol=1e-7)
    assert raw_returned.shape == (2, 33)
