import numpy as np
import pytest

from tempotrack_v10.qdic_features import projected_distribution_moments


def test_projected_mean_and_variance_match_linear_projection_identity():
    z = np.asarray([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    q = np.asarray([0.6, 0.8], dtype=np.float64)
    projected = q @ z.T
    mean = z.mean(axis=0)
    covariance = ((z - mean).T @ (z - mean)) / len(z)
    result = projected_distribution_moments(projected.astype(np.float32), recent_k=8)
    np.testing.assert_allclose(result[0], q @ mean, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result[1], q @ covariance @ q, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result[3], q @ mean, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result[4], q @ covariance @ q, rtol=0.0, atol=1e-6)


def test_projected_moments_use_population_variance_and_recent_window():
    result = projected_distribution_moments(
        np.asarray([[0.1, 0.2, 0.3, 0.9]], dtype=np.float32), recent_k=2
    )
    np.testing.assert_allclose(
        result,
        (0.6, 0.09, 0.645, 0.375, 0.096875, 0.4234375),
        rtol=0.0,
        atol=1e-7,
    )


def test_projected_moments_singleton_variance_is_zero():
    result = projected_distribution_moments(np.asarray([[0.7]], dtype=np.float32), recent_k=8)
    assert result[0] == pytest.approx(0.7, abs=1e-7)
    assert result[1] == 0.0
    assert result[2] == pytest.approx(0.7, abs=1e-7)
    assert result[3:] == pytest.approx((0.7, 0.0, 0.7), abs=1e-7)


def test_projected_moments_accepts_q1_and_rejects_invalid_inputs():
    np.testing.assert_allclose(
        projected_distribution_moments(np.asarray([0.2, 0.4], dtype=np.float32), recent_k=8),
        projected_distribution_moments(np.asarray([[0.2, 0.4]], dtype=np.float32), recent_k=8),
    )
    with pytest.raises(ValueError, match="recent_k"):
        projected_distribution_moments(np.ones((1, 2), dtype=np.float32), recent_k=0)
    with pytest.raises(ValueError, match="nonempty"):
        projected_distribution_moments(np.empty((1, 0), dtype=np.float32))
