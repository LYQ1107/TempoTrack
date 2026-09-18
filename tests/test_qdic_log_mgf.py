import numpy as np
import pytest

from tempotrack_v10.qdic_features import (
    empirical_log_mgf,
    projected_log_mgf,
)


def test_constant_samples_are_constant_for_any_beta():
    samples = np.asarray([0.4, 0.4, 0.4], dtype=np.float64)
    for beta in (1e-10, 0.25, 1.0, 100.0):
        assert empirical_log_mgf(samples, beta=beta) == pytest.approx(0.4, abs=1e-10)


def test_beta_to_zero_matches_mean():
    samples = np.asarray([-0.7, 0.1, 0.9, 0.4], dtype=np.float64)
    assert empirical_log_mgf(samples, beta=1e-10) == pytest.approx(float(samples.mean()), abs=1e-9)


def test_positive_beta_is_at_least_the_mean():
    samples = np.asarray([-0.7, 0.1, 0.9, 0.4], dtype=np.float64)
    assert empirical_log_mgf(samples, beta=1.0) >= float(samples.mean()) - 1e-12


def test_large_beta_approaches_maximum():
    samples = np.asarray([-0.7, 0.1, 0.9, 0.4], dtype=np.float64)
    assert empirical_log_mgf(samples, beta=100.0) == pytest.approx(0.9, abs=0.02)


def test_beta_one_is_close_to_second_order_for_low_variance_samples():
    samples = np.asarray([0.39, 0.40, 0.41, 0.40], dtype=np.float64)
    exact = empirical_log_mgf(samples, beta=1.0)
    second_order = float(samples.mean() + 0.5 * np.var(samples, ddof=0))
    assert exact == pytest.approx(second_order, abs=1e-5)
    assert not np.isclose(exact, second_order, rtol=0.0, atol=0.0)


def test_singleton_is_exactly_the_single_sample():
    assert empirical_log_mgf(np.asarray([0.37]), beta=1.0) == pytest.approx(0.37, abs=1e-12)


def test_recent_and_full_history_definitions():
    values = np.arange(1, 11, dtype=np.float32) / 10.0
    fast, slow = projected_log_mgf(values, recent_k=8, beta=1.0)
    assert fast == pytest.approx(empirical_log_mgf(values[-8:], beta=1.0), abs=1e-7)
    assert slow == pytest.approx(empirical_log_mgf(values, beta=1.0), abs=1e-7)
    short_fast, short_slow = projected_log_mgf(values[:7], recent_k=8, beta=1.0)
    assert short_fast == pytest.approx(short_slow, abs=1e-7)


def test_finite_cosine_range_is_supported():
    values = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
    result = empirical_log_mgf(values, beta=1.0)
    assert np.isfinite(result)


@pytest.mark.parametrize("sample", [[], [np.nan], [np.inf], [-np.inf]])
def test_invalid_samples_fail_closed(sample):
    with pytest.raises((ValueError, FloatingPointError)):
        empirical_log_mgf(np.asarray(sample, dtype=np.float64), beta=1.0)
