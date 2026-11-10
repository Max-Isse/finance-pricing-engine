import numpy as np
import pytest

from monitoring.drift_monitors import calculate_psi, classify_drift, run_drift_check


def test_psi_is_small_for_samples_from_the_same_distribution():
    rng = np.random.default_rng(0)
    assert calculate_psi(rng.normal(0.041, 0.002, 5000), rng.normal(0.041, 0.002, 5000)) < 0.10


def test_psi_flags_a_shifted_distribution():
    rng = np.random.default_rng(0)
    psi = calculate_psi(rng.normal(0.041, 0.002, 1000), rng.normal(0.049, 0.002, 1000))
    assert psi > 0.25
    assert classify_drift(psi) == "SIGNIFICANT_DRIFT"


def test_psi_counts_live_values_outside_the_baseline_range():
    baseline = np.linspace(0, 1, 1000)
    inside = calculate_psi(baseline, np.full(500, 0.95))
    outside = calculate_psi(baseline, np.full(500, 5.0))
    assert outside == pytest.approx(inside)
    assert outside > 0.25


def test_psi_is_zero_for_a_constant_baseline():
    assert calculate_psi(np.ones(100), np.ones(50)) == 0.0


def test_psi_rejects_empty_samples():
    with pytest.raises(ValueError):
        calculate_psi([], [1.0, 2.0])


def test_classification_thresholds():
    assert classify_drift(0.05) == "STABLE"
    assert classify_drift(0.15) == "MODERATE_DRIFT"
    assert classify_drift(0.30) == "SIGNIFICANT_DRIFT"


def test_drift_check_runs_without_redis():
    rng = np.random.default_rng(1)
    result = run_drift_check(rng.normal(size=500), rng.normal(size=500))
    assert result["status"] == "STABLE"
