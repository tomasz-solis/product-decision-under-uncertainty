import numpy as np
import pandas as pd
import pytest

from simulator.mvp_simulator import (
    simulate_option_do_nothing,
    simulate_option_stabilize_core,
    simulate_option_feature_extension,
    simulate_option_new_capability,
    sample_params,
    _sample_param,
)
from simulator.config import ParamSpec


def test_sample_param_triangular():
    spec = ParamSpec(low=0.0, mode=0.5, high=1.0, dist="tri")
    rng = np.random.default_rng(42)

    samples = [_sample_param(spec, rng) for _ in range(100)]

    assert all(0.0 <= s <= 1.0 for s in samples)
    assert len(samples) == 100


def test_sample_param_uniform():
    spec = ParamSpec(low=0.0, mode=0.5, high=1.0, dist="uniform")
    rng = np.random.default_rng(42)

    samples = [_sample_param(spec, rng) for _ in range(100)]

    assert all(0.0 <= s <= 1.0 for s in samples)


def test_sample_param_unknown_dist():
    spec = ParamSpec(low=0.0, mode=0.5, high=1.0, dist="unknown")
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError, match="Unknown dist"):
        _sample_param(spec, rng)


def test_sample_params_shape():
    specs = {
        "param1": ParamSpec(low=0.0, mode=0.5, high=1.0, dist="tri"),
        "param2": ParamSpec(low=10.0, mode=20.0, high=30.0, dist="uniform"),
    }

    df = sample_params(50, specs, seed=7)

    assert df.shape == (50, 2)
    assert "param1" in df.columns
    assert "param2" in df.columns


def test_simulate_option_do_nothing_shape():
    p = pd.DataFrame({
        "baseline_failure_rate": [0.05, 0.1],
        "rev_per_success_eur": [100.0, 100.0],
        "cost_per_failure_eur": [50.0, 50.0],
        "base_churn": [0.01, 0.02],
        "failure_to_churn_rel": [0.5, 0.5],
        "do_nothing_drift_cost_eur": [1000.0, 2000.0],
    })

    result = simulate_option_do_nothing(p, volume=1000)

    assert result.shape == (2,)
    assert isinstance(result, np.ndarray)


def test_simulate_option_do_nothing_basic_calculation():
    p = pd.DataFrame({
        "baseline_failure_rate": [0.1],
        "rev_per_success_eur": [100.0],
        "cost_per_failure_eur": [50.0],
        "base_churn": [0.0],
        "failure_to_churn_rel": [0.0],
        "do_nothing_drift_cost_eur": [0.0],
    })

    result = simulate_option_do_nothing(p, volume=1000)

    # 1000 * 0.9 * 100 - 1000 * 0.1 * 50 = 90000 - 5000 = 85000
    assert result[0] == pytest.approx(85000.0)


def test_simulate_option_stabilize_core_shape():
    p = pd.DataFrame({
        "baseline_failure_rate": [0.1, 0.2],
        "stabilize_failure_reduction": [0.5, 0.3],
        "rev_per_success_eur": [100.0, 100.0],
        "cost_per_failure_eur": [50.0, 50.0],
        "base_churn": [0.01, 0.02],
        "failure_to_churn_rel": [0.5, 0.5],
        "regression_prob": [0.1, 0.2],
        "regression_cost_eur": [1000.0, 2000.0],
    })

    result = simulate_option_stabilize_core(p, volume=1000)

    assert result.shape == (2,)


def test_simulate_option_feature_extension_shape():
    p = pd.DataFrame({
        "baseline_failure_rate": [0.1],
        "extension_exposure_reduction": [0.2],
        "rev_per_success_eur": [100.0],
        "extension_uptake": [0.3],
        "extension_rev_per_uptake_eur": [50.0],
        "extension_loss_rate": [0.1],
        "cost_per_failure_eur": [50.0],
        "base_churn": [0.01],
        "failure_to_churn_rel": [0.5],
        "regression_prob": [0.1],
        "regression_cost_eur": [1000.0],
    })

    result = simulate_option_feature_extension(p, volume=1000)

    assert result.shape == (1,)


def test_simulate_option_new_capability_shape():
    p = pd.DataFrame({
        "baseline_failure_rate": [0.1],
        "rev_per_success_eur": [100.0],
        "new_capability_uplift": [0.2],
        "cost_per_failure_eur": [50.0],
        "new_capability_regression_multiplier": [1.5],
        "base_churn": [0.01],
        "failure_to_churn_rel": [0.5],
        "regression_prob": [0.1],
        "regression_cost_eur": [1000.0],
    })

    result = simulate_option_new_capability(p, volume=1000)

    assert result.shape == (1,)


def test_division_by_zero_protection_stabilize():
    # Test the 1e-9 protection in stabilize_core
    p = pd.DataFrame({
        "baseline_failure_rate": [0.0],
        "stabilize_failure_reduction": [0.5],
        "rev_per_success_eur": [100.0],
        "cost_per_failure_eur": [50.0],
        "base_churn": [0.01],
        "failure_to_churn_rel": [0.5],
        "regression_prob": [0.0],
        "regression_cost_eur": [0.0],
    })

    result = simulate_option_stabilize_core(p, volume=1000)

    assert np.isfinite(result[0])


def test_division_by_zero_protection_extension():
    # Test the 1e-9 protection in feature_extension
    p = pd.DataFrame({
        "baseline_failure_rate": [0.0],
        "extension_exposure_reduction": [0.5],
        "rev_per_success_eur": [100.0],
        "extension_uptake": [0.3],
        "extension_rev_per_uptake_eur": [50.0],
        "extension_loss_rate": [0.1],
        "cost_per_failure_eur": [50.0],
        "base_churn": [0.01],
        "failure_to_churn_rel": [0.5],
        "regression_prob": [0.0],
        "regression_cost_eur": [0.0],
    })

    result = simulate_option_feature_extension(p, volume=1000)

    assert np.isfinite(result[0])


def test_all_options_return_same_length():
    # All simulation functions should return same length as input
    p = pd.DataFrame({
        "baseline_failure_rate": [0.05, 0.1, 0.15],
        "stabilize_failure_reduction": [0.5, 0.4, 0.3],
        "extension_exposure_reduction": [0.2, 0.3, 0.4],
        "extension_uptake": [0.3, 0.4, 0.5],
        "extension_rev_per_uptake_eur": [50.0, 60.0, 70.0],
        "extension_loss_rate": [0.1, 0.15, 0.2],
        "new_capability_uplift": [0.2, 0.3, 0.4],
        "new_capability_regression_multiplier": [1.5, 1.6, 1.7],
        "rev_per_success_eur": [100.0, 110.0, 120.0],
        "cost_per_failure_eur": [50.0, 55.0, 60.0],
        "base_churn": [0.01, 0.02, 0.03],
        "failure_to_churn_rel": [0.5, 0.6, 0.7],
        "regression_prob": [0.1, 0.2, 0.3],
        "regression_cost_eur": [1000.0, 1500.0, 2000.0],
        "do_nothing_drift_cost_eur": [1000.0, 1500.0, 2000.0],
    })

    r1 = simulate_option_do_nothing(p, volume=1000)
    r2 = simulate_option_stabilize_core(p, volume=1000)
    r3 = simulate_option_feature_extension(p, volume=1000)
    r4 = simulate_option_new_capability(p, volume=1000)

    assert len(r1) == len(r2) == len(r3) == len(r4) == 3
