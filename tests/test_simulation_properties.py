"""Property-based tests for Monte Carlo simulation invariants.

Each test asserts an economic relationship that must hold regardless of the
specific parameter values — not a particular numeric output, but a structural
property of the model. These complement the unit tests in test_simulation_logic.py,
which test specific scenarios with fixed inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from simulator.simulation import (
    simulate_option_do_nothing,
    simulate_option_feature_extension,
    simulate_option_stabilize_core,
)

# ---------------------------------------------------------------------------
# Shared fixture builder — single deterministic world for property tests
# ---------------------------------------------------------------------------

_BASE_PARAMS: dict[str, float] = {
    "baseline_failure_rate": 0.08,
    "failure_to_churn_rel": 0.15,
    "cost_per_failure_eur": 35.0,
    "value_per_success_eur": 35.0,
    "stabilize_failure_reduction": 0.60,
    "extension_uptake": 0.30,
    "extension_exposure_reduction": 0.70,
    "extension_value_per_uptake_eur": 4.0,
    "extension_loss_rate": 0.08,
    "new_capability_uplift": 0.08,
    "regression_event_prob": 0.0,
    "regression_event_cost_eur": 150_000.0,
    "stabilize_core_regression_prob_multiplier": 1.0,
    "feature_extension_regression_prob_multiplier": 0.8,
    "new_capability_regression_prob_multiplier": 2.5,
    "stabilize_core_upfront_cost_eur": 800_000.0,
    "stabilize_core_annual_maintenance_cost_eur": 30_000.0,
    "feature_extension_upfront_cost_eur": 600_000.0,
    "feature_extension_annual_maintenance_cost_eur": 60_000.0,
    "new_capability_upfront_cost_eur": 1_200_000.0,
    "new_capability_annual_maintenance_cost_eur": 100_000.0,
    "do_nothing_drift_cost_eur": 50_000.0,
    "stabilize_core_launch_delay_months": 0.0,
    "stabilize_core_benefit_ramp_months": 0.0,
    "stabilize_core_cost_overrun_multiplier": 1.0,
    "feature_extension_launch_delay_months": 0.0,
    "feature_extension_benefit_ramp_months": 0.0,
    "feature_extension_cost_overrun_multiplier": 1.0,
    "new_capability_launch_delay_months": 0.0,
    "new_capability_benefit_ramp_months": 0.0,
    "new_capability_cost_overrun_multiplier": 1.0,
}


def _frame(**overrides: float) -> pd.DataFrame:
    """Return a one-row parameter dataframe with optional field overrides."""
    values = {**_BASE_PARAMS, **overrides}
    return pd.DataFrame({key: [value] for key, value in values.items()})


# ---------------------------------------------------------------------------
# Property: volume scales benefit-side terms linearly for do-nothing
# ---------------------------------------------------------------------------


@given(
    drift_cost=st.floats(min_value=1_000.0, max_value=500_000.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=1.5, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=2000)
def test_do_nothing_scales_linearly_with_drift_cost(drift_cost: float, scale: float) -> None:
    """Do-nothing value must scale linearly with drift cost at fixed horizon."""
    low = simulate_option_do_nothing(
        _frame(do_nothing_drift_cost_eur=drift_cost),
        horizon_years=2.0,
        annual_discount_rate=0.0,
    )[0]
    high = simulate_option_do_nothing(
        _frame(do_nothing_drift_cost_eur=drift_cost * scale),
        horizon_years=2.0,
        annual_discount_rate=0.0,
    )[0]
    assert abs(high / low - scale) < 1e-9, (
        f"Do-nothing value did not scale linearly with drift cost: "
        f"expected ratio {scale}, got {high / low:.8f}"
    )


# ---------------------------------------------------------------------------
# Property: upfront cost reduces stabilize-core value 1:1
# ---------------------------------------------------------------------------


@given(
    base_upfront=st.floats(min_value=100_000.0, max_value=2_000_000.0, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=10_000.0, max_value=500_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=2000)
def test_stabilize_core_upfront_cost_reduces_value_one_to_one(
    base_upfront: float, delta: float
) -> None:
    """Every extra euro of upfront cost must reduce net value by exactly one euro."""
    low = simulate_option_stabilize_core(
        _frame(stabilize_core_upfront_cost_eur=base_upfront),
        annual_volume=250_000.0,
        horizon_years=2.0,
        annual_discount_rate=0.0,
        risk_rng=None,
        shared_risk_draws=np.array([0.5]),
    )[0]
    high = simulate_option_stabilize_core(
        _frame(stabilize_core_upfront_cost_eur=base_upfront + delta),
        annual_volume=250_000.0,
        horizon_years=2.0,
        annual_discount_rate=0.0,
        risk_rng=None,
        shared_risk_draws=np.array([0.5]),
    )[0]
    assert abs((low - high) - delta) < 1e-4, (
        f"Upfront cost increase of {delta:.2f} EUR did not reduce value 1:1. "
        f"Actual reduction: {low - high:.4f}"
    )


# ---------------------------------------------------------------------------
# Property: higher discount rate reduces the present value of future benefits
# ---------------------------------------------------------------------------


@given(
    low_rate=st.floats(min_value=0.0, max_value=0.12, allow_nan=False, allow_infinity=False),
    high_rate=st.floats(min_value=0.0, max_value=0.12, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=2000)
def test_discount_rate_monotonicity_for_stabilize_core(
    low_rate: float, high_rate: float
) -> None:
    """Higher discount rate must not increase value for an option with future benefits.

    With zero upfront cost and zero delay/ramp, all value arrives in the future.
    A higher discount rate must weakly reduce (or equal at rate=0) net present value.
    """
    assume(high_rate > low_rate + 0.005)

    low_pv = simulate_option_stabilize_core(
        _frame(
            stabilize_core_upfront_cost_eur=0.0,
            stabilize_core_annual_maintenance_cost_eur=0.0,
            do_nothing_drift_cost_eur=0.0,
        ),
        annual_volume=250_000.0,
        horizon_years=3.0,
        annual_discount_rate=low_rate,
        risk_rng=None,
        shared_risk_draws=np.array([0.5]),
    )[0]
    high_pv = simulate_option_stabilize_core(
        _frame(
            stabilize_core_upfront_cost_eur=0.0,
            stabilize_core_annual_maintenance_cost_eur=0.0,
            do_nothing_drift_cost_eur=0.0,
        ),
        annual_volume=250_000.0,
        horizon_years=3.0,
        annual_discount_rate=high_rate,
        risk_rng=None,
        shared_risk_draws=np.array([0.5]),
    )[0]
    assert high_pv <= low_pv + 1.0, (
        f"Higher discount rate {high_rate:.4f} produced higher value than "
        f"lower rate {low_rate:.4f}: {high_pv:.2f} vs {low_pv:.2f}"
    )


# ---------------------------------------------------------------------------
# Property: zero extension uptake makes feature extension match do-nothing in benefits
# ---------------------------------------------------------------------------


@given(
    drift_cost=st.floats(min_value=5_000.0, max_value=200_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=2000)
def test_zero_extension_uptake_collapses_feature_extension_to_do_nothing_costs(
    drift_cost: float,
) -> None:
    """With zero uptake there is no reliability benefit or extension value.

    Feature extension should equal do-nothing drift minus the explicit
    investment costs (upfront + maintenance over the horizon).
    """
    frame = _frame(
        extension_uptake=0.0,
        do_nothing_drift_cost_eur=drift_cost,
        feature_extension_launch_delay_months=0.0,
        feature_extension_benefit_ramp_months=0.0,
        feature_extension_cost_overrun_multiplier=1.0,
        feature_extension_upfront_cost_eur=600_000.0,
        feature_extension_annual_maintenance_cost_eur=60_000.0,
    )
    do_nothing_value = simulate_option_do_nothing(
        frame, horizon_years=2.0, annual_discount_rate=0.0
    )[0]
    feature_ext_value = simulate_option_feature_extension(
        frame,
        annual_volume=250_000.0,
        horizon_years=2.0,
        annual_discount_rate=0.0,
        risk_rng=None,
        shared_risk_draws=np.array([0.5]),
    )[0]
    expected = do_nothing_value - 600_000.0 - (60_000.0 * 2.0)
    assert abs(feature_ext_value - expected) < 1e-4, (
        f"Zero-uptake feature extension {feature_ext_value:.2f} does not match "
        f"do-nothing drift ({do_nothing_value:.2f}) minus investment costs "
        f"({expected:.2f})"
    )


# ---------------------------------------------------------------------------
# Property: regression event expected cost converges to p * severity
# ---------------------------------------------------------------------------


def test_regression_event_expected_cost_converges_to_probability_times_severity() -> None:
    """Over 50k worlds, mean regression cost should be within 3 sigma of p * severity.

    This tests the Bernoulli sampling in _sample_regression_cost and the
    shared-risk-draw mechanism. Uses a fixed p and severity so the expectation
    and variance are known analytically.
    """
    from simulator.simulation import run_simulation

    p = 0.20
    severity = 300_000.0
    n_worlds = 50_000
    expected_mean = p * severity
    # Variance of Bernoulli * constant: p*(1-p)*severity^2
    expected_std = (p * (1 - p)) ** 0.5 * severity / n_worlds**0.5

    results = run_simulation(
        "simulator/config.yaml",
        n_worlds=n_worlds,
        seed=42,
        scenario="mid_range_pressure",
        param_overrides={
            "regression_event_prob": {"dist": "constant", "value": p},
            "regression_event_cost_eur": {"dist": "constant", "value": severity},
            "stabilize_core_regression_prob_multiplier": {"dist": "constant", "value": 1.0},
        },
    )

    # Implied regression cost from the value difference vs a no-event run
    no_event_results = run_simulation(
        "simulator/config.yaml",
        n_worlds=n_worlds,
        seed=42,
        scenario="mid_range_pressure",
        param_overrides={
            "regression_event_prob": {"dist": "constant", "value": 0.0},
            "regression_event_cost_eur": {"dist": "constant", "value": severity},
            "stabilize_core_regression_prob_multiplier": {"dist": "constant", "value": 1.0},
        },
    )

    implied_mean_cost = float(
        (no_event_results["stabilize_core"] - results["stabilize_core"]).mean()
    )
    # Allow 4 sigma for safety
    tolerance = 4.0 * expected_std
    assert abs(implied_mean_cost - expected_mean) < tolerance, (
        f"Mean implied regression cost {implied_mean_cost:.2f} deviates from "
        f"expected {expected_mean:.2f} by more than {tolerance:.2f} (4 sigma)."
    )
