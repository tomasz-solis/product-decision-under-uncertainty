"""Business-logic tests for the simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simulator.analytics import summarize_results
from simulator.simulation import (
    run_all_scenarios,
    run_simulation,
    simulate_option_do_nothing,
    simulate_option_feature_extension,
    simulate_option_new_capability,
    simulate_option_stabilize_core,
)

CONFIG_PATH = "simulator/config.yaml"


def _single_world_frame(**overrides: float) -> pd.DataFrame:
    """Build a one-row dataframe for deterministic option tests."""

    values = {
        "baseline_failure_rate": 0.05,
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
        "regression_event_cost_eur": 150000.0,
        "stabilize_core_regression_prob_multiplier": 1.0,
        "feature_extension_regression_prob_multiplier": 0.8,
        "new_capability_regression_prob_multiplier": 2.5,
        "stabilize_core_upfront_cost_eur": 800000.0,
        "stabilize_core_annual_maintenance_cost_eur": 30000.0,
        "feature_extension_upfront_cost_eur": 600000.0,
        "feature_extension_annual_maintenance_cost_eur": 60000.0,
        "new_capability_upfront_cost_eur": 1200000.0,
        "new_capability_annual_maintenance_cost_eur": 100000.0,
        "do_nothing_drift_cost_eur": 50000.0,
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
    values.update(overrides)
    return pd.DataFrame({key: [value] for key, value in values.items()})


def test_feature_extension_zero_uptake_matches_do_nothing_plus_explicit_costs() -> None:
    """Zero uptake must not create a reliability benefit."""

    frame = _single_world_frame(extension_uptake=0.0)
    do_nothing = simulate_option_do_nothing(frame, horizon_years=2.0)[0]
    feature_extension = simulate_option_feature_extension(
        frame,
        annual_volume=250000.0,
        horizon_years=2.0,
        annual_discount_rate=0.0,
        risk_rng=np.random.default_rng(7),
    )[0]

    expected = do_nothing - 600000.0 - (60000.0 * 2.0)
    assert feature_extension == expected


def test_stabilize_core_value_moves_with_upfront_costs() -> None:
    """Upfront costs must flow straight into net value."""

    low_cost = _single_world_frame(stabilize_core_upfront_cost_eur=500000.0)
    high_cost = _single_world_frame(stabilize_core_upfront_cost_eur=700000.0)

    low_value = simulate_option_stabilize_core(
        low_cost,
        annual_volume=250000.0,
        horizon_years=2.0,
        annual_discount_rate=0.0,
        risk_rng=np.random.default_rng(11),
    )[0]
    high_value = simulate_option_stabilize_core(
        high_cost,
        annual_volume=250000.0,
        horizon_years=2.0,
        annual_discount_rate=0.0,
        risk_rng=np.random.default_rng(11),
    )[0]

    assert low_value - high_value == 200000.0


def test_new_capability_uses_value_per_success_for_successful_volume_uplift() -> None:
    """The new capability should scale with total value per successful unit."""

    low_value_frame = _single_world_frame(value_per_success_eur=30.0)
    high_value_frame = _single_world_frame(value_per_success_eur=40.0)

    low_value = simulate_option_new_capability(
        low_value_frame,
        annual_volume=250000.0,
        horizon_years=2.0,
        annual_discount_rate=0.0,
        risk_rng=np.random.default_rng(19),
    )[0]
    high_value = simulate_option_new_capability(
        high_value_frame,
        annual_volume=250000.0,
        horizon_years=2.0,
        annual_discount_rate=0.0,
        risk_rng=np.random.default_rng(19),
    )[0]

    successful_units = 250000.0 * 2.0 * (1.0 - 0.05)
    expected_delta = successful_units * 0.08 * (40.0 - 30.0)
    assert high_value - low_value == expected_delta


def test_launch_delay_reduces_stabilize_core_value() -> None:
    """A later launch should reduce discounted value for the same option."""

    early = _single_world_frame(stabilize_core_launch_delay_months=0.0)
    late = _single_world_frame(stabilize_core_launch_delay_months=6.0)

    early_value = simulate_option_stabilize_core(
        early,
        annual_volume=250000.0,
        horizon_years=2.0,
        annual_discount_rate=0.08,
        risk_rng=np.random.default_rng(21),
    )[0]
    late_value = simulate_option_stabilize_core(
        late,
        annual_volume=250000.0,
        horizon_years=2.0,
        annual_discount_rate=0.08,
        risk_rng=np.random.default_rng(21),
    )[0]

    assert early_value > late_value


def test_drift_cost_can_move_value_without_customer_harm_term() -> None:
    """Internal drift should still matter when failure-linked customer harm is zero."""

    no_drift = _single_world_frame(failure_to_churn_rel=0.0, do_nothing_drift_cost_eur=0.0)
    high_drift = _single_world_frame(failure_to_churn_rel=0.0, do_nothing_drift_cost_eur=80000.0)

    low_value = simulate_option_do_nothing(no_drift, horizon_years=2.0, annual_discount_rate=0.0)[0]
    high_value = simulate_option_do_nothing(
        high_drift,
        horizon_years=2.0,
        annual_discount_rate=0.0,
    )[0]

    assert low_value > high_value


def test_failure_linked_customer_harm_can_move_value_without_drift_cost() -> None:
    """Failure-linked customer harm should still matter when background drift is zero."""

    low_harm = _single_world_frame(failure_to_churn_rel=0.0, do_nothing_drift_cost_eur=0.0)
    high_harm = _single_world_frame(failure_to_churn_rel=0.25, do_nothing_drift_cost_eur=0.0)

    low_value = simulate_option_stabilize_core(
        low_harm,
        annual_volume=250000.0,
        horizon_years=2.0,
        annual_discount_rate=0.0,
        risk_rng=np.random.default_rng(7),
    )[0]
    high_value = simulate_option_stabilize_core(
        high_harm,
        annual_volume=250000.0,
        horizon_years=2.0,
        annual_discount_rate=0.0,
        risk_rng=np.random.default_rng(7),
    )[0]

    assert high_value > low_value


def test_event_based_regression_risk_changes_downside_metrics() -> None:
    """Higher event probability should worsen downside metrics."""

    no_event = run_simulation(
        CONFIG_PATH,
        n_worlds=3000,
        seed=42,
        scenario="mid_range_pressure",
        param_overrides={
            "regression_event_prob": {"dist": "constant", "value": 0.0},
            "regression_event_cost_eur": {"dist": "constant", "value": 250000.0},
        },
    )
    forced_event = run_simulation(
        CONFIG_PATH,
        n_worlds=3000,
        seed=42,
        scenario="mid_range_pressure",
        param_overrides={
            "regression_event_prob": {"dist": "constant", "value": 1.0},
            "regression_event_cost_eur": {"dist": "constant", "value": 250000.0},
        },
    )

    no_event_summary = summarize_results(no_event).set_index("option")
    forced_event_summary = summarize_results(forced_event).set_index("option")

    assert (
        forced_event_summary.loc["stabilize_core", "p05_value_eur"]
        < no_event_summary.loc["stabilize_core", "p05_value_eur"]
    )
    assert (
        forced_event_summary.loc["feature_extension", "mean_value_eur"]
        < no_event_summary.loc["feature_extension", "mean_value_eur"]
    )


def test_run_simulation_is_reproducible_for_the_same_seed() -> None:
    """The same seed should produce identical simulation outputs."""

    first = run_simulation(CONFIG_PATH, n_worlds=500, seed=42, scenario="mid_range_pressure")
    second = run_simulation(CONFIG_PATH, n_worlds=500, seed=42, scenario="mid_range_pressure")

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    ("low", "mode", "high"),
    [
        (0.05, 0.05, 0.05),
        (0.01, 0.01, 0.10),
        (0.01, 0.10, 0.10),
    ],
)
def test_triangular_degenerate_cases_do_not_raise(
    low: float,
    mode: float,
    high: float,
) -> None:
    """Boundary triangular configurations should still return finite samples."""

    from simulator.config import ParamSpec
    from simulator.simulation import _inverse_cdf_param

    spec = ParamSpec(
        dist="tri",
        low=low,
        mode=mode,
        high=high,
        value=None,
        median=None,
        p95=None,
    )
    uniforms = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    result = _inverse_cdf_param(spec, uniforms)

    assert result.shape == (5,)
    assert np.all(np.isfinite(result))


def test_scenario_worldviews_move_do_nothing_as_well() -> None:
    """Shared-environment scenarios should change the status-quo result too."""

    scenarios = run_all_scenarios(CONFIG_PATH, n_worlds=1000, seed=42)
    do_nothing = scenarios.loc[scenarios["option"] == "do_nothing"].set_index("scenario")

    assert (
        do_nothing.loc["reliability_crisis", "mean_value_eur"]
        != do_nothing.loc["mid_range_pressure", "mean_value_eur"]
    )
    assert (
        do_nothing.loc["growth_friendly_recovery", "mean_value_eur"]
        != do_nothing.loc["mid_range_pressure", "mean_value_eur"]
    )
