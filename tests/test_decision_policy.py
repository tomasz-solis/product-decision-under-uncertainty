"""Tests for the explicit decision-policy layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulator.config import AnalysisConfig, DecisionPolicyConfig
from simulator.policy import (
    build_policy_eligibility_table,
    payoff_delta_diagnostic,
    policy_frontier_analysis,
    policy_frontier_grid,
    select_recommendation,
)


def test_policy_prefers_lower_regret_within_tolerance_band() -> None:
    """A tighter regret profile should win when EV is inside the tolerance band."""

    summary = pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "mean_value_eur": 500000.0,
                "median_value_eur": 480000.0,
                "p05_value_eur": -200000.0,
                "p95_value_eur": 900000.0,
            },
            {
                "option": "feature_extension",
                "mean_value_eur": 450000.0,
                "median_value_eur": 430000.0,
                "p05_value_eur": -100000.0,
                "p95_value_eur": 800000.0,
            },
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "win_rate": 0.5,
                "mean_regret_eur": 220000.0,
                "median_regret_eur": 200000.0,
                "p95_regret_eur": 500000.0,
            },
            {
                "option": "feature_extension",
                "win_rate": 0.45,
                "mean_regret_eur": 90000.0,
                "median_regret_eur": 85000.0,
                "p95_regret_eur": 250000.0,
            },
        ]
    )
    policy = DecisionPolicyConfig(
        name="guardrailed_expected_value",
        minimum_p05_value_eur=-300000.0,
        maximum_mean_regret_eur=300000.0,
        ev_tolerance_eur=60000.0,
    )

    result = select_recommendation(summary, diagnostics, policy)

    assert result.selected_option == "feature_extension"
    assert result.binding_constraint == "ev_tolerance_then_regret"


def test_policy_falls_back_to_highest_ev_when_guardrails_fail() -> None:
    """If nothing clears the guardrails, the policy should relax and choose the EV leader."""

    summary = pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "mean_value_eur": 500000.0,
                "median_value_eur": 480000.0,
                "p05_value_eur": -500000.0,
                "p95_value_eur": 900000.0,
            },
            {
                "option": "feature_extension",
                "mean_value_eur": 450000.0,
                "median_value_eur": 430000.0,
                "p05_value_eur": -450000.0,
                "p95_value_eur": 800000.0,
            },
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "win_rate": 0.5,
                "mean_regret_eur": 700000.0,
                "median_regret_eur": 650000.0,
                "p95_regret_eur": 900000.0,
            },
            {
                "option": "feature_extension",
                "win_rate": 0.45,
                "mean_regret_eur": 600000.0,
                "median_regret_eur": 590000.0,
                "p95_regret_eur": 850000.0,
            },
        ]
    )
    policy = DecisionPolicyConfig(
        name="guardrailed_expected_value",
        minimum_p05_value_eur=-300000.0,
        maximum_mean_regret_eur=300000.0,
        ev_tolerance_eur=25000.0,
    )

    result = select_recommendation(summary, diagnostics, policy)

    assert result.selected_option == "stabilize_core"
    assert result.binding_constraint == "guardrails_relaxed"


def test_policy_marks_only_surviving_option_explicitly() -> None:
    """The policy output should explain when only one option survives guardrails."""

    summary = pd.DataFrame(
        [
            {
                "option": "do_nothing",
                "mean_value_eur": 100000.0,
                "median_value_eur": 95000.0,
                "p05_value_eur": -120000.0,
                "p95_value_eur": 250000.0,
            },
            {
                "option": "feature_extension",
                "mean_value_eur": 140000.0,
                "median_value_eur": 130000.0,
                "p05_value_eur": -520000.0,
                "p95_value_eur": 450000.0,
            },
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "option": "do_nothing",
                "win_rate": 0.45,
                "mean_regret_eur": 180000.0,
                "median_regret_eur": 150000.0,
                "p95_regret_eur": 340000.0,
            },
            {
                "option": "feature_extension",
                "win_rate": 0.55,
                "mean_regret_eur": 220000.0,
                "median_regret_eur": 190000.0,
                "p95_regret_eur": 480000.0,
            },
        ]
    )
    policy = DecisionPolicyConfig(
        name="guardrailed_expected_value",
        minimum_p05_value_eur=-300000.0,
        maximum_mean_regret_eur=300000.0,
        ev_tolerance_eur=25000.0,
    )

    result = select_recommendation(summary, diagnostics, policy)
    eligibility = build_policy_eligibility_table(summary, diagnostics, policy)

    assert result.selected_option == "do_nothing"
    assert result.binding_constraint == "only_option_passing_guardrails"
    assert result.eligible_option_count == 1
    assert result.runner_up_failure_reason == "misses_downside_floor"
    assert not bool(
        eligibility.loc[eligibility["option"] == "feature_extension", "eligible"].iloc[0]
    )


def test_payoff_delta_diagnostic_is_descriptive_not_threshold_like() -> None:
    """The payoff-delta helper should rank descriptive delta drivers only."""

    baseline_failure_rate = np.repeat(np.linspace(0.02, 0.10, 20), 20)
    extension_uptake = np.tile(np.linspace(0.10, 0.90, 20), 20)
    results = pd.DataFrame(
        {
            "baseline_failure_rate": baseline_failure_rate,
            "extension_uptake": extension_uptake,
            "stabilize_core": 150000.0 + (4000000.0 * baseline_failure_rate),
            "feature_extension": 120000.0 + (1200000.0 * extension_uptake),
            "do_nothing": -50000.0,
            "new_capability": -250000.0,
        }
    )
    analysis = AnalysisConfig(
        sensitivity_materiality_threshold_abs_spearman=0.0,
        sensitivity_max_rows_per_option=3,
        decision_boundary_top_parameters=2,
    )
    recommendation = select_recommendation(
        pd.DataFrame(
            [
                {
                    "option": "stabilize_core",
                    "mean_value_eur": 300000.0,
                    "median_value_eur": 300000.0,
                    "p05_value_eur": 0.0,
                    "p95_value_eur": 500000.0,
                },
                {
                    "option": "feature_extension",
                    "mean_value_eur": 250000.0,
                    "median_value_eur": 250000.0,
                    "p05_value_eur": 0.0,
                    "p95_value_eur": 450000.0,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "option": "stabilize_core",
                    "win_rate": 0.6,
                    "mean_regret_eur": 100000.0,
                    "median_regret_eur": 90000.0,
                    "p95_regret_eur": 220000.0,
                },
                {
                    "option": "feature_extension",
                    "win_rate": 0.4,
                    "mean_regret_eur": 120000.0,
                    "median_regret_eur": 110000.0,
                    "p95_regret_eur": 240000.0,
                },
            ]
        ),
        DecisionPolicyConfig(
            name="guardrailed_expected_value",
            minimum_p05_value_eur=-100000.0,
            maximum_mean_regret_eur=300000.0,
            ev_tolerance_eur=25000.0,
        ),
    )

    diagnostic = payoff_delta_diagnostic(results, recommendation, analysis)

    assert diagnostic["delta_rows"][0]["parameter"] == "extension_uptake"
    assert "observed_flip_region_low" not in diagnostic["delta_rows"][0]


def test_policy_frontier_grid_brackets_real_switching_region() -> None:
    """The full frontier should report the first actual switch, not the runner-up shortcut."""

    summary = pd.DataFrame(
        [
            {
                "option": "feature_extension",
                "mean_value_eur": 160000.0,
                "median_value_eur": 150000.0,
                "p05_value_eur": -540000.0,
                "p95_value_eur": 420000.0,
            },
            {
                "option": "do_nothing",
                "mean_value_eur": 110000.0,
                "median_value_eur": 105000.0,
                "p05_value_eur": -140000.0,
                "p95_value_eur": 240000.0,
            },
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "option": "feature_extension",
                "win_rate": 0.55,
                "mean_regret_eur": 210000.0,
                "median_regret_eur": 180000.0,
                "p95_regret_eur": 450000.0,
            },
            {
                "option": "do_nothing",
                "win_rate": 0.45,
                "mean_regret_eur": 250000.0,
                "median_regret_eur": 220000.0,
                "p95_regret_eur": 500000.0,
            },
        ]
    )
    policy = DecisionPolicyConfig(
        name="guardrailed_expected_value",
        minimum_p05_value_eur=-300000.0,
        maximum_mean_regret_eur=450000.0,
        ev_tolerance_eur=100000.0,
    )

    recommendation = select_recommendation(summary, diagnostics, policy)
    frontier = policy_frontier_analysis(summary, diagnostics, policy, recommendation)
    grid = policy_frontier_grid(summary, diagnostics, policy, recommendation)

    assert recommendation.selected_option == "do_nothing"
    assert frontier["frontier_rows"][0]["switching_value"] == -140000.0
    assert frontier["frontier_rows"][0]["first_switching_option"] == "feature_extension"
    assert frontier["frontier_rows"][0]["switch_type"] == "grid_bracket"
    downside_rows = grid.loc[grid["threshold_name"] == "minimum_p05_value_eur"]
    assert downside_rows["switch_from_baseline"].any()


def test_full_option_frontier_can_switch_to_non_runner_up_option() -> None:
    """The first actual policy switch can differ from the current runner-up."""

    summary = pd.DataFrame(
        [
            {
                "option": "feature_extension",
                "mean_value_eur": 130000.0,
                "median_value_eur": 125000.0,
                "p05_value_eur": -500000.0,
                "p95_value_eur": 420000.0,
            },
            {
                "option": "do_nothing",
                "mean_value_eur": 100000.0,
                "median_value_eur": 96000.0,
                "p05_value_eur": -80000.0,
                "p95_value_eur": 240000.0,
            },
            {
                "option": "stabilize_core",
                "mean_value_eur": 95000.0,
                "median_value_eur": 93000.0,
                "p05_value_eur": -90000.0,
                "p95_value_eur": 220000.0,
            },
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "option": "feature_extension",
                "win_rate": 0.50,
                "mean_regret_eur": 85000.0,
                "median_regret_eur": 82000.0,
                "p95_regret_eur": 180000.0,
            },
            {
                "option": "do_nothing",
                "win_rate": 0.30,
                "mean_regret_eur": 50000.0,
                "median_regret_eur": 48000.0,
                "p95_regret_eur": 140000.0,
            },
            {
                "option": "stabilize_core",
                "win_rate": 0.20,
                "mean_regret_eur": 30000.0,
                "median_regret_eur": 29000.0,
                "p95_regret_eur": 90000.0,
            },
        ]
    )
    policy = DecisionPolicyConfig(
        name="guardrailed_expected_value",
        minimum_p05_value_eur=-150000.0,
        maximum_mean_regret_eur=60000.0,
        ev_tolerance_eur=10000.0,
    )

    recommendation = select_recommendation(summary, diagnostics, policy)
    frontier = policy_frontier_analysis(summary, diagnostics, policy, recommendation)

    assert recommendation.selected_option == "stabilize_core"
    assert recommendation.runner_up == "feature_extension"
    assert frontier["frontier_rows"][0]["first_switching_option"] == "do_nothing"
    assert frontier["runner_up_comparison_rows"][0]["switching_value"] == -500000.0


def test_policy_frontier_grids_respect_non_negative_threshold_domains() -> None:
    """Regret-cap and EV-tolerance sweeps should never cross below zero."""

    summary = pd.DataFrame(
        [
            {
                "option": "do_nothing",
                "mean_value_eur": 1000.0,
                "median_value_eur": 1000.0,
                "p05_value_eur": 1000.0,
                "p95_value_eur": 1000.0,
            },
            {
                "option": "stabilize_core",
                "mean_value_eur": 700.0,
                "median_value_eur": 700.0,
                "p05_value_eur": 800.0,
                "p95_value_eur": 700.0,
            },
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "option": "do_nothing",
                "win_rate": 0.6,
                "mean_regret_eur": 250.0,
                "median_regret_eur": 250.0,
                "p95_regret_eur": 250.0,
            },
            {
                "option": "stabilize_core",
                "win_rate": 0.4,
                "mean_regret_eur": 50.0,
                "median_regret_eur": 50.0,
                "p95_regret_eur": 50.0,
            },
        ]
    )
    policy = DecisionPolicyConfig(
        name="guardrailed_expected_value",
        minimum_p05_value_eur=0.0,
        maximum_mean_regret_eur=100.0,
        ev_tolerance_eur=100.0,
    )

    recommendation = select_recommendation(summary, diagnostics, policy)
    grid = policy_frontier_grid(summary, diagnostics, policy, recommendation)

    regret_rows = grid.loc[grid["threshold_name"] == "maximum_mean_regret_eur"]
    tolerance_rows = grid.loc[grid["threshold_name"] == "ev_tolerance_eur"]
    assert regret_rows["tested_value"].min() >= 0.0
    assert tolerance_rows["tested_value"].min() >= 0.0
