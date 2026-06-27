"""Contract tests for the presentation display-table helpers.

These cover the formatting guarantees the app and docs rely on: column order,
human-readable formatting, failure-reason mapping, and the documented empty
shapes returned when an input frame has no rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulator.presentation import (
    comparison_frontier_display_table,
    diagnostics_display_table,
    eligibility_display_table,
    frontier_display_table,
    payoff_delta_display_table,
    runner_up_frontier_display_table,
    scenario_display_table,
    sensitivity_summary_note,
    stability_frequency_table,
    summary_display_table,
    top_sensitivity_rows,
)


def test_summary_display_table_formats_currency_and_labels() -> None:
    summary = pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "mean_value_eur": 500000.0,
                "median_value_eur": 480000.0,
                "p05_value_eur": -200000.0,
                "p95_value_eur": 900000.0,
            }
        ]
    )
    table = summary_display_table(summary)
    assert list(table.columns) == ["Option", "Expected Value", "P05", "Median", "P95"]
    row = table.iloc[0]
    assert row["Option"] == "Stabilize Core"
    assert row["Expected Value"] == "EUR 500,000"
    assert row["P05"] == "EUR -200,000"


def test_diagnostics_display_table_formats_rate_and_regret() -> None:
    diagnostics = pd.DataFrame(
        [
            {
                "option": "feature_extension",
                "win_rate": 0.48,
                "mean_regret_eur": 120000.0,
                "median_regret_eur": 110000.0,
                "p95_regret_eur": 320000.0,
            }
        ]
    )
    table = diagnostics_display_table(diagnostics)
    assert list(table.columns) == ["Option", "Win Rate", "Mean Regret", "P95 Regret"]
    assert table.iloc[0]["Win Rate"] == "48%"
    assert table.iloc[0]["Mean Regret"] == "EUR 120,000"


def test_eligibility_display_table_maps_failure_reason_and_eligibility() -> None:
    eligibility = pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "mean_value_eur": 500000.0,
                "p05_value_eur": -100000.0,
                "downside_slack_eur": 200000.0,
                "mean_regret_eur": 50000.0,
                "regret_slack_eur": 250000.0,
                "eligible": True,
                "failure_reason": np.nan,
            },
            {
                "option": "new_capability",
                "mean_value_eur": 100000.0,
                "p05_value_eur": -800000.0,
                "downside_slack_eur": -500000.0,
                "mean_regret_eur": 400000.0,
                "regret_slack_eur": -100000.0,
                "eligible": False,
                "failure_reason": "misses_downside_floor",
            },
        ]
    )
    table = eligibility_display_table(eligibility)
    assert table.iloc[0]["Eligible"] == "yes"
    assert table.iloc[0]["Failure Reason"] == "passes both guardrails"
    assert table.iloc[1]["Eligible"] == "no"
    # The internal code maps to a human-readable label (not the raw code).
    assert table.iloc[1]["Failure Reason"] != "misses_downside_floor"


def test_scenario_display_table_has_expected_columns() -> None:
    scenarios = pd.DataFrame(
        [
            {
                "scenario": "mid_range_pressure",
                "selected_option": "stabilize_core",
                "option": "stabilize_core",
                "mean_value_eur": 500000.0,
                "p05_value_eur": -100000.0,
                "mean_regret_eur": 50000.0,
                "eligible": True,
                "win_rate": 0.55,
            }
        ]
    )
    table = scenario_display_table(scenarios)
    assert list(table.columns) == [
        "Scenario",
        "Selected Option",
        "Option",
        "Expected Value",
        "P05",
        "Mean Regret",
        "Eligible",
        "Win Rate",
    ]
    assert table.iloc[0]["Selected Option"] == "Stabilize Core"


def test_payoff_delta_display_table_empty_and_populated() -> None:
    empty = payoff_delta_display_table(pd.DataFrame())
    assert list(empty.columns) == ["Parameter", "Delta rho", "Sampled range", "Interpretation"]
    assert empty.empty

    populated = payoff_delta_display_table(
        pd.DataFrame(
            [
                {
                    "parameter": "baseline_failure_rate",
                    "delta_spearman_corr": 0.42,
                    "sampled_min_value": 0.1,
                    "sampled_max_value": 0.3,
                    "interpretation_note": "stronger when failures rise",
                }
            ]
        )
    )
    assert populated.iloc[0]["Parameter"] == "Baseline Failure Rate"
    assert populated.iloc[0]["Delta rho"] == "+0.42"
    assert populated.iloc[0]["Sampled range"] == "0.100 to 0.300"


def test_frontier_display_table_handles_unobserved_switch() -> None:
    empty = frontier_display_table(pd.DataFrame())
    assert "Switch type" in empty.columns and empty.empty

    table = frontier_display_table(
        pd.DataFrame(
            [
                {
                    "threshold_label": "Downside floor",
                    "current_value": -300000.0,
                    "switching_value": np.nan,
                    "first_switching_option": np.nan,
                    "switch_type": "no_switch_within_range",
                    "switch_direction": np.nan,
                    "interpretation_note": "no flip within tested range",
                }
            ]
        )
    )
    assert table.iloc[0]["Switching value"] == "not observed"
    assert table.iloc[0]["Switching option"] == "not observed"
    assert table.iloc[0]["Direction"] == "not observed"


def test_comparison_frontier_table_and_runner_up_alias_match() -> None:
    rows = pd.DataFrame(
        [
            {
                "threshold_label": "Regret cap",
                "current_value": 300000.0,
                "switching_value": np.nan,
                "status": "already_ahead",
                "interpretation_note": "comparison already trails",
            }
        ]
    )
    table = comparison_frontier_display_table(rows)
    assert table.iloc[0]["Comparison threshold"] == "not needed"
    # The runner-up helper is documented as a legacy alias.
    pd.testing.assert_frame_equal(table, runner_up_frontier_display_table(rows))


def test_stability_frequency_table_empty_and_populated() -> None:
    assert list(stability_frequency_table([], "selected_option").columns) == ["Option", "Runs"]
    table = stability_frequency_table(
        [{"selected_option": "stabilize_core", "count": 18}], "selected_option"
    )
    assert table.iloc[0]["Option"] == "Stabilize Core"
    assert table.iloc[0]["Runs"] == 18


def test_top_sensitivity_rows_prefers_driver_columns_when_present() -> None:
    driver = pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "parameter": "baseline_failure_rate",
                "partial_rank_corr": 0.41,
                "ci_low": 0.20,
                "ci_high": 0.55,
            }
        ]
    )
    table = top_sensitivity_rows(driver, "stabilize_core", threshold=0.1, limit=5)
    assert list(table.columns) == ["Parameter", "Partial rank corr", "95% CI"]
    assert table.iloc[0]["95% CI"] == "+0.20 to +0.55"


def test_top_sensitivity_rows_falls_back_to_spearman() -> None:
    spearman = pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "parameter": "baseline_failure_rate",
                "spearman_corr": 0.55,
            }
        ]
    )
    table = top_sensitivity_rows(spearman, "stabilize_core", threshold=0.1, limit=5)
    assert list(table.columns) == ["Parameter", "Spearman"]
    assert table.iloc[0]["Spearman"] == "+0.55"


def test_sensitivity_summary_note_switches_on_available_columns() -> None:
    driver = pd.DataFrame(
        [{"option": "stabilize_core", "parameter": "baseline_failure_rate", "partial_rank_corr": 0.4}]
    )
    spearman = pd.DataFrame(
        [{"option": "stabilize_core", "parameter": "baseline_failure_rate", "spearman_corr": 0.4}]
    )
    # Single material row triggers the "only ..." note in both views.
    assert "Baseline Failure Rate" in (sensitivity_summary_note(driver, "stabilize_core", 0.1) or "")
    assert "Baseline Failure Rate" in (
        sensitivity_summary_note(spearman, "stabilize_core", 0.1) or ""
    )
