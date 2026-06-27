"""Render-contract tests for the Plotly chart builders that lacked coverage.

These assert the analytical contract of each figure - trace structure, series
names, and axis titles - rather than merely that the call does not raise.
"""

from __future__ import annotations

import pandas as pd

from simulator.visualizations import (
    create_executive_summary_table,
    create_guardrail_chart,
    create_ranked_payoff_profile,
    create_regret_comparison,
    create_risk_profile_chart,
    create_scenario_comparison,
    create_stability_chart,
)


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
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
                "p05_value_eur": -150000.0,
                "p95_value_eur": 800000.0,
            },
        ]
    )


def _diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "win_rate": 0.52,
                "mean_regret_eur": 200000.0,
                "p95_regret_eur": 500000.0,
            },
            {
                "option": "feature_extension",
                "win_rate": 0.48,
                "mean_regret_eur": 120000.0,
                "p95_regret_eur": 320000.0,
            },
        ]
    )


def test_ranked_payoff_profile_draws_range_mean_and_median_per_option() -> None:
    figure = create_ranked_payoff_profile(_summary(), recommended_option="stabilize_core")
    # Three traces per option: the P05-P95 range line, the mean marker, the median marker.
    assert len(figure.data) == 3 * len(_summary())
    assert "Discounted value" in figure.layout.xaxis.title.text


def test_guardrail_chart_marks_both_thresholds() -> None:
    eligibility = pd.DataFrame(
        [
            {"option": "stabilize_core", "mean_value_eur": 500000.0, "p05_value_eur": -100000.0,
             "mean_regret_eur": 50000.0},
            {"option": "feature_extension", "mean_value_eur": 450000.0, "p05_value_eur": -250000.0,
             "mean_regret_eur": 90000.0},
        ]
    )
    figure = create_guardrail_chart(
        eligibility,
        minimum_p05_value_eur=-300000.0,
        maximum_mean_regret_eur=300000.0,
        recommended_option="stabilize_core",
    )
    # Two markers (downside, regret) per option.
    assert len(figure.data) == 2 * len(eligibility)
    # One dashed guardrail line per panel.
    assert len(figure.layout.shapes) == 2


def test_stability_chart_empty_returns_no_traces() -> None:
    figure = create_stability_chart(pd.DataFrame())
    assert len(figure.data) == 0


def test_stability_chart_plots_ev_and_p05_ranges() -> None:
    runs = pd.DataFrame(
        [
            {"n_worlds": 5000, "selected_mean_value_eur": 100.0, "selected_p05_value_eur": -50.0},
            {"n_worlds": 5000, "selected_mean_value_eur": 140.0, "selected_p05_value_eur": -20.0},
            {"n_worlds": 20000, "selected_mean_value_eur": 110.0, "selected_p05_value_eur": -40.0},
            {"n_worlds": 20000, "selected_mean_value_eur": 120.0, "selected_p05_value_eur": -35.0},
        ]
    )
    figure = create_stability_chart(runs)
    assert [trace.name for trace in figure.data] == ["EV range", "P05 range"]
    assert figure.layout.yaxis.title.text == "Observed range (EUR)"


def test_risk_profile_chart_has_percentile_series() -> None:
    figure = create_risk_profile_chart(_summary())
    assert [trace.name for trace in figure.data] == ["P05", "Median", "P95"]
    assert figure.layout.barmode == "group"


def test_regret_comparison_has_mean_and_p95_series() -> None:
    figure = create_regret_comparison(_diagnostics())
    assert [trace.name for trace in figure.data] == ["Mean regret", "P95 regret"]


def test_scenario_comparison_uses_metadata_labels() -> None:
    scenario_results = pd.DataFrame(
        [
            {"option": "stabilize_core", "scenario": "mid_range_pressure", "mean_value_eur": 500.0},
            {"option": "stabilize_core", "scenario": "reliability_crisis", "mean_value_eur": 300.0},
            {"option": "feature_extension", "scenario": "mid_range_pressure", "mean_value_eur": 450.0},
            {"option": "feature_extension", "scenario": "reliability_crisis", "mean_value_eur": 200.0},
        ]
    )
    metadata = {
        "mid_range_pressure": {"label": "Mid-range pressure"},
        "reliability_crisis": {"label": "Reliability crisis"},
    }
    figure = create_scenario_comparison(scenario_results, metadata)
    assert [trace.name for trace in figure.data] == ["Stabilize Core", "Feature Extension"]
    assert list(figure.data[0].x) == ["Mid-range pressure", "Reliability crisis"]


def test_executive_summary_table_formats_merged_metrics() -> None:
    table = create_executive_summary_table(_summary(), _diagnostics())
    assert list(table.columns) == [
        "Option",
        "Expected Value",
        "P05",
        "P95",
        "Win Rate",
        "Mean Regret",
    ]
    top = table.iloc[0]
    assert top["Option"] == "Stabilize Core"
    assert top["Expected Value"] == "EUR 500,000"
    assert top["Win Rate"] == "52%"
