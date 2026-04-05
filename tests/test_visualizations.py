"""Semantic tests for the Plotly helpers."""

from __future__ import annotations

import pandas as pd

from simulator.visualizations import (
    create_decision_dashboard,
    create_sensitivity_waterfall,
    create_trade_off_matrix,
)


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return small summary, diagnostic, and sensitivity frames for chart tests."""

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
                "p05_value_eur": -150000.0,
                "p95_value_eur": 800000.0,
            },
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "win_rate": 0.52,
                "mean_regret_eur": 200000.0,
                "median_regret_eur": 180000.0,
                "p95_regret_eur": 500000.0,
            },
            {
                "option": "feature_extension",
                "win_rate": 0.48,
                "mean_regret_eur": 120000.0,
                "median_regret_eur": 110000.0,
                "p95_regret_eur": 320000.0,
            },
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "parameter": "baseline_failure_rate",
                "spearman_corr": 0.55,
            },
            {
                "option": "stabilize_core",
                "parameter": "cost_per_failure_eur",
                "spearman_corr": 0.09,
            },
            {
                "option": "stabilize_core",
                "parameter": "failure_to_churn_rel",
                "spearman_corr": 0.22,
            },
            {
                "option": "feature_extension",
                "parameter": "extension_uptake",
                "spearman_corr": 0.44,
            },
            {
                "option": "feature_extension",
                "parameter": "extension_value_per_uptake_eur",
                "spearman_corr": 0.31,
            },
        ]
    )
    return summary, diagnostics, sensitivity


def test_sensitivity_waterfall_filters_out_immaterial_rows() -> None:
    """The sensitivity chart should only show parameters above the materiality threshold."""

    _, _, sensitivity = _sample_inputs()
    figure = create_sensitivity_waterfall(sensitivity, "stabilize_core", threshold=0.10)

    assert list(figure.data[0].y) == ["Baseline Failure Rate", "Failure To Churn Rel"]
    assert list(figure.data[0].x) == [0.55, 0.22]
    assert figure.layout.xaxis.title.text == "Spearman correlation"


def test_trade_off_matrix_uses_value_and_regret_axes() -> None:
    """The trade-off matrix should expose the intended analytical comparison."""

    summary, diagnostics, _ = _sample_inputs()
    figure = create_trade_off_matrix(summary, diagnostics)

    assert figure.layout.xaxis.title.text == "Mean regret (EUR)"
    assert figure.layout.yaxis.title.text == "Expected value (EUR)"
    assert list(figure.data[0].text) == ["Stabilize Core", "Feature Extension"]


def test_decision_dashboard_labels_the_selected_recommendation_panel() -> None:
    """The dashboard should label the recommendation panel carefully."""

    summary, diagnostics, sensitivity = _sample_inputs()
    figure = create_decision_dashboard(
        summary,
        diagnostics,
        sensitivity,
        recommended_option="stabilize_core",
        sensitivity_threshold=0.10,
    )

    assert (
        figure.layout.annotations[3].text == "Material sensitivity for the selected recommendation"
    )
    assert list(figure.data[3].x) == ["Baseline Failure Rate", "Failure To Churn Rel"]


def test_decision_dashboard_uses_policy_selected_option_when_it_differs_from_ev_leader() -> None:
    """The recommendation panel should follow the selected option, not the EV leader."""

    summary, diagnostics, sensitivity = _sample_inputs()
    figure = create_decision_dashboard(
        summary,
        diagnostics,
        sensitivity,
        recommended_option="feature_extension",
        sensitivity_threshold=0.10,
    )

    assert list(figure.data[3].x) == ["Extension Uptake", "Extension Value Per Uptake Eur"]
