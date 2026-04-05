"""Plotly helpers aligned with the reporting and app contracts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator.output_utils import (
    clean_label,
    format_eur,
    labeled_option,
    material_sensitivity_rows,
)

COLORS = {
    "do_nothing": "#6c757d",
    "stabilize_core": "#15616d",
    "feature_extension": "#ff7d00",
    "new_capability": "#5a189a",
}


def create_decision_dashboard(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    sensitivity: pd.DataFrame | None = None,
    recommended_option: str | None = None,
    sensitivity_threshold: float = 0.10,
) -> go.Figure:
    """Create a compact four-panel decision dashboard."""

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Expected value",
            "Win rate",
            "Mean regret",
            "Material sensitivity for the selected recommendation",
        ),
        vertical_spacing=0.16,
    )

    summary_lookup = summary.set_index("option")
    diagnostics_lookup = diagnostics.set_index("option")
    selected_option = recommended_option or str(summary.iloc[0]["option"])

    for row, column_name, source in [
        (1, "mean_value_eur", summary_lookup),
        (2, "mean_regret_eur", diagnostics_lookup),
    ]:
        values = source[column_name]
        figure.add_trace(
            go.Bar(
                x=[labeled_option(option) for option in source.index],
                y=values,
                marker_color=[COLORS[str(option)] for option in source.index],
                text=[format_eur(value) for value in values],
                textposition="outside",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    figure.add_trace(
        go.Bar(
            x=[labeled_option(option) for option in diagnostics_lookup.index],
            y=diagnostics_lookup["win_rate"] * 100.0,
            marker_color=[COLORS[str(option)] for option in diagnostics_lookup.index],
            text=[f"{value:.0f}%" for value in diagnostics_lookup["win_rate"] * 100.0],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    if sensitivity is not None and not sensitivity.empty:
        selected_rows = material_sensitivity_rows(
            sensitivity=sensitivity,
            option=selected_option,
            threshold=sensitivity_threshold,
            limit=5,
        )
        figure.add_trace(
            go.Bar(
                x=[clean_label(value) for value in selected_rows["parameter"]],
                y=selected_rows["spearman_corr"],
                marker_color=COLORS[selected_option],
                text=[f"{value:+.2f}" for value in selected_rows["spearman_corr"]],
                textposition="outside",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    figure.update_layout(height=760, plot_bgcolor="white", paper_bgcolor="white")
    figure.update_yaxes(showgrid=True, gridcolor="#e6e6e6")
    figure.update_xaxes(showgrid=False)
    return figure


def create_risk_profile_chart(summary: pd.DataFrame) -> go.Figure:
    """Create grouped percentile bars for each option."""

    figure = go.Figure()
    labels = [labeled_option(option) for option in summary["option"]]
    for column_name, series_name, color in [
        ("p05_value_eur", "P05", "#c1121f"),
        ("median_value_eur", "Median", "#003049"),
        ("p95_value_eur", "P95", "#2a9d8f"),
    ]:
        figure.add_trace(
            go.Bar(
                name=series_name,
                x=labels,
                y=summary[column_name],
                text=[format_eur(value) for value in summary[column_name]],
                textposition="outside",
                marker_color=color,
            )
        )
    figure.update_layout(
        barmode="group",
        height=480,
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_title="EUR",
    )
    return figure


def create_regret_comparison(diagnostics: pd.DataFrame) -> go.Figure:
    """Create grouped regret bars for each option."""

    figure = go.Figure()
    labels = [labeled_option(option) for option in diagnostics["option"]]
    for column_name, series_name, color in [
        ("mean_regret_eur", "Mean regret", "#f77f00"),
        ("p95_regret_eur", "P95 regret", "#d62828"),
    ]:
        figure.add_trace(
            go.Bar(
                name=series_name,
                x=labels,
                y=diagnostics[column_name],
                text=[format_eur(value) for value in diagnostics[column_name]],
                textposition="outside",
                marker_color=color,
            )
        )
    figure.update_layout(
        barmode="group",
        height=480,
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_title="EUR",
    )
    return figure


def create_scenario_comparison(
    scenario_results: pd.DataFrame,
    scenario_metadata: dict[str, dict[str, str]],
) -> go.Figure:
    """Create one bar trace per option across named scenarios."""

    figure = go.Figure()
    for option_value, group in scenario_results.groupby("option", sort=False):
        option = str(option_value)
        figure.add_trace(
            go.Bar(
                name=labeled_option(option),
                x=[
                    scenario_metadata[str(value)]["label"]
                    if str(value) in scenario_metadata
                    else str(value)
                    for value in group["scenario"]
                ],
                y=group["mean_value_eur"],
                text=[format_eur(value) for value in group["mean_value_eur"]],
                textposition="outside",
                marker_color=COLORS[option],
            )
        )
    figure.update_layout(
        barmode="group",
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_title="EUR",
        xaxis_title="Scenario",
    )
    return figure


def create_sensitivity_waterfall(
    sensitivity: pd.DataFrame,
    option: str,
    threshold: float = 0.10,
) -> go.Figure:
    """Create a horizontal bar chart for the material sensitivity rows of one option."""

    rows = material_sensitivity_rows(
        sensitivity=sensitivity,
        option=option,
        threshold=threshold,
        limit=8,
    )
    figure = go.Figure(
        go.Bar(
            x=rows["spearman_corr"],
            y=[clean_label(value) for value in rows["parameter"]],
            orientation="h",
            marker_color=COLORS[option],
            text=[f"{value:+.2f}" for value in rows["spearman_corr"]],
            textposition="outside",
        )
    )
    figure.update_layout(
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Spearman correlation",
        yaxis_title="Parameter",
    )
    return figure


def create_executive_summary_table(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Create a formatted summary table used by the app and tests."""

    merged = summary.merge(
        diagnostics[["option", "win_rate", "mean_regret_eur"]],
        on="option",
        how="left",
    )
    merged["Option"] = merged["option"].map(labeled_option)
    merged["Expected Value"] = merged["mean_value_eur"].map(format_eur)
    merged["P05"] = merged["p05_value_eur"].map(format_eur)
    merged["P95"] = merged["p95_value_eur"].map(format_eur)
    merged["Win Rate"] = merged["win_rate"].map(lambda value: f"{value:.0%}")
    merged["Mean Regret"] = merged["mean_regret_eur"].map(format_eur)
    return merged[["Option", "Expected Value", "P05", "P95", "Win Rate", "Mean Regret"]]


def create_trade_off_matrix(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> go.Figure:
    """Create a regret-versus-value scatter plot."""

    merged = summary.merge(diagnostics[["option", "mean_regret_eur"]], on="option", how="left")
    figure = go.Figure(
        go.Scatter(
            x=merged["mean_regret_eur"],
            y=merged["mean_value_eur"],
            mode="markers+text",
            text=[labeled_option(option) for option in merged["option"]],
            textposition="top center",
            marker=dict(
                size=18,
                color=[COLORS[str(option)] for option in merged["option"]],
                line=dict(width=1, color="#111111"),
            ),
        )
    )
    figure.update_layout(
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Mean regret (EUR)",
        yaxis_title="Expected value (EUR)",
    )
    return figure
