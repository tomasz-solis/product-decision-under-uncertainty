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
from simulator.policy import PolicyFrontierResult

COLORS = {
    "do_nothing": "#9b9ea4",
    "stabilize_core": "#0b6e4f",
    "feature_extension": "#c46a2d",
    "new_capability": "#4f5d75",
}

BACKGROUND = "rgba(0, 0, 0, 0)"
GRID_COLOR = "#d8d1c5"
TEXT_COLOR = "#172026"
POLICY_SELECTED_COLOR = "#0b6e4f"
POLICY_COMPARISON_COLOR = "#c46a2d"
MUTED_COLOR = "#b8b2a7"
HOVER_BACKGROUND = "#fffdf8"


def create_ranked_payoff_profile(
    summary: pd.DataFrame,
    *,
    recommended_option: str | None = None,
    comparison_option: str | None = None,
) -> go.Figure:
    """Create a ranked payoff-range view centered on the decision trade-off."""

    ordered = summary.sort_values("mean_value_eur", ascending=False).reset_index(drop=True)
    figure = go.Figure()
    selected_option = recommended_option or str(ordered.iloc[0]["option"])
    labels = [labeled_option(str(option)) for option in ordered["option"]]
    y_positions = list(range(len(ordered)))[::-1]

    for y_position, (_, row) in zip(y_positions, ordered.iterrows(), strict=False):
        option = str(row["option"])
        color = _option_color(
            option,
            selected_option=selected_option,
            comparison_option=comparison_option,
        )
        figure.add_trace(
            go.Scatter(
                x=[float(row["p05_value_eur"]), float(row["p95_value_eur"])],
                y=[y_position, y_position],
                mode="lines",
                line=dict(color=color, width=8),
                hovertemplate=(
                    f"{labeled_option(option)}<br>"
                    "P05 to P95 range: %{x:,.0f} EUR<extra></extra>"
                ),
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[float(row["mean_value_eur"])],
                y=[y_position],
                mode="markers",
                marker=dict(color=color, size=16, line=dict(width=2, color="#ffffff")),
                name=labeled_option(option),
                hovertemplate=(
                    f"{labeled_option(option)}<br>"
                    "Expected value: %{x:,.0f} EUR<extra></extra>"
                ),
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[float(row["median_value_eur"])],
                y=[y_position],
                mode="markers",
                marker=dict(color="#ffffff", size=10, symbol="diamond", line=dict(width=2, color=color)),
                hovertemplate=(
                    f"{labeled_option(option)}<br>"
                    "Median value: %{x:,.0f} EUR<extra></extra>"
                ),
                showlegend=False,
            )
        )

    figure.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=labels,
        showgrid=False,
    )
    figure.update_xaxes(title_text="Discounted value (EUR)", zeroline=True, zerolinecolor=GRID_COLOR)
    figure.update_layout(
        title="Ranked payoff range",
        height=470,
        margin=dict(l=24, r=24, t=86, b=28),
    )
    _apply_currency_axis(figure, axis="x")
    _apply_theme(figure)
    return figure


def create_guardrail_chart(
    policy_eligibility: pd.DataFrame,
    *,
    minimum_p05_value_eur: float,
    maximum_mean_regret_eur: float,
    recommended_option: str,
    comparison_option: str | None = None,
) -> go.Figure:
    """Create a two-panel view that shows which guardrail binds for each option."""

    ordered = policy_eligibility.sort_values("mean_value_eur", ascending=False).reset_index(drop=True)
    labels = [labeled_option(str(option)) for option in ordered["option"]]

    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("Downside floor", "Regret cap"),
        horizontal_spacing=0.12,
    )
    for option, label, p05_value, regret_value in zip(
        ordered["option"],
        labels,
        ordered["p05_value_eur"],
        ordered["mean_regret_eur"],
        strict=False,
    ):
        color = _option_color(
            str(option),
            selected_option=recommended_option,
            comparison_option=comparison_option,
        )
        figure.add_trace(
            go.Scatter(
                x=[float(p05_value)],
                y=[label],
                mode="markers",
                marker=dict(color=color, size=15),
                showlegend=False,
                hovertemplate=_build_hover_template(
                    label,
                    "P05 value: %{x:,.0f} EUR",
                    "Must stay above the downside floor.",
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[float(regret_value)],
                y=[label],
                mode="markers",
                marker=dict(color=color, size=15),
                showlegend=False,
                hovertemplate=_build_hover_template(
                    label,
                    "Mean regret: %{x:,.0f} EUR",
                    "Must stay below the regret cap.",
                ),
            ),
            row=1,
            col=2,
        )

    figure.add_vline(
        x=float(minimum_p05_value_eur),
        line_color=POLICY_SELECTED_COLOR,
        line_dash="dash",
        row=1,
        col=1,
    )
    figure.add_vline(
        x=float(maximum_mean_regret_eur),
        line_color=POLICY_SELECTED_COLOR,
        line_dash="dash",
        row=1,
        col=2,
    )
    figure.update_layout(
        title="Guardrail position by option",
        height=460,
        margin=dict(l=24, r=24, t=86, b=28),
    )
    figure.update_xaxes(title_text="P05 value (EUR)", row=1, col=1)
    figure.update_xaxes(title_text="Mean regret (EUR)", row=1, col=2)
    _apply_currency_axis(figure, axis="x", row=1, col=1)
    _apply_currency_axis(figure, axis="x", row=1, col=2)
    _apply_theme(figure)
    return figure


def create_frontier_switch_chart(
    policy_frontier: PolicyFrontierResult,
    policy_frontier_grid: pd.DataFrame,
) -> go.Figure:
    """Create a threshold runway chart that shows current values and switch points."""

    frontier_rows = pd.DataFrame(policy_frontier["frontier_rows"])
    figure = go.Figure()
    for _, row in frontier_rows.iterrows():
        threshold_label = str(row["threshold_label"])
        threshold_grid = policy_frontier_grid.loc[
            policy_frontier_grid["threshold_name"] == row["threshold_name"]
        ]
        if not threshold_grid.empty:
            figure.add_trace(
                go.Scatter(
                    x=[
                        float(threshold_grid["tested_value"].min()),
                        float(threshold_grid["tested_value"].max()),
                    ],
                    y=[threshold_label, threshold_label],
                    mode="lines",
                    line=dict(color=MUTED_COLOR, width=10),
                    hovertemplate="Tested range: %{x:,.0f} EUR<extra></extra>",
                    showlegend=False,
                )
            )
        figure.add_trace(
            go.Scatter(
                x=[float(row["current_value"])],
                y=[threshold_label],
                mode="markers",
                marker=dict(color=POLICY_SELECTED_COLOR, size=16, symbol="diamond"),
                name="Current policy",
                hovertemplate=_build_hover_template(
                    threshold_label,
                    "Current value: %{x:,.0f} EUR",
                ),
                showlegend=False,
            )
        )
        if pd.notna(row["switching_value"]):
            figure.add_trace(
                go.Scatter(
                    x=[float(row["switching_value"])],
                    y=[threshold_label],
                    mode="markers",
                    marker=dict(color=POLICY_COMPARISON_COLOR, size=16),
                    hovertemplate=_build_hover_template(
                        threshold_label,
                        "First switching value: %{x:,.0f} EUR",
                    ),
                    showlegend=False,
                )
            )

    figure.update_layout(
        title="Threshold switch runway",
        height=400,
        margin=dict(l=24, r=24, t=86, b=24),
        xaxis_title="Threshold value (EUR)",
    )
    _apply_currency_axis(figure, axis="x")
    _apply_theme(figure)
    return figure


def create_stability_chart(stability_runs: pd.DataFrame) -> go.Figure:
    """Create a compact published-case stability chart focused on spread by world count."""

    figure = go.Figure()

    if stability_runs.empty:
        _apply_theme(figure)
        figure.update_layout(height=360, margin=dict(l=24, r=24, t=86, b=24))
        return figure

    spread = (
        stability_runs.groupby("n_worlds", observed=True)
        .agg(
            selected_ev_range_eur=(
                "selected_mean_value_eur",
                lambda series: float(series.max() - series.min()),
            ),
            selected_p05_range_eur=(
                "selected_p05_value_eur",
                lambda series: float(series.max() - series.min()),
            ),
        )
        .reset_index()
    )
    figure.add_trace(
        go.Scatter(
            x=spread["n_worlds"],
            y=spread["selected_ev_range_eur"],
            mode="lines+markers",
            name="EV range",
            line=dict(color=POLICY_COMPARISON_COLOR, width=3),
            hovertemplate="%{x:,} worlds<br>EV range: %{y:,.0f} EUR<extra></extra>",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=spread["n_worlds"],
            y=spread["selected_p05_range_eur"],
            mode="lines+markers",
            name="P05 range",
            line=dict(color=POLICY_SELECTED_COLOR, width=3),
            hovertemplate="%{x:,} worlds<br>P05 range: %{y:,.0f} EUR<extra></extra>",
        )
    )
    figure.update_layout(
        title="Metric spread by world count",
        height=400,
        margin=dict(l=24, r=24, t=86, b=24),
        xaxis_title="Simulation worlds",
        yaxis_title="Observed range (EUR)",
    )
    _apply_currency_axis(figure, axis="y")
    _apply_theme(figure)
    return figure


def create_decision_dashboard(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    sensitivity: pd.DataFrame | None = None,
    recommended_option: str | None = None,
    sensitivity_threshold: float = 0.10,
) -> go.Figure:
    """Create a compact four-panel dashboard for value, win rate, regret, and sensitivity."""

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

    figure.add_trace(
        go.Bar(
            x=[labeled_option(str(option)) for option in summary_lookup.index],
            y=summary_lookup["mean_value_eur"],
            marker_color=[
                _option_color(str(option), selected_option=selected_option)
                for option in summary_lookup.index
            ],
            text=[format_eur(value) for value in summary_lookup["mean_value_eur"]],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=[labeled_option(str(option)) for option in diagnostics_lookup.index],
            y=diagnostics_lookup["win_rate"] * 100.0,
            marker_color=[
                _option_color(str(option), selected_option=selected_option)
                for option in diagnostics_lookup.index
            ],
            text=[f"{value:.0%}" for value in diagnostics_lookup["win_rate"]],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=[labeled_option(str(option)) for option in diagnostics_lookup.index],
            y=diagnostics_lookup["mean_regret_eur"],
            marker_color=[
                _option_color(str(option), selected_option=selected_option)
                for option in diagnostics_lookup.index
            ],
            text=[format_eur(value) for value in diagnostics_lookup["mean_regret_eur"]],
            textposition="outside",
            showlegend=False,
        ),
        row=2,
        col=1,
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
                marker_color=_option_color(selected_option, selected_option=selected_option),
                text=[f"{value:+.2f}" for value in selected_rows["spearman_corr"]],
                textposition="outside",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    figure.update_yaxes(title_text="EUR", row=1, col=1)
    figure.update_yaxes(title_text="Win rate (%)", row=1, col=2)
    figure.update_yaxes(title_text="EUR", row=2, col=1)
    figure.update_yaxes(title_text="Spearman correlation", row=2, col=2)
    figure.update_layout(height=760, margin=dict(l=24, r=24, t=70, b=24))
    _apply_theme(figure)
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
        yaxis_title="EUR",
    )
    _apply_currency_axis(figure, axis="y")
    _apply_theme(figure)
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
        yaxis_title="EUR",
    )
    _apply_currency_axis(figure, axis="y")
    _apply_theme(figure)
    return figure


def create_scenario_comparison(
    scenario_results: pd.DataFrame,
    scenario_metadata: dict[str, dict[str, str]],
) -> go.Figure:
    """Create one bar trace per option across named scenarios."""

    figure = go.Figure()
    for option_value, group in scenario_results.groupby("option", sort=False):
        option = str(option_value)
        label = labeled_option(option)
        figure.add_trace(
            go.Bar(
                name=label,
                x=[
                    scenario_metadata[str(value)]["label"]
                    if str(value) in scenario_metadata
                    else str(value)
                    for value in group["scenario"]
                ],
                y=group["mean_value_eur"],
                marker_color=COLORS[option],
                hovertemplate=_build_hover_template(
                    label,
                    "Scenario: %{x}",
                    "Expected value: %{y:,.0f} EUR",
                ),
            )
        )
    figure.update_layout(
        title="Expected value by scenario",
        barmode="group",
        height=440,
        margin=dict(l=24, r=24, t=86, b=96),
        bargap=0.24,
        yaxis_title="EUR",
        xaxis_title="Scenario",
    )
    _apply_currency_axis(figure, axis="y")
    _apply_theme(figure)
    figure.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0, 0, 0, 0)",
        )
    )
    return figure


def create_sensitivity_waterfall(
    sensitivity: pd.DataFrame,
    option: str,
    threshold: float = 0.10,
) -> go.Figure:
    """Create a horizontal bar chart for the material sensitivity rows of one option."""

    metric_column = "spearman_corr"
    title = "Spearman correlation"
    rows = sensitivity
    if "partial_rank_corr" in sensitivity.columns:
        metric_column = "partial_rank_corr"
        title = "Partial rank correlation"
        rows = (
            sensitivity.loc[sensitivity["option"] == option]
            .assign(abs_metric=lambda frame: frame[metric_column].abs())
            .sort_values("abs_metric", ascending=False)
            .loc[lambda frame: frame["abs_metric"] >= threshold]
            .head(8)
            .copy()
        )
    else:
        rows = material_sensitivity_rows(
            sensitivity=sensitivity,
            option=option,
            threshold=threshold,
            limit=8,
        )
    figure = go.Figure(
        go.Bar(
            x=rows[metric_column],
            y=[clean_label(value) for value in rows["parameter"]],
            orientation="h",
            marker_color=COLORS.get(option, POLICY_SELECTED_COLOR),
            text=[f"{value:+.2f}" for value in rows[metric_column]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=_build_hover_template(
                labeled_option(option),
                "Parameter: %{y}",
                f"{title}: %{{x:+.2f}}",
            ),
        )
    )
    figure.update_layout(
        title=f"Decision drivers for {labeled_option(option)}",
        height=500,
        margin=dict(l=24, r=56, t=86, b=24),
        xaxis_title=title,
        yaxis_title="Parameter",
    )
    if not rows.empty:
        max_abs_value = max(abs(float(rows[metric_column].min())), abs(float(rows[metric_column].max())))
        padding = max(0.12, max_abs_value * 0.20)
        figure.update_xaxes(range=[-(max_abs_value + padding), max_abs_value + padding])
    _apply_theme(figure)
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
        xaxis_title="Mean regret (EUR)",
        yaxis_title="Expected value (EUR)",
        margin=dict(l=24, r=24, t=86, b=24),
    )
    _apply_currency_axis(figure, axis="x")
    _apply_currency_axis(figure, axis="y")
    _apply_theme(figure)
    return figure


def _build_hover_template(title: str, *lines: str) -> str:
    """Build a Plotly hover template without evaluating Plotly placeholders in Python."""

    return "<br>".join((title, *lines, "<extra></extra>"))


def _option_color(
    option: str,
    *,
    selected_option: str | None = None,
    comparison_option: str | None = None,
) -> str:
    """Return the semantic color for one option."""

    if selected_option is not None and option == selected_option:
        return POLICY_SELECTED_COLOR
    if comparison_option is not None and option == comparison_option:
        return POLICY_COMPARISON_COLOR
    return COLORS.get(option, MUTED_COLOR)


def _apply_currency_axis(
    figure: go.Figure,
    *,
    axis: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Format one Plotly axis with readable euro abbreviations."""

    updater = figure.update_xaxes if axis == "x" else figure.update_yaxes
    updater(
        tickprefix="€",
        tickformat="~s",
        exponentformat="none",
        separatethousands=True,
        row=row,
        col=col,
    )


def _apply_theme(figure: go.Figure) -> None:
    """Apply the portfolio chart theme in place."""

    figure.update_layout(
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(
            family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
            color=TEXT_COLOR,
            size=13,
        ),
        title_font=dict(size=20, color=TEXT_COLOR),
        hoverlabel=dict(
            bgcolor=HOVER_BACKGROUND,
            bordercolor=GRID_COLOR,
            font=dict(color=TEXT_COLOR, size=13),
        ),
        uniformtext=dict(minsize=11, mode="hide"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=12, color=TEXT_COLOR),
            itemclick=False,
            itemdoubleclick=False,
        ),
    )
    figure.update_annotations(font=dict(size=15, color=TEXT_COLOR))
    figure.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont=dict(size=12, color=TEXT_COLOR),
        title_font=dict(size=14, color=TEXT_COLOR),
        automargin=True,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont=dict(size=12, color=TEXT_COLOR),
        title_font=dict(size=14, color=TEXT_COLOR),
        automargin=True,
    )
