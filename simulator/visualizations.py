"""
Visualization functions for decision analysis outputs.
Provides executive-level charts with consistent styling.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

COLOR_SCHEME = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#06A77D",
    "warning": "#F18F01",
    "danger": "#C73E1D",
    "neutral": "#6C757D",
    "options": ["#2E86AB", "#06A77D", "#F18F01", "#A23B72"],
    "percentiles": {"p05": "#C73E1D", "p50": "#2E86AB", "p95": "#06A77D"}
}

OPTION_LABELS = {
    "do_nothing": "Do Nothing",
    "stabilize_core": "Stabilize Core",
    "feature_extension": "Feature Extension",
    "new_capability": "New Capability",
}


def clean_label(text: str) -> str:
    """
    Convert snake_case, camelCase, PascalCase, or kebab-case to Title Case.
    Examples:
        feature_extension -> Feature Extension
        featureExtension -> Feature Extension
        FeatureExtension -> Feature Extension
        feature-extension -> Feature Extension
    """
    import re

    # Replace underscores and hyphens with spaces
    text = text.replace('_', ' ').replace('-', ' ')

    # Insert space before capital letters (for camelCase/PascalCase)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Title case
    return text.title()


def create_decision_dashboard(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    sensitivity: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Create comprehensive executive dashboard with all key metrics.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Expected Value (higher is better)",
            "Win Rate (higher = more robust)",
            "Mean Regret (lower = less risk)",
            "Top Parameter Drivers"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.25,
        horizontal_spacing=0.18
    )

    options = summary['option'].tolist()
    clean_options = [clean_label(OPTION_LABELS.get(opt, opt)) for opt in options]
    colors = [COLOR_SCHEME["options"][i] for i in range(len(options))]

    # Panel 1: EV - smart text positioning
    ev_vals = summary['mean_value_eur'].tolist()
    ev_max = max(ev_vals)
    ev_text_pos = ['inside' if val > ev_max * 0.15 else 'outside' for val in ev_vals]
    ev_text_colors = ['white' if val > ev_max * 0.15 else 'black' for val in ev_vals]

    fig.add_trace(
        go.Bar(
            x=clean_options,
            y=ev_vals,
            marker_color=colors,
            text=[f"€{val/1000:.0f}k" for val in ev_vals],
            textposition=ev_text_pos,
            textfont=dict(size=16, color=ev_text_colors),
        ),
        row=1, col=1
    )

    # Panel 2: Win Rates - smart text positioning
    if not diagnostics.empty:
        clean_diag_options = [clean_label(OPTION_LABELS.get(opt, opt)) for opt in diagnostics['option']]
        win_rates = (diagnostics['win_rate'] * 100).tolist()
        win_max = max(win_rates)
        win_text_pos = ['inside' if val > win_max * 0.15 else 'outside' for val in win_rates]
        win_text_colors = ['white' if val > win_max * 0.15 else 'black' for val in win_rates]

        fig.add_trace(
            go.Bar(
                x=clean_diag_options,
                y=win_rates,
                marker_color=colors,
                text=[f"{v:.0f}%" for v in win_rates],
                textposition=win_text_pos,
                textfont=dict(size=16, color=win_text_colors),
            ),
            row=1, col=2
        )

        # Panel 3: Regret - smart text positioning
        regret_vals = diagnostics['mean_regret_eur'].tolist()
        regret_max = max(regret_vals)
        regret_text_pos = ['inside' if val > regret_max * 0.15 else 'outside' for val in regret_vals]
        regret_text_colors = ['white' if val > regret_max * 0.15 else 'black' for val in regret_vals]

        fig.add_trace(
            go.Bar(
                x=clean_diag_options,
                y=regret_vals,
                marker_color=COLOR_SCHEME["danger"],
                text=[f"€{val/1000:.0f}k" for val in regret_vals],
                textposition=regret_text_pos,
                textfont=dict(size=16, color=regret_text_colors),
            ),
            row=2, col=1
        )

    # Panel 4: Sensitivity - smart text positioning
    if sensitivity is not None and not sensitivity.empty:
        top_sens = sensitivity.nlargest(5, 'spearman_corr', keep='first')
        clean_params = [clean_label(p) for p in top_sens['parameter']]
        sens_vals = top_sens['spearman_corr'].tolist()
        sens_max = max(sens_vals)
        sens_text_pos = ['inside' if val > sens_max * 0.15 else 'outside' for val in sens_vals]
        sens_text_colors = ['white' if val > sens_max * 0.15 else 'black' for val in sens_vals]

        fig.add_trace(
            go.Bar(
                x=clean_params,
                y=sens_vals,
                marker_color=COLOR_SCHEME["primary"],
                text=[f"{val:.2f}" for val in sens_vals],
                textposition=sens_text_pos,
                textfont=dict(size=16, color=sens_text_colors),
            ),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        showlegend=False,
        height=800,
        font=dict(size=16, color='black'),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=100, b=80)
    )

    # Update axes with high contrast
    fig.update_xaxes(showgrid=False, tickfont=dict(size=14, color='black'), tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor='#d0d0d0', tickfont=dict(size=14, color='black'))

    # Axis labels with high contrast
    fig.update_yaxes(title_text="Value (k€)", title_font=dict(size=16, color='black'), row=1, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", title_font=dict(size=16, color='black'), row=1, col=2)
    fig.update_yaxes(title_text="Regret (k€)", title_font=dict(size=16, color='black'), row=2, col=1)
    fig.update_yaxes(title_text="Correlation", title_font=dict(size=16, color='black'), row=2, col=2)

    # Update subplot titles with high contrast and consistent positioning
    for annotation in fig.layout.annotations:
        annotation.font.size = 15
        annotation.font.color = 'black'
        annotation.y = annotation.y + 0.02

    return fig


def create_risk_profile_chart(summary: pd.DataFrame) -> go.Figure:
    """
    Create detailed risk profile showing P05/P50/P95 for each option.
    Higher floor (P05) = less downside risk. Tighter range = more predictable.
    """
    fig = go.Figure()

    options = summary['option'].tolist()
    clean_options = [clean_label(OPTION_LABELS.get(opt, opt)) for opt in options]

    # P05 - Downside
    fig.add_trace(go.Bar(
        name='P05 (Downside)',
        x=clean_options,
        y=summary['p05_value_eur'],
        marker_color=COLOR_SCHEME["percentiles"]["p05"],
        text=[f"€{val/1000:.0f}k" for val in summary['p05_value_eur']],
        textposition='outside',
        textfont=dict(size=16, color='black'),
    ))

    # P50 - Median
    fig.add_trace(go.Bar(
        name='P50 (Median)',
        x=clean_options,
        y=summary['median_value_eur'],
        marker_color=COLOR_SCHEME["percentiles"]["p50"],
        text=[f"€{val/1000:.0f}k" for val in summary['median_value_eur']],
        textposition='outside',
        textfont=dict(size=16, color='black'),
    ))

    # P95 - Upside
    fig.add_trace(go.Bar(
        name='P95 (Upside)',
        x=clean_options,
        y=summary['p95_value_eur'],
        marker_color=COLOR_SCHEME["percentiles"]["p95"],
        text=[f"€{val/1000:.0f}k" for val in summary['p95_value_eur']],
        textposition='outside',
        textfont=dict(size=16, color='black'),
    ))

    fig.update_layout(
        title=dict(
            text="Risk Profile: Look for higher floor (P05) and tighter range",
            font=dict(size=16, color='black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="",
        yaxis_title="Value (k€)",
        barmode='group',
        height=550,
        font=dict(size=16, color='black'),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=100, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=14, color='black')
        )
    )

    fig.update_xaxes(showgrid=False, tickfont=dict(size=14, color='black'))
    fig.update_yaxes(showgrid=True, gridcolor='#d0d0d0', tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))

    return fig


def create_regret_comparison(diagnostics: pd.DataFrame) -> go.Figure:
    """
    Create regret comparison showing mean and P95 regret for each option.
    Regret = missed opportunity cost. Lower is better.
    """
    if diagnostics.empty:
        return go.Figure()

    fig = go.Figure()

    clean_options = [clean_label(OPTION_LABELS.get(opt, opt)) for opt in diagnostics['option']]

    fig.add_trace(go.Bar(
        name='Mean Regret',
        x=clean_options,
        y=diagnostics['mean_regret_eur'],
        marker_color=COLOR_SCHEME["warning"],
        text=[f"€{val/1000:.0f}k" for val in diagnostics['mean_regret_eur']],
        textposition='outside',
        textfont=dict(size=16, color='black'),
    ))

    fig.add_trace(go.Bar(
        name='P95 Regret',
        x=clean_options,
        y=diagnostics['p95_regret_eur'],
        marker_color=COLOR_SCHEME["danger"],
        text=[f"€{val/1000:.0f}k" for val in diagnostics['p95_regret_eur']],
        textposition='outside',
        textfont=dict(size=16, color='black'),
    ))

    fig.update_layout(
        title=dict(
            text="Regret Analysis: Lower means less pain when you're wrong",
            font=dict(size=16, color='black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="",
        yaxis_title="Regret (k€)",
        barmode='group',
        height=550,
        font=dict(size=16, color='black'),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=100, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=14, color='black')
        )
    )

    fig.update_xaxes(showgrid=False, tickfont=dict(size=14, color='black'))
    fig.update_yaxes(showgrid=True, gridcolor='#d0d0d0', tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))

    return fig


def create_scenario_comparison(scenario_results: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Create scenario comparison showing EV across base/conservative/aggressive.

    Args:
        scenario_results: Dict mapping scenario name to summary DataFrame

    Returns:
        Plotly figure with scenario comparison
    """
    fig = go.Figure()

    scenarios = list(scenario_results.keys())
    options = scenario_results[scenarios[0]]['option'].tolist()
    colors = [COLOR_SCHEME["options"][i] for i in range(len(options))]

    for i, option in enumerate(options):
        evs = []
        for scenario in scenarios:
            summary = scenario_results[scenario]
            ev = summary[summary['option'] == option]['mean'].iloc[0]
            evs.append(ev)

        fig.add_trace(go.Bar(
            name=OPTION_LABELS.get(option, option),
            x=scenarios,
            y=evs,
            marker_color=colors[i],
            text=[f"€{val:,.0f}" for val in evs],
            textposition='outside',
        ))

    fig.update_layout(
        title="Scenario Robustness: Expected Value Across Worldviews",
        xaxis_title="Scenario",
        yaxis_title="Expected Value (€)",
        barmode='group',
        height=500,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_sensitivity_waterfall(sensitivity: pd.DataFrame, option: str = "feature_extension") -> go.Figure:
    """
    Create waterfall chart showing how key parameters affect outcome.

    Args:
        sensitivity: DataFrame with parameter correlations
        option: Which option to analyze

    Returns:
        Plotly figure with sensitivity waterfall
    """
    # Get top 10 parameters by absolute correlation
    top_sens = sensitivity.nlargest(10, 'correlation', keep='first')

    fig = go.Figure(go.Waterfall(
        name="Parameter Impact",
        orientation="h",
        y=top_sens['parameter'],
        x=top_sens['correlation'],
        connector={"line": {"color": "lightgray"}},
        decreasing={"marker": {"color": COLOR_SCHEME["danger"]}},
        increasing={"marker": {"color": COLOR_SCHEME["success"]}},
        totals={"marker": {"color": COLOR_SCHEME["primary"]}},
    ))

    fig.update_layout(
        title=f"Sensitivity Analysis: Top 10 Drivers for {OPTION_LABELS.get(option, option)}",
        xaxis_title="Spearman Correlation with Outcome",
        yaxis_title="Parameter",
        height=600,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="white",
        showlegend=False,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')

    return fig


def create_distribution_comparison(results: pd.DataFrame, options: List[str]) -> go.Figure:
    """
    Create overlaid distribution plots for multiple options.

    Args:
        results: DataFrame with simulation results (each row = one world)
        options: List of option column names to compare

    Returns:
        Plotly figure with overlaid histograms
    """
    fig = go.Figure()

    colors = [COLOR_SCHEME["options"][i] for i in range(len(options))]

    for i, option in enumerate(options):
        fig.add_trace(go.Histogram(
            x=results[option],
            name=OPTION_LABELS.get(option, option),
            opacity=0.6,
            marker_color=colors[i],
            nbinsx=50,
        ))

    fig.update_layout(
        title="Outcome Distributions: Full Uncertainty Range",
        xaxis_title="Net Value (€)",
        yaxis_title="Frequency",
        barmode='overlay',
        height=500,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98
        )
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_executive_summary_table(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    """
    Create formatted table for executive summary.

    Args:
        summary: DataFrame with summary statistics (from summarize_results)
        diagnostics: DataFrame with diagnostics (from decision_diagnostics)

    Returns:
        Formatted DataFrame ready for display
    """
    export_df = pd.DataFrame()

    export_df['Option'] = [OPTION_LABELS.get(opt, opt) for opt in summary['option']]
    export_df['Expected Value'] = [f"€{val:,.0f}" for val in summary['mean_value_eur']]
    export_df['Downside (P05)'] = [f"€{val:,.0f}" for val in summary['p05_value_eur']]
    export_df['Median (P50)'] = [f"€{val:,.0f}" for val in summary['median_value_eur']]
    export_df['Upside (P95)'] = [f"€{val:,.0f}" for val in summary['p95_value_eur']]

    # Merge diagnostics data
    if not diagnostics.empty and 'win_rate' in diagnostics.columns:
        diag_dict = diagnostics.set_index('option').to_dict('index')
        export_df['Win Rate'] = [f"{diag_dict[opt]['win_rate']*100:.0f}%" for opt in summary['option']]
        export_df['Mean Regret'] = [f"€{diag_dict[opt]['mean_regret_eur']:,.0f}" for opt in summary['option']]

    return export_df


def create_trade_off_matrix(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> go.Figure:
    """
    Create scatter plot showing EV vs Regret trade-off.
    Upper-left quadrant = ideal (high value, low regret).
    """
    if diagnostics.empty:
        return go.Figure()

    options = summary['option'].tolist()
    evs = summary['mean_value_eur'].tolist()

    diag_dict = diagnostics.set_index('option')['mean_regret_eur'].to_dict()
    regrets = [diag_dict[opt] for opt in options]
    colors = [COLOR_SCHEME["options"][i] for i in range(len(options))]

    avg_ev = np.mean(evs)
    avg_regret = np.mean(regrets)

    # Determine axis ranges for quadrants
    x_min = min(regrets) * 0.8
    x_max = max(regrets) * 1.2
    y_min = min(evs) * 0.8
    y_max = max(evs) * 1.2

    fig = go.Figure()

    # Add colored quadrant backgrounds
    # Upper-left: High EV, Low Regret (Green - Ideal)
    fig.add_shape(type="rect",
        x0=x_min, y0=avg_ev, x1=avg_regret, y1=y_max,
        fillcolor="#06A77D", opacity=0.15, layer="below", line_width=0
    )
    # Upper-right: High EV, High Regret (Yellow - Acceptable)
    fig.add_shape(type="rect",
        x0=avg_regret, y0=avg_ev, x1=x_max, y1=y_max,
        fillcolor="#F18F01", opacity=0.15, layer="below", line_width=0
    )
    # Lower-left: Low EV, Low Regret (Yellow - Acceptable)
    fig.add_shape(type="rect",
        x0=x_min, y0=y_min, x1=avg_regret, y1=avg_ev,
        fillcolor="#F18F01", opacity=0.15, layer="below", line_width=0
    )
    # Lower-right: Low EV, High Regret (Red - Not Advisable)
    fig.add_shape(type="rect",
        x0=avg_regret, y0=y_min, x1=x_max, y1=avg_ev,
        fillcolor="#C73E1D", opacity=0.15, layer="below", line_width=0
    )

    # Add quadrant labels in corners to avoid overlap
    # Upper-left: Ideal
    fig.add_annotation(
        x=x_min + (avg_regret - x_min) * 0.15,
        y=y_max - (y_max - avg_ev) * 0.15,
        text="<b>Ideal</b><br>High EV, Low Regret",
        showarrow=False,
        font=dict(size=12, color="#06A77D"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        borderpad=4
    )
    # Upper-right: Acceptable
    fig.add_annotation(
        x=x_max - (x_max - avg_regret) * 0.15,
        y=y_max - (y_max - avg_ev) * 0.15,
        text="<b>Acceptable</b><br>High EV, High Regret",
        showarrow=False,
        font=dict(size=12, color="#F18F01"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        borderpad=4
    )
    # Lower-left: Safe
    fig.add_annotation(
        x=x_min + (avg_regret - x_min) * 0.15,
        y=y_min + (avg_ev - y_min) * 0.15,
        text="<b>Safe</b><br>Low EV, Low Regret",
        showarrow=False,
        font=dict(size=12, color="#F18F01"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        borderpad=4
    )
    # Lower-right: Not Advisable
    fig.add_annotation(
        x=x_max - (x_max - avg_regret) * 0.15,
        y=y_min + (avg_ev - y_min) * 0.15,
        text="<b>Not Advisable</b><br>Low EV, High Regret",
        showarrow=False,
        font=dict(size=12, color="#C73E1D"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        borderpad=4
    )

    # Smart text positioning: left side = right-aligned text, right side = left-aligned text
    text_positions = []
    for regret in regrets:
        if regret < avg_regret:
            text_positions.append("middle right")
        else:
            text_positions.append("middle left")

    fig.add_trace(go.Scatter(
        x=regrets,
        y=evs,
        mode='markers+text',
        marker=dict(
            size=25,
            color=colors,
            line=dict(width=3, color='black')
        ),
        text=[clean_label(OPTION_LABELS.get(opt, opt)) for opt in options],
        textposition=text_positions,
        textfont=dict(size=15, color='black'),
        showlegend=False
    ))

    fig.add_hline(y=avg_ev, line_dash="dash", line_color="#666", opacity=0.6, line_width=2)
    fig.add_vline(x=avg_regret, line_dash="dash", line_color="#666", opacity=0.6, line_width=2)

    fig.update_layout(
        title=dict(
            text="Trade-off Matrix: Quadrants show EV vs Regret trade-offs",
            font=dict(size=16, color='black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Mean Regret (k€) - lower is better",
        yaxis_title="Expected Value (k€) - higher is better",
        height=600,
        font=dict(size=16, color='black'),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=100, b=80),
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max])
    )

    fig.update_xaxes(showgrid=True, gridcolor='#d0d0d0', tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
    fig.update_yaxes(showgrid=True, gridcolor='#d0d0d0', tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))

    return fig


