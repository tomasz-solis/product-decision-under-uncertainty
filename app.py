# app.py
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate limiting constants
MAX_SIMULATIONS_PER_HOUR = 100
MAX_CSV_SIZE_MB = 10

from simulator.config import load_config, get_seed, get_simulation_settings
from simulator.mvp_simulator import (
    run_simulation,
    summarize_results,
    decision_diagnostics,
    sensitivity_analysis,
    run_all_scenarios,
)
from simulator.visualizations import (
    create_decision_dashboard,
    create_risk_profile_chart,
    create_regret_comparison,
    create_scenario_comparison,
    create_trade_off_matrix,
    create_executive_summary_table,
    clean_label,
    OPTION_LABELS,
)

CONFIG_PATH = "simulator/config.yaml"

OPTION_COLS = ["do_nothing", "stabilize_core", "feature_extension", "new_capability"]

# --- Small UI helpers ---------------------------------------------------------

def _reset_param_state(cfg: dict) -> None:
    """Reset all param low/mode/high widgets back to YAML defaults."""
    params = cfg.get("params", {})
    for p_name, spec in params.items():
        if not isinstance(spec, dict):
            continue
        if not all(k in spec for k in ("low", "mode", "high")):
            continue
        for k in ("low", "mode", "high"):
            st_key = f"{p_name}__{k}"
            if k in spec and _is_number(spec[k]):
                st.session_state[st_key] = float(spec[k])


def _group_for_param(name: str) -> str:
    # Keep this simple and explicit. It’s UI, not ontology.
    if name.startswith("baseline_") or name in {"base_churn"}:
        return "Base rates"
    if name.startswith("stabilize_") or name.startswith("regression_") or name.startswith("regression"):
        return "Stabilize core"
    if name.startswith("extension_"):
        return "Feature extension"
    if name.startswith("new_capability_"):
        return "New capability"
    if name.startswith("do_nothing_"):
        return "Do nothing"
    return "Other"


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _check_rate_limit() -> bool:
    """Check if user hit rate limit. Returns True if ok to proceed."""
    import time
    now = time.time()

    if 'simulation_timestamps' not in st.session_state:
        st.session_state.simulation_timestamps = []

    one_hour_ago = now - 3600
    st.session_state.simulation_timestamps = [
        ts for ts in st.session_state.simulation_timestamps
        if ts > one_hour_ago
    ]

    if len(st.session_state.simulation_timestamps) >= MAX_SIMULATIONS_PER_HOUR:
        return False

    st.session_state.simulation_timestamps.append(now)
    return True


def _check_csv_size(uploaded_file) -> bool:
    """Check if uploaded CSV is under size limit."""
    if uploaded_file is None:
        return True

    size_mb = uploaded_file.size / (1024 * 1024)
    return size_mb <= MAX_CSV_SIZE_MB


def _bootstrap_ci(data: np.ndarray, percentile: float, n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """Bootstrap CI for a percentile. Returns (lower, upper)."""
    np.random.seed(42)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(np.percentile(sample, percentile))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def _test_normality(data: pd.Series) -> Dict[str, Any]:
    """Test if data is normal. Uses Shapiro-Wilk or Anderson-Darling."""
    clean = data.dropna()
    if len(clean) < 3:
        return {"error": "Insufficient data"}

    if len(clean) > 5000:
        result = stats.anderson(clean, dist='norm')
        is_normal = result.statistic < result.critical_values[2]
        return {"test": "anderson", "is_normal": is_normal, "statistic": float(result.statistic)}
    else:
        statistic, p_value = stats.shapiro(clean)
        is_normal = p_value > 0.05
        return {"test": "shapiro", "is_normal": is_normal, "p_value": float(p_value)}


def _detect_time_series_pattern(data: pd.Series, dates: Optional[pd.Series] = None) -> Optional[Dict[str, Any]]:
    """Detect seasonality and trend using STL decomposition."""
    clean = data.dropna()
    if len(clean) < 14:
        return None

    try:
        if dates is not None:
            ts = pd.Series(clean.values, index=pd.to_datetime(dates.dropna()))
        else:
            ts = pd.Series(clean.values)

        result = seasonal_decompose(ts, model='additive', period=min(7, len(ts) // 2), extrapolate_trend='freq')

        seasonal_strength = float(np.var(result.seasonal)) / float(np.var(result.seasonal + result.resid))
        trend_strength = float(np.var(result.trend.dropna())) / float(np.var(result.observed.dropna()))

        return {
            "has_seasonality": seasonal_strength > 0.1,
            "has_trend": trend_strength > 0.1,
            "seasonal_strength": seasonal_strength,
            "trend_strength": trend_strength,
        }
    except Exception:
        return None


def _calculate_correlation_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """Calculate correlation matrix and find strong correlations (|r| > 0.7)."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame(), []

    corr_matrix = numeric_df.corr()

    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    float(corr_val)
                ))

    return corr_matrix, strong_corr


def _calculate_quality_score(quality_metrics: Dict[str, Any]) -> int:
    """Score data quality 0-100 based on sample size, variance, outliers, skewness."""
    score = 100
    n = quality_metrics.get('n', 0)
    cv = quality_metrics.get('cv', 0)
    outlier_pct = quality_metrics.get('outlier_pct', 0)
    skewness = abs(quality_metrics.get('skewness', 0))

    if n < 10:
        score -= 40
    elif n < 30:
        score -= 30
    elif n < 50:
        score -= 15
    elif n < 100:
        score -= 5

    if cv > 1.0:
        score -= 20
    elif cv > 0.5:
        score -= 10

    if outlier_pct > 10:
        score -= 15
    elif outlier_pct > 5:
        score -= 7

    if skewness > 2:
        score -= 15
    elif skewness > 1:
        score -= 7

    if quality_metrics.get('normality', {}).get('is_normal', False):
        score += 5

    return max(0, min(100, score))


def _analyze_data_quality(data: pd.Series, col_name: str) -> Dict[str, Any]:
    """Run stats on a column and return metrics, warnings, insights."""
    clean = data.dropna()
    n = len(clean)

    if n == 0:
        return {"error": "No valid data"}

    mean = float(clean.mean())
    std = float(clean.std())
    median = float(clean.median())
    cv = abs(std / mean) if mean != 0 else float('inf')

    skewness = float(clean.skew())
    kurtosis = float(clean.kurtosis())

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((clean < lower_bound) | (clean > upper_bound)).sum()
    outlier_pct = 100 * outliers / n

    normality = _test_normality(clean)

    ci_low, ci_mid, ci_high = (None, None), (None, None), (None, None)
    if n >= 30:
        ci_low = _bootstrap_ci(clean.values, 5, n_bootstrap=500)
        ci_mid = _bootstrap_ci(clean.values, 50, n_bootstrap=500)
        ci_high = _bootstrap_ci(clean.values, 95, n_bootstrap=500)

    warnings = []
    insights = []

    if n < 30:
        warnings.append(f"⚠️ Only {n} samples - might be noisy")
    else:
        insights.append(f"✓ Good size (n={n})")

    if cv > 0.5:
        warnings.append(f"⚠️ High variance (CV={cv:.2f})")
    elif cv < 0.1:
        insights.append(f"✓ Stable (CV={cv:.2f})")

    if outlier_pct > 5:
        warnings.append(f"⚠️ {outliers} outliers ({outlier_pct:.1f}%)")

    if abs(skewness) > 1:
        warnings.append(f"⚠️ Skewed (skew={skewness:.2f})")

    if not normality.get("is_normal", True):
        insights.append(f"ℹ️ Non-normal. Percentile ranges ok.")

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "cv": cv,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "outliers": outliers,
        "outlier_pct": outlier_pct,
        "normality": normality,
        "ci_p05": ci_low,
        "ci_p50": ci_mid,
        "ci_p95": ci_high,
        "warnings": warnings,
        "insights": insights
    }


def _create_distribution_plot(data: pd.Series, col_name: str, p05: float, p50: float, p95: float) -> go.Figure:
    """Histogram with p5/p50/p95 lines."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data,
        name="Distribution",
        nbinsx=30,
        marker=dict(color='lightblue', line=dict(color='darkblue', width=1))
    ))

    fig.add_vline(x=p05, line_dash="dash", line_color="orange", annotation_text="p5")
    fig.add_vline(x=p50, line_dash="solid", line_color="red", annotation_text="p50 (median)")
    fig.add_vline(x=p95, line_dash="dash", line_color="orange", annotation_text="p95")

    fig.update_layout(
        title=f"Distribution: {col_name}",
        xaxis_title=col_name,
        yaxis_title="Count",
        showlegend=False,
        height=300
    )

    return fig


def _create_comparison_chart(csv_col: str, csv_ranges: Dict, yaml_ranges: Dict) -> go.Figure:
    """Before/after bar chart."""
    fig = go.Figure()

    categories = ['Low (p5)', 'Mode (p50)', 'High (p95)']

    fig.add_trace(go.Bar(
        name='Current (YAML)',
        x=categories,
        y=[yaml_ranges.get('low', 0), yaml_ranges.get('mode', 0), yaml_ranges.get('high', 0)],
        marker_color='lightgray'
    ))

    fig.add_trace(go.Bar(
        name='Suggested (Data)',
        x=categories,
        y=[csv_ranges['low'], csv_ranges['mode'], csv_ranges['high']],
        marker_color='lightblue'
    ))

    fig.update_layout(
        title=f"Range Comparison: {csv_col}",
        yaxis_title="Value",
        barmode='group',
        height=300
    )

    return fig


def _create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title="Correlation Matrix",
        xaxis_title="Parameters",
        yaxis_title="Parameters",
        height=500,
        width=500
    )

    return fig


def _get_quality_color(score: int) -> str:
    """Return emoji based on quality score."""
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟡"
    else:
        return "🔴"


def _clamp_triplet(low: float, mode: float, high: float) -> Tuple[float, float, float]:
    # Ensure low <= mode <= high. If user enters weird values, we repair gently.
    if low > high:
        low, high = high, low
    mode = max(low, min(mode, high))
    return low, mode, high


def _build_overrides_from_state(cfg: dict) -> Dict[str, Dict[str, float]]:
    """
    Returns param_overrides in the shape:
    {"param": {"low": ..., "mode": ..., "high": ...}, ...}
    Only includes params where something changed vs YAML.
    """
    overrides: Dict[str, Dict[str, float]] = {}

    params = cfg.get("params", {})
    for p_name, spec in params.items():
        if not isinstance(spec, dict):
            continue

        key_low = f"{p_name}__low"
        key_mode = f"{p_name}__mode"
        key_high = f"{p_name}__high"

        if key_low not in st.session_state:
            continue

        low_raw = float(st.session_state[key_low])
        mode_raw = float(st.session_state[key_mode])
        high_raw = float(st.session_state[key_high])

        low, mode, high = _clamp_triplet(low_raw, mode_raw, high_raw)

        low0 = float(spec.get("low"))
        mode0 = float(spec.get("mode"))
        high0 = float(spec.get("high"))

        if (low, mode, high) != (low0, mode0, high0):
            overrides[p_name] = {"low": low, "mode": mode, "high": high}

    return overrides


def _diff_table(cfg: dict, overrides: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    params = cfg.get("params", {})
    for p_name, ov in overrides.items():
        base = params.get(p_name, {})
        rows.append(
            {
                "parameter": p_name,
                "base_low": base.get("low"),
                "base_mode": base.get("mode"),
                "base_high": base.get("high"),
                "new_low": ov.get("low"),
                "new_mode": ov.get("mode"),
                "new_high": ov.get("high"),
            }
        )
    return pd.DataFrame(rows)


def _long_values(df: pd.DataFrame) -> pd.DataFrame:
    d = df[OPTION_COLS].copy()
    d["draw"] = range(len(d))
    return d.melt(id_vars=["draw"], value_vars=OPTION_COLS, var_name="option", value_name="value_eur")


def _regret_long(df: pd.DataFrame) -> pd.DataFrame:
    # Regret per draw vs the best alternative in that draw.
    values = df[OPTION_COLS].copy()
    best = values.max(axis=1)
    reg = (best.values.reshape(-1, 1) - values.values)
    reg_df = pd.DataFrame(reg, columns=OPTION_COLS)
    reg_df["draw"] = range(len(reg_df))
    return reg_df.melt(id_vars=["draw"], value_vars=OPTION_COLS, var_name="option", value_name="regret_eur")


def _format_money_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c.endswith("_eur") or c.endswith("_value_eur") or c.endswith("_regret_eur"):
            out[c] = out[c].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
    return out


def _format_rate_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c.endswith("win_rate") or c == "win_rate":
            out[c] = out[c].map(lambda x: f"{x:.3f}" if pd.notnull(x) else x)
        if c.endswith("spearman_corr") or c == "spearman_corr":
            out[c] = out[c].map(lambda x: f"{x:.3f}" if pd.notnull(x) else x)
    return out


def _clean_table_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and option/parameter names in tables for display."""
    out = df.copy()

    # Clean column names
    out.columns = [clean_label(col) for col in out.columns]

    # Clean 'option' column if it exists
    if 'Option' in out.columns:
        out['Option'] = out['Option'].apply(lambda x: clean_label(OPTION_LABELS.get(x, x)))

    # Clean 'parameter' column if it exists
    if 'Parameter' in out.columns:
        out['Parameter'] = out['Parameter'].apply(clean_label)

    # Clean 'scenario' column if it exists
    if 'Scenario' in out.columns:
        out['Scenario'] = out['Scenario'].str.title()

    return out

# --- App ----------------------------------------------------------------------

st.set_page_config(page_title="Decision Quality Simulator", layout="wide")

st.title("Decision Quality Simulator")
st.caption("Compare options under uncertainty by editing assumptions. Not a forecast. Not a recommendation.")
st.caption("This tool is designed to evolve into a data product, but today it exists to make decision quality visible.")

cfg = load_config(CONFIG_PATH)
sim = get_simulation_settings(cfg)

scenario_keys = list(cfg.get("scenarios", {}).keys()) or ["base"]

# Initialise session_state defaults from YAML once
params = cfg.get("params", {})
for p_name, spec in params.items():
    if not isinstance(spec, dict):
        continue
    for k in ("low", "mode", "high"):
        st_key = f"{p_name}__{k}"
        if st_key not in st.session_state and k in spec and _is_number(spec[k]):
            st.session_state[st_key] = float(spec[k])

with st.sidebar:
    st.header("Run")

    n_worlds = st.number_input(
        "Simulation runs",
        min_value=1_000,
        max_value=100_000,
        value=int(sim.get("n_worlds", 20_000)),
        step=1_000,
    )

    seed_default = int(get_seed(cfg))
    seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=seed_default, step=1)

    scenario_default = sim.get("scenario", "base")
    scenario_idx = scenario_keys.index(scenario_default) if scenario_default in scenario_keys else 0

    st.markdown("""
    **Scenario definitions:**
    - **Base**: Default parameter ranges from config
    - **Conservative**: Lower effectiveness (↓ adoption, ↓ impact)
    - **Aggressive**: Higher effectiveness (↑ adoption, ↑ impact)
    """)

    scenario = st.selectbox("Scenario", options=scenario_keys, index=scenario_idx)

    st.divider()
    st.header("Assumptions")

    with st.expander("Upload data to suggest ranges", expanded=False):
        st.write("Upload a CSV with historical data. The app will suggest parameter ranges based on your data.")

        with st.expander("📋 What format do I need?", expanded=False):
            st.markdown("""
            **Required format:**
            - CSV file with column headers
            - At least one numeric column
            - At least 10 rows of data (more is better)

            **Example:**
            ```
            date,failure_rate,revenue,cost,churn
            2024-01-01,0.045,102.5,48.2,0.012
            2024-01-02,0.052,98.3,51.0,0.015
            2024-01-03,0.048,105.1,49.5,0.011
            ...
            ```

            **Tips:**
            - Column names should match your parameter names (e.g., `baseline_failure_rate`)
            - Non-numeric columns (like dates) are automatically skipped
            - The app calculates 5th/50th/95th percentiles from your data
            """)

            # Generate example CSV for download
            example_csv = """date,baseline_failure_rate,revenue_per_success,cost_per_failure,churn_rate
2024-01-01,0.045,102.5,48.2,0.012
2024-01-02,0.052,98.3,51.0,0.015
2024-01-03,0.048,105.1,49.5,0.011
2024-01-04,0.061,95.8,52.3,0.018
2024-01-05,0.044,103.2,47.8,0.010
2024-01-06,0.055,99.7,50.1,0.014
2024-01-07,0.047,101.8,48.9,0.012
2024-01-08,0.050,100.5,49.7,0.013
2024-01-09,0.058,97.2,51.8,0.016
2024-01-10,0.046,104.3,48.5,0.011"""

            st.download_button(
                label="Download example CSV template",
                data=example_csv,
                file_name="example_data.csv",
                mime="text/csv",
            )

        st.caption(f"Max file size: {MAX_CSV_SIZE_MB}MB")
        uploaded_file = st.file_uploader("Choose CSV file", type="csv", label_visibility="visible")

        if uploaded_file is not None:
            # Check CSV size limit
            if not _check_csv_size(uploaded_file):
                st.error(f"❌ File too large (max {MAX_CSV_SIZE_MB}MB)")
                st.stop()

            try:
                df_upload = pd.read_csv(uploaded_file)
                logger.info(f"CSV uploaded: {len(df_upload)} rows, {len(df_upload.columns)} columns, size: {uploaded_file.size / 1024:.1f}KB")
                st.write(f"Loaded {len(df_upload)} rows, {len(df_upload.columns)} columns")

                # Show preview
                with st.expander("Preview data", expanded=False):
                    st.dataframe(df_upload.head(10))

                # Find numeric columns
                numeric_cols = [col for col in df_upload.columns if pd.api.types.is_numeric_dtype(df_upload[col])]

                if not numeric_cols:
                    st.warning("No numeric columns found in CSV")
                else:
                    st.write(f"Found {len(numeric_cols)} numeric columns")

                    # Advanced Analytics Section
                    st.divider()

                    # Correlation Analysis
                    if len(numeric_cols) >= 2:
                        with st.expander("🔗 Correlation Analysis", expanded=False):
                            corr_matrix, strong_corr = _calculate_correlation_matrix(df_upload[numeric_cols])

                            if not corr_matrix.empty:
                                fig_corr = _create_correlation_heatmap(corr_matrix)
                                st.plotly_chart(fig_corr, width='stretch')

                                if strong_corr:
                                    st.warning(f"**Found {len(strong_corr)} strong correlations (|r| > 0.7):**")
                                    for col1, col2, corr_val in strong_corr:
                                        st.write(f"• **{col1}** ↔ **{col2}**: r = {corr_val:.3f}")
                                    st.caption("⚠️ These params move together - might affect ranges")
                                else:
                                    st.success("✓ No strong correlations. Params look independent.")

                    # Time Series Detection (if date column exists)
                    date_cols = [col for col in df_upload.columns if 'date' in col.lower()]
                    if date_cols:
                        with st.expander("📅 Time Series Patterns", expanded=False):
                            date_col = date_cols[0]
                            st.write(f"Detected date column: **{date_col}**")

                            for csv_col in numeric_cols:
                                ts_pattern = _detect_time_series_pattern(df_upload[csv_col], df_upload[date_col])
                                if ts_pattern:
                                    st.write(f"**{csv_col}:**")
                                    if ts_pattern.get('has_seasonality'):
                                        st.warning(f"⚠️ Seasonality detected (strength: {ts_pattern['seasonal_strength']:.2f})")
                                    if ts_pattern.get('has_trend'):
                                        st.warning(f"⚠️ Trend detected (strength: {ts_pattern['trend_strength']:.2f})")
                                    if ts_pattern.get('has_seasonality') or ts_pattern.get('has_trend'):
                                        st.caption("Check if historical ranges still make sense")

                    # Quality Score Dashboard
                    st.divider()
                    st.subheader("📊 Data Quality Dashboard")

                    # First pass: analyze all columns to show quality dashboard
                    quality_scores_all = {}
                    for csv_col in numeric_cols:
                        col_data = df_upload[csv_col].dropna()
                        if len(col_data) >= 10:
                            quality = _analyze_data_quality(col_data, csv_col)
                            score = _calculate_quality_score(quality)
                            quality_scores_all[csv_col] = {
                                "score": score,
                                "n": quality.get('n', 0),
                                "cv": quality.get('cv', 0),
                                "outliers": quality.get('outliers', 0)
                            }

                    # Display quality dashboard
                    if quality_scores_all:
                        cols_dash = st.columns(min(4, len(quality_scores_all)))
                        for idx, (col_name, scores) in enumerate(quality_scores_all.items()):
                            with cols_dash[idx % len(cols_dash)]:
                                color = _get_quality_color(scores['score'])
                                st.metric(
                                    label=f"{color} {col_name}",
                                    value=f"{scores['score']}/100",
                                    delta=f"n={scores['n']}"
                                )

                    # Let user map columns to parameters
                    st.divider()
                    st.subheader("Map columns to parameters")
                    st.caption("Select which CSV columns should update which parameters. Uncheck to skip.")

                    suggested_ranges = {}
                    mappings = {}
                    quality_issues = []

                    for csv_col in numeric_cols:
                        col_data = df_upload[csv_col].dropna()
                        if len(col_data) < 10:
                            continue

                        # Analyze data quality
                        quality = _analyze_data_quality(col_data, csv_col)
                        score = _calculate_quality_score(quality)

                        # Calculate suggested ranges
                        p05 = float(np.percentile(col_data, 5))
                        p50 = float(np.percentile(col_data, 50))
                        p95 = float(np.percentile(col_data, 95))

                        suggested_ranges[csv_col] = {
                            "low": p05,
                            "mode": p50,
                            "high": p95,
                            "quality": quality
                        }

                        # Track columns with quality issues
                        if quality.get("warnings"):
                            quality_issues.append((csv_col, quality["warnings"]))

                        # Find matching parameter name (if any)
                        matching_param = None
                        for p_name in params.keys():
                            if csv_col.lower().replace("_", "") in p_name.lower().replace("_", ""):
                                matching_param = p_name
                                break

                        # Create mapping UI
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            use_col = st.checkbox(csv_col, value=(matching_param is not None), key=f"use_{csv_col}")
                        with col2:
                            if use_col:
                                target_param = st.selectbox(
                                    "maps to",
                                    options=list(params.keys()),
                                    index=list(params.keys()).index(matching_param) if matching_param else 0,
                                    key=f"map_{csv_col}",
                                    label_visibility="collapsed"
                                )
                                mappings[csv_col] = target_param
                        with col3:
                            if use_col:
                                st.caption(f"{p05:.4f} / {p50:.4f} / {p95:.4f}")

                        # Show detailed quality info for selected columns
                        if use_col:
                            # Warnings
                            if quality.get("warnings"):
                                for warning in quality["warnings"]:
                                    st.caption(warning)

                            # Insights
                            if quality.get("insights"):
                                for insight in quality["insights"]:
                                    st.caption(insight)

                            # Show distribution plot
                            with st.expander(f"📊 View distribution: {csv_col}", expanded=False):
                                # Distribution histogram
                                fig = _create_distribution_plot(col_data, csv_col, p05, p50, p95)
                                st.plotly_chart(fig, width='stretch')

                                # Statistical summary
                                st.write("**Statistical Summary:**")
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Mean", f"{quality['mean']:.4f}")
                                    st.metric("Median", f"{quality['median']:.4f}")
                                with col_b:
                                    st.metric("Std Dev", f"{quality['std']:.4f}")
                                    st.metric("CV", f"{quality['cv']:.3f}")
                                with col_c:
                                    st.metric("Skewness", f"{quality['skewness']:.3f}")
                                    st.metric("Outliers", f"{quality['outliers']}")

                                # Bootstrap CIs if available
                                if quality.get('ci_p05') and quality['ci_p05'][0] is not None:
                                    st.write("**95% Confidence Intervals:**")
                                    st.caption(f"p5:  [{quality['ci_p05'][0]:.4f}, {quality['ci_p05'][1]:.4f}]")
                                    st.caption(f"p50: [{quality['ci_p50'][0]:.4f}, {quality['ci_p50'][1]:.4f}]")
                                    st.caption(f"p95: [{quality['ci_p95'][0]:.4f}, {quality['ci_p95'][1]:.4f}]")

                                # Comparison with current YAML if parameter exists
                                if target_param in params:
                                    st.write("**Comparison with current config:**")
                                    yaml_ranges = params[target_param]
                                    comparison_fig = _create_comparison_chart(
                                        csv_col,
                                        {"low": p05, "mode": p50, "high": p95},
                                        yaml_ranges
                                    )
                                    st.plotly_chart(comparison_fig, width='stretch')

                    # Show overall data quality summary if there are issues
                    if quality_issues:
                        with st.expander("⚠️ Data Quality Summary", expanded=False):
                            for col, warnings in quality_issues:
                                st.write(f"**{col}:**")
                                for w in warnings:
                                    st.write(f"  {w}")

                    if mappings:
                        st.divider()

                        # Bulk Comparison View
                        with st.expander("📋 Bulk Comparison: All Parameters", expanded=False):
                            comparison_data = []
                            for csv_col, param_name in mappings.items():
                                ranges = suggested_ranges[csv_col]
                                yaml_range = params.get(param_name, {})
                                quality_info = ranges['quality']
                                quality_score = _calculate_quality_score(quality_info)

                                comparison_data.append({
                                    "CSV Column": csv_col,
                                    "→ Parameter": param_name,
                                    "Quality": f"{_get_quality_color(quality_score)} {quality_score}/100",
                                    "Current Low": yaml_range.get('low', '-'),
                                    "Suggested Low": ranges['low'],
                                    "Current Mode": yaml_range.get('mode', '-'),
                                    "Suggested Mode": ranges['mode'],
                                    "Current High": yaml_range.get('high', '-'),
                                    "Suggested High": ranges['high'],
                                    "Sample Size": quality_info.get('n', '-'),
                                })

                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, width='stretch')
                                st.caption("Sort by Quality to see best data first")

                        if st.button("Apply suggested ranges", type="primary"):
                            # Apply mappings to session state
                            applied = []
                            for csv_col, param_name in mappings.items():
                                ranges = suggested_ranges[csv_col]
                                st.session_state[f"{param_name}__low"] = ranges["low"]
                                st.session_state[f"{param_name}__mode"] = ranges["mode"]
                                st.session_state[f"{param_name}__high"] = ranges["high"]
                                applied.append(f"{csv_col} → {param_name}")

                            # Log the application
                            logger.info(f"Applied {len(mappings)} parameter ranges from CSV: {', '.join(applied)}")

                            st.success(f"Applied {len(mappings)} parameter ranges from your data")
                            st.rerun()
                    else:
                        st.info("Check at least one column to apply ranges")

            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")

    with st.expander("Edit parameter ranges (in-memory)", expanded=False):
        st.write("These changes are temporary. They won’t modify the YAML.")

        search = st.text_input("Filter parameters", value="", placeholder="e.g. failure, churn, uplift")

        c_reset, c_hint = st.columns([1, 2])
        with c_reset:
            if st.button("Reset overrides", width='stretch'):
                _reset_param_state(cfg)
                st.rerun()
        with c_hint:
            st.caption("Resets all low/mode/high fields back to YAML.")

        # Group params for readability
        grouped: Dict[str, list[str]] = {}
        search_norm = search.strip().lower()

        for p_name, spec in params.items():
            if not isinstance(spec, dict):
                continue
            if not all(k in spec for k in ("low", "mode", "high")):
                continue

            if search_norm and search_norm not in p_name.lower():
                continue

            grouped.setdefault(_group_for_param(p_name), []).append(p_name)

        for group_name in ["Base rates", "Stabilize core", "Feature extension", "New capability", "Do nothing", "Other"]:
            if group_name not in grouped:
                continue

            st.subheader(group_name)
            for p_name in grouped[group_name]:
                spec = params[p_name]
                dist = spec.get("dist", "triangular")

                # Keep controls numeric and explicit (no fancy inference about units).
                col_a, col_b, col_c = st.columns(3)

                low_key = f"{p_name}__low"
                mode_key = f"{p_name}__mode"
                high_key = f"{p_name}__high"

                # Use number inputs (more precise than sliders for this use case)
                st.markdown(f"**{p_name}**")

                with col_a:
                    low = st.number_input("low", key=low_key, format="%.6f", label_visibility="collapsed")
                with col_b:
                    mode = st.number_input("mode", key=mode_key, format="%.6f", label_visibility="collapsed")
                with col_c:
                    high = st.number_input("high", key=high_key, format="%.6f", label_visibility="collapsed")

                low2, mode2, high2 = _clamp_triplet(float(low), float(mode), float(high))

                # Don’t mutate widget state; just surface it.
                if (low2, mode2, high2) != (float(low), float(mode), float(high)):
                    st.caption("Note: low/mode/high should satisfy low ≤ mode ≤ high. The simulator will clamp values internally.")

                st.caption(f"{dist}")

    st.divider()
    run_btn = st.button("Run simulation", type="primary")


if run_btn:
    if not _check_rate_limit():
        st.error(f"❌ Rate limit hit ({MAX_SIMULATIONS_PER_HOUR}/hour). Try again later.")
        st.stop()

    overrides = _build_overrides_from_state(cfg)

    if overrides:
        with st.expander("What changed (vs YAML)", expanded=True):
            st.dataframe(_diff_table(cfg, overrides), width='stretch')
    else:
        st.caption("No assumption overrides applied (using YAML as-is).")

    try:
        logger.info(f"Running simulation: n_worlds={n_worlds}, seed={seed}, scenario={scenario}, overrides={len(overrides) if overrides else 0}")
        df = run_simulation(
            CONFIG_PATH,
            n_worlds=int(n_worlds),
            seed=int(seed),
            scenario=str(scenario),
            param_overrides=overrides if overrides else None,
        )
        logger.info(f"Simulation completed successfully: {len(df)} worlds simulated")
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        st.error(f"Simulation failed: {str(e)}")
        st.stop()

    # --- Charts ------------------------------------------------------
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Executive Dashboard", "Distributions", "Details", "Methodology", "Export"])

    with tab1:
        summary_df = summarize_results(df)
        diag = decision_diagnostics(df)
        sens = sensitivity_analysis(df)

        # 1. Dashboard - builds metric understanding
        st.plotly_chart(create_decision_dashboard(summary_df, diag, sens), width='stretch')

        # 2. Trade-off Matrix - shows what to avoid with context
        st.plotly_chart(create_trade_off_matrix(summary_df, diag), width='stretch')

        # 3. Executive Summary Table - numbers for those who want them
        st.subheader("Executive Summary Table")
        export_table = create_executive_summary_table(summary_df, diag)
        st.dataframe(export_table, width='stretch')

        # 4. Risk Profile - deeper risk analysis
        st.subheader("Risk Profile")
        st.plotly_chart(create_risk_profile_chart(summary_df), width='stretch')

        # 5. Regret Analysis - deeper regret analysis
        st.subheader("Regret Analysis")
        st.plotly_chart(create_regret_comparison(diag), width='stretch')

    with tab2:
        st.subheader("Outcome distributions across plausible futures")
        st.caption("Each point is one simulated world under the current assumptions.")

        v = _long_values(df)
        v["option"] = pd.Categorical(v["option"], categories=OPTION_COLS, ordered=True)

        # Apply clean labels
        v["option_clean"] = v["option"].apply(lambda x: clean_label(OPTION_LABELS.get(x, x)))

        fig1 = px.violin(
            v,
            x="option_clean",
            y="value_eur",
            box=True,
            points=False,
        )
        fig1.update_layout(
            yaxis_title="Value (k€)",
            xaxis_title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=16, color='black'),
            height=550,
            margin=dict(l=80, r=80, t=80, b=80)
        )
        fig1.update_xaxes(showgrid=False, tickfont=dict(size=14, color='black'))
        fig1.update_yaxes(showgrid=True, gridcolor='#d0d0d0', tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
        st.plotly_chart(fig1, width='stretch')

        st.subheader("Regret across plausible futures")
        st.caption("Regret is measured per world against the best-performing option in that same world.")

        r = _regret_long(df)
        r["option"] = pd.Categorical(r["option"], categories=OPTION_COLS, ordered=True)

        # Apply clean labels
        r["option_clean"] = r["option"].apply(lambda x: clean_label(OPTION_LABELS.get(x, x)))

        fig2 = px.box(
            r,
            x="option_clean",
            y="regret_eur",
            points=False,
        )
        fig2.update_layout(
            yaxis_title="Regret (k€)",
            xaxis_title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=16, color='black'),
            height=550,
            margin=dict(l=80, r=80, t=80, b=80)
        )
        fig2.update_xaxes(showgrid=False, tickfont=dict(size=14, color='black'))
        fig2.update_yaxes(showgrid=True, gridcolor='#d0d0d0', tickfont=dict(size=14, color='black'), title_font=dict(size=16, color='black'))
        st.plotly_chart(fig2, width='stretch')

    with tab3:
        st.subheader("Raw Tables")
        st.caption("Detailed numerical breakdowns for deep analysis.")

        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Summary")
            st.dataframe(_clean_table_names(_format_money_cols(summary_df)), width='stretch')

        with c2:
            st.subheader("Decision Diagnostics")
            st.dataframe(_clean_table_names(_format_rate_cols(_format_money_cols(diag))), width='stretch')

        st.subheader("Sensitivity (Feature Extension vs Stabilize Core)")
        st.dataframe(_clean_table_names(_format_rate_cols(sens)), width='stretch')

        st.subheader("Scenario Comparison")
        st.caption("Compare how options perform under different parameter assumptions (base/conservative/aggressive).")
        sc = run_all_scenarios(CONFIG_PATH, n_worlds=int(n_worlds), seed=int(seed))
        st.dataframe(_clean_table_names(_format_rate_cols(_format_money_cols(sc))), width='stretch')

    with tab4:
        st.subheader("Methodology")
        st.markdown("""
        This framework uses Monte Carlo simulation to compare decision options under uncertainty.

        ### How It Works (Plain Language)

        1. **Define uncertainty**: Each parameter (adoption rate, cost, failure rate) is expressed as a range rather than a single number.

        2. **Generate possible futures**: The simulation creates thousands of plausible scenarios by randomly sampling from these ranges.

        3. **Identical conditions**: All options are evaluated under the same scenarios, so differences reflect decision quality, not luck.

        4. **Comparative metrics**: Rather than predicting outcomes, we compare options using multiple lenses (expected value, downside risk, regret, robustness).

        ---

        ### Key Metrics Explained

        **Expected Value (EV)**
        Average outcome across all simulated futures. Shows what tends to work well on average.
        Higher is better for value maximizers.

        **P05 / P50 / P95 (Percentiles)**
        - P05: 5th percentile (downside scenario)
        - P50: Median (typical scenario)
        - P95: 95th percentile (upside scenario)

        Higher floor (P05) means less downside risk. Tighter range means more predictability.

        **Win Rate**
        Percentage of scenarios where this option delivers the highest value.
        Higher win rate = more robust across different futures.

        **Regret**
        Missed opportunity cost. How much worse you'd do compared to the best alternative in each scenario.
        Lower regret means less pain when you're wrong.

        **Sensitivity (Correlation)**
        Which input parameters most strongly affect outcomes.
        Identifies which uncertainties matter most for the decision.

        ---

        ### Technical Details

        **Sampling Method**
        Triangular distributions for each parameter (low, mode, high). Assumes independence between parameters.

        **Simulation Size**
        20,000 scenarios by default. Large enough for stable percentile estimates.

        **Common Random Numbers**
        All options use the same random seed per scenario. This ensures fair comparison by eliminating sampling noise.

        **Regret Calculation**
        For each scenario *i*:
        `Regret(option, i) = max(all options in scenario i) - value(option, i)`

        **Sensitivity Analysis**
        Spearman rank correlation between each input parameter and the outcome.
        Measures monotonic relationship strength (−1 to +1).

        **Win Rate Calculation**
        `Win Rate(option) = count(scenarios where option has max value) / total scenarios`

        ---

        ### Limitations

        - Assumes parameter independence (no correlation modeling)
        - Single time horizon (not multi-period decisions)
        - Triangular distributions (simplifying assumption)
        - Does not account for real option value or learning opportunities

        ---

        ### References

        - Howard, R. A. (1988). Decision analysis: Practice and promise. *Management Science*.
        - Savage, S. L. (2009). *The Flaw of Averages*.
        - Raiffa, H., & Schlaifer, R. (1961). *Applied Statistical Decision Theory*.
        """)

    with tab5:
        st.subheader("Export Results")
        st.caption("Download simulation data for further analysis or archival.")

        col_export1, col_export2, col_export3 = st.columns(3)

        with col_export1:
            # Export raw simulation data
            csv_raw = df.to_csv(index=False)
            st.download_button(
                label="Download raw simulation data (CSV)",
                data=csv_raw,
                file_name=f"simulation_raw_{scenario}_{seed}.csv",
                mime="text/csv",
            )

        with col_export2:
            # Export summary
            csv_summary = summary_df.to_csv(index=False)
            st.download_button(
                label="Download summary (CSV)",
                data=csv_summary,
                file_name=f"simulation_summary_{scenario}_{seed}.csv",
                mime="text/csv",
            )

        with col_export3:
            # Export diagnostics
            csv_diag = diag.to_csv(index=False)
            st.download_button(
                label="Download diagnostics (CSV)",
                data=csv_diag,
                file_name=f"simulation_diagnostics_{scenario}_{seed}.csv",
                mime="text/csv",
            )

else:
    st.write("Use the sidebar to run the simulation.")
    st.write("You can optionally edit parameter ranges. Changes are applied in-memory only.")