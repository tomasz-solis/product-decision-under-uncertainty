# app.py
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from simulator.config import load_config, get_seed, get_simulation_settings
from simulator.mvp_simulator import (
    run_simulation,
    summarize_results,
    decision_diagnostics,
    sensitivity_analysis,
    run_all_scenarios,
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

        uploaded_file = st.file_uploader("Choose CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
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

                    # Let user map columns to parameters
                    st.subheader("Map columns to parameters")
                    st.caption("Select which CSV columns should update which parameters. Uncheck to skip.")

                    suggested_ranges = {}
                    mappings = {}

                    for csv_col in numeric_cols:
                        col_data = df_upload[csv_col].dropna()
                        if len(col_data) < 10:
                            continue

                        # Calculate suggested ranges
                        p05 = float(np.percentile(col_data, 5))
                        p50 = float(np.percentile(col_data, 50))
                        p95 = float(np.percentile(col_data, 95))

                        suggested_ranges[csv_col] = {"low": p05, "mode": p50, "high": p95}

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

                    if mappings:
                        st.divider()
                        if st.button("Apply suggested ranges", type="primary"):
                            # Apply mappings to session state
                            for csv_col, param_name in mappings.items():
                                ranges = suggested_ranges[csv_col]
                                st.session_state[f"{param_name}__low"] = ranges["low"]
                                st.session_state[f"{param_name}__mode"] = ranges["mode"]
                                st.session_state[f"{param_name}__high"] = ranges["high"]
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
    overrides = _build_overrides_from_state(cfg)

    if overrides:
        with st.expander("What changed (vs YAML)", expanded=True):
            st.dataframe(_diff_table(cfg, overrides), width='stretch')
    else:
        st.caption("No assumption overrides applied (using YAML as-is).")

    try:
        df = run_simulation(
            CONFIG_PATH,
            n_worlds=int(n_worlds),
            seed=int(seed),
            scenario=str(scenario),
            param_overrides=overrides if overrides else None,
        )
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        st.stop()

    # --- Tables ---------------------------------------------------------------
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Summary")
        st.dataframe(_format_money_cols(summarize_results(df)), width='stretch')

    with c2:
        st.subheader("Decision diagnostics")
        st.dataframe(_format_rate_cols(_format_money_cols(decision_diagnostics(df))), width='stretch')

    st.subheader("Sensitivity (feature_extension vs stabilize_core)")
    st.dataframe(_format_rate_cols(sensitivity_analysis(df)), width='stretch')

    st.subheader("Scenario comparison")
    sc = run_all_scenarios(CONFIG_PATH, n_worlds=int(n_worlds), seed=int(seed))
    st.dataframe(_format_rate_cols(_format_money_cols(sc)), width='stretch')

    # --- Charts (Plotly) ------------------------------------------------------
    st.subheader("Outcome distributions across plausible futures")
    st.caption("Each point is one simulated world under the current assumptions.")

    v = _long_values(df)
    v["option"] = pd.Categorical(v["option"], categories=OPTION_COLS, ordered=True)

    fig1 = px.violin(
        v,
        x="option",
        y="value_eur",
        box=True,
        points=False,
    )
    fig1.update_layout(yaxis_title="Value (EUR)", xaxis_title="")
    st.plotly_chart(fig1, width='stretch')

    st.subheader("Regret across plausible futures")
    st.caption("Regret is measured per world against the best-performing option in that same world.")

    r = _regret_long(df)
    r["option"] = pd.Categorical(r["option"], categories=OPTION_COLS, ordered=True)
    fig2 = px.box(
        r,
        x="option",
        y="regret_eur",
        points=False,
    )
    fig2.update_layout(yaxis_title="Regret (EUR)", xaxis_title="")
    st.plotly_chart(fig2, width='stretch')

else:
    st.write("Use the sidebar to run the simulation.")
    st.write("You can optionally edit parameter ranges. Changes are applied in-memory only.")