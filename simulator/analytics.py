"""Analytics helpers for simulation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simulator.config import get_decision_policy, get_simulation_settings, load_config
from simulator.policy import select_recommendation
from simulator.simulation import OPTION_COLUMNS, OPTION_LABELS, run_simulation

DEFAULT_STABILITY_SEEDS = (11, 17, 23, 31, 42)
DEFAULT_STABILITY_WORLD_COUNTS = (5000, 10000, 20000)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary metrics for each option."""

    rows = []
    for option in OPTION_COLUMNS:
        values = df[option].to_numpy()
        rows.append(
            {
                "option": option,
                "mean_value_eur": float(np.mean(values)),
                "median_value_eur": float(np.median(values)),
                "p05_value_eur": float(np.quantile(values, 0.05)),
                "p95_value_eur": float(np.quantile(values, 0.95)),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_value_eur", ascending=False).reset_index(drop=True)


def decision_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Return win-rate and regret diagnostics for each option."""

    values = df[OPTION_COLUMNS].to_numpy()
    best = values.max(axis=1)
    rows = []
    for index, option in enumerate(OPTION_COLUMNS):
        option_values = values[:, index]
        regret = best - option_values
        rows.append(
            {
                "option": option,
                "win_rate": float(np.mean(option_values == best)),
                "mean_regret_eur": float(np.mean(regret)),
                "median_regret_eur": float(np.median(regret)),
                "p95_regret_eur": float(np.quantile(regret, 0.95)),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_regret_eur").reset_index(drop=True)


def sensitivity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-option Spearman correlations between parameters and outcomes."""

    params = [column for column in df.columns if column not in {*OPTION_COLUMNS, "scenario"}]
    rows = []
    for option in OPTION_COLUMNS:
        option_values = pd.Series(df[option])
        for parameter in params:
            parameter_values = pd.Series(df[parameter])
            if (
                parameter_values.nunique(dropna=False) <= 1
                or option_values.nunique(dropna=False) <= 1
            ):
                corr = 0.0
            else:
                corr = parameter_values.corr(option_values, method="spearman")
            rows.append(
                {
                    "option": option,
                    "parameter": parameter,
                    "spearman_corr": 0.0 if pd.isna(corr) else float(corr),
                }
            )

    return (
        pd.DataFrame(rows)
        .assign(abs_spearman=lambda frame: frame["spearman_corr"].abs())
        .sort_values(["option", "abs_spearman"], ascending=[True, False])
        .drop(columns=["abs_spearman"])
        .reset_index(drop=True)
    )


def decision_delta_sensitivity(
    results: pd.DataFrame,
    selected_option: str,
    runner_up: str,
) -> pd.DataFrame:
    """Return descriptive sensitivity of the selected-vs-runner-up delta."""

    delta = pd.Series(results[selected_option] - results[runner_up], name="delta_value_eur")
    params = [column for column in results.columns if column not in {*OPTION_COLUMNS, "scenario"}]
    rows = []
    for parameter in params:
        parameter_values = pd.Series(results[parameter])
        if parameter_values.nunique(dropna=False) <= 1 or delta.nunique(dropna=False) <= 1:
            corr = 0.0
        else:
            corr = parameter_values.corr(delta, method="spearman")
        rows.append(
            {
                "parameter": parameter,
                "delta_spearman_corr": 0.0 if pd.isna(corr) else float(corr),
            }
        )

    return (
        pd.DataFrame(rows)
        .assign(abs_delta_spearman=lambda frame: frame["delta_spearman_corr"].abs())
        .sort_values("abs_delta_spearman", ascending=False)
        .drop(columns=["abs_delta_spearman"])
        .reset_index(drop=True)
    )


def stability_analysis(
    config_path: str | Path,
    seeds: tuple[int, ...] = DEFAULT_STABILITY_SEEDS,
    world_counts: tuple[int, ...] = DEFAULT_STABILITY_WORLD_COUNTS,
) -> pd.DataFrame:
    """Run a modest stability sweep across seeds and world counts."""

    cfg = load_config(config_path)
    simulation = get_simulation_settings(cfg)
    policy = get_decision_policy(cfg)
    default_scenario = str(simulation["scenario"])

    rows: list[dict[str, object]] = []
    for world_count in world_counts:
        for seed in seeds:
            results = run_simulation(
                config_path,
                n_worlds=int(world_count),
                seed=int(seed),
                scenario=default_scenario,
            )
            summary = summarize_results(results)
            diagnostics = decision_diagnostics(results)
            recommendation = select_recommendation(summary, diagnostics, policy)
            ev_leader = str(summary.iloc[0]["option"])
            runner_p05 = float(
                summary.loc[summary["option"] == recommendation.runner_up, "p05_value_eur"].iloc[0]
            )
            rows.append(
                {
                    "scenario": default_scenario,
                    "n_worlds": int(world_count),
                    "seed": int(seed),
                    "selected_option": recommendation.selected_option,
                    "runner_up": recommendation.runner_up,
                    "ev_leader": ev_leader,
                    "selected_mean_value_eur": recommendation.selected_mean_value_eur,
                    "runner_up_mean_value_eur": recommendation.runner_up_mean_value_eur,
                    "selected_p05_value_eur": recommendation.selected_p05_value_eur,
                    "runner_up_p05_value_eur": runner_p05,
                    "binding_constraint": recommendation.binding_constraint,
                }
            )

    return pd.DataFrame(rows).sort_values(["n_worlds", "seed"]).reset_index(drop=True)


def stability_summary(stability_runs: pd.DataFrame) -> dict[str, object]:
    """Summarize recommendation and ranking stability across reruns."""

    if stability_runs.empty:
        return {
            "run_count": 0,
            "recommendation_frequency": [],
            "ev_leader_frequency": [],
            "selected_p05_range_eur": None,
            "runner_up_p05_range_eur": None,
        }

    recommendation_frequency = (
        stability_runs.groupby("selected_option", observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    ev_leader_frequency = (
        stability_runs.groupby("ev_leader", observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return {
        "run_count": int(len(stability_runs)),
        "recommendation_frequency": recommendation_frequency.to_dict(orient="records"),
        "ev_leader_frequency": ev_leader_frequency.to_dict(orient="records"),
        "selected_p05_range_eur": float(
            stability_runs["selected_p05_value_eur"].max()
            - stability_runs["selected_p05_value_eur"].min()
        ),
        "runner_up_p05_range_eur": float(
            stability_runs["runner_up_p05_value_eur"].max()
            - stability_runs["runner_up_p05_value_eur"].min()
        ),
    }


__all__ = [
    "OPTION_COLUMNS",
    "OPTION_LABELS",
    "decision_delta_sensitivity",
    "decision_diagnostics",
    "sensitivity_analysis",
    "stability_analysis",
    "stability_summary",
    "summarize_results",
]
