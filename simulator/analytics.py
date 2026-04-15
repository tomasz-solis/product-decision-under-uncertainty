"""Analytics helpers for simulation outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from simulator.config import get_decision_policy, get_simulation_settings, load_config
from simulator.policy import select_recommendation
from simulator.simulation import OPTION_COLUMNS, OPTION_LABELS, run_simulation

logger = logging.getLogger(__name__)
_HAS_LOGGED_SINGULAR_WARNING = False

DEFAULT_STABILITY_SEEDS = (11, 17, 23, 31, 42, 57)
DEFAULT_STABILITY_WORLD_COUNTS = (5000, 10000, 20000, 40000)
DEFAULT_DRIVER_BOOTSTRAP_SAMPLES = 40
DEFAULT_DRIVER_CONFIDENCE_LEVEL = 0.95


def spearman_rank_correlation(left: pd.Series, right: pd.Series) -> float:
    """Return a Spearman rank correlation without requiring SciPy."""

    paired = pd.concat(
        [
            pd.Series(left, copy=False, name="left"),
            pd.Series(right, copy=False, name="right"),
        ],
        axis=1,
    ).dropna()
    if paired.empty:
        return 0.0

    left_values = paired["left"]
    right_values = paired["right"]
    if left_values.nunique(dropna=False) <= 1 or right_values.nunique(dropna=False) <= 1:
        return 0.0

    ranked = paired.rank(method="average")
    correlation = ranked["left"].corr(ranked["right"], method="pearson")
    if pd.isna(correlation):
        return 0.0
    return float(correlation)


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
                corr = spearman_rank_correlation(parameter_values, option_values)
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


def driver_analysis(
    df: pd.DataFrame,
    *,
    bootstrap_samples: int = DEFAULT_DRIVER_BOOTSTRAP_SAMPLES,
    confidence_level: float = DEFAULT_DRIVER_CONFIDENCE_LEVEL,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Return partial-rank driver estimates with bootstrap intervals."""

    params = [column for column in df.columns if column not in {*OPTION_COLUMNS, "scenario"}]
    if not params:
        return pd.DataFrame(
            columns=[
                "option",
                "parameter",
                "partial_rank_corr",
                "ci_low",
                "ci_high",
                "signal_class",
                "method",
                "bootstrap_samples",
            ]
        )

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for option in OPTION_COLUMNS:
        analysis_frame = df[params + [option]].dropna().reset_index(drop=True)
        if analysis_frame.empty:
            continue
        point_estimates = _partial_rank_correlations(
            analysis_frame[params],
            pd.Series(analysis_frame[option], name=option),
        )
        bootstrap_estimates = _bootstrap_partial_rank_correlations(
            analysis_frame,
            feature_columns=params,
            target_column=option,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        for parameter in params:
            estimate = float(point_estimates.get(parameter, 0.0))
            ci_low, ci_high = _bootstrap_interval(
                bootstrap_estimates.get(parameter, []),
                confidence_level=confidence_level,
                fallback=estimate,
            )
            rows.append(
                {
                    "option": option,
                    "parameter": parameter,
                    "partial_rank_corr": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "signal_class": "decision_support",
                    "method": "partial_rank_correlation",
                    "bootstrap_samples": int(bootstrap_samples),
                }
            )

    return (
        pd.DataFrame(rows)
        .assign(abs_partial_rank_corr=lambda frame: frame["partial_rank_corr"].abs())
        .sort_values(["option", "abs_partial_rank_corr"], ascending=[True, False])
        .drop(columns=["abs_partial_rank_corr"])
        .reset_index(drop=True)
    )


def decision_delta_sensitivity(
    results: pd.DataFrame,
    selected_option: str,
    comparison_option: str,
) -> pd.DataFrame:
    """Return descriptive sensitivity of the selected-vs-runner-up delta."""

    delta = pd.Series(results[selected_option] - results[comparison_option], name="delta_value_eur")
    params = [column for column in results.columns if column not in {*OPTION_COLUMNS, "scenario"}]
    rows = []
    for parameter in params:
        parameter_values = pd.Series(results[parameter])
        if parameter_values.nunique(dropna=False) <= 1 or delta.nunique(dropna=False) <= 1:
            corr = 0.0
        else:
            corr = spearman_rank_correlation(parameter_values, delta)
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
            comparison_option = recommendation.comparison_option
            comparison_role = recommendation.comparison_option_role or "no_comparison_available"
            comparison_p05 = None
            comparison_mean_value = recommendation.comparison_mean_value_eur
            if comparison_option is not None:
                comparison_p05 = float(
                    summary.loc[summary["option"] == comparison_option, "p05_value_eur"].iloc[0]
                )
            rows.append(
                {
                    "scenario": default_scenario,
                    "n_worlds": int(world_count),
                    "seed": int(seed),
                    "selected_option": recommendation.selected_option,
                    "comparison_option": comparison_option,
                    "comparison_option_role": comparison_role,
                    "ev_leader": ev_leader,
                    "selected_mean_value_eur": recommendation.selected_mean_value_eur,
                    "comparison_mean_value_eur": comparison_mean_value,
                    "selected_p05_value_eur": recommendation.selected_p05_value_eur,
                    "comparison_p05_value_eur": comparison_p05,
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
            "comparison_p05_range_eur": None,
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
        "comparison_p05_range_eur": float(
            stability_runs["comparison_p05_value_eur"].dropna().max()
            - stability_runs["comparison_p05_value_eur"].dropna().min()
        )
        if stability_runs["comparison_p05_value_eur"].notna().any()
        else None,
    }


def independence_ablation(
    config_path: str | Path,
    *,
    n_worlds: int = 10_000,
    seed: int = 42,
    scenario: str | None = None,
) -> dict[str, object]:
    """Compare the published dependency model against full independence."""

    cfg = load_config(config_path)
    simulation = get_simulation_settings(cfg)
    chosen_scenario = str(simulation["scenario"] if scenario is None else scenario)
    policy = get_decision_policy(cfg)

    correlated_results = run_simulation(
        config_path,
        n_worlds=n_worlds,
        seed=seed,
        scenario=chosen_scenario,
    )
    independent_results = run_simulation(
        config_path,
        n_worlds=n_worlds,
        seed=seed,
        scenario=chosen_scenario,
        dependency_overrides={},
    )

    correlated_summary = summarize_results(correlated_results)
    independent_summary = summarize_results(independent_results)
    correlated_diagnostics = decision_diagnostics(correlated_results)
    independent_diagnostics = decision_diagnostics(independent_results)
    correlated_recommendation = select_recommendation(
        correlated_summary,
        correlated_diagnostics,
        policy,
    )
    independent_recommendation = select_recommendation(
        independent_summary,
        independent_diagnostics,
        policy,
    )

    correlated_indexed = correlated_summary.set_index("option")
    independent_indexed = independent_summary.set_index("option")
    selected_option = correlated_recommendation.selected_option
    correlated_selected = correlated_indexed.loc[selected_option]
    independent_selected = independent_indexed.loc[selected_option]

    comparison_rows: list[dict[str, object]] = []
    for option in OPTION_COLUMNS:
        correlated_row = correlated_indexed.loc[option]
        independent_row = independent_indexed.loc[option]
        comparison_rows.append(
            {
                "option": option,
                "option_label": OPTION_LABELS[option],
                "correlated_mean_value_eur": float(correlated_row["mean_value_eur"]),
                "independent_mean_value_eur": float(independent_row["mean_value_eur"]),
                "mean_value_delta_eur": float(
                    correlated_row["mean_value_eur"] - independent_row["mean_value_eur"]
                ),
                "correlated_p05_value_eur": float(correlated_row["p05_value_eur"]),
                "independent_p05_value_eur": float(independent_row["p05_value_eur"]),
                "p05_value_delta_eur": float(
                    correlated_row["p05_value_eur"] - independent_row["p05_value_eur"]
                ),
            }
        )

    logger.debug(
        "Completed dependency ablation for scenario '%s' (n_worlds=%d, seed=%d).",
        chosen_scenario,
        n_worlds,
        seed,
    )
    return {
        "scenario": chosen_scenario,
        "n_worlds": int(n_worlds),
        "seed": int(seed),
        "selected_option": selected_option,
        "selected_option_label": OPTION_LABELS[selected_option],
        "correlated_selected_option": correlated_recommendation.selected_option,
        "independent_selected_option": independent_recommendation.selected_option,
        "recommendation_changed": (
            correlated_recommendation.selected_option
            != independent_recommendation.selected_option
        ),
        "selected_option_p05_correlated_eur": float(correlated_selected["p05_value_eur"]),
        "selected_option_p05_independent_eur": float(independent_selected["p05_value_eur"]),
        "selected_option_p05_delta_eur": float(
            correlated_selected["p05_value_eur"] - independent_selected["p05_value_eur"]
        ),
        "selected_option_mean_correlated_eur": float(correlated_selected["mean_value_eur"]),
        "selected_option_mean_independent_eur": float(independent_selected["mean_value_eur"]),
        "selected_option_mean_delta_eur": float(
            correlated_selected["mean_value_eur"] - independent_selected["mean_value_eur"]
        ),
        "comparison_rows": comparison_rows,
        "interpretation": (
            "Negative delta means the configured dependency layer makes the metric worse than "
            "the independence baseline for the same option and run settings."
        ),
    }


def _partial_rank_correlations(
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Return partial-rank correlations for each feature against one target."""

    combined = pd.concat([features.reset_index(drop=True), target.reset_index(drop=True)], axis=1)
    combined = combined.dropna()
    if combined.empty:
        return {column: 0.0 for column in features.columns}

    ranked = combined.rank(method="average")
    direct_correlation = ranked.corr(method="pearson").fillna(0.0).iloc[:-1, -1]
    perfect_driver_mask = direct_correlation.abs() >= (1.0 - 1e-12)
    if perfect_driver_mask.any():
        return {
            str(feature_name): (
                float(np.sign(direct_correlation.loc[feature_name]))
                if bool(perfect_driver_mask.loc[feature_name])
                else 0.0
            )
            for feature_name in features.columns
        }

    corr_matrix = ranked.corr(method="pearson").fillna(0.0).to_numpy(dtype=float)
    condition_number = float(np.linalg.cond(corr_matrix))
    if not np.isfinite(condition_number) or condition_number > 1e10:
        _log_singular_partial_corr_warning(condition_number)
        return {column: 0.0 for column in features.columns}

    precision = np.linalg.pinv(corr_matrix)
    target_index = len(corr_matrix) - 1

    rows: dict[str, float] = {}
    for feature_index, feature_name in enumerate(features.columns):
        denominator = precision[feature_index, feature_index] * precision[target_index, target_index]
        if denominator <= 0.0:
            rows[str(feature_name)] = 0.0
            continue
        rows[str(feature_name)] = float(
            np.clip(
                -precision[feature_index, target_index] / np.sqrt(denominator),
                -1.0,
                1.0,
            )
        )
    return rows


def _bootstrap_partial_rank_correlations(
    analysis_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """Return bootstrap draws for the partial-rank driver estimates."""

    if bootstrap_samples < 1 or analysis_frame.empty:
        return {column: [] for column in feature_columns}

    draws: dict[str, list[float]] = {column: [] for column in feature_columns}
    n_rows = len(analysis_frame)
    for _ in range(bootstrap_samples):
        sample_indices = rng.integers(0, n_rows, size=n_rows)
        sampled = analysis_frame.iloc[sample_indices].reset_index(drop=True)
        estimates = _partial_rank_correlations(
            sampled[feature_columns],
            pd.Series(sampled[target_column], name=target_column),
        )
        for column in feature_columns:
            draws[column].append(float(estimates.get(column, 0.0)))
    return draws


def _bootstrap_interval(
    draws: list[float],
    *,
    confidence_level: float,
    fallback: float,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for one estimate."""

    if not draws:
        return fallback, fallback
    alpha = max(0.0, min(1.0, 1.0 - confidence_level))
    return (
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - (alpha / 2.0))),
    )


def _log_singular_partial_corr_warning(condition_number: float) -> None:
    """Log the near-singular partial-correlation warning once per process."""

    global _HAS_LOGGED_SINGULAR_WARNING

    if _HAS_LOGGED_SINGULAR_WARNING:
        logger.debug(
            "Suppressed repeated near-singular partial-correlation warning "
            "(condition number=%.1e).",
            condition_number,
        )
        return

    logger.debug(
        "Rank correlation matrix is near-singular (condition number=%.1e). "
        "Partial rank correlations will be zeroed out. Further occurrences "
        "will be suppressed for this process.",
        condition_number,
    )
    _HAS_LOGGED_SINGULAR_WARNING = True


__all__ = [
    "OPTION_COLUMNS",
    "OPTION_LABELS",
    "decision_delta_sensitivity",
    "decision_diagnostics",
    "driver_analysis",
    "independence_ablation",
    "sensitivity_analysis",
    "spearman_rank_correlation",
    "stability_analysis",
    "stability_summary",
    "summarize_results",
]
