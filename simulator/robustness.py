"""Decision-robustness helpers beyond descriptive sensitivity tables."""

from __future__ import annotations

import logging
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

from simulator.analytics import (
    DEFAULT_STABILITY_SEEDS,
    DEFAULT_STABILITY_WORLD_COUNTS,
    decision_diagnostics,
    independence_ablation,
    summarize_results,
)
from simulator.config import (
    get_analysis_settings,
    get_decision_policy,
    get_dependency_settings,
    get_simulation_settings,
    load_config,
    parse_param_specs,
)
from simulator.output_utils import labeled_option, material_driver_rows
from simulator.policy import policy_frontier_analysis, select_recommendation
from simulator.simulation import run_simulation

logger = logging.getLogger(__name__)


def build_robustness_report(
    config_path: str | Path,
    stability_runs: pd.DataFrame,
    driver_analysis: pd.DataFrame,
    selected_option: str,
) -> dict[str, Any]:
    """Build a compact robustness artifact for the current published case."""

    cfg = load_config(config_path)
    analysis = get_analysis_settings(cfg)
    logger.info("Building robustness report for '%s'.", selected_option)
    convergence = _convergence_rows(stability_runs)
    frontier_stability = _frontier_stability_rows(
        config_path,
        seeds=DEFAULT_STABILITY_SEEDS,
        world_counts=DEFAULT_STABILITY_WORLD_COUNTS,
    )
    metric_error_rows = _metric_error_rows(
        config_path,
        seeds=DEFAULT_STABILITY_SEEDS,
        world_counts=DEFAULT_STABILITY_WORLD_COUNTS,
    )
    stress_tests = _stress_test_rows(
        config_path=config_path,
        driver_analysis=driver_analysis,
        selected_option=selected_option,
        threshold=analysis.sensitivity_materiality_threshold_abs_spearman,
    )
    dependency_check = independence_ablation(
        config_path,
        n_worlds=int(cfg["simulation"]["n_worlds"]),
        seed=int(cfg["project"]["seed"]),
        scenario=str(cfg["simulation"]["scenario"]),
    )
    dependency_frontier = _dependency_value_frontier(
        config_path,
        n_worlds=min(int(cfg["simulation"]["n_worlds"]), 5000),
        seed=int(cfg["project"]["seed"]),
    )
    report = {
        "selected_option": selected_option,
        "convergence_rows": convergence,
        "frontier_stability_rows": frontier_stability,
        "metric_error_rows": metric_error_rows,
        "stress_test_rows": stress_tests,
        "dependency_ablation": dependency_check,
        "dependency_value_frontier": dependency_frontier,
    }
    logger.info("Robustness report complete.")
    return report


def build_robustness_markdown(payload: dict[str, Any]) -> str:
    """Render a compact markdown summary of the robustness artifact."""

    lines = [
        f"- Baseline selected option: **{labeled_option(str(payload['selected_option']))}**.",
        "- This artifact separates convergence, frontier stability, and directional driver stress.",
    ]

    dependency_ablation_payload = payload.get("dependency_ablation")
    if isinstance(dependency_ablation_payload, dict):
        lines.extend(
            [
                (
                    "- Dependency ablation: the correlated run selects "
                    f"**{labeled_option(str(dependency_ablation_payload['correlated_selected_option']))}**, "
                    "while the independence rerun selects "
                    f"**{labeled_option(str(dependency_ablation_payload['independent_selected_option']))}**."
                ),
                (
                    "- For the correlated selected option, downside P05 moves from "
                    f"**{_markdown_scalar(dependency_ablation_payload['selected_option_p05_independent_eur'])}** "
                    "under independence to "
                    f"**{_markdown_scalar(dependency_ablation_payload['selected_option_p05_correlated_eur'])}** "
                    "with dependencies."
                ),
                "",
                "Dependency ablation by option:",
                "",
                _markdown_table(
                    pd.DataFrame(dependency_ablation_payload["comparison_rows"]).reindex(
                        columns=[
                            "option_label",
                            "correlated_mean_value_eur",
                            "independent_mean_value_eur",
                            "correlated_p05_value_eur",
                            "independent_p05_value_eur",
                        ]
                    ).rename(
                        columns={
                            "option_label": "Option",
                            "correlated_mean_value_eur": "Correlated EV (EUR)",
                            "independent_mean_value_eur": "Independent EV (EUR)",
                            "correlated_p05_value_eur": "Correlated P05 (EUR)",
                            "independent_p05_value_eur": "Independent P05 (EUR)",
                        }
                    )
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Convergence by world count:",
            "",
            _markdown_table(
                pd.DataFrame(payload["convergence_rows"]).reindex(
                    columns=[
                        "world_count",
                        "run_count",
                        "recommendation_consistency",
                        "selected_ev_std_eur",
                        "selected_p05_std_eur",
                    ]
                ).rename(
                    columns={
                        "world_count": "Worlds",
                        "run_count": "Runs",
                        "recommendation_consistency": "Recommendation consistency",
                        "selected_ev_std_eur": "Selected EV std (EUR)",
                        "selected_p05_std_eur": "Selected P05 std (EUR)",
                    }
                )
            ),
            "",
            "Frontier stability across the published seed/world-count grid:",
            "",
            _markdown_table(
                pd.DataFrame(payload["frontier_stability_rows"]).reindex(
                    columns=[
                        "threshold_label",
                        "switching_options_observed",
                        "switching_value_min",
                        "switching_value_max",
                        "switch_type_modes",
                    ]
                ).rename(
                    columns={
                        "threshold_label": "Threshold",
                        "switching_options_observed": "Switching option(s)",
                        "switching_value_min": "Min boundary",
                        "switching_value_max": "Max boundary",
                        "switch_type_modes": "Observed switch type(s)",
                    }
                )
            ),
            "",
            "Monte Carlo error bands by option and world count:",
            "",
            _markdown_table(
                pd.DataFrame(payload["metric_error_rows"]).reindex(
                    columns=[
                        "world_count",
                        "option",
                        "mean_value_range_eur",
                        "p05_value_range_eur",
                    ]
                ).assign(option=lambda frame: frame["option"].map(labeled_option)).rename(
                    columns={
                        "world_count": "Worlds",
                        "option": "Option",
                        "mean_value_range_eur": "EV range (EUR)",
                        "p05_value_range_eur": "P05 range (EUR)",
                    }
                )
            ),
            "",
            "Directional stress tests on the strongest material drivers:",
            "",
            _markdown_table(
                pd.DataFrame(payload["stress_test_rows"]).reindex(
                    columns=[
                        "parameter",
                        "stress_level",
                        "tested_value",
                        "selected_option",
                        "recommendation_changed",
                    ]
                ).assign(
                    selected_option=lambda frame: frame["selected_option"].map(labeled_option),
                    recommendation_changed=lambda frame: frame["recommendation_changed"].map(
                        lambda value: "yes" if bool(value) else "no"
                    ),
                ).rename(
                    columns={
                        "parameter": "Parameter",
                        "stress_level": "Stress level",
                        "tested_value": "Tested value",
                        "selected_option": "Selected option",
                        "recommendation_changed": "Recommendation changed?",
                    }
                )
            ),
        ]
    )

    dvf_rows = payload.get("dependency_value_frontier", [])
    if dvf_rows:
        lines.extend(
            [
                "",
                "Dependency-value frontier (each pair swept across rho grid):",
                "",
                "- Rows show whether the recommendation changed when a single dependency "
                "correlation was moved to a tested value while others stayed fixed.",
                "",
                _markdown_table(
                    pd.DataFrame(dvf_rows)
                    .reindex(
                        columns=[
                            "pair",
                            "base_rho",
                            "tested_rho",
                            "selected_option",
                            "recommendation_changed",
                        ]
                    )
                    .assign(
                        selected_option=lambda frame: frame["selected_option"].map(labeled_option),
                        recommendation_changed=lambda frame: frame["recommendation_changed"].map(
                            lambda value: "yes" if bool(value) else "no"
                        ),
                    )
                    .rename(
                        columns={
                            "pair": "Dependency pair",
                            "base_rho": "Base rho",
                            "tested_rho": "Tested rho",
                            "selected_option": "Selected option",
                            "recommendation_changed": "Recommendation changed?",
                        }
                    )
                ),
            ]
        )

    return "\n".join(lines)


def _convergence_rows(stability_runs: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize recommendation stability and metric noise by world count."""

    rows: list[dict[str, Any]] = []
    for _world_count, group in stability_runs.groupby("n_worlds", observed=True):
        recommendation_consistency = 0.0
        if not group.empty:
            recommendation_consistency = float(
                group["selected_option"].value_counts(normalize=True).iloc[0]
            )
        world_count_int = int(group["n_worlds"].astype(int).iloc[0])
        rows.append(
            {
                "world_count": world_count_int,
                "run_count": int(len(group)),
                "recommendation_consistency": round(recommendation_consistency, 6),
                "consistency_target_met": recommendation_consistency >= 0.80,
                "selected_ev_std_eur": round(
                    float(group["selected_mean_value_eur"].astype(float).std(ddof=0)),
                    6,
                ),
                "selected_p05_std_eur": round(
                    float(group["selected_p05_value_eur"].astype(float).std(ddof=0)),
                    6,
                ),
                "selected_ev_range_eur": round(
                    float(
                        group["selected_mean_value_eur"].astype(float).max()
                        - group["selected_mean_value_eur"].astype(float).min()
                    ),
                    6,
                ),
                "selected_p05_range_eur": round(
                    float(
                        group["selected_p05_value_eur"].astype(float).max()
                        - group["selected_p05_value_eur"].astype(float).min()
                    ),
                    6,
                ),
            }
        )
    return sorted(rows, key=lambda row: row["world_count"])


def _frontier_stability_rows(
    config_path: str | Path,
    seeds: tuple[int, ...],
    world_counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Summarize how the full-option frontier moves across repeated reruns."""

    cfg = load_config(config_path)
    simulation = cfg["simulation"]
    scenario = str(simulation["scenario"])
    policy = get_decision_policy(cfg)
    rows: list[dict[str, Any]] = []
    for world_count in world_counts:
        for seed in seeds:
            results = run_simulation(
                config_path,
                n_worlds=int(world_count),
                seed=int(seed),
                scenario=scenario,
            )
            summary = summarize_results(results)
            diagnostics = decision_diagnostics(results)
            recommendation = select_recommendation(summary, diagnostics, policy)
            frontier = policy_frontier_analysis(summary, diagnostics, policy, recommendation)
            for row in frontier["frontier_rows"]:
                rows.append(
                    {
                        "threshold_name": row["threshold_name"],
                        "threshold_label": row["threshold_label"],
                        "switching_value": row["switching_value"],
                        "switch_type": row["switch_type"],
                        "first_switching_option": row["first_switching_option"],
                    }
                )
    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for threshold_name, group in frame.groupby("threshold_name", observed=True):
        label = str(group.iloc[0]["threshold_label"])
        switching_values = group["switching_value"].dropna().astype(float)
        switching_options = ", ".join(
            sorted(
                {
                    labeled_option(str(value))
                    for value in group["first_switching_option"].dropna().tolist()
                }
            )
        ) or "none"
        switch_types = ", ".join(sorted({str(value) for value in group["switch_type"].tolist()}))
        summary_rows.append(
            {
                "threshold_name": threshold_name,
                "threshold_label": label,
                "switching_options_observed": switching_options,
                "switching_value_min": None
                if switching_values.empty
                else round(float(switching_values.min()), 6),
                "switching_value_max": None
                if switching_values.empty
                else round(float(switching_values.max()), 6),
                "switch_type_modes": switch_types,
            }
        )
    return sorted(summary_rows, key=lambda row: row["threshold_name"])


def _dependency_value_frontier(
    config_path: str | Path,
    *,
    n_worlds: int = 5000,
    seed: int = 42,
    grid: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8),
) -> list[dict[str, Any]]:
    """Sweep each configured dependency rho across a grid and record recommendation flips.

    For each dependency pair (left, right), holds all other correlations at
    their configured values and varies only the tested pair across the grid.
    Records the selected option at each point and whether it differs from the
    baseline (all correlations at their configured values).
    """

    cfg = load_config(config_path)
    policy = get_decision_policy(cfg)
    simulation = get_simulation_settings(cfg)
    base_deps = get_dependency_settings(cfg)
    scenario = str(simulation["scenario"])

    base_results = run_simulation(config_path, n_worlds=n_worlds, seed=seed, scenario=scenario)
    base_summary = summarize_results(base_results)
    base_diagnostics = decision_diagnostics(base_results)
    base_recommendation = select_recommendation(base_summary, base_diagnostics, policy)
    base_selected = base_recommendation.selected_option

    rows: list[dict[str, Any]] = []
    for left, targets in base_deps.items():
        for right, base_rho in targets.items():
            for tested_rho in grid:
                overrides = {
                    left_key: dict(right_targets)
                    for left_key, right_targets in base_deps.items()
                }
                overrides[left][right] = tested_rho
                results = run_simulation(
                    config_path,
                    n_worlds=n_worlds,
                    seed=seed,
                    scenario=scenario,
                    dependency_overrides=overrides,
                )
                summary = summarize_results(results)
                diagnostics = decision_diagnostics(results)
                rec = select_recommendation(summary, diagnostics, policy)
                rows.append(
                    {
                        "pair": f"{left}:{right}",
                        "base_rho": round(float(base_rho), 3),
                        "tested_rho": round(float(tested_rho), 3),
                        "selected_option": rec.selected_option,
                        "recommendation_changed": rec.selected_option != base_selected,
                    }
                )
    logger.debug(
        "Dependency-value frontier complete: %d pair/rho combinations.",
        len(rows),
    )
    return rows


def _stress_test_rows(
    *,
    config_path: str | Path,
    driver_analysis: pd.DataFrame,
    selected_option: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Run one-way and paired stress tests for the strongest material drivers."""

    cfg = load_config(config_path)
    policy = get_decision_policy(cfg)
    specs = parse_param_specs(cfg)
    base_results = run_simulation(config_path, scenario=str(cfg["simulation"]["scenario"]))
    base_summary = summarize_results(base_results)
    base_diagnostics = decision_diagnostics(base_results)
    base_recommendation = select_recommendation(base_summary, base_diagnostics, policy)

    drivers = material_driver_rows(
        driver_analysis=driver_analysis,
        option=selected_option,
        threshold=threshold,
        limit=2,
    )
    if drivers.empty:
        return []

    rows: list[dict[str, Any]] = []
    driver_specs: list[dict[str, float | str]] = [
        {
            "name": str(row["parameter"]),
            "estimate": float(row["partial_rank_corr"]),
        }
        for _, row in drivers.iterrows()
    ]
    driver_names = [str(driver["name"]) for driver in driver_specs]
    for driver in driver_specs:
        driver_name = str(driver["name"])
        levels = _stress_levels_for_spec(specs[driver_name])
        for level_name, level_value in levels.items():
            results = run_simulation(
                config_path,
                seed=int(cfg["project"]["seed"]),
                n_worlds=int(cfg["simulation"]["n_worlds"]),
                scenario=str(cfg["simulation"]["scenario"]),
                param_overrides={driver_name: {"dist": "constant", "value": float(level_value)}},
            )
            summary = summarize_results(results)
            diagnostics = decision_diagnostics(results)
            recommendation = select_recommendation(summary, diagnostics, policy)
            rows.append(
                {
                    "parameter": driver_name,
                    "stress_level": level_name,
                    "tested_value": round(float(level_value), 6),
                    "selected_option": recommendation.selected_option,
                    "recommendation_changed": recommendation.selected_option
                    != base_recommendation.selected_option,
                }
            )

    if len(driver_specs) >= 2:
        paired = _paired_driver_overrides(driver_specs[:2], specs)
        for level_name, overrides in paired.items():
            results = run_simulation(
                config_path,
                seed=int(cfg["project"]["seed"]),
                n_worlds=int(cfg["simulation"]["n_worlds"]),
                scenario=str(cfg["simulation"]["scenario"]),
                param_overrides=overrides,
            )
            summary = summarize_results(results)
            diagnostics = decision_diagnostics(results)
            recommendation = select_recommendation(summary, diagnostics, policy)
            rows.append(
                {
                    "parameter": " + ".join(driver_names[:2]),
                    "stress_level": level_name,
                    "tested_value": "paired override",
                    "selected_option": recommendation.selected_option,
                    "recommendation_changed": recommendation.selected_option
                    != base_recommendation.selected_option,
                }
            )
    return rows


def _metric_error_rows(
    config_path: str | Path,
    seeds: tuple[int, ...],
    world_counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Return per-option EV and P05 ranges across the stability grid."""

    cfg = load_config(config_path)
    scenario = str(cfg["simulation"]["scenario"])
    rows: list[dict[str, Any]] = []
    for world_count in world_counts:
        for seed in seeds:
            results = run_simulation(
                config_path,
                n_worlds=int(world_count),
                seed=int(seed),
                scenario=scenario,
            )
            summary = summarize_results(results)
            for _, row in summary.iterrows():
                rows.append(
                    {
                        "world_count": int(world_count),
                        "seed": int(seed),
                        "option": str(row["option"]),
                        "mean_value_eur": float(row["mean_value_eur"]),
                        "p05_value_eur": float(row["p05_value_eur"]),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    return (
        frame.groupby(["world_count", "option"], observed=True)
        .agg(
            mean_value_range_eur=("mean_value_eur", lambda series: float(series.max() - series.min())),
            p05_value_range_eur=("p05_value_eur", lambda series: float(series.max() - series.min())),
        )
        .reset_index()
        .sort_values(["world_count", "option"])
        .to_dict(orient="records")
    )


def _stress_levels_for_spec(spec: Any) -> dict[str, float]:
    """Return low/base/high values for one parameter specification."""

    if spec.dist == "tri":
        return {"low": float(spec.low), "base": float(spec.mode), "high": float(spec.high)}
    if spec.dist == "uniform":
        midpoint = (float(spec.low) + float(spec.high)) / 2.0
        return {"low": float(spec.low), "base": midpoint, "high": float(spec.high)}
    if spec.dist == "constant":
        return {"low": float(spec.value), "base": float(spec.value), "high": float(spec.value)}
    return {"low": float(spec.median), "base": float(spec.median), "high": float(spec.p95)}


def _paired_driver_overrides(
    drivers: list[dict[str, float | str]],
    specs: dict[str, Any],
) -> dict[str, dict[str, dict[str, float | str]]]:
    """Return multi-driver stress cases, including opposing directions."""

    all_adverse: dict[str, dict[str, float | str]] = {}
    all_supportive: dict[str, dict[str, float | str]] = {}
    opposing_challenge: dict[str, dict[str, float | str]] = {}
    opposing_relief: dict[str, dict[str, float | str]] = {}

    for index, driver in enumerate(drivers):
        name = str(driver["name"])
        estimate = float(driver["estimate"])
        all_adverse[name] = _stress_override_for_direction(specs[name], estimate, "adverse")
        all_supportive[name] = _stress_override_for_direction(specs[name], estimate, "supportive")
        direction = "adverse" if index == 0 else "supportive"
        opposing_challenge[name] = _stress_override_for_direction(specs[name], estimate, direction)
        opposite_direction = "supportive" if index == 0 else "adverse"
        opposing_relief[name] = _stress_override_for_direction(
            specs[name],
            estimate,
            opposite_direction,
        )

    return {
        "paired_all_adverse": all_adverse,
        "paired_all_supportive": all_supportive,
        "paired_opposing_challenge": opposing_challenge,
        "paired_opposing_relief": opposing_relief,
    }


def _stress_override_for_direction(
    spec: Any,
    driver_estimate: float,
    direction: str,
) -> dict[str, float | str]:
    """Return the constant override that either helps or hurts the selected option."""

    levels = _stress_levels_for_spec(spec)
    hurts_when_higher = driver_estimate < 0.0
    if direction == "adverse":
        target_level = "high" if hurts_when_higher else "low"
    else:
        target_level = "low" if hurts_when_higher else "high"
    return {"dist": "constant", "value": float(levels[target_level])}


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small markdown table without optional dependencies."""

    headers = [str(column) for column in frame.columns]
    rows = [[_markdown_scalar(value) for value in row] for row in frame.values.tolist()]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _markdown_scalar(value: Any) -> str:
    """Return one stable markdown-cell string."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if pd.isna(number):
            return ""
        return f"{round(number, 6):.6f}".rstrip("0").rstrip(".")
    return str(value)
