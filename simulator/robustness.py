"""Decision-robustness helpers beyond descriptive sensitivity tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from simulator.analytics import (
    DEFAULT_STABILITY_SEEDS,
    DEFAULT_STABILITY_WORLD_COUNTS,
    decision_diagnostics,
    summarize_results,
)
from simulator.config import (
    get_analysis_settings,
    get_decision_policy,
    load_config,
    parse_param_specs,
)
from simulator.output_utils import labeled_option, material_sensitivity_rows
from simulator.policy import policy_frontier_analysis, select_recommendation
from simulator.simulation import run_simulation


def build_robustness_report(
    config_path: str | Path,
    stability_runs: pd.DataFrame,
    sensitivity: pd.DataFrame,
    selected_option: str,
) -> dict[str, Any]:
    """Build a compact robustness artifact for the current published case."""

    cfg = load_config(config_path)
    analysis = get_analysis_settings(cfg)
    convergence = _convergence_rows(stability_runs)
    frontier_stability = _frontier_stability_rows(
        config_path,
        seeds=DEFAULT_STABILITY_SEEDS,
        world_counts=DEFAULT_STABILITY_WORLD_COUNTS,
    )
    stress_tests = _stress_test_rows(
        config_path=config_path,
        sensitivity=sensitivity,
        selected_option=selected_option,
        threshold=analysis.sensitivity_materiality_threshold_abs_spearman,
    )
    return {
        "selected_option": selected_option,
        "convergence_rows": convergence,
        "frontier_stability_rows": frontier_stability,
        "stress_test_rows": stress_tests,
    }


def build_robustness_markdown(payload: dict[str, Any]) -> str:
    """Render a compact markdown summary of the robustness artifact."""

    lines = [
        f"- Baseline selected option: **{labeled_option(str(payload['selected_option']))}**.",
        "- This artifact separates convergence, frontier stability, and directional driver stress.",
        "",
        "Convergence by world count:",
        "",
        _markdown_table(
            pd.DataFrame(payload["convergence_rows"])[
                [
                    "world_count",
                    "run_count",
                    "recommendation_consistency",
                    "selected_ev_std_eur",
                    "selected_p05_std_eur",
                ]
            ].rename(
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
            pd.DataFrame(payload["frontier_stability_rows"])[
                [
                    "threshold_label",
                    "switching_options_observed",
                    "switching_value_min",
                    "switching_value_max",
                    "switch_type_modes",
                ]
            ].rename(
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
        "Directional stress tests on the strongest material drivers:",
        "",
        _markdown_table(
            pd.DataFrame(payload["stress_test_rows"])[
                [
                    "parameter",
                    "stress_level",
                    "tested_value",
                    "selected_option",
                    "recommendation_changed",
                ]
            ].assign(
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
                "selected_ev_std_eur": round(
                    float(group["selected_mean_value_eur"].astype(float).std(ddof=0)),
                    6,
                ),
                "selected_p05_std_eur": round(
                    float(group["selected_p05_value_eur"].astype(float).std(ddof=0)),
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


def _stress_test_rows(
    *,
    config_path: str | Path,
    sensitivity: pd.DataFrame,
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

    drivers = material_sensitivity_rows(
        sensitivity=sensitivity,
        option=selected_option,
        threshold=threshold,
        limit=2,
    )
    if drivers.empty:
        return []

    rows: list[dict[str, Any]] = []
    driver_names = [str(value) for value in drivers["parameter"].tolist()]
    for driver_name in driver_names:
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

    if len(driver_names) >= 2:
        paired: dict[str, dict[str, dict[str, float | str]]] = {
            "paired_low": {
                name: {
                    "dist": "constant",
                    "value": float(_stress_levels_for_spec(specs[name])["low"]),
                }
                for name in driver_names[:2]
            },
            "paired_high": {
                name: {
                    "dist": "constant",
                    "value": float(_stress_levels_for_spec(specs[name])["high"]),
                }
                for name in driver_names[:2]
            },
        }
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


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small markdown table without optional dependencies."""

    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.values.tolist()]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
