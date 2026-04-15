"""Artifact generation and markdown helpers for the synthetic case study."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from simulator.analytics import (
    DEFAULT_STABILITY_SEEDS,
    DEFAULT_STABILITY_WORLD_COUNTS,
    decision_diagnostics,
    driver_analysis,
    sensitivity_analysis,
    stability_analysis,
    stability_summary,
    summarize_results,
)
from simulator.artifact_freshness import build_artifact_metadata, sha256_file
from simulator.config import (
    AnalysisConfig,
    get_analysis_settings,
    get_decision_policy,
    get_scenario_metadata,
    get_seed,
    get_simulation_settings,
    load_config,
    validate_config,
)
from simulator.evidence import profile_public_evidence, summarize_public_evidence
from simulator.output_utils import (
    format_eur_markdown,
    format_number,
    format_pct,
    format_threshold_eur_markdown,
    labeled_option,
    material_driver_rows,
)
from simulator.policy import (
    FAILURE_REASON_LABELS,
    FULL_FRONTIER_SWITCH_LABELS,
    RUNNER_UP_FRONTIER_STATUS_LABELS,
    PayoffDeltaDiagnostic,
    PolicyFrontierResult,
    RecommendationResult,
    build_policy_eligibility_table,
    payoff_delta_diagnostic,
    policy_frontier_analysis,
    policy_frontier_grid,
    select_recommendation,
)
from simulator.project_paths import (
    ASSUMPTION_REGISTRY_PATH,
    GENERATOR_SCRIPT_PATH,
    PARAMETER_REGISTRY_PATH,
    PUBLIC_DATA_DIR,
    SOURCE_MANIFEST_PATH,
)
from simulator.provenance import (
    build_assumption_manifest,
    load_assumption_registry,
    load_parameter_registry,
    validate_parameter_registry,
)
from simulator.robustness import build_robustness_markdown, build_robustness_report
from simulator.simulation import run_all_scenarios, run_simulation
from simulator.visualizations import create_ranked_payoff_profile

logger = logging.getLogger(__name__)
DECISION_SUMMARY_DIV_ID = "decision-summary-chart"

CASE_STUDY_MARKERS = {
    "recommendation": (
        "<!-- GENERATED:CASE_STUDY_RECOMMENDATION:START -->",
        "<!-- GENERATED:CASE_STUDY_RECOMMENDATION:END -->",
    ),
    "base_summary": (
        "<!-- GENERATED:CASE_STUDY_BASE_SUMMARY:START -->",
        "<!-- GENERATED:CASE_STUDY_BASE_SUMMARY:END -->",
    ),
    "regret": (
        "<!-- GENERATED:CASE_STUDY_REGRET:START -->",
        "<!-- GENERATED:CASE_STUDY_REGRET:END -->",
    ),
    "eligibility": (
        "<!-- GENERATED:CASE_STUDY_ELIGIBILITY:START -->",
        "<!-- GENERATED:CASE_STUDY_ELIGIBILITY:END -->",
    ),
    "scenarios": (
        "<!-- GENERATED:CASE_STUDY_SCENARIOS:START -->",
        "<!-- GENERATED:CASE_STUDY_SCENARIOS:END -->",
    ),
    "payoff_delta": (
        "<!-- GENERATED:CASE_STUDY_PAYOFF_DELTA:START -->",
        "<!-- GENERATED:CASE_STUDY_PAYOFF_DELTA:END -->",
    ),
    "frontier": (
        "<!-- GENERATED:CASE_STUDY_FRONTIER:START -->",
        "<!-- GENERATED:CASE_STUDY_FRONTIER:END -->",
    ),
    "stability": (
        "<!-- GENERATED:CASE_STUDY_STABILITY:START -->",
        "<!-- GENERATED:CASE_STUDY_STABILITY:END -->",
    ),
    "sensitivity": (
        "<!-- GENERATED:CASE_STUDY_SENSITIVITY:START -->",
        "<!-- GENERATED:CASE_STUDY_SENSITIVITY:END -->",
    ),
}

EXEC_SUMMARY_MARKERS = {
    "recommendation": (
        "<!-- GENERATED:EXEC_SUMMARY_RECOMMENDATION:START -->",
        "<!-- GENERATED:EXEC_SUMMARY_RECOMMENDATION:END -->",
    ),
    "results": (
        "<!-- GENERATED:EXEC_SUMMARY_RESULTS:START -->",
        "<!-- GENERATED:EXEC_SUMMARY_RESULTS:END -->",
    ),
}

@dataclass(frozen=True)
class CaseStudyArtifacts:
    """Generated outputs for the published synthetic case study."""

    metadata: dict[str, Any]
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    scenario_results: pd.DataFrame
    sensitivity: pd.DataFrame
    driver_analysis: pd.DataFrame
    parameter_registry: pd.DataFrame
    assumption_manifest: dict[str, Any]
    recommendation: RecommendationResult
    policy_eligibility: pd.DataFrame
    payoff_delta: PayoffDeltaDiagnostic
    policy_frontier: PolicyFrontierResult
    policy_frontier_grid: pd.DataFrame
    stability_runs: pd.DataFrame
    stability_summary: dict[str, Any]
    evidence_profile: dict[str, Any]
    evidence_summary: dict[str, Any]
    robustness: dict[str, Any]
    scenario_metadata: dict[str, dict[str, str]]
    analysis: AnalysisConfig


def build_case_study_artifacts(
    config_path: str | Path,
    n_worlds: int | None = None,
    seed: int | None = None,
) -> CaseStudyArtifacts:
    """Run the case-study model and assemble the reporting outputs."""

    config_path = Path(config_path)
    cfg = load_config(config_path)
    validate_config(cfg)

    simulation = get_simulation_settings(cfg)
    chosen_seed = get_seed(cfg) if seed is None else int(seed)
    if n_worlds is not None:
        simulation["n_worlds"] = int(n_worlds)
    logger.info(
        "Building case study artifacts (n_worlds=%d, seed=%d).",
        int(simulation["n_worlds"]),
        int(chosen_seed),
    )

    base_results = run_simulation(
        config_path,
        n_worlds=simulation["n_worlds"],
        seed=chosen_seed,
        scenario=simulation["scenario"],
    )
    summary = summarize_results(base_results)
    diagnostics = decision_diagnostics(base_results)
    sensitivity = sensitivity_analysis(base_results)
    driver_rows = driver_analysis(base_results)
    scenario_results = run_all_scenarios(
        config_path,
        n_worlds=simulation["n_worlds"],
        seed=chosen_seed,
    )

    analysis = get_analysis_settings(cfg)
    decision_policy = get_decision_policy(cfg)
    recommendation = select_recommendation(summary, diagnostics, decision_policy)
    eligibility = build_policy_eligibility_table(summary, diagnostics, decision_policy)
    payoff_delta = payoff_delta_diagnostic(base_results, recommendation, analysis)
    frontier = policy_frontier_analysis(summary, diagnostics, decision_policy, recommendation)
    frontier_grid = policy_frontier_grid(summary, diagnostics, decision_policy, recommendation)

    parameter_registry_path = config_path.parent / PARAMETER_REGISTRY_PATH.name
    parameter_registry = validate_parameter_registry(
        load_parameter_registry(parameter_registry_path),
        cfg["params"],
    )
    assumption_registry_path = config_path.parent / ASSUMPTION_REGISTRY_PATH.name
    assumption_manifest = build_assumption_manifest(
        cfg,
        parameter_registry=parameter_registry,
        assumption_registry=load_assumption_registry(assumption_registry_path),
    )

    config_sha = sha256_file(config_path)
    parameter_registry_sha = sha256_file(parameter_registry_path)
    assumption_registry_sha = sha256_file(assumption_registry_path)
    stability_runs = _cached_stability_runs(
        str(config_path),
        config_sha,
        parameter_registry_sha,
        assumption_registry_sha,
        DEFAULT_STABILITY_SEEDS,
        DEFAULT_STABILITY_WORLD_COUNTS,
    ).copy()
    stability_report = stability_summary(stability_runs)
    evidence_profile = profile_public_evidence(PUBLIC_DATA_DIR, SOURCE_MANIFEST_PATH)
    evidence_summary = summarize_public_evidence(evidence_profile)
    robustness = build_robustness_report(
        config_path=config_path,
        stability_runs=stability_runs,
        driver_analysis=driver_rows,
        selected_option=recommendation.selected_option,
    )
    scenario_metadata = get_scenario_metadata(cfg)

    metadata = build_artifact_metadata(
        cfg=cfg,
        config_path=config_path,
        parameter_registry_path=parameter_registry_path,
        assumption_registry_path=assumption_registry_path,
        generator_script_path=GENERATOR_SCRIPT_PATH,
        n_worlds=int(simulation["n_worlds"]),
        seed=int(chosen_seed),
        annual_volume=int(simulation["annual_volume"]),
        time_horizon_years=int(simulation["time_horizon_years"]),
        discount_rate_annual=float(simulation["discount_rate_annual"]),
        published_scenario=str(simulation["scenario"]),
    )
    artifacts = CaseStudyArtifacts(
        metadata=metadata,
        summary=summary,
        diagnostics=diagnostics,
        scenario_results=scenario_results,
        sensitivity=sensitivity,
        driver_analysis=driver_rows,
        parameter_registry=parameter_registry,
        assumption_manifest=assumption_manifest,
        recommendation=recommendation,
        policy_eligibility=eligibility,
        payoff_delta=payoff_delta,
        policy_frontier=frontier,
        policy_frontier_grid=frontier_grid,
        stability_runs=stability_runs,
        stability_summary=stability_report,
        evidence_profile=evidence_profile,
        evidence_summary=evidence_summary,
        robustness=robustness,
        scenario_metadata=scenario_metadata,
        analysis=analysis,
    )
    logger.info(
        "Case study artifact payload assembled (summary_rows=%d, scenario_rows=%d).",
        len(summary),
        len(scenario_results),
    )
    return artifacts


def write_case_study_artifacts(
    artifacts: CaseStudyArtifacts,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, CSV, and markdown fragments for the case study."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json_records(out_dir / "summary.json", artifacts.summary)
    _write_json_records(out_dir / "diagnostics.json", artifacts.diagnostics)
    _write_json_records(out_dir / "scenario_results.json", artifacts.scenario_results)
    _write_json_records(out_dir / "sensitivity.json", artifacts.sensitivity)
    _write_json_records(out_dir / "driver_analysis.json", artifacts.driver_analysis)
    _write_json_records(out_dir / "parameter_registry.json", artifacts.parameter_registry)
    _write_json_records(out_dir / "policy_eligibility.json", artifacts.policy_eligibility)
    _write_json_records(out_dir / "policy_frontier_grid.json", artifacts.policy_frontier_grid)
    _write_json_records(out_dir / "stability_runs.json", artifacts.stability_runs)
    _write_json_object(out_dir / "assumption_manifest.json", artifacts.assumption_manifest)
    _write_json_object(out_dir / "recommendation.json", artifacts.recommendation.to_dict())
    _write_json_object(out_dir / "payoff_delta_diagnostic.json", artifacts.payoff_delta)
    _write_json_object(out_dir / "policy_frontier.json", artifacts.policy_frontier)
    _write_json_object(out_dir / "stability_summary.json", artifacts.stability_summary)
    _write_json_object(out_dir / "evidence_summary.json", artifacts.evidence_summary)
    _write_json_object(out_dir / "robustness.json", artifacts.robustness)
    _write_json_object(out_dir / "metadata.json", artifacts.metadata)
    _write_csv(out_dir / "parameter_registry.csv", artifacts.parameter_registry)
    _write_plotly_html(
        out_dir / "decision_summary.html",
        create_ranked_payoff_profile(
            artifacts.summary,
            recommended_option=artifacts.recommendation.selected_option,
            comparison_option=artifacts.recommendation.comparison_option,
        ),
        div_id=DECISION_SUMMARY_DIV_ID,
    )

    sensitivity_markdown = build_sensitivity_markdown(artifacts)
    fragments = {
        "recommendation.md": build_recommendation_markdown(artifacts),
        "summary_table.md": build_summary_markdown(artifacts.summary),
        "diagnostics_table.md": build_diagnostics_markdown(artifacts.diagnostics),
        "policy_eligibility.md": build_policy_eligibility_markdown(artifacts),
        "scenario_table.md": build_scenarios_markdown(
            artifacts.scenario_results,
            artifacts.scenario_metadata,
        ),
        "payoff_delta_diagnostic.md": build_payoff_delta_markdown(artifacts),
        "policy_frontier.md": build_policy_frontier_markdown(artifacts),
        "stability.md": build_stability_markdown(artifacts),
        "driver_analysis.md": sensitivity_markdown,
        "sensitivity_table.md": sensitivity_markdown,
        "robustness.md": build_robustness_markdown(artifacts.robustness),
    }
    for filename, content in fragments.items():
        (out_dir / filename).write_text(content, encoding="utf-8")
    logger.info(
        "Artifacts complete. %d files written to %s.",
        19 + len(fragments),
        out_dir,
    )
    return fragments


def update_case_study_docs(
    case_study_path: str | Path,
    executive_summary_path: str | Path,
    fragments: Mapping[str, str],
) -> None:
    """Replace the generated sections in the checked-in markdown docs."""

    case_study_content = Path(case_study_path).read_text(encoding="utf-8")
    case_study_content = _replace_section(
        case_study_content,
        *CASE_STUDY_MARKERS["recommendation"],
        fragments["recommendation.md"],
    )
    case_study_content = _replace_section(
        case_study_content,
        *CASE_STUDY_MARKERS["base_summary"],
        fragments["summary_table.md"],
    )
    case_study_content = _replace_section(
        case_study_content,
        *CASE_STUDY_MARKERS["regret"],
        fragments["diagnostics_table.md"],
    )
    case_study_content = _replace_section(
        case_study_content,
        *CASE_STUDY_MARKERS["eligibility"],
        fragments["policy_eligibility.md"],
    )
    case_study_content = _replace_section(
        case_study_content,
        *CASE_STUDY_MARKERS["scenarios"],
        fragments["scenario_table.md"],
    )
    case_study_content = _replace_section(
        case_study_content,
        *CASE_STUDY_MARKERS["payoff_delta"],
        fragments["payoff_delta_diagnostic.md"],
    )
    case_study_content = _replace_section(
        case_study_content,
        *CASE_STUDY_MARKERS["frontier"],
        fragments["policy_frontier.md"],
    )
    case_study_content = _replace_section(
        case_study_content,
        *CASE_STUDY_MARKERS["stability"],
        fragments["stability.md"],
    )
    case_study_content = _replace_section(
        case_study_content,
        *CASE_STUDY_MARKERS["sensitivity"],
        fragments["driver_analysis.md"],
    )
    Path(case_study_path).write_text(case_study_content, encoding="utf-8")

    executive_content = Path(executive_summary_path).read_text(encoding="utf-8")
    executive_content = _replace_section(
        executive_content,
        *EXEC_SUMMARY_MARKERS["recommendation"],
        fragments["recommendation.md"],
    )
    executive_content = _replace_section(
        executive_content,
        *EXEC_SUMMARY_MARKERS["results"],
        fragments["summary_table.md"],
    )
    Path(executive_summary_path).write_text(executive_content, encoding="utf-8")


def build_recommendation_markdown(artifacts: CaseStudyArtifacts) -> str:
    """Render the recommendation block from the current policy result."""

    return "\n".join(
        recommendation_lines(artifacts.recommendation, artifacts.summary, artifacts.metadata)
    )


def recommendation_lines(
    recommendation: RecommendationResult,
    summary: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> list[str]:
    """Return stable recommendation bullets for docs and the app."""

    selected_label = labeled_option(recommendation.selected_option)
    mean_value_line = _selection_margin_line(recommendation)

    lines = [
        f"- Recommendation: **{selected_label}**.",
        f"- Policy: `{recommendation.policy_name}`.",
        f"- Why it wins: {_selection_reason_line(recommendation)}",
    ]
    if recommendation.policy_runner_up is not None:
        lines.append(
            f"- Policy runner-up: **{labeled_option(recommendation.policy_runner_up)}**."
        )
        if _should_highlight_best_excluded_alternative(recommendation):
            lines.append(
                f"- Best excluded alternative: **{labeled_option(str(recommendation.best_excluded_option))}** "
                f"stays out of policy scope because {_best_excluded_failure_detail(recommendation)}."
            )
    elif recommendation.selected_reason_type == "only_option_passing_guardrails":
        lines.append(
            f"- Best excluded alternative: **{labeled_option(str(recommendation.best_excluded_option))}** "
            f"has the strongest excluded EV case, but {_best_excluded_failure_detail(recommendation)}."
        )
    elif recommendation.selected_reason_type == "guardrails_relaxed_highest_ev":
        lines.append(
            "- Guardrail reality: no option passes both guardrails, "
            f"so **{selected_label}** wins on expected value."
        )
        if recommendation.best_excluded_option is not None:
            lines.append(
                f"- Best remaining excluded alternative: **{labeled_option(recommendation.best_excluded_option)}**."
            )
    if mean_value_line:
        lines.append(f"- Expected-value comparison: {mean_value_line}")
    lines.append(
        f"- Published run: `{metadata['n_worlds']:,}` worlds, seed `{metadata['seed']}`, "
        f"annual volume `{metadata['annual_volume']:,}`, horizon "
        f"`{metadata['time_horizon_years']}` years, discount rate "
        f"`{metadata['discount_rate_annual']:.0%}`, declared model version "
        f"`{metadata['declared_model_version']}`."
    )
    return lines


def build_summary_markdown(summary: pd.DataFrame) -> str:
    """Render the base summary table as markdown."""

    export = summary.copy()
    export["Option"] = export["option"].map(labeled_option)
    export["Expected Value"] = export["mean_value_eur"].map(format_eur_markdown)
    export["P05"] = export["p05_value_eur"].map(format_eur_markdown)
    export["Median"] = export["median_value_eur"].map(format_eur_markdown)
    export["P95"] = export["p95_value_eur"].map(format_eur_markdown)
    return _markdown_table(export[["Option", "Expected Value", "P05", "Median", "P95"]])


def build_diagnostics_markdown(diagnostics: pd.DataFrame) -> str:
    """Render the regret table as markdown."""

    export = diagnostics.copy()
    export["Option"] = export["option"].map(labeled_option)
    export["Win Rate"] = export["win_rate"].map(format_pct)
    export["Mean Regret"] = export["mean_regret_eur"].map(format_eur_markdown)
    export["P95 Regret"] = export["p95_regret_eur"].map(format_eur_markdown)
    return _markdown_table(export[["Option", "Win Rate", "Mean Regret", "P95 Regret"]])


def build_policy_eligibility_markdown(artifacts: CaseStudyArtifacts) -> str:
    """Render the guardrail-eligibility table as markdown."""

    export = artifacts.policy_eligibility.copy()
    export["Option"] = export["option"].map(labeled_option)
    export["Expected Value"] = export["mean_value_eur"].map(format_eur_markdown)
    export["P05"] = export["p05_value_eur"].map(format_eur_markdown)
    export["Downside Slack"] = export["downside_slack_eur"].map(format_eur_markdown)
    export["Mean Regret"] = export["mean_regret_eur"].map(format_eur_markdown)
    export["Regret Slack"] = export["regret_slack_eur"].map(format_eur_markdown)
    export["Eligible"] = export["eligible"].map(lambda value: "yes" if bool(value) else "no")
    export["Failure Reason"] = export["failure_reason"].map(_failure_reason_label)
    lines = [
        "- This is the policy-defining table for the current run.",
        "- An option must pass both the downside floor and the regret cap to stay eligible.",
        "",
        _markdown_table(
            export[
                [
                    "Option",
                    "Expected Value",
                    "P05",
                    "Downside Slack",
                    "Mean Regret",
                    "Regret Slack",
                    "Eligible",
                    "Failure Reason",
                ]
            ]
        ),
    ]
    return "\n".join(lines)


def build_scenarios_markdown(
    scenario_results: pd.DataFrame,
    scenario_metadata: dict[str, dict[str, str]],
) -> str:
    """Render the scenario comparison table as markdown."""

    export = scenario_results.copy()
    export["Scenario"] = export["scenario"].map(
        lambda value: scenario_metadata.get(str(value), {}).get("label", str(value))
    )
    export["Selected Option"] = export["selected_option"].map(labeled_option)
    export["Option"] = export["option"].map(labeled_option)
    export["Expected Value"] = export["mean_value_eur"].map(format_eur_markdown)
    export["P05"] = export["p05_value_eur"].map(format_eur_markdown)
    export["Mean Regret"] = export["mean_regret_eur"].map(format_eur_markdown)
    export["Eligible"] = export["eligible"].map(lambda value: "yes" if bool(value) else "no")

    lines = ["Scenario descriptions:"]
    for scenario_name, metadata in scenario_metadata.items():
        label = metadata.get("label", scenario_name)
        description = metadata.get("description", "")
        lines.append(f"- `{label}` (`{scenario_name}`): {description}")
    lines.append("")
    lines.append(
        _markdown_table(
            export[
                [
                    "Scenario",
                    "Selected Option",
                    "Option",
                    "Expected Value",
                    "P05",
                    "Mean Regret",
                    "Eligible",
                ]
            ]
        )
    )
    return "\n".join(lines)


def build_payoff_delta_markdown(artifacts: CaseStudyArtifacts) -> str:
    """Render the descriptive selected-vs-runner-up payoff diagnostic."""

    diagnostic = artifacts.payoff_delta
    unit_lookup = artifacts.parameter_registry.set_index("parameter_name")["unit"].to_dict()
    delta_mean = float(diagnostic["mean_delta_eur"])
    delta_line = (
        "selected option leads the comparison option"
        if delta_mean >= 0.0
        else "selected option trails the comparison option"
    )
    comparison_label = labeled_option(str(diagnostic["comparison_option"]))
    lines = [
        f"- Selected option: **{labeled_option(diagnostic['selected_option'])}**.",
        (
            f"- {_comparison_role_heading(str(diagnostic['comparison_option_role']))}: "
            f"**{comparison_label}**."
        ),
        f"- Mean payoff delta: {format_eur_markdown(delta_mean)} ({delta_line}).",
        f"- P05 payoff delta: {format_eur_markdown(float(diagnostic['p05_delta_eur']))}.",
        f"- Win rate vs comparison: {format_pct(float(diagnostic['win_rate_vs_comparison']))}.",
        (
            "- This section is descriptive. It ranks parameters by association "
            "with the selected-minus-comparison payoff delta inside the sampled worlds."
        ),
    ]
    rows = diagnostic["delta_rows"]
    if not rows:
        lines.append("- No parameter cleared the materiality threshold for this delta diagnostic.")
        return "\n".join(lines)

    table = pd.DataFrame(rows)
    table["Parameter"] = table["parameter"]
    table["Unit"] = table["parameter"].map(lambda value: unit_lookup.get(str(value), ""))
    table["Delta rho"] = table["delta_spearman_corr"].map(lambda value: f"{value:+.2f}")
    table["Sampled range"] = table.apply(
        lambda row: f"{row['sampled_min_value']:,.3f} to {row['sampled_max_value']:,.3f}",
        axis=1,
    )
    table["Interpretation"] = table["interpretation_note"]
    lines.append("")
    lines.append(
        _markdown_table(
            table[["Parameter", "Unit", "Delta rho", "Sampled range", "Interpretation"]]
        )
    )
    return "\n".join(lines)


def build_policy_frontier_markdown(artifacts: CaseStudyArtifacts) -> str:
    """Render the full-option frontier plus the runner-up comparison view."""

    frontier = artifacts.policy_frontier
    sweep = artifacts.policy_frontier_grid
    export = pd.DataFrame(frontier["frontier_rows"]).copy()
    export["Threshold"] = export["threshold_label"]
    export["Current value"] = export["current_value"].map(format_threshold_eur_markdown)
    export["Raw switching value"] = export["switching_value"].map(
        lambda value: "not observed" if pd.isna(value) else format_number(float(value))
    )
    export["Display switching value"] = export["switching_value"].map(
        lambda value: "not observed"
        if pd.isna(value)
        else format_threshold_eur_markdown(float(value))
    )
    export["First switching option"] = export["first_switching_option"].map(
        lambda value: "not observed" if pd.isna(value) else labeled_option(str(value))
    )
    export["Switch type"] = export["switch_type"].map(
        lambda value: FULL_FRONTIER_SWITCH_LABELS.get(str(value), str(value))
    )
    export["Direction"] = export["switch_direction"].map(
        lambda value: "not observed" if pd.isna(value) else str(value).replace("_", " ")
    )
    export["Interpretation"] = export["interpretation_note"]

    comparison_export = pd.DataFrame(frontier["secondary_comparison_rows"]).copy()
    comparison_export["Threshold"] = comparison_export["threshold_label"]
    comparison_export["Current value"] = comparison_export["current_value"].map(
        format_threshold_eur_markdown
    )
    comparison_export["Comparison threshold"] = comparison_export["switching_value"].map(
        lambda value: "not needed"
        if pd.isna(value)
        else format_threshold_eur_markdown(float(value))
    )
    comparison_export["Status"] = comparison_export["status"].map(
        lambda value: RUNNER_UP_FRONTIER_STATUS_LABELS.get(str(value), str(value))
    )
    comparison_export["Interpretation"] = comparison_export["interpretation_note"]

    sweep_summary = (
        sweep.groupby(["threshold_name", "threshold_label"], observed=True)
        .agg(
            min_tested_value=("tested_value", "min"),
            max_tested_value=("tested_value", "max"),
            switch_count=("switch_from_baseline", "sum"),
            switching_options=(
                "selected_option",
                lambda series: ", ".join(
                    sorted(
                        {
                            labeled_option(str(value))
                            for value in series
                            if pd.notna(value) and str(value) != frontier["selected_option"]
                        }
                    )
                ),
            ),
        )
        .reset_index()
    )
    sweep_summary["Threshold"] = sweep_summary["threshold_label"]
    sweep_summary["Tested range"] = sweep_summary.apply(
        lambda row: (
            f"{format_threshold_eur_markdown(float(row['min_tested_value']))} to "
            f"{format_threshold_eur_markdown(float(row['max_tested_value']))}"
        ),
        axis=1,
    )
    sweep_summary["Selection switched?"] = sweep_summary["switch_count"].map(
        lambda value: "yes" if int(value) > 0 else "no"
    )
    sweep_summary["Switching option(s)"] = sweep_summary["switching_options"].replace("", "none")

    lines = [
        (
            "- The first table is the full-option frontier. It re-runs the whole policy and "
            "records the first threshold change that flips the recommendation."
        ),
        "",
        _markdown_table(
            export[
                [
                    "Threshold",
                    "Current value",
                    "Raw switching value",
                    "Display switching value",
                    "First switching option",
                    "Switch type",
                    "Direction",
                    "Interpretation",
                ]
            ]
        ),
    ]
    if not comparison_export.empty:
        lines.extend(
            [
                (
                    "- The second table is secondary context. It follows the main comparison "
                    "option, which can be the policy runner-up or the best excluded alternative "
                    "depending on the branch."
                ),
                "",
                _markdown_table(
                    comparison_export[
                        [
                            "Threshold",
                            "Current value",
                            "Comparison threshold",
                            "Status",
                            "Interpretation",
                        ]
                    ]
                ),
            ]
        )
    lines.extend(
        [
            "",
            _markdown_table(
                sweep_summary[
                    ["Threshold", "Tested range", "Selection switched?", "Switching option(s)"]
                ]
            ),
        ]
    )
    return "\n".join(lines)


def build_stability_markdown(artifacts: CaseStudyArtifacts) -> str:
    """Render the Monte Carlo stability summary as markdown."""

    summary = artifacts.stability_summary
    lines = [
        (
            f"- Stability runs: `{summary['run_count']}` published-case reruns "
            "across multiple seeds and world counts."
        ),
        (
            "- Selected-option P05 range: "
            f"{format_eur_markdown(float(summary['selected_p05_range_eur']))}."
        ),
    ]
    if summary["comparison_p05_range_eur"] is not None:
        lines.append(
            "- Comparison-option P05 range: "
            f"{format_eur_markdown(float(summary['comparison_p05_range_eur']))}."
        )
    recommendation_frequency = pd.DataFrame(summary["recommendation_frequency"])
    ev_leader_frequency = pd.DataFrame(summary["ev_leader_frequency"])

    if not recommendation_frequency.empty:
        recommendation_frequency["Option"] = recommendation_frequency["selected_option"].map(
            labeled_option
        )
        recommendation_frequency["Share"] = recommendation_frequency["count"].map(
            lambda value: format_pct(float(value) / float(summary["run_count"]))
        )
        lines.extend(["", "Recommendation frequency:", ""])
        lines.append(
            _markdown_table(
                recommendation_frequency[["Option", "count", "Share"]].rename(
                    columns={"count": "Runs"}
                )
            )
        )

    if not ev_leader_frequency.empty:
        ev_leader_frequency["EV leader"] = ev_leader_frequency["ev_leader"].map(labeled_option)
        ev_leader_frequency["Share"] = ev_leader_frequency["count"].map(
            lambda value: format_pct(float(value) / float(summary["run_count"]))
        )
        lines.extend(["", "EV leader frequency:", ""])
        lines.append(
            _markdown_table(
                ev_leader_frequency[["EV leader", "count", "Share"]].rename(
                    columns={"count": "Runs"}
                )
            )
        )
    return "\n".join(lines)


def build_sensitivity_markdown(artifacts: CaseStudyArtifacts) -> str:
    """Render the decision-support driver view and keep Spearman secondary."""

    threshold = artifacts.analysis.sensitivity_materiality_threshold_abs_spearman
    limit = artifacts.analysis.sensitivity_max_rows_per_option
    sections: list[str] = [
        (
            "This section is the decision-support view. It uses partial rank correlation "
            "with bootstrap intervals. The descriptive Spearman output still exists in "
            "`artifacts/case_study/sensitivity.json` for quick inspection."
        ),
        "",
    ]
    for option in sorted(artifacts.driver_analysis["option"].unique()):
        rows = material_driver_rows(
            driver_analysis=artifacts.driver_analysis,
            option=str(option),
            threshold=threshold,
            limit=limit,
        )
        sections.append(f"### {labeled_option(str(option))}")
        if rows.empty:
            sections.append(
                "No decision-support driver cleared the current materiality threshold of "
                f"|partial rho| >= {threshold:.2f}."
            )
            sections.append("")
            continue
        rows["Parameter"] = rows["parameter"]
        rows["Partial rank corr"] = rows["partial_rank_corr"].map(lambda value: f"{value:+.2f}")
        rows["95% CI"] = rows.apply(
            lambda row: f"{float(row['ci_low']):+.2f} to {float(row['ci_high']):+.2f}",
            axis=1,
        )
        sections.append(_markdown_table(rows[["Parameter", "Partial rank corr", "95% CI"]]))
        sections.append("")
    return "\n".join(sections).strip()


def _selection_reason_line(recommendation: RecommendationResult) -> str:
    """Return one plain-English line for the selected policy branch."""

    selected = labeled_option(recommendation.selected_option)
    if recommendation.selected_reason_type == "only_option_passing_guardrails":
        return f"{selected} is the only option that clears both guardrails."
    if recommendation.selected_reason_type == "guardrails_relaxed_highest_ev":
        return "No option clears both guardrails, so the policy falls back to expected value."
    if recommendation.selected_reason_type == "ev_tolerance_override":
        return f"{selected} stays inside the EV tolerance band and wins the regret tie-break."
    return f"{selected} clears the guardrails and leads expected value inside the eligible set."


def _best_excluded_failure_detail(recommendation: RecommendationResult) -> str:
    """Describe why the best excluded alternative did not survive the policy."""

    reason = recommendation.best_excluded_failure_reason
    if reason == "misses_downside_floor":
        return (
            "it misses the downside floor by about "
            f"{format_eur_markdown(abs(float(recommendation.best_excluded_downside_slack_eur or 0.0)))}"
        )
    if reason == "misses_regret_cap":
        return (
            "it misses the regret cap by about "
            f"{format_eur_markdown(abs(float(recommendation.best_excluded_regret_slack_eur or 0.0)))}"
        )
    if reason == "misses_downside_floor_and_regret_cap":
        return (
            "it misses the downside floor by about "
            f"{format_eur_markdown(abs(float(recommendation.best_excluded_downside_slack_eur or 0.0)))} "
            "and the regret cap by about "
            f"{format_eur_markdown(abs(float(recommendation.best_excluded_regret_slack_eur or 0.0)))}"
        )
    return "it stays behind after the policy tie-break"


def _selection_margin_line(recommendation: RecommendationResult) -> str | None:
    """Describe whether the selected option leads or trails on expected value."""

    margin = recommendation.comparison_margin_eur
    comparison_option = recommendation.comparison_option
    if margin is None or comparison_option is None:
        return None
    amount = format_eur_markdown(abs(margin))
    comparison = labeled_option(comparison_option)
    if margin < 0.0:
        return f"the selected option trails **{comparison}** by {amount}."
    if margin > 0.0:
        return f"the selected option leads **{comparison}** by {amount}."
    return f"the selected option is tied with **{comparison}** on expected value."


def _should_highlight_best_excluded_alternative(recommendation: RecommendationResult) -> bool:
    """Return whether the best excluded alternative adds useful policy context."""

    if (
        recommendation.best_excluded_option is None
        or recommendation.best_excluded_mean_value_eur is None
    ):
        return False
    return recommendation.best_excluded_mean_value_eur >= recommendation.selected_mean_value_eur


def _comparison_role_heading(role: str) -> str:
    """Return a readable heading for the comparison option."""

    if role == "policy_runner_up":
        return "Policy runner-up"
    if role == "best_excluded_option":
        return "Best excluded alternative"
    return "Comparison option"


def _failure_reason_label(value: object) -> str:
    """Return a public-facing failure label from an internal failure code."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "passes both guardrails"
    return FAILURE_REASON_LABELS.get(str(value), str(value))


def _replace_section(content: str, start_marker: str, end_marker: str, replacement: str) -> str:
    """Replace one generated markdown section."""

    if start_marker not in content or end_marker not in content:
        raise ValueError(f"Missing generated section markers: {start_marker} / {end_marker}")
    start_index = content.index(start_marker) + len(start_marker)
    end_index = content.index(end_marker)
    return content[:start_index] + "\n" + replacement.strip() + "\n" + content[end_index:]


def _write_json_records(path: Path, frame: pd.DataFrame) -> None:
    """Write a dataframe to JSON records with stable float rounding."""

    records = [_stable_json_value(record) for record in frame.to_dict(orient="records")]
    _write_json_object(path, records)


def _write_json_object(path: Path, payload: Any) -> None:
    """Write a JSON payload with stable ordering."""

    stable_payload = _stable_json_value(payload)
    path.write_text(
        json.dumps(stable_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a dataframe to CSV with stable quoting."""

    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _write_plotly_html(path: Path, figure: go.Figure, *, div_id: str) -> None:
    """Write one Plotly HTML artifact with stable IDs and rounded numeric payloads."""

    pio.write_html(
        _stable_plotly_figure(figure),
        file=path,
        include_plotlyjs="cdn",
        div_id=div_id,
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a simple markdown table without optional dependencies."""

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


def _stable_json_value(payload: Any) -> Any:
    """Round floats recursively so generated artifacts diff cleanly."""

    if payload is None:
        return None
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, Integral):
        return int(payload)
    if isinstance(payload, Real):
        number = float(payload)
        if pd.isna(number):
            return None
        return round(number, 6)
    if isinstance(payload, (date, datetime)):
        return payload.isoformat()
    if isinstance(payload, dict):
        return {key: _stable_json_value(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_stable_json_value(value) for value in payload]
    return payload


def _stable_plotly_figure(figure: go.Figure) -> go.Figure:
    """Return a figure copy with rounded numeric payloads for deterministic HTML export."""

    stable_payload = _stable_json_value(figure.to_plotly_json())
    return go.Figure(stable_payload)


@lru_cache(maxsize=4)
def _cached_stability_runs(
    config_path: str,
    config_sha256: str,
    parameter_registry_sha256: str,
    assumption_registry_sha256: str,
    seeds: tuple[int, ...],
    world_counts: tuple[int, ...],
) -> pd.DataFrame:
    """Cache the stability sweep by artifact-defining content, not just by path."""

    del config_sha256, parameter_registry_sha256, assumption_registry_sha256
    return stability_analysis(config_path, seeds=seeds, world_counts=world_counts)
