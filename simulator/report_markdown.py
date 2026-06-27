"""Markdown section builders for the published case-study artifacts.

These functions turn the assembled :class:`CaseStudyArtifacts` payload into the
markdown fragments spliced into CASE_STUDY.md / EXECUTIVE_SUMMARY.md. They are
pure rendering: no simulation, IO, or file writing happens here (that lives in
``simulator.reporting``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from simulator.config import AnalysisConfig
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
)
from simulator.serialization import markdown_table


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
    value_of_information: dict[str, Any]


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
    return markdown_table(export[["Option", "Expected Value", "P05", "Median", "P95"]])


def build_diagnostics_markdown(diagnostics: pd.DataFrame) -> str:
    """Render the regret table as markdown."""

    export = diagnostics.copy()
    export["Option"] = export["option"].map(labeled_option)
    export["Win Rate"] = export["win_rate"].map(format_pct)
    export["Mean Regret"] = export["mean_regret_eur"].map(format_eur_markdown)
    export["P95 Regret"] = export["p95_regret_eur"].map(format_eur_markdown)
    return markdown_table(export[["Option", "Win Rate", "Mean Regret", "P95 Regret"]])


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
        markdown_table(
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
        markdown_table(
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
        markdown_table(
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
        markdown_table(
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
                markdown_table(
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
            markdown_table(
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
            markdown_table(
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
            markdown_table(
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
        sections.append(markdown_table(rows[["Parameter", "Partial rank corr", "95% CI"]]))
        sections.append("")
    return "\n".join(sections).strip()


VALUE_OF_INFORMATION_MAX_ROWS = 8


VALUE_OF_INFORMATION_MIN_EUR = 1.0


def build_value_of_information_markdown(artifacts: CaseStudyArtifacts) -> str:
    """Render the value-of-information view (EVPI and per-parameter EVPPI)."""

    voi = artifacts.value_of_information
    registry = artifacts.parameter_registry.set_index("parameter_name")
    unit_lookup = registry["unit"].to_dict()
    meaning_lookup = registry["business_meaning"].to_dict()

    evpi = float(voi["evpi_eur"])
    ev_optimal = voi["ev_optimal_option"]
    ev_optimal_label = labeled_option(str(ev_optimal)) if ev_optimal is not None else "n/a"
    lines = [
        (
            "- This view answers which uncertainty is worth resolving before deciding. "
            "It is computed against the expected-value-optimal action, not the policy pick."
        ),
        f"- Expected-value-optimal action under full uncertainty: **{ev_optimal_label}**.",
        f"- EVPI (value of resolving every uncertainty before deciding): {format_eur_markdown(evpi)}.",
        (
            "- EVPPI below is the value of resolving one parameter on its own. Each EVPPI is "
            "bounded by the EVPI, and the values are not additive across parameters."
        ),
    ]

    material_rows = [
        row
        for row in voi["evppi_rows"]
        if float(row["evppi_eur"]) >= VALUE_OF_INFORMATION_MIN_EUR
    ]
    if not material_rows:
        lines.append("- No single parameter carries material decision-relevant information.")
        return "\n".join(lines)

    table = pd.DataFrame(material_rows[:VALUE_OF_INFORMATION_MAX_ROWS])
    table["Parameter"] = table["parameter"]
    table["Unit"] = table["parameter"].map(lambda value: unit_lookup.get(str(value), ""))
    table["EVPPI"] = table["evppi_eur"].map(lambda value: format_eur_markdown(float(value)))
    table["Share of EVPI"] = table["share_of_evpi"].map(lambda value: format_pct(float(value)))
    table["What it measures"] = table["parameter"].map(
        lambda value: meaning_lookup.get(str(value), "")
    )
    lines.append("")
    lines.append(
        markdown_table(
            table[["Parameter", "Unit", "EVPPI", "Share of EVPI", "What it measures"]]
        )
    )
    return "\n".join(lines)


def _published_driver_analysis_rows(
    driver_analysis: pd.DataFrame,
    *,
    threshold: float,
    limit: int,
) -> pd.DataFrame:
    """Return the stable decision-support rows that belong in the published JSON artifact."""

    if driver_analysis.empty:
        return driver_analysis.copy()

    published_rows: list[pd.DataFrame] = []
    for option in sorted(driver_analysis["option"].astype(str).unique()):
        option_rows = material_driver_rows(
            driver_analysis=driver_analysis,
            option=option,
            threshold=threshold,
            limit=limit,
        )
        if option_rows.empty:
            continue
        published_rows.append(option_rows)

    if not published_rows:
        return driver_analysis.iloc[0:0].copy()

    return (
        pd.concat(published_rows, ignore_index=True)
        .assign(abs_partial_rank_corr=lambda frame: frame["partial_rank_corr"].abs())
        .sort_values(
            ["option", "abs_partial_rank_corr", "parameter"],
            ascending=[True, False, True],
        )
        .drop(columns=["abs_partial_rank_corr"])
        .reset_index(drop=True)
    )


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
