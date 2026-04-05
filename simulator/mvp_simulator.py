"""Compatibility wrapper around the refactored simulation modules."""

from __future__ import annotations

from pathlib import Path

from simulator.analytics import decision_diagnostics, sensitivity_analysis, summarize_results
from simulator.output_utils import material_sensitivity_rows, sensitivity_note
from simulator.reporting import build_case_study_artifacts, recommendation_lines
from simulator.simulation import (
    OPTION_COLUMNS,
    OPTION_LABELS,
    run_all_scenarios,
    run_simulation,
    sample_params,
    simulate_option_do_nothing,
    simulate_option_feature_extension,
    simulate_option_new_capability,
    simulate_option_stabilize_core,
)

__all__ = [
    "OPTION_COLUMNS",
    "OPTION_LABELS",
    "decision_diagnostics",
    "run_all_scenarios",
    "run_simulation",
    "sample_params",
    "sensitivity_analysis",
    "simulate_option_do_nothing",
    "simulate_option_feature_extension",
    "simulate_option_new_capability",
    "simulate_option_stabilize_core",
    "summarize_results",
]


def main(config_path: str | Path = "simulator/config.yaml") -> None:
    """Print a compact summary for the checked-in scenario."""

    artifacts = build_case_study_artifacts(config_path)
    threshold = artifacts.analysis.sensitivity_materiality_threshold_abs_spearman
    limit = artifacts.analysis.sensitivity_max_rows_per_option
    selected_option = artifacts.recommendation.selected_option
    material_rows = material_sensitivity_rows(
        artifacts.sensitivity,
        selected_option,
        threshold=threshold,
        limit=limit,
    )

    print("Platform investment case study")
    print()
    print("Recommendation")
    for line in recommendation_lines(
        artifacts.recommendation, artifacts.summary, artifacts.metadata
    ):
        print(line)
    print()
    print("Guardrail eligibility")
    print(artifacts.policy_eligibility.to_string(index=False))
    print()
    print("Summary")
    print(artifacts.summary.to_string(index=False))
    print()
    print("Diagnostics")
    print(artifacts.diagnostics.to_string(index=False))
    print()
    print("Selected-vs-runner-up payoff diagnostic")
    payoff_delta_rows = artifacts.payoff_delta["delta_rows"]
    if payoff_delta_rows:
        print(
            f"Mean payoff delta: {artifacts.payoff_delta['mean_delta_eur']:.0f} EUR, "
            f"P05 payoff delta: {artifacts.payoff_delta['p05_delta_eur']:.0f} EUR"
        )
        print(payoff_delta_rows)
    else:
        print("No parameter cleared the payoff-diagnostic materiality threshold.")
    print()
    print("Policy frontier")
    print(artifacts.policy_frontier["frontier_rows"])
    print()
    print("Scenario comparison")
    print(artifacts.scenario_results.to_string(index=False))
    print()
    print("Published-case stability")
    print(artifacts.stability_summary)
    print()
    print(f"Material sensitivity for {selected_option}")
    if material_rows.empty:
        print(
            sensitivity_note(artifacts.sensitivity, selected_option, threshold)
            or "No material rows."
        )
    else:
        print(material_rows.to_string(index=False))
        note = sensitivity_note(artifacts.sensitivity, selected_option, threshold)
        if note:
            print(note)


if __name__ == "__main__":
    main()
