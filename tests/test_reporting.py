"""Tests for reporting helpers and generated markdown fragments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pandas as pd

from simulator.output_utils import material_driver_rows
from simulator.policy import PolicyFrontierRow
from simulator.reporting import (
    build_case_study_artifacts,
    build_payoff_delta_markdown,
    build_policy_frontier_markdown,
    build_recommendation_markdown,
    build_sensitivity_markdown,
    build_stability_markdown,
    write_case_study_artifacts,
)

CONFIG_PATH = "simulator/config.yaml"


def test_write_case_study_artifacts_emits_traceable_outputs(tmp_path: Path) -> None:
    """Artifact generation should write the expected JSON, CSV, and markdown files."""

    artifacts = build_case_study_artifacts(CONFIG_PATH, n_worlds=1200, seed=42)
    write_case_study_artifacts(artifacts, tmp_path)

    expected_files = {
        "summary.json",
        "diagnostics.json",
        "scenario_results.json",
        "sensitivity.json",
        "parameter_registry.json",
        "parameter_registry.csv",
        "policy_eligibility.json",
        "assumption_manifest.json",
        "recommendation.json",
        "payoff_delta_diagnostic.json",
        "policy_frontier.json",
        "policy_frontier_grid.json",
        "stability_runs.json",
        "stability_summary.json",
        "evidence_summary.json",
        "robustness.json",
        "metadata.json",
        "recommendation.md",
        "summary_table.md",
        "diagnostics_table.md",
        "policy_eligibility.md",
        "scenario_table.md",
        "payoff_delta_diagnostic.md",
        "policy_frontier.md",
        "stability.md",
        "sensitivity_table.md",
        "robustness.md",
    }

    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})


def test_sensitivity_markdown_suppresses_noise_for_do_nothing() -> None:
    """The published sensitivity fragment should not pretend that noise is insight."""

    artifacts = build_case_study_artifacts(CONFIG_PATH, n_worlds=1500, seed=42)
    markdown = build_sensitivity_markdown(artifacts)
    do_nothing_section = _section_for_option(markdown, "Do Nothing")

    assert "do_nothing_drift_cost_eur" in do_nothing_section
    assert "Partial rank corr" in do_nothing_section
    assert "baseline_failure_rate" not in do_nothing_section
    assert "stabilize_core_upfront_cost_eur" not in do_nothing_section


def test_material_driver_rows_drop_intervals_that_cross_zero() -> None:
    """Borderline driver rows should not survive when the interval still includes zero."""

    driver_analysis = pd.DataFrame(
        [
            {
                "option": "do_nothing",
                "parameter": "do_nothing_drift_cost_eur",
                "partial_rank_corr": 1.0,
                "ci_low": 1.0,
                "ci_high": 1.0,
            },
            {
                "option": "do_nothing",
                "parameter": "stabilize_core_upfront_cost_eur",
                "partial_rank_corr": 0.11,
                "ci_low": -0.04,
                "ci_high": 0.05,
            },
            {
                "option": "do_nothing",
                "parameter": "feature_extension_regression_prob_multiplier",
                "partial_rank_corr": 0.10,
                "ci_low": 0.02,
                "ci_high": 0.16,
            },
        ]
    )

    rows = material_driver_rows(
        driver_analysis=driver_analysis,
        option="do_nothing",
        threshold=0.10,
        limit=3,
    )

    assert list(rows["parameter"]) == [
        "do_nothing_drift_cost_eur",
        "feature_extension_regression_prob_multiplier",
    ]


def test_payoff_delta_markdown_stays_descriptive_and_drops_fake_thresholds() -> None:
    """The payoff-diagnostic markdown should not pretend it is a policy threshold view."""

    artifacts = build_case_study_artifacts(CONFIG_PATH, n_worlds=1500, seed=42)
    markdown = build_payoff_delta_markdown(artifacts)

    assert "This section is descriptive" in markdown
    assert "Observed flip region" not in markdown
    assert "0.289982" not in markdown


def test_policy_frontier_markdown_reports_actual_switching_thresholds() -> None:
    """The reporting layer should publish threshold changes that can flip selection."""

    artifacts = build_case_study_artifacts(CONFIG_PATH, n_worlds=1500, seed=42)
    markdown = build_policy_frontier_markdown(artifacts)

    assert "full-option frontier" in markdown
    assert "Display switching value" in markdown
    assert "First switching option" in markdown
    assert "Switching option(s)" in markdown


def test_policy_frontier_markdown_keeps_small_currency_thresholds_visible() -> None:
    """Small positive thresholds should not be rounded down to zero in the frontier output."""

    artifacts = build_case_study_artifacts(CONFIG_PATH, n_worlds=1500, seed=42)
    row = cast(
        PolicyFrontierRow,
        {
        "threshold_name": "ev_tolerance_eur",
        "threshold_label": "EV tolerance",
        "unit": "eur",
        "current_value": 100.0,
        "switching_value": 300.0,
        "first_switching_option": "stabilize_core",
        "switch_type": "exact_match",
        "switch_direction": "more_permissive",
        "all_options_considered": True,
        "interpretation_note": "Example row.",
        },
    )
    artifacts.policy_frontier["frontier_rows"][0] = row
    markdown = build_policy_frontier_markdown(artifacts)

    assert "€300" in markdown
    assert "€0" not in markdown


def test_recommendation_markdown_explains_guardrail_fallback_when_none_survive() -> None:
    """The recommendation copy should say when the policy falls back to expected value."""

    artifacts = build_case_study_artifacts(CONFIG_PATH, n_worlds=1500, seed=42)
    markdown = build_recommendation_markdown(artifacts)

    assert "No option clears both guardrails" in markdown
    assert "Best remaining excluded alternative" in markdown
    assert "Expected-value comparison" in markdown


def test_stability_markdown_reports_frequency_tables() -> None:
    """The stability fragment should summarize rerun frequency cleanly."""

    artifacts = build_case_study_artifacts(CONFIG_PATH, n_worlds=1500, seed=42)
    markdown = build_stability_markdown(artifacts)

    assert "Recommendation frequency" in markdown
    assert "EV leader frequency" in markdown


def test_formula_appendix_covers_the_key_model_terms() -> None:
    """The docs should link to an equation view that uses the current parameter names."""

    case_study = Path("CASE_STUDY.md").read_text(encoding="utf-8")
    formulas = Path("simulator/formulas.md").read_text(encoding="utf-8")

    assert "simulator/formulas.md" in case_study
    for term in [
        "value_per_success_eur",
        "regression_event_prob",
        "delay_o",
        "Value_stabilize",
        "Value_extension",
    ]:
        assert term in formulas


def _section_for_option(markdown: str, heading: str) -> str:
    """Extract one option section from the sensitivity markdown fragment."""

    pattern = rf"### {re.escape(heading)}\n(.*?)(?:\n### |\Z)"
    match = re.search(pattern, markdown, re.DOTALL)
    if not match:
        raise AssertionError(f"Could not find section for {heading!r}.")
    return match.group(1)
