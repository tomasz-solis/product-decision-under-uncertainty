"""Artifact generation and markdown helpers for the synthetic case study."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping
from functools import lru_cache
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
    value_of_information,
)
from simulator.artifact_freshness import build_artifact_metadata, sha256_file
from simulator.config import (
    get_analysis_settings,
    get_decision_policy,
    get_scenario_metadata,
    get_seed,
    get_simulation_settings,
    load_config,
    validate_config,
)
from simulator.evidence import profile_public_evidence, summarize_public_evidence
from simulator.policy import (
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
from simulator.report_markdown import (
    CaseStudyArtifacts,
    _published_driver_analysis_rows,
    build_diagnostics_markdown,
    build_payoff_delta_markdown,
    build_policy_eligibility_markdown,
    build_policy_frontier_markdown,
    build_recommendation_markdown,
    build_scenarios_markdown,
    build_sensitivity_markdown,
    build_stability_markdown,
    build_summary_markdown,
    build_value_of_information_markdown,
)
from simulator.robustness import build_robustness_markdown, build_robustness_report
from simulator.serialization import stable_json_value
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
    "value_of_information": (
        "<!-- GENERATED:CASE_STUDY_VALUE_OF_INFORMATION:START -->",
        "<!-- GENERATED:CASE_STUDY_VALUE_OF_INFORMATION:END -->",
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
    value_of_information_result = value_of_information(base_results)

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
        value_of_information=value_of_information_result,
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
    published_driver_analysis = _published_driver_analysis_rows(
        artifacts.driver_analysis,
        threshold=artifacts.analysis.sensitivity_materiality_threshold_abs_spearman,
        limit=artifacts.analysis.sensitivity_max_rows_per_option,
    )

    _write_json_records(out_dir / "summary.json", artifacts.summary)
    _write_json_records(out_dir / "diagnostics.json", artifacts.diagnostics)
    _write_json_records(out_dir / "scenario_results.json", artifacts.scenario_results)
    _write_json_records(out_dir / "sensitivity.json", artifacts.sensitivity)
    _write_json_records(out_dir / "driver_analysis.json", published_driver_analysis)
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
    _write_json_object(out_dir / "value_of_information.json", artifacts.value_of_information)
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
        "value_of_information.md": build_value_of_information_markdown(artifacts),
    }
    for filename, content in fragments.items():
        (out_dir / filename).write_text(content, encoding="utf-8")
    logger.info(
        "Artifacts complete. %d files written to %s.",
        20 + len(fragments),
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
    case_study_content = _replace_section(
        case_study_content,
        *CASE_STUDY_MARKERS["value_of_information"],
        fragments["value_of_information.md"],
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


def _replace_section(content: str, start_marker: str, end_marker: str, replacement: str) -> str:
    """Replace one generated markdown section."""

    if start_marker not in content or end_marker not in content:
        raise ValueError(f"Missing generated section markers: {start_marker} / {end_marker}")
    start_index = content.index(start_marker) + len(start_marker)
    end_index = content.index(end_marker)
    return content[:start_index] + "\n" + replacement.strip() + "\n" + content[end_index:]


def _write_json_records(path: Path, frame: pd.DataFrame) -> None:
    """Write a dataframe to JSON records with stable float rounding."""

    records = [stable_json_value(record) for record in frame.to_dict(orient="records")]
    _write_json_object(path, records)


def _write_json_object(path: Path, payload: Any) -> None:
    """Write a JSON payload with stable ordering."""

    stable_payload = stable_json_value(payload)
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


def _stable_plotly_figure(figure: go.Figure) -> go.Figure:
    """Return a figure copy with rounded numeric payloads for deterministic HTML export."""

    stable_payload = stable_json_value(figure.to_plotly_json())
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
