"""Streamlit app for exploring the Product Decision Under Uncertainty case study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from simulator.analytics import decision_diagnostics, sensitivity_analysis, summarize_results
from simulator.artifact_freshness import build_artifact_metadata, compare_artifact_metadata
from simulator.config import (
    apply_scenario,
    get_analysis_settings,
    get_decision_policy,
    get_declared_model_version,
    get_scenario_metadata,
    get_seed,
    get_simulation_settings,
    load_config,
)
from simulator.output_utils import build_run_context
from simulator.policy import (
    PayoffDeltaDiagnostic,
    PolicyFrontierResult,
    RecommendationResult,
    build_policy_eligibility_table,
    payoff_delta_diagnostic,
    policy_frontier_analysis,
    policy_frontier_grid,
    select_recommendation,
)
from simulator.presentation import (
    diagnostics_display_table,
    eligibility_display_table,
    frontier_display_table,
    payoff_delta_display_table,
    runner_up_frontier_display_table,
    scenario_display_table,
    sensitivity_summary_note,
    stability_frequency_table,
    summary_display_table,
    top_sensitivity_rows,
)
from simulator.project_paths import (
    ASSUMPTION_REGISTRY_PATH,
    CASE_STUDY_ARTIFACTS_DIR,
    CONFIG_PATH,
    GENERATOR_SCRIPT_PATH,
    PARAMETER_REGISTRY_PATH,
)
from simulator.provenance import (
    build_assumption_manifest,
    load_assumption_registry,
    load_parameter_registry,
    validate_parameter_registry,
)
from simulator.reporting import build_case_study_artifacts, recommendation_lines
from simulator.simulation import OPTION_LABELS, run_all_scenarios, run_simulation
from simulator.visualizations import (
    create_decision_dashboard,
    create_regret_comparison,
    create_risk_profile_chart,
    create_scenario_comparison,
    create_sensitivity_waterfall,
    create_trade_off_matrix,
)


@dataclass(frozen=True)
class AppOutputs:
    """Typed container for the app's cached simulation outputs."""

    simulation_settings: dict[str, Any]
    results: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    sensitivity: pd.DataFrame
    scenario_results: pd.DataFrame
    recommendation: RecommendationResult
    policy_eligibility: pd.DataFrame
    payoff_delta: PayoffDeltaDiagnostic
    policy_frontier: PolicyFrontierResult
    policy_frontier_grid: pd.DataFrame


@dataclass(frozen=True)
class PublishedGovernance:
    """Published governance and provenance payload used in the app sidebar and panels."""

    metadata: dict[str, Any]
    stability_summary: dict[str, Any]
    evidence_summary: dict[str, Any]
    manifest_counts: dict[str, int]
    freshness_status: str
    freshness_message: str
    stale_fields: tuple[str, ...]
    evidence_note_path: str
    frontier_semantics: str


def published_case_caption() -> str:
    """Return the caption that distinguishes the published run from reruns."""

    return (
        "The checked-in markdown artifacts come from the default scenario and the "
        "default run settings."
    )


def governance_warning_message(governance: PublishedGovernance) -> str | None:
    """Return the governance freshness banner text, if the app should show one."""

    if governance.freshness_status in {"stale", "unknown"}:
        return governance.freshness_message
    return None


@st.cache_data(show_spinner=False)
def compute_outputs(n_worlds: int, seed: int, scenario: str) -> AppOutputs:
    """Run the simulator and cache the main analytical outputs for the app."""

    cfg = load_config(CONFIG_PATH)
    analysis = get_analysis_settings(cfg)
    policy = get_decision_policy(cfg)
    scenario_cfg = apply_scenario(cfg, scenario)
    simulation_settings = get_simulation_settings(scenario_cfg)
    simulation_settings["n_worlds"] = int(n_worlds)

    results = run_simulation(CONFIG_PATH, n_worlds=n_worlds, seed=seed, scenario=scenario)
    summary = summarize_results(results)
    diagnostics = decision_diagnostics(results)
    sensitivity = sensitivity_analysis(results)
    scenario_results = run_all_scenarios(CONFIG_PATH, n_worlds=n_worlds, seed=seed)
    recommendation = select_recommendation(summary, diagnostics, policy)
    policy_eligibility = build_policy_eligibility_table(summary, diagnostics, policy)
    payoff_delta = payoff_delta_diagnostic(results, recommendation, analysis)
    frontier = policy_frontier_analysis(summary, diagnostics, policy, recommendation)
    frontier_grid = policy_frontier_grid(summary, diagnostics, policy, recommendation)
    return AppOutputs(
        simulation_settings=simulation_settings,
        results=results,
        summary=summary,
        diagnostics=diagnostics,
        sensitivity=sensitivity,
        scenario_results=scenario_results,
        recommendation=recommendation,
        policy_eligibility=policy_eligibility,
        payoff_delta=payoff_delta,
        policy_frontier=frontier,
        policy_frontier_grid=frontier_grid,
    )


@st.cache_data(show_spinner=False)
def load_published_governance() -> PublishedGovernance:
    """Load the published governance summaries plus live manifest counts."""

    cfg = load_config(CONFIG_PATH)
    parameter_registry = validate_parameter_registry(
        load_parameter_registry(PARAMETER_REGISTRY_PATH),
        cfg["params"],
    )
    manifest = build_assumption_manifest(
        cfg,
        parameter_registry,
        load_assumption_registry(ASSUMPTION_REGISTRY_PATH),
    )
    manifest_counts = {
        "params": len(manifest["params"]),
        "simulation": len(manifest["simulation"]),
        "decision_policy": len(manifest["decision_policy"]),
        "analysis": len(manifest["analysis"]),
        "dependencies": len(manifest["dependencies"]),
        "scenarios": len(manifest["scenarios"]),
    }
    simulation = get_simulation_settings(cfg)
    live_metadata = build_artifact_metadata(
        cfg=cfg,
        config_path=Path(CONFIG_PATH),
        parameter_registry_path=Path(PARAMETER_REGISTRY_PATH),
        assumption_registry_path=Path(ASSUMPTION_REGISTRY_PATH),
        generator_script_path=GENERATOR_SCRIPT_PATH,
        n_worlds=int(simulation["n_worlds"]),
        seed=int(get_seed(cfg)),
        annual_volume=int(simulation["annual_volume"]),
        time_horizon_years=int(simulation["time_horizon_years"]),
        discount_rate_annual=float(simulation["discount_rate_annual"]),
        published_scenario=str(simulation["scenario"]),
    )

    metadata_path = CASE_STUDY_ARTIFACTS_DIR / "metadata.json"
    stability_path = CASE_STUDY_ARTIFACTS_DIR / "stability_summary.json"
    evidence_path = CASE_STUDY_ARTIFACTS_DIR / "evidence_summary.json"
    if metadata_path.exists() and stability_path.exists() and evidence_path.exists():
        metadata = _load_json(metadata_path)
        evidence_summary = _load_json(evidence_path)
        freshness = compare_artifact_metadata(metadata, live_metadata)
        return PublishedGovernance(
            metadata=metadata,
            stability_summary=_load_json(stability_path),
            evidence_summary=evidence_summary,
            manifest_counts=manifest_counts,
            freshness_status=freshness.status,
            freshness_message=freshness.guidance,
            stale_fields=freshness.mismatched_fields,
            evidence_note_path=str(evidence_summary.get("note_artifact_path", "")),
            frontier_semantics="full-option switch frontier",
        )

    fallback = build_case_study_artifacts(CONFIG_PATH)
    return PublishedGovernance(
        metadata=fallback.metadata,
        stability_summary=fallback.stability_summary,
        evidence_summary=fallback.evidence_summary,
        manifest_counts=manifest_counts,
        freshness_status="unknown",
        freshness_message=(
            "Published governance artifacts were missing, so the app rebuilt them live."
        ),
        stale_fields=(),
        evidence_note_path=str(fallback.evidence_summary.get("note_artifact_path", "")),
        frontier_semantics="full-option switch frontier",
    )


def render_app() -> None:
    """Render the Streamlit interface."""

    st.set_page_config(page_title="Product Decision Under Uncertainty", layout="wide")
    st.title("Product Decision Under Uncertainty")

    cfg = load_config(CONFIG_PATH)
    analysis = get_analysis_settings(cfg)
    simulation = get_simulation_settings(cfg)
    scenario_metadata = get_scenario_metadata(cfg)
    scenario_names = list(scenario_metadata)
    published_governance = load_published_governance()
    warning_message = governance_warning_message(published_governance)
    if warning_message:
        if published_governance.freshness_status == "stale":
            st.warning(warning_message)
        else:
            st.info(warning_message)

    with st.sidebar:
        st.header("Run settings")
        n_worlds = st.number_input(
            "Simulation runs",
            min_value=1_000,
            max_value=100_000,
            value=int(simulation["n_worlds"]),
            step=1_000,
        )
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=10_000_000,
            value=int(get_seed(cfg)),
            step=1,
        )
        scenario = st.selectbox(
            "Scenario",
            options=scenario_names,
            index=scenario_names.index(simulation["scenario"]),
            format_func=lambda key: scenario_metadata[key]["label"],
        )
        st.markdown("**Scenario descriptions**")
        for name in scenario_names:
            label = scenario_metadata[name]["label"]
            description = scenario_metadata[name]["description"]
            st.caption(f"`{label}`: {description}")

    run_context = build_run_context(
        selected_settings={
            "scenario": scenario,
            "scenario_label": str(scenario_metadata[scenario]["label"]),
            "seed": int(seed),
            "n_worlds": int(n_worlds),
        },
        published_settings={
            "scenario": str(simulation["scenario"]),
            "scenario_label": str(scenario_metadata[str(simulation["scenario"])]["label"]),
            "seed": int(get_seed(cfg)),
            "n_worlds": int(simulation["n_worlds"]),
        },
    )
    st.caption(published_case_caption())
    st.subheader(str(run_context["heading"]))
    st.caption(str(run_context["detail"]))
    if not bool(run_context["matches_published"]):
        st.caption(str(run_context["note"]))

    outputs = compute_outputs(int(n_worlds), int(seed), scenario)
    current_metadata = {
        "seed": int(seed),
        "n_worlds": int(n_worlds),
        "annual_volume": int(outputs.simulation_settings["annual_volume"]),
        "time_horizon_years": int(outputs.simulation_settings["time_horizon_years"]),
        "discount_rate_annual": float(outputs.simulation_settings["discount_rate_annual"]),
        "declared_model_version": get_declared_model_version(cfg),
    }

    st.subheader("Recommendation")
    for line in recommendation_lines(outputs.recommendation, outputs.summary, current_metadata):
        st.markdown(line)

    st.subheader("Guardrail eligibility")
    st.dataframe(
        eligibility_display_table(outputs.policy_eligibility),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Summary")
    st.dataframe(summary_display_table(outputs.summary), width="stretch", hide_index=True)
    st.plotly_chart(
        create_decision_dashboard(
            outputs.summary,
            outputs.diagnostics,
            outputs.sensitivity,
            recommended_option=outputs.recommendation.selected_option,
            sensitivity_threshold=analysis.sensitivity_materiality_threshold_abs_spearman,
        ),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Risk profile")
        st.plotly_chart(create_risk_profile_chart(outputs.summary), width="stretch")
    with right:
        st.subheader("Regret")
        st.plotly_chart(create_regret_comparison(outputs.diagnostics), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Trade-off matrix")
        st.plotly_chart(
            create_trade_off_matrix(outputs.summary, outputs.diagnostics),
            width="stretch",
        )
    with right:
        st.subheader("Scenario comparison")
        st.plotly_chart(
            create_scenario_comparison(outputs.scenario_results, scenario_metadata),
            width="stretch",
        )

    st.subheader("Sensitivity")
    selected_option = st.selectbox(
        "Inspect option",
        options=list(OPTION_LABELS.keys()),
        index=list(OPTION_LABELS.keys()).index(outputs.recommendation.selected_option),
        format_func=lambda option: OPTION_LABELS[option],
    )
    sensitivity_left, sensitivity_right = st.columns([3, 2])
    with sensitivity_left:
        st.plotly_chart(
            create_sensitivity_waterfall(
                outputs.sensitivity,
                selected_option,
                threshold=analysis.sensitivity_materiality_threshold_abs_spearman,
            ),
            width="stretch",
        )
    with sensitivity_right:
        st.dataframe(
            top_sensitivity_rows(
                outputs.sensitivity,
                selected_option,
                threshold=analysis.sensitivity_materiality_threshold_abs_spearman,
                limit=analysis.sensitivity_max_rows_per_option,
            ),
            width="stretch",
            hide_index=True,
        )
        note = sensitivity_summary_note(
            outputs.sensitivity,
            selected_option,
            threshold=analysis.sensitivity_materiality_threshold_abs_spearman,
        )
        if note:
            st.caption(note)

    st.subheader("Selected-vs-runner-up payoff diagnostic")
    st.caption(
        "This section is descriptive. It shows which sampled parameters move with the "
        "selected-minus-runner-up payoff delta. It does not define the policy."
    )
    st.dataframe(
        payoff_delta_display_table(pd.DataFrame(outputs.payoff_delta["delta_rows"])),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Policy frontier")
    st.caption(
        "Full-option switch frontier. Each row re-runs the full policy and records the first "
        "threshold move that changes the selected option."
    )
    st.dataframe(
        frontier_display_table(pd.DataFrame(outputs.policy_frontier["frontier_rows"])),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Secondary runner-up view. These rows show when the current runner-up clears its own "
        "blocking threshold, which is useful context but not the same as the first switch."
    )
    st.dataframe(
        runner_up_frontier_display_table(
            pd.DataFrame(outputs.policy_frontier["runner_up_comparison_rows"])
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Published-case stability")
    stability = published_governance.stability_summary
    st.caption(
        f"Published-case reruns: {stability['run_count']} across multiple seeds and world counts. "
        f"Selected-option P05 range: EUR {float(stability['selected_p05_range_eur']):,.0f}. "
        f"Runner-up P05 range: EUR {float(stability['runner_up_p05_range_eur']):,.0f}."
    )
    stability_left, stability_right = st.columns(2)
    with stability_left:
        st.markdown("**Recommendation frequency**")
        st.dataframe(
            stability_frequency_table(
                stability["recommendation_frequency"],
                key="selected_option",
            ),
            width="stretch",
            hide_index=True,
        )
    with stability_right:
        st.markdown("**EV leader frequency**")
        st.dataframe(
            stability_frequency_table(stability["ev_leader_frequency"], key="ev_leader"),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Provenance and evidence")
    evidence = published_governance.evidence_summary
    metadata = published_governance.metadata
    evidence_status_line = (
        "No public sources registered yet"
        if int(evidence["source_count"]) == 0
        else f"{evidence['source_count']} public source(s) registered"
    )
    st.markdown(
        "\n".join(
            [
                f"- Declared model version: `{metadata['declared_model_version']}`.",
                f"- Freshness status: `{published_governance.freshness_status}`.",
                f"- Code fingerprint: `{str(metadata['code_sha256'])[:12]}`.",
                f"- Lockfile fingerprint: `{str(metadata['lockfile_sha256'])[:12]}`.",
                (
                    "- Assumption manifest coverage: "
                    f"`{published_governance.manifest_counts['params']}` "
                    "parameters, "
                    f"`{published_governance.manifest_counts['dependencies']}` dependencies, "
                    f"`{published_governance.manifest_counts['scenarios']}` scenarios."
                ),
                f"- Public evidence status: {evidence_status_line}.",
                f"- Evidence artifact path: `{published_governance.evidence_note_path}`.",
                f"- Frontier semantics: `{published_governance.frontier_semantics}`.",
                f"- Evidence note: {evidence['note']}",
            ]
        )
    )

    st.subheader("Downloads")
    download_left, download_middle, download_third, download_right = st.columns(4)
    with download_left:
        st.download_button(
            "Summary CSV",
            data=outputs.summary.to_csv(index=False),
            file_name="summary.csv",
            mime="text/csv",
        )
    with download_middle:
        st.download_button(
            "Diagnostics CSV",
            data=outputs.diagnostics.to_csv(index=False),
            file_name="diagnostics.csv",
            mime="text/csv",
        )
    with download_third:
        st.download_button(
            "Eligibility CSV",
            data=outputs.policy_eligibility.to_csv(index=False),
            file_name="policy_eligibility.csv",
            mime="text/csv",
        )
    with download_right:
        st.download_button(
            "Scenario CSV",
            data=outputs.scenario_results.to_csv(index=False),
            file_name="scenario_results.csv",
            mime="text/csv",
        )

    with st.expander("Raw tables", expanded=False):
        st.markdown("**Diagnostics**")
        st.dataframe(
            diagnostics_display_table(outputs.diagnostics),
            width="stretch",
            hide_index=True,
        )
        st.markdown("**Scenario table**")
        st.dataframe(
            scenario_display_table(
                outputs.scenario_results.assign(
                    scenario=outputs.scenario_results["scenario"].map(
                        lambda value: scenario_metadata[str(value)]["label"]
                    )
                )
            ),
            width="stretch",
            hide_index=True,
        )


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file into a dictionary."""

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    render_app()
