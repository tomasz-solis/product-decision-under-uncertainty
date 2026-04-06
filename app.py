"""Streamlit app for exploring the Product Decision Under Uncertainty case study."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from simulator.analytics import (
    decision_diagnostics,
    driver_analysis,
    sensitivity_analysis,
    summarize_results,
)
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
from simulator.output_utils import build_run_context, format_eur, labeled_option
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
    comparison_frontier_display_table,
    diagnostics_display_table,
    eligibility_display_table,
    frontier_display_table,
    payoff_delta_display_table,
    scenario_display_table,
    sensitivity_summary_note,
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
from simulator.reporting import build_case_study_artifacts
from simulator.simulation import OPTION_LABELS, run_all_scenarios, run_simulation
from simulator.visualizations import (
    create_frontier_switch_chart,
    create_guardrail_chart,
    create_ranked_payoff_profile,
    create_scenario_comparison,
    create_sensitivity_waterfall,
    create_stability_chart,
)


@dataclass(frozen=True)
class AppOutputs:
    """Typed container for the app's cached simulation outputs."""

    simulation_settings: dict[str, Any]
    results: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    sensitivity: pd.DataFrame
    driver_analysis: pd.DataFrame
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
    stability_runs: pd.DataFrame
    stability_summary: dict[str, Any]
    evidence_summary: dict[str, Any]
    manifest_counts: dict[str, int]
    freshness_status: str
    freshness_message: str
    stale_fields: tuple[str, ...]
    evidence_note_path: str
    frontier_semantics: str


PLOTLY_CONFIG: dict[str, object] = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
}


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


def inject_app_styles() -> None:
    """Apply the app-specific visual system on top of Streamlit defaults."""

    st.markdown(
        """
        <style>
        :root {
            --app-bg-top: #f7f4ec;
            --app-bg-bottom: #fcfbf7;
            --panel-bg: #fcfaf5;
            --panel-border: #e5dfd3;
            --panel-shadow: rgba(23, 32, 38, 0.05);
            --ink: #172026;
            --muted: #58656d;
            --accent: #0b6e4f;
            --sidebar-bg: #eff2f6;
        }

        .stApp {
            background: linear-gradient(180deg, var(--app-bg-top) 0%, var(--app-bg-bottom) 22%, #ffffff 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3.2rem;
            max-width: 1380px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--sidebar-bg) 0%, #f8f9fb 100%);
            border-right: 1px solid #d9e0e7;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.55;
        }

        h1, h2, h3 {
            letter-spacing: -0.03em;
        }

        h2 {
            margin-top: 0.25rem;
        }

        [data-testid="stCaptionContainer"] {
            margin-bottom: 0.8rem;
        }

        [data-testid="stTabs"] {
            margin-top: 1rem;
        }

        [data-testid="stTabs"] button[role="tab"] {
            border-radius: 999px;
            border: 1px solid var(--panel-border);
            background: #f4efe3;
            padding: 0.45rem 0.95rem;
        }

        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: var(--ink);
            border-color: var(--ink);
        }

        [data-testid="stTabs"] button[role="tab"] p {
            font-size: 0.95rem;
            font-weight: 600;
        }

        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
            color: #ffffff;
        }

        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"],
        [data-testid="stExpander"] {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            box-shadow: 0 12px 32px var(--panel-shadow);
        }

        [data-testid="stPlotlyChart"] {
            padding: 0.7rem 0.9rem 0.4rem 0.9rem;
            margin-bottom: 1.4rem;
        }

        [data-testid="stDataFrame"] {
            padding: 0.4rem;
            margin-bottom: 1.2rem;
        }

        .section-copy {
            max-width: 64rem;
            color: var(--muted);
            line-height: 1.55;
            margin: 0 0 0.9rem 0;
        }

        .note-card {
            background: #f6f2e8;
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin: 0 0 0.9rem 0;
        }

        .note-card__title {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--accent);
            margin: 0 0 0.35rem 0;
        }

        .note-card__body {
            color: var(--ink);
            line-height: 1.5;
            margin: 0;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.85rem;
            margin: 0.5rem 0 1.25rem 0;
        }

        .metric-card {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            box-shadow: 0 12px 28px var(--panel-shadow);
            padding: 0.95rem 1rem;
        }

        .metric-card__label {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--muted);
            margin: 0 0 0.25rem 0;
        }

        .metric-card__value {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            line-height: 1.2;
            margin: 0;
        }

        .metric-card__detail {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0.35rem 0 0 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_copy(text: str) -> None:
    """Render one section explainer paragraph in the app's muted body style."""

    st.markdown(
        f"<p class='section-copy'>{html.escape(text)}</p>",
        unsafe_allow_html=True,
    )


def render_note_card(title: str, body: str) -> None:
    """Render a compact explanatory note card."""

    st.markdown(
        (
            "<div class='note-card'>"
            f"<p class='note-card__title'>{html.escape(title)}</p>"
            f"<p class='note-card__body'>{html.escape(body)}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_plotly_figure(figure: Any) -> None:
    """Render one Plotly figure with a consistent Streamlit wrapper config."""

    st.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        config=PLOTLY_CONFIG,
    )


def render_metric_card_grid(cards: list[dict[str, str]]) -> None:
    """Render a compact grid of metric cards from label, value, and detail text."""

    markup = "".join(
        (
            "<div class='metric-card'>"
            f"<p class='metric-card__label'>{html.escape(card['label'])}</p>"
            f"<p class='metric-card__value'>{html.escape(card['value'])}</p>"
            f"<p class='metric-card__detail'>{html.escape(card['detail'])}</p>"
            "</div>"
        )
        for card in cards
    )
    st.markdown(f"<div class='metric-grid'>{markup}</div>", unsafe_allow_html=True)


def _policy_status_short(recommendation: RecommendationResult) -> str:
    """Return the short policy-state label used in executive metric cards."""

    if recommendation.selected_reason_type == "guardrails_relaxed_highest_ev":
        return "Fallback to EV"
    if recommendation.selected_reason_type == "only_option_passing_guardrails":
        return "Only eligible option"
    if recommendation.selected_reason_type == "ev_tolerance_override":
        return "EV tolerance tie-break"
    return "Passes guardrails"


def _policy_status_detail(recommendation: RecommendationResult) -> str:
    """Return a plain-English explanation of the current policy branch."""

    selected_label = labeled_option(recommendation.selected_option)
    if recommendation.selected_reason_type == "guardrails_relaxed_highest_ev":
        return (
            f"No option cleared both guardrails, so the policy fell back to expected value and selected {selected_label}."
        )
    if recommendation.selected_reason_type == "only_option_passing_guardrails":
        return f"{selected_label} is the only option that clears both the downside floor and the regret cap."
    if recommendation.selected_reason_type == "ev_tolerance_override":
        return (
            f"{selected_label} stayed inside the EV tolerance band and then won the regret tie-break."
        )
    return f"{selected_label} clears the guardrails and still leads the eligible set on expected value."


def _selected_option_snapshot(outputs: AppOutputs) -> dict[str, object]:
    """Return the selected option's main summary, regret, and guardrail metrics."""

    selected_option = outputs.recommendation.selected_option
    summary_row = outputs.summary.set_index("option").loc[selected_option]
    diagnostics_row = outputs.diagnostics.set_index("option").loc[selected_option]
    eligibility_row = outputs.policy_eligibility.set_index("option").loc[selected_option]
    return {
        "option": selected_option,
        "label": labeled_option(selected_option),
        "mean_value_eur": float(summary_row["mean_value_eur"]),
        "p05_value_eur": float(summary_row["p05_value_eur"]),
        "mean_regret_eur": float(diagnostics_row["mean_regret_eur"]),
        "downside_slack_eur": float(eligibility_row["downside_slack_eur"]),
        "regret_slack_eur": float(eligibility_row["regret_slack_eur"]),
    }


def render_executive_metric_cards(outputs: AppOutputs) -> None:
    """Render the headline metrics for the currently selected recommendation."""

    snapshot = _selected_option_snapshot(outputs)
    recommendation = outputs.recommendation
    comparison_option = recommendation.comparison_option
    comparison_margin = recommendation.comparison_margin_eur
    comparison_label = "No direct comparator"
    comparison_detail = "The policy did not surface a meaningful comparison option."
    if comparison_option is not None and comparison_margin is not None:
        comparison_label = (
            f"EV lead vs {labeled_option(comparison_option)}"
            if comparison_margin >= 0.0
            else f"EV gap vs {labeled_option(comparison_option)}"
        )
        comparison_detail = (
            "Positive means the selected option leads on expected value."
            if comparison_margin >= 0.0
            else "Negative means the selected option trails the comparison option."
        )

    cards = [
        {
            "label": "Recommendation",
            "value": str(snapshot["label"]),
            "detail": _policy_status_short(recommendation),
        },
        {
            "label": "Expected value",
            "value": format_eur(float(snapshot["mean_value_eur"])),
            "detail": "Average discounted outcome across the simulated worlds.",
        },
        {
            "label": comparison_label,
            "value": (
                "Not needed"
                if comparison_margin is None
                else format_eur(abs(float(comparison_margin)))
            ),
            "detail": comparison_detail,
        },
        {
            "label": "Downside P05",
            "value": format_eur(float(snapshot["p05_value_eur"])),
            "detail": "A conservative downside point: 95% of simulated outcomes land above it.",
        },
        {
            "label": "Mean regret",
            "value": format_eur(float(snapshot["mean_regret_eur"])),
            "detail": "Average gap to the hindsight-best option. Lower is better.",
        },
    ]
    render_metric_card_grid(cards)


def render_recommendation_summary(outputs: AppOutputs, current_metadata: dict[str, Any]) -> None:
    """Render the recommendation summary in plain language for executive readers."""

    snapshot = _selected_option_snapshot(outputs)
    recommendation = outputs.recommendation
    comparison_option = recommendation.comparison_option
    comparison_margin = recommendation.comparison_margin_eur

    lines = [
        f"- Current call: **{snapshot['label']}**.",
        f"- Policy readout: {_policy_status_detail(recommendation)}",
        (
            "- Core numbers: expected value "
            f"**{format_eur(float(snapshot['mean_value_eur']))}**, downside P05 "
            f"**{format_eur(float(snapshot['p05_value_eur']))}**, and mean regret "
            f"**{format_eur(float(snapshot['mean_regret_eur']))}**."
        ),
    ]
    if comparison_option is not None and comparison_margin is not None:
        direction = "leads" if comparison_margin >= 0.0 else "trails"
        lines.append(
            f"- Honest comparison: the selected option {direction} **{labeled_option(comparison_option)}** "
            f"by **{format_eur(abs(float(comparison_margin)))}** on expected value."
        )
    lines.append(
        f"- Published run settings for this view: `{current_metadata['n_worlds']:,}` worlds, seed `{current_metadata['seed']}`, "
        f"horizon `{current_metadata['time_horizon_years']}` years, and discount rate "
        f"`{current_metadata['discount_rate_annual']:.0%}`."
    )
    for line in lines:
        st.markdown(line)


def scenario_selection_display_table(
    scenario_results: pd.DataFrame,
    scenario_metadata: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Return one row per scenario with the selected option and its core metrics."""

    selected_rows = scenario_results.loc[
        scenario_results["option"] == scenario_results["selected_option"]
    ].copy()
    selected_rows["Scenario"] = selected_rows["scenario"].map(
        lambda value: scenario_metadata.get(str(value), {}).get("label", str(value))
    )
    selected_rows["Selected option"] = selected_rows["selected_option"].map(labeled_option)
    selected_rows["Expected value"] = selected_rows["mean_value_eur"].map(format_eur)
    selected_rows["P05"] = selected_rows["p05_value_eur"].map(format_eur)
    selected_rows["Mean regret"] = selected_rows["mean_regret_eur"].map(format_eur)
    selected_rows["Eligible"] = selected_rows["eligible"].map(
        lambda value: "Yes" if bool(value) else "No"
    )
    return selected_rows[
        ["Scenario", "Selected option", "Expected value", "P05", "Mean regret", "Eligible"]
    ]


def governance_display_table(governance: PublishedGovernance) -> pd.DataFrame:
    """Return a compact provenance and evidence table for the published artifacts."""

    evidence = governance.evidence_summary
    metadata = governance.metadata
    evidence_status = (
        "No public sources registered yet"
        if int(evidence["source_count"]) == 0
        else f"{evidence['source_count']} public source(s) registered"
    )
    rows = [
        {"Field": "Declared model version", "Value": str(metadata["declared_model_version"])},
        {"Field": "Freshness status", "Value": str(governance.freshness_status)},
        {"Field": "Code fingerprint", "Value": str(metadata["code_sha256"])[:12]},
        {"Field": "Lockfile fingerprint", "Value": str(metadata["lockfile_sha256"])[:12]},
        {
            "Field": "Assumption manifest coverage",
            "Value": (
                f"{governance.manifest_counts['params']} parameters, "
                f"{governance.manifest_counts['dependencies']} dependencies, "
                f"{governance.manifest_counts['scenarios']} scenarios"
            ),
        },
        {"Field": "Public evidence status", "Value": evidence_status},
        {"Field": "Evidence artifact path", "Value": str(governance.evidence_note_path)},
        {"Field": "Frontier semantics", "Value": str(governance.frontier_semantics)},
    ]
    return pd.DataFrame(rows)


def render_stability_summary_cards(stability: dict[str, Any]) -> None:
    """Render compact stability summary cards instead of a misleading frequency bar."""

    run_count = int(stability["run_count"])
    recommendation_rows = list(stability["recommendation_frequency"])
    ev_leader_rows = list(stability["ev_leader_frequency"])
    selected_row = recommendation_rows[0] if recommendation_rows else None
    ev_leader_row = ev_leader_rows[0] if ev_leader_rows else None

    cards = [
        {
            "label": "Recommendation held",
            "value": (
                "No reruns"
                if selected_row is None
                else f"{int(selected_row['count'])}/{run_count}"
            ),
            "detail": (
                "No published-case reruns were available."
                if selected_row is None
                else f"{labeled_option(str(selected_row['selected_option']))} remained selected across the reruns."
            ),
        },
        {
            "label": "EV leader consistency",
            "value": (
                "No reruns"
                if ev_leader_row is None
                else f"{int(ev_leader_row['count'])}/{run_count}"
            ),
            "detail": (
                "No EV-leader check was available."
                if ev_leader_row is None
                else f"{labeled_option(str(ev_leader_row['ev_leader']))} led expected value in the reruns."
            ),
        },
        {
            "label": "Selected-option P05 spread",
            "value": format_eur(float(stability["selected_p05_range_eur"])),
            "detail": "Lower spread means the downside view moved less as seeds and world counts changed.",
        },
    ]
    render_metric_card_grid(cards)


def metric_dictionary_sections() -> dict[str, pd.DataFrame]:
    """Return grouped glossary tables for the app's dictionary tab."""

    return {
        "Decision rule": pd.DataFrame(
            [
                {
                    "Term": "Guardrailed expected value",
                    "Meaning": "The app first removes options that fail policy guardrails, then picks the best remaining expected value.",
                    "How to use it": "Treat it as the formal decision rule, not as a generic label for expected value.",
                },
                {
                    "Term": "Eligible option",
                    "Meaning": "An option that clears both the downside floor and the regret cap.",
                    "How to use it": "Only eligible options can win before the fallback logic is considered.",
                },
                {
                    "Term": "Downside floor",
                    "Meaning": "The minimum acceptable P05 value for an option.",
                    "How to use it": "If an option sits below this floor, its downside is judged too weak for the current policy.",
                },
                {
                    "Term": "Regret cap",
                    "Meaning": "The maximum acceptable mean regret for an option.",
                    "How to use it": "If an option sits above this cap, it gives up too much value versus the hindsight-best action.",
                },
                {
                    "Term": "Policy frontier",
                    "Meaning": "The threshold movement needed before the recommendation changes.",
                    "How to use it": "Use it to see how much policy slack you have before another option becomes preferred.",
                },
            ]
        ),
        "Risk and payoff": pd.DataFrame(
            [
                {
                    "Term": "Expected value",
                    "Meaning": "The average discounted outcome across all simulated worlds.",
                    "How to use it": "This is the central value metric, but it should be read together with downside and regret.",
                },
                {
                    "Term": "Downside P05",
                    "Meaning": "The 5th percentile outcome. Ninety-five percent of simulated outcomes land above it.",
                    "How to use it": "Use it as the main downside anchor when the business cares about avoiding bad tails.",
                },
                {
                    "Term": "P95",
                    "Meaning": "The 95th percentile outcome.",
                    "How to use it": "Use it to understand upside potential, not as the main policy constraint.",
                },
                {
                    "Term": "Mean regret",
                    "Meaning": "The average gap between an option and the hindsight-best option in each world.",
                    "How to use it": "Lower is better. It captures the cost of picking the wrong option under uncertainty.",
                },
                {
                    "Term": "P95 regret",
                    "Meaning": "The 95th percentile regret level.",
                    "How to use it": "Use it as a tail-regret diagnostic when the average regret looks acceptable but tails still worry you.",
                },
            ]
        ),
        "Drivers": pd.DataFrame(
            [
                {
                    "Term": "Partial rank correlation",
                    "Meaning": "A driver score that estimates how one assumption moves the selected option after controlling for the others.",
                    "How to use it": "Magnitude shows importance; sign shows direction. Positive means more of the input tends to help the option.",
                },
                {
                    "Term": "95% CI",
                    "Meaning": "A bootstrap confidence interval around the partial rank correlation.",
                    "How to use it": "Tighter intervals mean the driver signal is more stable across resamples.",
                },
                {
                    "Term": "Delta rho",
                    "Meaning": "A descriptive rank association between a parameter and the selected-minus-comparison payoff gap.",
                    "How to use it": "This helps explain the diagnostic comparison, but it does not define policy.",
                },
                {
                    "Term": "Selected-vs-best-excluded payoff diagnostic",
                    "Meaning": "A descriptive view of what moves the gap between the recommendation and its honest comparison option.",
                    "How to use it": "Use it to understand the contest around the current decision, not to set thresholds.",
                },
            ]
        ),
        "Governance": pd.DataFrame(
            [
                {
                    "Term": "Published-case stability",
                    "Meaning": "A rerun check that varies seeds and world counts on the published configuration.",
                    "How to use it": "If the recommendation keeps flipping, the published case may still be too sampling-sensitive.",
                },
                {
                    "Term": "Freshness status",
                    "Meaning": "Whether the checked-in governance artifacts still match the current code, config, and dependency state.",
                    "How to use it": "Treat a stale status as a governance warning, not just a cosmetic mismatch.",
                },
                {
                    "Term": "Assumption manifest",
                    "Meaning": "The traceable registry of parameter, policy, dependency, and scenario assumptions.",
                    "How to use it": "Use it to audit what was assumed, why, and by whom.",
                },
            ]
        ),
    }


def render_metric_dictionary() -> None:
    """Render the grouped metric dictionary inside the dedicated app tab."""

    dictionary_tabs = st.tabs(list(metric_dictionary_sections()))
    for tab, (label, table) in zip(
        dictionary_tabs,
        metric_dictionary_sections().items(),
        strict=False,
    ):
        with tab:
            st.dataframe(table, width="stretch", hide_index=True)


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
    driver_rows = driver_analysis(results)
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
        driver_analysis=driver_rows,
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
    stability_runs_path = CASE_STUDY_ARTIFACTS_DIR / "stability_runs.json"
    stability_path = CASE_STUDY_ARTIFACTS_DIR / "stability_summary.json"
    evidence_path = CASE_STUDY_ARTIFACTS_DIR / "evidence_summary.json"
    if (
        metadata_path.exists()
        and stability_runs_path.exists()
        and stability_path.exists()
        and evidence_path.exists()
    ):
        metadata = _load_json(metadata_path)
        evidence_summary = _load_json(evidence_path)
        freshness = compare_artifact_metadata(metadata, live_metadata)
        return PublishedGovernance(
            metadata=metadata,
            stability_runs=pd.DataFrame(_load_json(stability_runs_path)),
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
        stability_runs=fallback.stability_runs,
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
    inject_app_styles()
    st.title("Product Decision Under Uncertainty")

    cfg = load_config(CONFIG_PATH)
    analysis = get_analysis_settings(cfg)
    decision_policy = get_decision_policy(cfg)
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

    render_executive_metric_cards(outputs)
    hero_left, hero_right = st.columns([3, 2])
    with hero_left:
        st.subheader("Recommendation")
        render_section_copy(
            "Start here for the decision call, the core value and risk numbers, and the honest comparison against the nearest alternative."
        )
        render_recommendation_summary(outputs, current_metadata)
    with hero_right:
        render_note_card(
            "How to use this page",
            "Read the recommendation first, then use Guardrail eligibility and Policy frontier to understand why the rule selected it. Use Driver analysis and Published-case stability to judge how fragile the call is.",
        )
        render_note_card(
            "Metric dictionary",
            "If a phrase sounds internal, use the Metric dictionary tab below. It explains which numbers are policy-defining and which are descriptive only.",
        )

    st.divider()

    st.subheader("Payoff distribution")
    render_section_copy(
        "Each option is shown as a range from P05 to P95. The circle marks expected value and the diamond marks the median, so you can separate central value from tail risk."
    )
    render_plotly_figure(
        create_ranked_payoff_profile(
            outputs.summary,
            recommended_option=outputs.recommendation.selected_option,
            comparison_option=outputs.recommendation.comparison_option,
        )
    )
    st.dataframe(summary_display_table(outputs.summary), width="stretch", hide_index=True)
    render_note_card(
        "Read this chart",
        "A wider range means more uncertainty. A higher P05 means a safer downside. Use the table below the chart when you need exact numbers rather than visual ranking.",
    )

    st.subheader("Guardrail eligibility")
    render_section_copy(
        "An option stays eligible only if its downside P05 stays above the floor and its mean regret stays below the cap. This is the policy-defining filter, not just a descriptive chart."
    )
    render_plotly_figure(
        create_guardrail_chart(
            outputs.policy_eligibility,
            minimum_p05_value_eur=decision_policy.minimum_p05_value_eur,
            maximum_mean_regret_eur=decision_policy.maximum_mean_regret_eur,
            recommended_option=outputs.recommendation.selected_option,
            comparison_option=outputs.recommendation.comparison_option,
        )
    )
    st.dataframe(
        eligibility_display_table(outputs.policy_eligibility),
        width="stretch",
        hide_index=True,
    )
    render_note_card(
        "Read this chart",
        "On the downside panel, points to the right of the dashed line pass. On the regret panel, points to the left of the dashed line pass. If no option passes both, the policy falls back to expected value.",
    )

    st.subheader("Policy frontier")
    render_section_copy(
        "The frontier shows how far each threshold could move before the recommendation changes. It answers the practical question: how much policy slack do we really have?"
    )
    render_plotly_figure(
        create_frontier_switch_chart(
            outputs.policy_frontier,
            outputs.policy_frontier_grid,
        )
    )
    st.dataframe(
        frontier_display_table(pd.DataFrame(outputs.policy_frontier["frontier_rows"])),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Scenario comparison")
    render_section_copy(
        "This compares the options across declared business scenarios so you can see whether the recommendation is a single-scenario artifact or a broader pattern."
    )
    render_plotly_figure(create_scenario_comparison(outputs.scenario_results, scenario_metadata))
    st.dataframe(
        scenario_selection_display_table(outputs.scenario_results, scenario_metadata),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("Driver analysis")
    render_section_copy(
        "Partial rank correlation estimates which assumptions really move the selected option after controlling for the others. This is where to look when you want to know what could change the call."
    )
    selected_option = st.selectbox(
        "Inspect option",
        options=list(OPTION_LABELS.keys()),
        index=list(OPTION_LABELS.keys()).index(outputs.recommendation.selected_option),
        format_func=labeled_option,
    )
    sensitivity_left, sensitivity_right = st.columns([3, 2])
    with sensitivity_left:
        render_plotly_figure(
            create_sensitivity_waterfall(
                outputs.driver_analysis,
                selected_option,
                threshold=analysis.sensitivity_materiality_threshold_abs_spearman,
            )
        )
    with sensitivity_right:
        st.dataframe(
            top_sensitivity_rows(
                outputs.driver_analysis,
                selected_option,
                threshold=analysis.sensitivity_materiality_threshold_abs_spearman,
                limit=analysis.sensitivity_max_rows_per_option,
            ),
            width="stretch",
            hide_index=True,
        )
        note = sensitivity_summary_note(
            outputs.driver_analysis,
            selected_option,
            threshold=analysis.sensitivity_materiality_threshold_abs_spearman,
        )
        if note:
            render_note_card("Driver note", note.replace("`", ""))
        render_note_card(
            "Read this chart",
            "Positive values mean the parameter tends to help the option as it rises. Negative values mean it tends to hurt. Larger absolute values matter more than small ones.",
        )

    comparison_heading = (
        "Selected-vs-policy-runner-up payoff diagnostic"
        if outputs.recommendation.policy_runner_up is not None
        else "Selected-vs-best-excluded payoff diagnostic"
    )
    st.subheader(comparison_heading)
    render_section_copy(
        "This diagnostic is descriptive. It shows which sampled parameters move with the selected-minus-comparison payoff gap. It helps explain the contest around the decision, but it does not define policy."
    )
    st.dataframe(
        payoff_delta_display_table(pd.DataFrame(outputs.payoff_delta["delta_rows"])),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Published-case stability")
    stability = published_governance.stability_summary
    comparison_p05_range = stability["comparison_p05_range_eur"]
    stability_caption = (
        f"Published-case reruns: {stability['run_count']} across multiple seeds and world counts. "
        f"Selected-option P05 range: {format_eur(float(stability['selected_p05_range_eur']))}."
    )
    if comparison_p05_range is not None:
        stability_caption += (
            f" Comparison-option P05 range: {format_eur(float(comparison_p05_range))}."
        )
    render_section_copy(
        "This is a sampling-robustness check on the published configuration. Stable frequency and narrower metric ranges mean the recommendation is less sensitive to Monte Carlo noise."
    )
    st.caption(stability_caption)
    render_stability_summary_cards(stability)
    render_plotly_figure(create_stability_chart(published_governance.stability_runs))

    st.divider()

    reference_dictionary, reference_provenance, reference_downloads = st.tabs(
        ["Metric dictionary", "Provenance and evidence", "Downloads and raw tables"]
    )

    with reference_dictionary:
        st.subheader("Metric dictionary")
        render_section_copy(
            "Use this tab when a phrase sounds technical or overloaded. The key distinction is whether a metric drives the policy itself or only helps explain the result."
        )
        render_metric_dictionary()

    with reference_provenance:
        st.subheader("Provenance and evidence")
        render_section_copy(
            "This ties the published artifacts back to the current model version, dependency state, assumption manifest, and public evidence profile."
        )
        st.dataframe(
            governance_display_table(published_governance),
            width="stretch",
            hide_index=True,
        )
        render_note_card(
            "Evidence note",
            str(published_governance.evidence_summary["note"]),
        )

    with reference_downloads:
        st.subheader("Downloads")
        render_section_copy(
            "Use the CSV downloads for external review or follow-on analysis. The raw tables below mirror the main app views with exact formatted values."
        )
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
            st.markdown("**Summary**")
            st.dataframe(summary_display_table(outputs.summary), width="stretch", hide_index=True)
            st.markdown("**Guardrail eligibility**")
            st.dataframe(
                eligibility_display_table(outputs.policy_eligibility),
                width="stretch",
                hide_index=True,
            )
            st.markdown("**Diagnostics**")
            st.dataframe(
                diagnostics_display_table(outputs.diagnostics),
                width="stretch",
                hide_index=True,
            )
            st.markdown("**Policy frontier**")
            st.dataframe(
                frontier_display_table(pd.DataFrame(outputs.policy_frontier["frontier_rows"])),
                width="stretch",
                hide_index=True,
            )
            st.markdown("**Secondary comparison frontier**")
            st.dataframe(
                comparison_frontier_display_table(
                    pd.DataFrame(outputs.policy_frontier["secondary_comparison_rows"])
                ),
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
