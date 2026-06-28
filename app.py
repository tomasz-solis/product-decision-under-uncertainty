"""Streamlit app for exploring the Product Decision Under Uncertainty case study."""

from __future__ import annotations

import atexit
import copy
import html
import json
import re
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import streamlit as st
import yaml

from simulator.analytics import (
    decision_diagnostics,
    driver_analysis,
    sensitivity_analysis,
    summarize_results,
)
from simulator.app_state import AppOutputs, PublishedGovernance
from simulator.artifact_freshness import build_artifact_metadata, compare_artifact_metadata
from simulator.config import (
    DecisionPolicyConfig,
    apply_scenario,
    get_analysis_settings,
    get_decision_policy,
    get_declared_model_version,
    get_scenario_metadata,
    get_seed,
    get_simulation_settings,
    load_config,
    validate_config,
)
from simulator.output_utils import build_run_context, format_eur, labeled_option
from simulator.policy import (
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
    driver_analysis_interpretation_note,
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
    EVIDENCE_ARTIFACTS_DIR,
    GENERATOR_SCRIPT_PATH,
    PARAMETER_REGISTRY_PATH,
)
from simulator.provenance import (
    build_assumption_manifest,
    load_assumption_registry,
    load_parameter_registry,
    validate_parameter_registry,
)
from simulator.report_markdown import recommendation_lines
from simulator.reporting import build_case_study_artifacts
from simulator.serialization import markdown_table
from simulator.simulation import OPTION_LABELS, run_all_scenarios, run_simulation
from simulator.visualizations import (
    create_frontier_switch_chart,
    create_guardrail_chart,
    create_ranked_payoff_profile,
    create_scenario_comparison,
    create_sensitivity_waterfall,
    create_stability_chart,
)


class SelectedOptionSnapshot(TypedDict):
    """Typed summary of the currently selected option's core decision metrics."""

    option: str
    label: str
    mean_value_eur: float
    p05_value_eur: float
    mean_regret_eur: float
    downside_slack_eur: float
    regret_slack_eur: float


PLOTLY_CONFIG: dict[str, object] = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
}


def published_case_caption() -> str:
    """Caption distinguishing the published run from interactive reruns."""

    return (
        "The checked-in markdown artifacts come from the default scenario and the "
        "default run settings."
    )


def governance_warning_message(governance: PublishedGovernance) -> str | None:
    """Return the governance freshness banner text, if the app should show one."""

    if governance.freshness_status in {"stale", "unknown"}:
        return governance.freshness_message
    return None


_APP_STYLES_PATH = Path(__file__).parent / "static" / "app_styles.css"


def inject_app_styles() -> None:
    """Apply the app-specific visual system on top of Streamlit defaults."""
    css = _APP_STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


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


def _selected_option_snapshot(outputs: AppOutputs) -> SelectedOptionSnapshot:
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
    for tab, (_label, table) in zip(
        dictionary_tabs,
        metric_dictionary_sections().items(),
        strict=False,
    ):
        with tab:
            # st.table renders a static HTML table whose cells wrap onto new
            # lines, so the long definitions fit the page width instead of forcing
            # the horizontal scroll / truncation of the fixed-height dataframe grid.
            st.table(table.set_index("Term"))


def _persist_uploaded_config(raw: bytes) -> Path:
    """Write an uploaded config to a per-session private dir under a content hash.

    The engine consumes a path (``load_config``/``run_simulation`` re-read it on every
    rerun), so the upload must live on disk. ``mkdtemp`` gives an owner-only random
    directory rather than a world-readable predictable path, and the full SHA-256
    filename keeps the path content-addressed so the cache key stays stable and
    identical content de-duplicates.
    """

    upload_dir = st.session_state.get("_pduu_upload_dir")
    if upload_dir is None or not Path(upload_dir).exists():
        upload_dir = tempfile.mkdtemp(prefix="pduu_")
        st.session_state["_pduu_upload_dir"] = upload_dir
        atexit.register(shutil.rmtree, upload_dir, ignore_errors=True)
    path = Path(upload_dir) / f"{sha256(raw).hexdigest()}.yaml"
    if not path.exists():
        path.write_bytes(raw)
    return path


def build_config_from_form(base_cfg: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    """Overlay headline economic levers from the no-YAML form onto the built-in config.

    Returns a new config dict for the caller to validate via ``validate_config``. The
    triangular params keep a sensible spread around the chosen typical value so
    ``low <= mode <= high`` always holds.
    """

    cfg = copy.deepcopy(base_cfg)
    simulation = cfg["simulation"]
    simulation["annual_volume"] = int(values["annual_volume"])
    simulation["time_horizon_years"] = int(values["time_horizon_years"])
    simulation["discount_rate_annual"] = float(values["discount_rate_annual"])

    params = cfg["params"]
    failure_mode = float(values["baseline_failure_rate"])
    params["baseline_failure_rate"] = {
        "dist": "tri",
        "low": round(max(0.0, failure_mode * 0.6), 6),
        "mode": failure_mode,
        "high": round(min(1.0, max(failure_mode, failure_mode * 1.5)), 6),
    }
    drift_mode = float(values["do_nothing_drift_cost_eur"])
    params["do_nothing_drift_cost_eur"] = {
        "dist": "tri",
        "low": round(max(0.0, drift_mode * 0.6), 6),
        "mode": drift_mode,
        "high": round(max(drift_mode, drift_mode * 1.6), 6),
    }
    for key in (
        "stabilize_core_upfront_cost_eur",
        "feature_extension_upfront_cost_eur",
        "new_capability_upfront_cost_eur",
    ):
        params[key] = {"dist": "constant", "value": float(values[key])}
    return cfg


def _render_config_builder() -> tuple[Path, str, str | None] | None:
    """Render the no-YAML config form; return an active (path, key, error) or None.

    A built config is remembered in session state so it survives reruns (e.g. while the
    guardrail sliders move) until the user clears it or uploads a file.
    """

    base_cfg = load_config(CONFIG_PATH)
    simulation = base_cfg["simulation"]
    params = base_cfg["params"]
    with st.expander("Build a config (no YAML)"):
        st.caption("Set the headline economics; the four options and their structure stay fixed.")
        values: dict[str, float] = {
            "annual_volume": st.number_input(
                "Annual volume (units)", min_value=1_000, max_value=100_000_000,
                value=int(simulation["annual_volume"]), step=10_000,
            ),
            "time_horizon_years": st.number_input(
                "Time horizon (years)", min_value=1, max_value=10,
                value=int(simulation["time_horizon_years"]), step=1,
            ),
            "discount_rate_annual": st.number_input(
                "Discount rate (annual)", min_value=0.0, max_value=0.5,
                value=float(simulation["discount_rate_annual"]), step=0.01, format="%.2f",
            ),
            "baseline_failure_rate": st.number_input(
                "Baseline failure rate (typical)", min_value=0.0, max_value=1.0,
                value=float(params["baseline_failure_rate"]["mode"]), step=0.01, format="%.2f",
            ),
            "do_nothing_drift_cost_eur": st.number_input(
                "Do-nothing drift cost / yr (EUR)", min_value=0.0, max_value=10_000_000.0,
                value=float(params["do_nothing_drift_cost_eur"]["mode"]), step=10_000.0,
            ),
            "stabilize_core_upfront_cost_eur": st.number_input(
                "Stabilize Core upfront (EUR)", min_value=0.0, max_value=50_000_000.0,
                value=float(params["stabilize_core_upfront_cost_eur"]["value"]), step=50_000.0,
            ),
            "feature_extension_upfront_cost_eur": st.number_input(
                "Feature Extension upfront (EUR)", min_value=0.0, max_value=50_000_000.0,
                value=float(params["feature_extension_upfront_cost_eur"]["value"]), step=50_000.0,
            ),
            "new_capability_upfront_cost_eur": st.number_input(
                "New Capability upfront (EUR)", min_value=0.0, max_value=50_000_000.0,
                value=float(params["new_capability_upfront_cost_eur"]["value"]), step=50_000.0,
            ),
        }
        apply_clicked = st.button("Use these values")
        clear_clicked = st.button("Clear (back to built-in)")

    if clear_clicked:
        st.session_state.pop("_pduu_form_config", None)
        return None
    if apply_clicked:
        try:
            built = build_config_from_form(base_cfg, values)
            validate_config(built)
            raw = yaml.safe_dump(built, sort_keys=False).encode("utf-8")
            path = _persist_uploaded_config(raw)
            st.session_state["_pduu_form_config"] = str(path)
            return path, sha256(raw).hexdigest(), None
        except Exception:
            st.session_state.pop("_pduu_form_config", None)
            return (
                Path(CONFIG_PATH),
                "default",
                "Those values didn't make a valid config (costs must be non-negative and "
                "rates between 0 and 1). Showing the built-in case.",
            )

    stored = st.session_state.get("_pduu_form_config")
    if stored and Path(stored).exists():
        return Path(stored), sha256(Path(stored).read_bytes()).hexdigest(), None
    return None


def _evidence_candidate_value() -> float | None:
    """Return the ready evidence-backed baseline-failure-rate candidate, if available."""

    path = EVIDENCE_ARTIFACTS_DIR / "parameter_candidates.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for candidate in payload.get("candidates", []):
        if (
            candidate.get("target_name") == "baseline_failure_rate"
            and candidate.get("candidate_status") == "candidate_ready"
            and candidate.get("candidate_value") is not None
        ):
            return float(candidate["candidate_value"])
    return None


def _overlay_evidence(cfg: dict[str, Any], value: float) -> dict[str, Any]:
    """Return a config copy with baseline_failure_rate pinned around the evidence value.

    Kept a tight triangular rather than a constant so the parameter stays valid as a
    Gaussian-copula dependency anchor (the engine rejects constant dependency parameters).
    """

    overlaid = copy.deepcopy(cfg)
    pinned = round(float(value), 6)
    overlaid["params"]["baseline_failure_rate"] = {
        "dist": "tri",
        "low": round(max(0.0, pinned * 0.9), 6),
        "mode": pinned,
        "high": round(min(1.0, pinned * 1.1), 6),
    }
    return overlaid


def _apply_evidence_overlay(base_path: Path) -> tuple[Path, str, str | None]:
    """Overlay the evidence-backed baseline failure rate onto a config and persist it.

    Guards: the value flows through ``validate_config`` before use, and the overlaid config
    is content-addressed so toggling it is part of the cache key (never served stale).
    """

    candidate = _evidence_candidate_value()
    if candidate is None:
        return Path(CONFIG_PATH), "default", None
    try:
        overlaid = _overlay_evidence(load_config(base_path), candidate)
        validate_config(overlaid)
        raw = yaml.safe_dump(overlaid, sort_keys=False).encode("utf-8")
        path = _persist_uploaded_config(raw)
        return path, sha256(raw).hexdigest(), None
    except Exception:
        return (
            Path(CONFIG_PATH),
            "default",
            "Couldn't apply the evidence-backed value to this config; showing the built-in case.",
        )


def _resolve_data_source() -> tuple[Path, str, str | None]:
    """Render the data-source controls and return (config_path, cache_key, error).

    Sources in precedence order: a freshly uploaded ``config.yaml``, a config built from the
    no-YAML form, or the built-in case — with an optional evidence-backed overlay on top.
    Anything that fails validation falls back to the built-in case with a readable reason.
    """

    with st.sidebar:
        st.header("Data source")
        st.caption(
            "Explore the built-in checkout case, upload your own config.yaml, or build one "
            "below — all re-parameterize this four-option platform-investment model with your "
            "own costs, volumes, rates, scenarios, and risk guardrails."
        )
        uploaded = st.file_uploader(
            "Upload config.yaml",
            type=["yaml", "yml"],
            help="Start from the template below, edit the costs, rates, and guardrails, then upload.",
        )
        st.download_button(
            "Download config template",
            data=Path(CONFIG_PATH).read_text(encoding="utf-8"),
            file_name="config.template.yaml",
            mime="text/yaml",
        )
        form_result = _render_config_builder()
        candidate = _evidence_candidate_value()
        use_evidence = st.toggle(
            "Use evidence-backed baseline failure rate",
            key="_pduu_use_evidence",
            disabled=candidate is None,
            help=(
                f"Overlay the HM Land Registry search-completion proxy ({candidate:.1%}) onto "
                "baseline_failure_rate, through the validated config path."
                if candidate is not None
                else "No evidence-backed candidate is available."
            ),
        )

    if uploaded is not None:
        raw = uploaded.getvalue()
        try:
            base_path = _persist_uploaded_config(raw)
            validate_config(load_config(base_path))
        except Exception:
            return (
                Path(CONFIG_PATH),
                "default",
                "We couldn't use that config file. Check it against the template — every "
                "required parameter present, distributions well-formed, values in range — "
                "then re-upload. Showing the built-in case for now.",
            )
        base = (base_path, sha256(raw).hexdigest())
    elif form_result is not None:
        if form_result[2] is not None:
            return form_result
        base = (form_result[0], form_result[1])
    else:
        base = (Path(CONFIG_PATH), "default")

    if use_evidence and candidate is not None:
        return _apply_evidence_overlay(base[0])
    return base[0], base[1], None


def _md_inline_to_html(text: str) -> str:
    """Convert the small markdown subset used in brief bullets to safe HTML."""

    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
    return escaped


def build_decision_brief(
    outputs: AppOutputs,
    current_metadata: dict[str, Any],
    selected_policy: DecisionPolicyConfig,
) -> tuple[str, str]:
    """Return a one-page board readout as (markdown, self-contained html).

    Reuses the same recommendation wording and display tables the app renders,
    so an exported brief never drifts from what the stakeholder saw on screen.
    """

    rec_lines = recommendation_lines(outputs.recommendation, outputs.summary, current_metadata)
    verdict = _verdict_line(outputs.recommendation)
    policy_line = f"Guardrails — {format_guardrails(selected_policy)}."
    title = f"Decision brief — {current_metadata.get('scenario_label', 'Published case')}"
    tables = {
        "Option payoffs": summary_display_table(outputs.summary),
        "Guardrail eligibility": eligibility_display_table(outputs.policy_eligibility),
        "What would change the call (policy frontier)": frontier_display_table(
            pd.DataFrame(outputs.policy_frontier["frontier_rows"])
        ),
    }
    disclaimer = (
        "Generated from the Product Decision Under Uncertainty app. The recommendation is an "
        "output of the configured risk preferences, not a substitute for them."
    )

    markdown_lines = [
        f"# {title}",
        "",
        f"**{verdict}**",
        "",
        "## Recommendation",
        *rec_lines,
        "",
        f"- {policy_line}",
    ]
    for heading, table in tables.items():
        markdown_lines.extend(["", f"## {heading}", "", markdown_table(table)])
    markdown_lines.extend(["", f"_{disclaimer}_"])
    markdown = "\n".join(markdown_lines)

    bullets = "".join(
        f"<li>{_md_inline_to_html(line.lstrip('- '))}</li>" for line in rec_lines
    )
    sections = "".join(
        f"<h2>{html.escape(heading)}</h2>{table.to_html(index=False, border=0, classes='t')}"
        for heading, table in tables.items()
    )
    brief_html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "max-width:900px;margin:2rem auto;padding:0 1.25rem;color:#1a1a1a;line-height:1.45}"
        "h1{font-size:1.5rem;margin-bottom:.25rem}h2{font-size:1.05rem;margin-top:1.5rem}"
        "ul{padding-left:1.1rem}table.t{border-collapse:collapse;width:100%;font-size:.85rem;margin-top:.25rem}"
        "table.t th,table.t td{border:1px solid #ddd;padding:6px 9px;text-align:left}"
        "table.t th{background:#f5f5f7}.muted{color:#666;font-size:.8rem;margin-top:1.5rem}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p><strong>{html.escape(verdict)}</strong></p>"
        f"<h2>Recommendation</h2><ul>{bullets}</ul>"
        f"<p>{_md_inline_to_html(policy_line)}</p>{sections}"
        f"<p class='muted'>{html.escape(disclaimer)}</p></body></html>"
    )
    return markdown, brief_html


@st.cache_data(show_spinner=False, max_entries=16)
def _compute_simulation(
    config_path: str,
    n_worlds: int,
    seed: int,
    scenario: str,
) -> dict[str, Any]:
    """Run the threshold-independent Monte Carlo layer (cached, no guardrail args).

    The guardrail thresholds are a policy overlay that never change the simulated
    worlds, so the expensive sampling/analytics live here keyed only on the run
    settings. Moving a guardrail slider then hits this cache instead of re-simulating.
    """

    results = run_simulation(config_path, n_worlds=n_worlds, seed=seed, scenario=scenario)
    return {
        "results": results,
        "summary": summarize_results(results),
        "diagnostics": decision_diagnostics(results),
        "sensitivity": sensitivity_analysis(results),
        "driver_analysis": driver_analysis(results),
        "scenario_results": run_all_scenarios(config_path, n_worlds=n_worlds, seed=seed),
    }


@st.cache_data(show_spinner=False, max_entries=64)
def compute_outputs(
    config_path: str,
    n_worlds: int,
    seed: int,
    scenario: str,
    minimum_p05_value_eur: float,
    maximum_mean_regret_eur: float,
    ev_tolerance_eur: float,
) -> AppOutputs:
    """Assemble app outputs: the cached sim layer plus a cheap guardrail-dependent overlay.

    ``config_path`` is content-addressed for uploads, so it keys the cache on content
    without a separate hash. The three guardrail values feed only the fast policy layer,
    so dragging a threshold re-runs the recommendation/frontier but not the simulation.
    """

    cfg = load_config(config_path)
    analysis = get_analysis_settings(cfg)
    policy = DecisionPolicyConfig(
        name=get_decision_policy(cfg).name,
        minimum_p05_value_eur=float(minimum_p05_value_eur),
        maximum_mean_regret_eur=float(maximum_mean_regret_eur),
        ev_tolerance_eur=float(ev_tolerance_eur),
    )
    sim = _compute_simulation(config_path, int(n_worlds), int(seed), str(scenario))
    results = sim["results"]
    summary = sim["summary"]
    diagnostics = sim["diagnostics"]

    simulation_settings = get_simulation_settings(apply_scenario(cfg, scenario))
    simulation_settings["n_worlds"] = int(n_worlds)
    simulation_settings["scenario"] = str(scenario)

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
        sensitivity=sim["sensitivity"],
        driver_analysis=sim["driver_analysis"],
        scenario_results=sim["scenario_results"],
        recommendation=recommendation,
        policy_eligibility=policy_eligibility,
        payoff_delta=payoff_delta,
        policy_frontier=frontier,
        policy_frontier_grid=frontier_grid,
    )


def load_published_governance() -> PublishedGovernance:
    """Load the published governance summaries plus live manifest counts.

    Deliberately uncached: the freshness verdict compares the committed metadata
    against live fingerprints of the current code/config/lockfile. Caching it for
    the server's lifetime would pin a stale verdict, so regenerating artifacts in a
    side terminal would still show the "do not match" banner until a restart. The
    work here is light (YAML loads + hashing ~20 small files); the Monte Carlo run
    stays cached separately.
    """

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


def _show_governance_warning(governance: PublishedGovernance) -> None:
    """Render the freshness banner for the published governance bundle."""

    warning_message = governance_warning_message(governance)
    if warning_message is None:
        return
    if governance.freshness_status == "stale":
        st.warning(warning_message)
    else:
        st.info(warning_message)


def _render_sidebar(
    cfg: dict[str, Any],
    default_policy: DecisionPolicyConfig,
) -> tuple[int, int, str, DecisionPolicyConfig]:
    """Render the sidebar controls and return the selected run settings and policy."""

    simulation = get_simulation_settings(cfg)
    scenario_metadata = get_scenario_metadata(cfg)
    scenario_names = list(scenario_metadata)

    with st.sidebar:
        st.header("Run settings")
        worlds_value = int(simulation["n_worlds"])
        seed_value = int(get_seed(cfg))
        n_worlds = st.number_input(
            "Simulation runs",
            min_value=min(1_000, worlds_value),
            max_value=max(100_000, worlds_value),
            value=worlds_value,
            step=1_000,
        )
        seed = st.number_input(
            "Seed",
            min_value=min(0, seed_value),
            max_value=max(10_000_000, seed_value),
            value=seed_value,
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
            st.caption(f"{label}: {description}")

        st.markdown("**Policy guardrails**")
        st.caption(
            "These encode the team's risk tolerance. Move them to see what would "
            "change the call — the recommendation, charts, and frontier update live."
        )
        floor_value = float(default_policy.minimum_p05_value_eur)
        regret_value = float(default_policy.maximum_mean_regret_eur)
        tolerance_value = float(default_policy.ev_tolerance_eur)
        floor_lo, floor_hi = guardrail_widget_bounds(floor_value, -2_000_000.0, 0.0)
        regret_lo, regret_hi = guardrail_widget_bounds(regret_value, 0.0, 2_000_000.0)
        tolerance_lo, tolerance_hi = guardrail_widget_bounds(tolerance_value, 0.0, 1_000_000.0)
        minimum_p05_value_eur = st.number_input(
            "Downside floor — keep P05 at or above (EUR)",
            min_value=floor_lo,
            max_value=floor_hi,
            value=floor_value,
            step=25_000.0,
        )
        maximum_mean_regret_eur = st.number_input(
            "Regret cap — keep mean regret at or below (EUR)",
            min_value=regret_lo,
            max_value=regret_hi,
            value=regret_value,
            step=25_000.0,
        )
        ev_tolerance_eur = st.number_input(
            "EV tolerance band (EUR)",
            min_value=tolerance_lo,
            max_value=tolerance_hi,
            value=tolerance_value,
            step=10_000.0,
        )

    selected_policy = DecisionPolicyConfig(
        name=default_policy.name,
        minimum_p05_value_eur=float(minimum_p05_value_eur),
        maximum_mean_regret_eur=float(maximum_mean_regret_eur),
        ev_tolerance_eur=float(ev_tolerance_eur),
    )
    return int(n_worlds), int(seed), str(scenario), selected_policy


def guardrail_widget_bounds(value: float, floor: float, ceil: float) -> tuple[float, float]:
    """Widen ``[floor, ceil]`` so a validated config value is always inside the slider.

    ``validate_config`` accepts policy values beyond the default widget ranges (e.g. a
    positive downside floor, or an unusually large regret cap), so seeding a
    ``number_input`` with such a value would make Streamlit raise outside the upload
    try/except. Widening the bounds to include the value removes that crash class for any
    config the engine accepts, from any source, without clamping (mis-stating) the value.
    """

    return (min(floor, value), max(ceil, value))


def format_guardrails(policy: DecisionPolicyConfig) -> str:
    """Return the shared 'floor / cap / tolerance' clause for the caption and the brief.

    One formatter keeps the on-screen "Active guardrails" caption and the exported brief
    from ever drifting apart.
    """

    return (
        f"downside floor {format_eur(policy.minimum_p05_value_eur)}, "
        f"regret cap {format_eur(policy.maximum_mean_regret_eur)}, "
        f"EV tolerance {format_eur(policy.ev_tolerance_eur)}"
    )


def _verdict_line(recommendation: RecommendationResult) -> str:
    """One-sentence plain-English verdict for the hero and the exported brief."""

    label = labeled_option(recommendation.selected_option)
    if recommendation.selected_reason_type == "guardrails_relaxed_highest_ev":
        return f"We recommend {label} — the least-bad option, because no option clears both safety bars."
    if recommendation.selected_reason_type == "only_option_passing_guardrails":
        return f"We recommend {label} — the only option that clears both safety bars."
    if recommendation.selected_reason_type == "ev_tolerance_override":
        return f"We recommend {label} — within the EV-tolerance band it wins the lower-regret tie-break."
    return f"We recommend {label} — it clears both guardrails and leads on expected value."


def _render_run_summary(
    cfg: dict[str, Any],
    *,
    scenario: str,
    seed: int,
    n_worlds: int,
) -> None:
    """Render the current run context above the main analytical sections."""

    simulation = get_simulation_settings(cfg)
    scenario_metadata = get_scenario_metadata(cfg)
    published_scenario = str(simulation["scenario"])
    run_context = build_run_context(
        selected_settings={
            "scenario": scenario,
            "scenario_label": str(scenario_metadata[scenario]["label"]),
            "seed": int(seed),
            "n_worlds": int(n_worlds),
        },
        published_settings={
            "scenario": published_scenario,
            "scenario_label": str(scenario_metadata[published_scenario]["label"]),
            "seed": int(get_seed(cfg)),
            "n_worlds": int(simulation["n_worlds"]),
        },
    )
    st.caption(published_case_caption())
    st.subheader(str(run_context["heading"]))
    st.caption(str(run_context["detail"]))
    if not bool(run_context["matches_published"]):
        st.caption(str(run_context["note"]))


def _build_current_metadata(
    cfg: dict[str, Any],
    *,
    n_worlds: int,
    seed: int,
    scenario: str,
    scenario_metadata: dict[str, dict[str, str]],
    outputs: AppOutputs,
) -> dict[str, Any]:
    """Build the metadata payload used in the recommendation summary and brief."""

    return {
        "seed": int(seed),
        "n_worlds": int(n_worlds),
        "scenario": str(scenario),
        "scenario_label": str(scenario_metadata.get(scenario, {}).get("label", scenario)),
        "annual_volume": int(outputs.simulation_settings["annual_volume"]),
        "time_horizon_years": int(outputs.simulation_settings["time_horizon_years"]),
        "discount_rate_annual": float(outputs.simulation_settings["discount_rate_annual"]),
        "declared_model_version": get_declared_model_version(cfg),
    }


def _render_hero_section(outputs: AppOutputs, current_metadata: dict[str, Any]) -> None:
    """Render the recommendation summary and payoff distribution section."""

    render_executive_metric_cards(outputs)
    hero_left, hero_right = st.columns([3, 2])
    with hero_left:
        st.subheader("Recommendation")
        st.markdown(f"### {_verdict_line(outputs.recommendation)}")
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


def _render_policy_section(
    outputs: AppOutputs,
    decision_policy: Any,
    scenario_metadata: dict[str, dict[str, str]],
) -> None:
    """Render the policy-defining charts and the scenario comparison."""

    st.subheader("Guardrail eligibility")
    render_section_copy(
        "An option stays eligible only if its downside P05 stays above the floor and its mean regret stays below the cap. This is the policy-defining filter, not just a descriptive chart."
    )
    st.caption(
        f"Active guardrails — {format_guardrails(decision_policy)}. Adjust them in the sidebar."
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


def _render_analysis_section(
    outputs: AppOutputs,
    analysis: Any,
    published_governance: PublishedGovernance,
) -> None:
    """Render driver analysis, descriptive diagnostics, and stability checks."""

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
    st.caption(
        driver_analysis_interpretation_note(outputs.driver_analysis, selected_option)
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
    st.subheader("Published-case stability")
    render_section_copy(
        "This is a sampling-robustness check on the published configuration. Stable frequency and narrower metric ranges mean the recommendation is less sensitive to Monte Carlo noise."
    )
    st.caption(stability_caption)
    render_stability_summary_cards(stability)
    render_plotly_figure(create_stability_chart(published_governance.stability_runs))
    st.divider()


def _render_reference_section(
    outputs: AppOutputs,
    scenario_metadata: dict[str, dict[str, str]],
    published_governance: PublishedGovernance,
    selected_policy: DecisionPolicyConfig,
    current_metadata: dict[str, Any],
    is_custom_config: bool,
) -> None:
    """Render the reference tabs with provenance, downloads, and raw tables."""

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
        if is_custom_config:
            st.warning(
                "These provenance, evidence, and freshness details describe the built-in "
                "published case. Your uploaded run is exploratory and is not covered by "
                "these governance artifacts."
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
        st.subheader("Board readout")
        render_section_copy(
            "Export a one-page decision brief for a stakeholder review. It carries the recommendation, the active guardrails, the option payoffs, and what would change the call — so the deck never drifts from what the app showed."
        )
        brief_markdown, brief_html = build_decision_brief(
            outputs,
            current_metadata,
            selected_policy,
        )
        brief_left, brief_right = st.columns(2)
        with brief_left:
            st.download_button(
                "Decision brief (HTML)",
                data=brief_html,
                file_name="decision_brief.html",
                mime="text/html",
                help="Open in a browser and print to PDF for a slide-ready one-pager.",
            )
        with brief_right:
            st.download_button(
                "Decision brief (Markdown)",
                data=brief_markdown,
                file_name="decision_brief.md",
                mime="text/markdown",
                help="Paste into a doc, ticket, or wiki.",
            )

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


def render_app() -> None:
    """Entry point for the Streamlit interface."""

    st.set_page_config(page_title="Product Decision Under Uncertainty", layout="wide")
    inject_app_styles()
    st.title("Product Decision Under Uncertainty")

    config_path, config_key, config_error = _resolve_data_source()
    is_custom_config = config_key != "default"
    cfg = load_config(config_path)
    analysis = get_analysis_settings(cfg)
    default_policy = get_decision_policy(cfg)
    scenario_metadata = get_scenario_metadata(cfg)
    published_governance = load_published_governance()

    _show_governance_warning(published_governance)
    if config_error is not None:
        st.warning(config_error)
    n_worlds, seed, scenario, selected_policy = _render_sidebar(cfg, default_policy)
    _render_run_summary(cfg, scenario=scenario, seed=seed, n_worlds=n_worlds)
    if bool(st.session_state.get("_pduu_use_evidence")):
        st.info(
            "Using the evidence-backed baseline failure rate (HM Land Registry proxy). Every "
            "other input remains an elicited assumption; this run diverges from the published "
            "elicited case."
        )
    elif is_custom_config:
        st.info(
            "Running your uploaded or built config. Every input is an elicited assumption you "
            "provided; the Provenance & evidence and freshness panels describe the built-in "
            "published case, not this run."
        )
    else:
        st.caption(
            "All model inputs are elicited assumptions with documented provenance, "
            "not fitted to private data."
        )

    outputs = compute_outputs(
        str(config_path),
        n_worlds,
        seed,
        scenario,
        selected_policy.minimum_p05_value_eur,
        selected_policy.maximum_mean_regret_eur,
        selected_policy.ev_tolerance_eur,
    )
    current_metadata = _build_current_metadata(
        cfg,
        n_worlds=n_worlds,
        seed=seed,
        scenario=scenario,
        scenario_metadata=scenario_metadata,
        outputs=outputs,
    )

    _render_hero_section(outputs, current_metadata)
    _render_policy_section(outputs, selected_policy, scenario_metadata)
    _render_analysis_section(outputs, analysis, published_governance)
    _render_reference_section(
        outputs,
        scenario_metadata,
        published_governance,
        selected_policy,
        current_metadata,
        is_custom_config,
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file and return the parsed dict."""

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    render_app()
