"""Tests for the interactive sidebar wiring, self-serve config, and decision brief."""

from __future__ import annotations

import inspect

from app import (
    _compute_simulation,
    _evidence_candidate_value,
    _overlay_evidence,
    _verdict_line,
    build_config_from_form,
    build_decision_brief,
    compute_outputs,
    format_guardrails,
    guardrail_widget_bounds,
)
from simulator.analytics import (
    decision_diagnostics,
    driver_analysis,
    sensitivity_analysis,
    summarize_results,
)
from simulator.app_state import AppOutputs
from simulator.config import (
    DecisionPolicyConfig,
    apply_scenario,
    get_analysis_settings,
    get_decision_policy,
    get_simulation_settings,
    load_config,
    validate_config,
)
from simulator.output_utils import labeled_option
from simulator.policy import (
    build_policy_eligibility_table,
    payoff_delta_diagnostic,
    policy_frontier_analysis,
    policy_frontier_grid,
    select_recommendation,
)
from simulator.project_paths import CONFIG_PATH
from simulator.simulation import run_all_scenarios, run_simulation


def _build_outputs(
    *,
    scenario: str = "mid_range_pressure",
    n_worlds: int = 3000,
    seed: int = 42,
    policy: DecisionPolicyConfig | None = None,
) -> AppOutputs:
    """Assemble an AppOutputs the same way the cached app path does, for tests."""

    cfg = load_config(CONFIG_PATH)
    analysis = get_analysis_settings(cfg)
    policy = policy or get_decision_policy(cfg)
    results = run_simulation(CONFIG_PATH, n_worlds=n_worlds, seed=seed, scenario=scenario)
    summary = summarize_results(results)
    diagnostics = decision_diagnostics(results)
    recommendation = select_recommendation(summary, diagnostics, policy)
    simulation_settings = get_simulation_settings(apply_scenario(cfg, scenario))
    simulation_settings["n_worlds"] = n_worlds
    simulation_settings["scenario"] = scenario
    return AppOutputs(
        simulation_settings=simulation_settings,
        results=results,
        summary=summary,
        diagnostics=diagnostics,
        sensitivity=sensitivity_analysis(results),
        driver_analysis=driver_analysis(results),
        scenario_results=run_all_scenarios(CONFIG_PATH, n_worlds=n_worlds, seed=seed),
        recommendation=recommendation,
        policy_eligibility=build_policy_eligibility_table(summary, diagnostics, policy),
        payoff_delta=payoff_delta_diagnostic(results, recommendation, analysis),
        policy_frontier=policy_frontier_analysis(summary, diagnostics, policy, recommendation),
        policy_frontier_grid=policy_frontier_grid(summary, diagnostics, policy, recommendation),
    )


def _metadata(scenario_label: str = "Mid-range pressure") -> dict[str, object]:
    return {
        "seed": 42,
        "n_worlds": 3000,
        "scenario": "mid_range_pressure",
        "scenario_label": scenario_label,
        "annual_volume": 250000,
        "time_horizon_years": 2,
        "discount_rate_annual": 0.08,
        "declared_model_version": "5.0.0",
    }


def test_compute_outputs_exposes_thresholds_and_splits_the_cache() -> None:
    """compute_outputs takes the guardrails; the cached sim layer must not (C2 cache split)."""

    params = inspect.signature(compute_outputs).parameters
    for name in (
        "config_path",
        "minimum_p05_value_eur",
        "maximum_mean_regret_eur",
        "ev_tolerance_eur",
    ):
        assert name in params
    assert "config_hash" not in params  # P1: content-addressed path keys the cache

    sim_params = set(inspect.signature(_compute_simulation).parameters)
    assert sim_params == {"config_path", "n_worlds", "seed", "scenario"}
    for guardrail in ("minimum_p05_value_eur", "maximum_mean_regret_eur", "ev_tolerance_eur"):
        assert guardrail not in sim_params  # moving a guardrail must not re-simulate


def test_loosening_the_regret_cap_makes_do_nothing_eligible() -> None:
    """The sidebar regret-cap control should change which options pass the policy."""

    cfg = load_config(CONFIG_PATH)
    base = get_decision_policy(cfg)
    results = run_simulation(CONFIG_PATH, n_worlds=6000, seed=42, scenario="mid_range_pressure")
    summary = summarize_results(results)
    diagnostics = decision_diagnostics(results)

    strict = build_policy_eligibility_table(summary, diagnostics, base).set_index("option")
    loosened_policy = DecisionPolicyConfig(
        name=base.name,
        minimum_p05_value_eur=base.minimum_p05_value_eur,
        maximum_mean_regret_eur=800_000.0,
        ev_tolerance_eur=base.ev_tolerance_eur,
    )
    loosened = build_policy_eligibility_table(summary, diagnostics, loosened_policy).set_index(
        "option"
    )

    assert not bool(strict.loc["do_nothing", "eligible"])
    assert bool(loosened.loc["do_nothing", "eligible"])


def test_decision_brief_carries_recommendation_scenario_and_tables() -> None:
    """The exported brief should mirror the on-screen recommendation and tables."""

    outputs = _build_outputs()
    policy = get_decision_policy(load_config(CONFIG_PATH))
    markdown, brief_html = build_decision_brief(outputs, _metadata(), policy)

    assert markdown.startswith("# Decision brief")
    assert "## Recommendation" in markdown
    assert "Mid-range pressure" in markdown
    assert "Guardrails" in markdown
    assert "| Option |" in markdown

    assert "<table" in brief_html
    assert "Mid-range pressure" in brief_html
    assert "<strong>" in brief_html  # markdown bold survived into the HTML brief


def test_decision_brief_html_escapes_scenario_label() -> None:
    """A hostile scenario label must not inject raw markup into the HTML brief."""

    outputs = _build_outputs()
    policy = get_decision_policy(load_config(CONFIG_PATH))
    _markdown, brief_html = build_decision_brief(
        outputs,
        _metadata(scenario_label="<script>x</script>"),
        policy,
    )

    assert "<script>x</script>" not in brief_html
    assert "&lt;script&gt;" in brief_html


def test_guardrail_widget_bounds_widen_to_include_out_of_range_values() -> None:
    """A validated config value beyond the default slider range must not crash it (C1)."""

    # validate_config accepts a positive downside floor; default ceiling is 0.0.
    lo, hi = guardrail_widget_bounds(100_000.0, -2_000_000.0, 0.0)
    assert lo <= 100_000.0 <= hi
    # a regret cap above the default 2M ceiling.
    lo, hi = guardrail_widget_bounds(3_000_000.0, 0.0, 2_000_000.0)
    assert lo <= 3_000_000.0 <= hi
    # an in-range value leaves the default bounds untouched.
    assert guardrail_widget_bounds(-300_000.0, -2_000_000.0, 0.0) == (-2_000_000.0, 0.0)


def test_format_guardrails_and_verdict_are_plain_and_shared() -> None:
    """One guardrail formatter for caption+brief (P2); verdict names the option (A5)."""

    policy = get_decision_policy(load_config(CONFIG_PATH))
    clause = format_guardrails(policy)
    assert "downside floor" in clause
    assert "regret cap" in clause
    assert "EV tolerance" in clause

    outputs = _build_outputs()
    verdict = _verdict_line(outputs.recommendation)
    assert verdict.startswith("We recommend")
    assert labeled_option(outputs.recommendation.selected_option) in verdict


def test_build_config_from_form_produces_a_valid_config() -> None:
    """The no-YAML form must overlay levers into a config the engine accepts (A2)."""

    base = load_config(CONFIG_PATH)
    values = {
        "annual_volume": 400_000,
        "time_horizon_years": 3,
        "discount_rate_annual": 0.10,
        "baseline_failure_rate": 0.12,
        "do_nothing_drift_cost_eur": 80_000.0,
        "stabilize_core_upfront_cost_eur": 900_000.0,
        "feature_extension_upfront_cost_eur": 500_000.0,
        "new_capability_upfront_cost_eur": 1_300_000.0,
    }
    built = build_config_from_form(base, values)
    validate_config(built)  # raises if the form produced an invalid config

    assert built["simulation"]["annual_volume"] == 400_000
    assert built["params"]["baseline_failure_rate"]["mode"] == 0.12
    low = built["params"]["baseline_failure_rate"]["low"]
    high = built["params"]["baseline_failure_rate"]["high"]
    assert low <= 0.12 <= high
    assert base["simulation"]["annual_volume"] != 400_000  # deep copy left the base untouched


def test_evidence_overlay_pins_baseline_and_stays_valid() -> None:
    """The evidence toggle must pin baseline_failure_rate via a validated config (A4)."""

    value = _evidence_candidate_value()
    assert value is not None
    assert 0.0 < value < 1.0  # the HM Land Registry proxy candidate is available

    overlaid = _overlay_evidence(load_config(CONFIG_PATH), value)
    validate_config(overlaid)  # raises if the overlay broke the config
    spec = overlaid["params"]["baseline_failure_rate"]
    assert spec["dist"] == "tri"  # a distribution, not a constant (stays a copula anchor)
    assert spec["low"] <= round(value, 6) <= spec["high"]
