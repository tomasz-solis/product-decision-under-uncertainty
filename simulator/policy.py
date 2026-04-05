"""Decision-policy helpers for recommendation selection and frontier analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict

import numpy as np
import pandas as pd

from simulator.config import AnalysisConfig, DecisionPolicyConfig
from simulator.output_utils import labeled_option

FAILURE_REASON_LABELS = {
    "misses_downside_floor": "fails the downside floor",
    "misses_regret_cap": "fails the regret cap",
    "misses_downside_floor_and_regret_cap": "fails both guardrails",
}

FULL_FRONTIER_SWITCH_LABELS = {
    "exact_match": "exact match",
    "grid_bracket": "grid bracket",
    "no_switch_observed": "no switch observed",
}

RUNNER_UP_FRONTIER_STATUS_LABELS = {
    "already_non_binding": "already non-binding",
    "not_binding": "not binding",
    "not_needed": "not needed",
    "switch_observed": "runner-up threshold reached",
}


@dataclass(frozen=True)
class RecommendationResult:
    """Structured recommendation output for one summary/diagnostic pair."""

    policy_name: str
    selected_option: str
    runner_up: str
    selection_reason: str
    binding_constraint: str
    selection_margin_eur: float
    selected_mean_value_eur: float
    runner_up_mean_value_eur: float
    selected_p05_value_eur: float
    selected_mean_regret_eur: float
    eligible_option_count: int
    eligible_options: tuple[str, ...]
    selected_reason_type: str
    runner_up_failure_reason: str | None
    selected_downside_slack_eur: float
    selected_regret_slack_eur: float
    runner_up_downside_slack_eur: float
    runner_up_regret_slack_eur: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary representation."""

        return asdict(self)


class PayoffDeltaRow(TypedDict):
    """One parameter row from the selected-vs-runner-up payoff diagnostic."""

    parameter: str
    delta_spearman_corr: float
    sampled_min_value: float
    sampled_max_value: float
    interpretation_note: str


class PayoffDeltaDiagnostic(TypedDict):
    """Typed selected-vs-runner-up payoff diagnostic."""

    selected_option: str
    runner_up: str
    mean_delta_eur: float
    p05_delta_eur: float
    win_rate_vs_runner_up: float
    delta_rows: list[PayoffDeltaRow]


class PolicyFrontierRow(TypedDict):
    """One full-option frontier row for a single policy threshold."""

    threshold_name: str
    threshold_label: str
    unit: str
    current_value: float
    switching_value: float | None
    first_switching_option: str | None
    switch_type: str
    switch_direction: str | None
    all_options_considered: bool
    interpretation_note: str


class RunnerUpFrontierRow(TypedDict):
    """One selected-vs-runner-up threshold comparison row."""

    threshold_name: str
    threshold_label: str
    unit: str
    current_value: float
    switching_value: float | None
    status: str
    interpretation_note: str


class PolicyFrontierResult(TypedDict):
    """Typed policy-frontier output for reporting and the app."""

    selected_option: str
    baseline_selected_option: str
    runner_up: str
    frontier_rows: list[PolicyFrontierRow]
    runner_up_comparison_rows: list[RunnerUpFrontierRow]


@dataclass(frozen=True)
class _FrontierObservation:
    """One candidate threshold that flips the full recommendation."""

    boundary_value: float
    observed_value: float
    switching_option: str
    switch_type: str


def build_policy_eligibility_table(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    policy: DecisionPolicyConfig,
) -> pd.DataFrame:
    """Return one row per option showing guardrail pass/fail status."""

    scored = summary.merge(
        diagnostics[["option", "win_rate", "mean_regret_eur", "p95_regret_eur"]],
        on="option",
        how="left",
    ).copy()
    scored["downside_slack_eur"] = scored["p05_value_eur"] - policy.minimum_p05_value_eur
    scored["regret_slack_eur"] = policy.maximum_mean_regret_eur - scored["mean_regret_eur"]
    scored["passes_downside_floor"] = scored["downside_slack_eur"] >= 0.0
    scored["passes_regret_cap"] = scored["regret_slack_eur"] >= 0.0
    scored["eligible"] = scored["passes_downside_floor"] & scored["passes_regret_cap"]
    scored["failure_reason"] = scored.apply(_failure_reason, axis=1)
    return scored.sort_values(
        ["mean_value_eur", "mean_regret_eur"], ascending=[False, True]
    ).reset_index(drop=True)


def select_recommendation(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    policy: DecisionPolicyConfig,
) -> RecommendationResult:
    """Select a winner from summary and diagnostics using an explicit policy."""

    scored = build_policy_eligibility_table(summary, diagnostics, policy)
    eligible = scored.loc[scored["eligible"]].copy()

    if eligible.empty:
        ranked = scored.sort_values(["mean_value_eur", "mean_regret_eur"], ascending=[False, True])
        selected = ranked.iloc[0]
        runner_up = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
        binding_constraint = "guardrails_relaxed"
        selected_reason_type = "guardrails_relaxed_highest_ev"
    elif len(eligible) == 1:
        selected = eligible.iloc[0]
        runner_candidates = scored.loc[scored["option"] != selected["option"]].sort_values(
            ["mean_value_eur", "mean_regret_eur"],
            ascending=[False, True],
        )
        runner_up = runner_candidates.iloc[0] if not runner_candidates.empty else selected
        binding_constraint = "only_option_passing_guardrails"
        selected_reason_type = "only_option_passing_guardrails"
    else:
        eligible = eligible.sort_values(
            ["mean_value_eur", "mean_regret_eur"], ascending=[False, True]
        )
        ev_leader = eligible.iloc[0]
        tolerance_band = eligible.loc[
            eligible["mean_value_eur"] >= ev_leader["mean_value_eur"] - policy.ev_tolerance_eur
        ].copy()
        policy_sorted = tolerance_band.sort_values(
            ["mean_regret_eur", "mean_value_eur"],
            ascending=[True, False],
        )
        selected = policy_sorted.iloc[0]
        if str(selected["option"]) != str(ev_leader["option"]):
            binding_constraint = "ev_tolerance_then_regret"
            selected_reason_type = "ev_tolerance_override"
        else:
            binding_constraint = "highest_ev_eligible"
            selected_reason_type = "highest_ev_eligible"

        runner_candidates = scored.loc[scored["option"] != selected["option"]].sort_values(
            ["mean_value_eur", "mean_regret_eur"],
            ascending=[False, True],
        )
        runner_up = runner_candidates.iloc[0] if not runner_candidates.empty else selected

    eligible_options = tuple(str(option) for option in eligible["option"].tolist())
    reason = _selection_reason(
        selected=selected,
        runner_up=runner_up,
        policy=policy,
        selected_reason_type=selected_reason_type,
        eligible_option_count=len(eligible),
    )

    return RecommendationResult(
        policy_name=policy.name,
        selected_option=str(selected["option"]),
        runner_up=str(runner_up["option"]),
        selection_reason=reason,
        binding_constraint=binding_constraint,
        selection_margin_eur=float(selected["mean_value_eur"] - runner_up["mean_value_eur"]),
        selected_mean_value_eur=float(selected["mean_value_eur"]),
        runner_up_mean_value_eur=float(runner_up["mean_value_eur"]),
        selected_p05_value_eur=float(selected["p05_value_eur"]),
        selected_mean_regret_eur=float(selected["mean_regret_eur"]),
        eligible_option_count=int(len(eligible)),
        eligible_options=eligible_options,
        selected_reason_type=selected_reason_type,
        runner_up_failure_reason=_as_optional_reason(runner_up["failure_reason"]),
        selected_downside_slack_eur=float(selected["downside_slack_eur"]),
        selected_regret_slack_eur=float(selected["regret_slack_eur"]),
        runner_up_downside_slack_eur=float(runner_up["downside_slack_eur"]),
        runner_up_regret_slack_eur=float(runner_up["regret_slack_eur"]),
    )


def payoff_delta_diagnostic(
    results: pd.DataFrame,
    recommendation: RecommendationResult,
    analysis: AnalysisConfig,
) -> PayoffDeltaDiagnostic:
    """Summarize selected-versus-runner-up payoff associations honestly."""

    from simulator.analytics import decision_delta_sensitivity

    delta = results[recommendation.selected_option] - results[recommendation.runner_up]
    delta_sensitivity = decision_delta_sensitivity(
        results,
        selected_option=recommendation.selected_option,
        runner_up=recommendation.runner_up,
    )
    candidates = (
        delta_sensitivity.assign(
            abs_delta_spearman=lambda frame: frame["delta_spearman_corr"].abs()
        )
        .loc[
            lambda frame: (
                frame["abs_delta_spearman"]
                >= analysis.sensitivity_materiality_threshold_abs_spearman
            )
        ]
        .head(analysis.decision_boundary_top_parameters)
    )

    rows: list[PayoffDeltaRow] = []
    for _, row in candidates.iterrows():
        parameter = str(row["parameter"])
        rows.append(
            {
                "parameter": parameter,
                "delta_spearman_corr": float(row["delta_spearman_corr"]),
                "sampled_min_value": float(results[parameter].min()),
                "sampled_max_value": float(results[parameter].max()),
                "interpretation_note": (
                    "Descriptive rank association with the selected-minus-runner-up "
                    "payoff delta inside the sampled worlds."
                ),
            }
        )

    return {
        "selected_option": recommendation.selected_option,
        "runner_up": recommendation.runner_up,
        "mean_delta_eur": float(delta.mean()),
        "p05_delta_eur": float(np.quantile(delta, 0.05)),
        "win_rate_vs_runner_up": float(np.mean(delta > 0.0)),
        "delta_rows": rows,
    }


def policy_frontier_analysis(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    policy: DecisionPolicyConfig,
    recommendation: RecommendationResult,
) -> PolicyFrontierResult:
    """Return full-option switching frontiers plus a runner-up comparison table."""

    scored = build_policy_eligibility_table(summary, diagnostics, policy).set_index("option")
    selected = scored.loc[recommendation.selected_option]
    runner_up = scored.loc[recommendation.runner_up]

    frontier_rows = [
        policy_frontier_first_switch(
            summary,
            diagnostics,
            policy,
            recommendation,
            threshold_name="minimum_p05_value_eur",
            threshold_label="Downside floor",
        ),
        policy_frontier_first_switch(
            summary,
            diagnostics,
            policy,
            recommendation,
            threshold_name="maximum_mean_regret_eur",
            threshold_label="Regret cap",
        ),
        policy_frontier_first_switch(
            summary,
            diagnostics,
            policy,
            recommendation,
            threshold_name="ev_tolerance_eur",
            threshold_label="EV tolerance",
        ),
    ]
    runner_up_rows = [
        _downside_frontier_row(policy, runner_up, recommendation),
        _regret_frontier_row(policy, runner_up, recommendation),
        _tolerance_frontier_row(policy, selected, runner_up, recommendation),
    ]
    return {
        "selected_option": recommendation.selected_option,
        "baseline_selected_option": recommendation.selected_option,
        "runner_up": recommendation.runner_up,
        "frontier_rows": frontier_rows,
        "runner_up_comparison_rows": runner_up_rows,
    }


def policy_frontier_first_switch(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    policy: DecisionPolicyConfig,
    recommendation: RecommendationResult,
    threshold_name: str,
    threshold_label: str,
) -> PolicyFrontierRow:
    """Return the first threshold change that flips the recommendation."""

    current_value = _policy_value(policy, threshold_name)
    observation = _nearest_switch_observation(
        summary=summary,
        diagnostics=diagnostics,
        policy=policy,
        threshold_name=threshold_name,
        current_value=current_value,
        baseline_selected_option=recommendation.selected_option,
    )
    if observation is None:
        return _frontier_row(
            threshold_name=threshold_name,
            threshold_label=threshold_label,
            unit="eur",
            current_value=current_value,
            switching_value=None,
            first_switching_option=None,
            switch_type="no_switch_observed",
            switch_direction=None,
            all_options_considered=True,
            interpretation_note=(
                "Re-evaluating the full option set across the tested threshold domain "
                "did not change the selected option."
            ),
        )

    switch_direction = _switch_direction(threshold_name, current_value, observation.observed_value)
    direction_phrase = _direction_phrase(threshold_name, switch_direction)
    selected_label = labeled_option(recommendation.selected_option)
    switching_label = labeled_option(observation.switching_option)
    if observation.switch_type == "exact_match":
        note = (
            f"Full-option sweep: {selected_label} switches to {switching_label} when the "
            f"{threshold_label.lower()} moves {direction_phrase} to "
            f"{observation.boundary_value:,.2f} EUR."
        )
    else:
        note = (
            f"Full-option sweep: {selected_label} switches to {switching_label} once the "
            f"{threshold_label.lower()} moves {direction_phrase} past "
            f"{observation.boundary_value:,.2f} EUR."
        )
    return _frontier_row(
        threshold_name=threshold_name,
        threshold_label=threshold_label,
        unit="eur",
        current_value=current_value,
        switching_value=observation.boundary_value,
        first_switching_option=observation.switching_option,
        switch_type=observation.switch_type,
        switch_direction=switch_direction,
        all_options_considered=True,
        interpretation_note=note,
    )


def policy_frontier_grid(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    policy: DecisionPolicyConfig,
    recommendation: RecommendationResult,
) -> pd.DataFrame:
    """Evaluate the policy on valid threshold grids around the switching regions."""

    frontier = policy_frontier_analysis(summary, diagnostics, policy, recommendation)
    rows: list[dict[str, object]] = []
    for frontier_row in frontier["frontier_rows"]:
        for tested_value in _grid_values(frontier_row):
            varied_policy = _policy_with_value(policy, frontier_row["threshold_name"], tested_value)
            varied_result = select_recommendation(summary, diagnostics, varied_policy)
            rows.append(
                {
                    "threshold_name": frontier_row["threshold_name"],
                    "threshold_label": frontier_row["threshold_label"],
                    "tested_value": float(tested_value),
                    "selected_option": varied_result.selected_option,
                    "runner_up": varied_result.runner_up,
                    "binding_constraint": varied_result.binding_constraint,
                    "eligible_option_count": varied_result.eligible_option_count,
                    "baseline_selected_option": recommendation.selected_option,
                    "switch_from_baseline": varied_result.selected_option
                    != recommendation.selected_option,
                }
            )
    return pd.DataFrame(rows).sort_values(["threshold_name", "tested_value"]).reset_index(drop=True)


def decision_boundary_analysis(
    results: pd.DataFrame,
    recommendation: RecommendationResult,
    analysis: AnalysisConfig,
) -> PayoffDeltaDiagnostic:
    """Return the renamed payoff-delta diagnostic for backward compatibility."""

    return payoff_delta_diagnostic(results, recommendation, analysis)


def _nearest_switch_observation(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    policy: DecisionPolicyConfig,
    threshold_name: str,
    current_value: float,
    baseline_selected_option: str,
) -> _FrontierObservation | None:
    """Return the nearest exact or bracketed threshold change that flips selection."""

    observations: list[tuple[float, float, _FrontierObservation]] = []
    for boundary_value in _boundary_candidates(summary, diagnostics, policy, threshold_name):
        exact_result = _evaluate_threshold(
            summary,
            diagnostics,
            policy,
            threshold_name,
            boundary_value,
        )
        if exact_result.selected_option != baseline_selected_option:
            observations.append(
                (
                    abs(boundary_value - current_value),
                    boundary_value,
                    _FrontierObservation(
                        boundary_value=float(boundary_value),
                        observed_value=float(boundary_value),
                        switching_option=exact_result.selected_option,
                        switch_type="exact_match",
                    ),
                )
            )
            continue

        for nudged_value in _boundary_neighbors(threshold_name, boundary_value):
            nudged_result = _evaluate_threshold(
                summary,
                diagnostics,
                policy,
                threshold_name,
                nudged_value,
            )
            if nudged_result.selected_option == baseline_selected_option:
                continue
            observations.append(
                (
                    abs(nudged_value - current_value),
                    boundary_value,
                    _FrontierObservation(
                        boundary_value=float(boundary_value),
                        observed_value=float(nudged_value),
                        switching_option=nudged_result.selected_option,
                        switch_type="grid_bracket",
                    ),
                )
            )
            break

    if not observations:
        return None

    observations.sort(key=lambda item: (item[0], item[1]))
    return observations[0][2]


def _boundary_candidates(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    policy: DecisionPolicyConfig,
    threshold_name: str,
) -> list[float]:
    """Return exact threshold boundaries that can alter the policy state."""

    if threshold_name == "minimum_p05_value_eur":
        values = summary["p05_value_eur"].astype(float).tolist()
        return sorted({float(value) for value in values})

    if threshold_name == "maximum_mean_regret_eur":
        values = diagnostics["mean_regret_eur"].astype(float).tolist()
        return sorted({max(0.0, float(value)) for value in values})

    scored = build_policy_eligibility_table(summary, diagnostics, policy)
    eligible = scored.loc[scored["eligible"]].copy()
    if len(eligible) <= 1:
        return []
    ev_leader = eligible.sort_values(
        ["mean_value_eur", "mean_regret_eur"],
        ascending=[False, True],
    ).iloc[0]
    values = [
        max(0.0, float(ev_leader["mean_value_eur"] - row["mean_value_eur"]))
        for _, row in eligible.iterrows()
        if str(row["option"]) != str(ev_leader["option"])
    ]
    return sorted({float(value) for value in values})


def _boundary_neighbors(threshold_name: str, boundary_value: float) -> list[float]:
    """Return valid floating-point neighbors around one threshold boundary."""

    neighbors = [
        float(np.nextafter(boundary_value, -np.inf)),
        float(np.nextafter(boundary_value, np.inf)),
    ]
    if threshold_name in {"maximum_mean_regret_eur", "ev_tolerance_eur"}:
        neighbors = [value for value in neighbors if value >= 0.0]
    return [value for value in neighbors if np.isfinite(value)]


def _evaluate_threshold(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    policy: DecisionPolicyConfig,
    threshold_name: str,
    tested_value: float,
) -> RecommendationResult:
    """Return the recommendation after varying one threshold."""

    varied_policy = _policy_with_value(policy, threshold_name, tested_value)
    return select_recommendation(summary, diagnostics, varied_policy)


def _selection_reason(
    selected: pd.Series,
    runner_up: pd.Series,
    policy: DecisionPolicyConfig,
    selected_reason_type: str,
    eligible_option_count: int,
) -> str:
    """Return plain-English recommendation reasoning tied to the actual policy branch."""

    selected_label = labeled_option(str(selected["option"]))
    runner_label = labeled_option(str(runner_up["option"]))
    if selected_reason_type == "only_option_passing_guardrails":
        failure_detail = _failure_detail(runner_up)
        value_direction = (
            "higher" if runner_up["mean_value_eur"] > selected["mean_value_eur"] else "lower"
        )
        return (
            f"{selected_label} is the only option that passes both guardrails. "
            f"{runner_label} has {value_direction} expected value, but {failure_detail}."
        )
    if selected_reason_type == "ev_tolerance_override":
        return (
            f"{selected_label} and {runner_label} both pass the guardrails. "
            f"They sit inside the {policy.ev_tolerance_eur:,.0f} EUR EV tolerance band, "
            f"so the lower-regret option wins."
        )
    if selected_reason_type == "guardrails_relaxed_highest_ev":
        return (
            "No option passes both guardrails, so the policy falls back to the highest "
            "expected value."
        )
    return (
        f"{selected_label} leads expected value among the {eligible_option_count} "
        "eligible option(s)."
    )


def _failure_reason(row: pd.Series) -> str | None:
    """Return one technical failure reason for a policy-eligibility row."""

    if bool(row["passes_downside_floor"]) and bool(row["passes_regret_cap"]):
        return None
    if not bool(row["passes_downside_floor"]) and not bool(row["passes_regret_cap"]):
        return "misses_downside_floor_and_regret_cap"
    if not bool(row["passes_downside_floor"]):
        return "misses_downside_floor"
    return "misses_regret_cap"


def _failure_detail(row: pd.Series) -> str:
    """Return plain-English detail for why one option failed the policy."""

    reason = _as_optional_reason(row["failure_reason"])
    if reason == "misses_downside_floor":
        return (
            f"it misses the downside floor by roughly "
            f"{abs(float(row['downside_slack_eur'])):,.0f} EUR"
        )
    if reason == "misses_regret_cap":
        return f"it misses the regret cap by roughly {abs(float(row['regret_slack_eur'])):,.0f} EUR"
    if reason == "misses_downside_floor_and_regret_cap":
        return (
            "it misses the downside floor by roughly "
            f"{abs(float(row['downside_slack_eur'])):,.0f} EUR and the regret cap by roughly "
            f"{abs(float(row['regret_slack_eur'])):,.0f} EUR"
        )
    return "it remains in policy scope"


def _downside_frontier_row(
    policy: DecisionPolicyConfig,
    runner_up: pd.Series,
    recommendation: RecommendationResult,
) -> RunnerUpFrontierRow:
    """Return the runner-up downside-threshold comparison row."""

    runner_downside_slack = float(runner_up["downside_slack_eur"])
    if runner_downside_slack >= 0.0:
        note = f"{labeled_option(recommendation.runner_up)} already passes the downside floor."
        return _runner_up_frontier_row(
            "minimum_p05_value_eur",
            "Downside floor",
            "eur",
            policy.minimum_p05_value_eur,
            None,
            "already_non_binding",
            note,
        )

    switching_value = float(runner_up["p05_value_eur"])
    note = (
        f"{labeled_option(recommendation.runner_up)} becomes eligible on downside once the "
        f"floor is relaxed to {switching_value:,.2f} EUR."
    )
    return _runner_up_frontier_row(
        "minimum_p05_value_eur",
        "Downside floor",
        "eur",
        policy.minimum_p05_value_eur,
        switching_value,
        "switch_observed",
        note,
    )


def _regret_frontier_row(
    policy: DecisionPolicyConfig,
    runner_up: pd.Series,
    recommendation: RecommendationResult,
) -> RunnerUpFrontierRow:
    """Return the runner-up regret-threshold comparison row."""

    runner_regret_slack = float(runner_up["regret_slack_eur"])
    if runner_regret_slack >= 0.0:
        note = f"{labeled_option(recommendation.runner_up)} already passes the regret cap."
        return _runner_up_frontier_row(
            "maximum_mean_regret_eur",
            "Regret cap",
            "eur",
            policy.maximum_mean_regret_eur,
            None,
            "already_non_binding",
            note,
        )

    switching_value = float(runner_up["mean_regret_eur"])
    note = (
        f"{labeled_option(recommendation.runner_up)} becomes eligible on regret once the cap "
        f"rises to {switching_value:,.2f} EUR."
    )
    return _runner_up_frontier_row(
        "maximum_mean_regret_eur",
        "Regret cap",
        "eur",
        policy.maximum_mean_regret_eur,
        switching_value,
        "switch_observed",
        note,
    )


def _tolerance_frontier_row(
    policy: DecisionPolicyConfig,
    selected: pd.Series,
    runner_up: pd.Series,
    recommendation: RecommendationResult,
) -> RunnerUpFrontierRow:
    """Return the runner-up EV-tolerance comparison row."""

    selected_ev = float(selected["mean_value_eur"])
    runner_ev = float(runner_up["mean_value_eur"])
    selected_regret = float(selected["mean_regret_eur"])
    runner_regret = float(runner_up["mean_regret_eur"])

    if not bool(selected["eligible"]) or not bool(runner_up["eligible"]):
        note = (
            "EV tolerance is not the binding threshold because both options are not yet eligible."
        )
        return _runner_up_frontier_row(
            "ev_tolerance_eur",
            "EV tolerance",
            "eur",
            policy.ev_tolerance_eur,
            None,
            "not_binding",
            note,
        )

    if selected_ev == runner_ev or selected_regret == runner_regret:
        note = "The selected-vs-runner-up pair does not need an EV tolerance override."
        return _runner_up_frontier_row(
            "ev_tolerance_eur",
            "EV tolerance",
            "eur",
            policy.ev_tolerance_eur,
            0.0,
            "not_needed",
            note,
        )

    if runner_ev > selected_ev and runner_regret <= selected_regret:
        note = (
            f"{labeled_option(recommendation.runner_up)} would not need an EV tolerance override "
            "once it is eligible because it already leads on EV and regret."
        )
        return _runner_up_frontier_row(
            "ev_tolerance_eur",
            "EV tolerance",
            "eur",
            policy.ev_tolerance_eur,
            None,
            "not_needed",
            note,
        )

    if selected_ev > runner_ev and selected_regret <= runner_regret:
        note = (
            f"{labeled_option(recommendation.selected_option)} already leads on EV and regret, "
            "so tolerance does not bind."
        )
        return _runner_up_frontier_row(
            "ev_tolerance_eur",
            "EV tolerance",
            "eur",
            policy.ev_tolerance_eur,
            None,
            "not_needed",
            note,
        )

    switching_value = abs(selected_ev - runner_ev)
    lower_regret = recommendation.selected_option
    if runner_regret < selected_regret:
        lower_regret = recommendation.runner_up
    note = (
        f"{labeled_option(lower_regret)} would need an EV tolerance of {switching_value:,.2f} EUR "
        "to win on regret despite trailing on EV."
    )
    return _runner_up_frontier_row(
        "ev_tolerance_eur",
        "EV tolerance",
        "eur",
        policy.ev_tolerance_eur,
        switching_value,
        "switch_observed",
        note,
    )


def _frontier_row(
    threshold_name: str,
    threshold_label: str,
    unit: str,
    current_value: float,
    switching_value: float | None,
    first_switching_option: str | None,
    switch_type: str,
    switch_direction: str | None,
    all_options_considered: bool,
    interpretation_note: str,
) -> PolicyFrontierRow:
    """Build one full-option policy-frontier row."""

    return {
        "threshold_name": threshold_name,
        "threshold_label": threshold_label,
        "unit": unit,
        "current_value": float(current_value),
        "switching_value": None if switching_value is None else float(switching_value),
        "first_switching_option": first_switching_option,
        "switch_type": switch_type,
        "switch_direction": switch_direction,
        "all_options_considered": all_options_considered,
        "interpretation_note": interpretation_note,
    }


def _runner_up_frontier_row(
    threshold_name: str,
    threshold_label: str,
    unit: str,
    current_value: float,
    switching_value: float | None,
    status: str,
    interpretation_note: str,
) -> RunnerUpFrontierRow:
    """Build one runner-up comparison row."""

    return {
        "threshold_name": threshold_name,
        "threshold_label": threshold_label,
        "unit": unit,
        "current_value": float(current_value),
        "switching_value": None if switching_value is None else float(switching_value),
        "status": status,
        "interpretation_note": interpretation_note,
    }


def _grid_values(frontier_row: PolicyFrontierRow) -> list[float]:
    """Return a threshold grid that stays inside the valid policy domain."""

    threshold_name = str(frontier_row["threshold_name"])
    current = float(frontier_row["current_value"])
    switching_value = frontier_row["switching_value"]
    if threshold_name == "minimum_p05_value_eur":
        return _downside_floor_grid(current, switching_value)
    if threshold_name == "maximum_mean_regret_eur":
        return _regret_cap_grid(current, switching_value)
    if threshold_name == "ev_tolerance_eur":
        return _ev_tolerance_grid(current, switching_value)
    raise ValueError(f"Unknown threshold for policy frontier grid: {threshold_name}")


def _downside_floor_grid(current_value: float, switching_value: float | None) -> list[float]:
    """Return a local grid for downside-floor sweeps."""

    return _threshold_grid(
        current_value=current_value,
        switching_value=switching_value,
        minimum_value=None,
        default_step=50_000.0,
    )


def _regret_cap_grid(current_value: float, switching_value: float | None) -> list[float]:
    """Return a non-negative local grid for regret-cap sweeps."""

    return _threshold_grid(
        current_value=current_value,
        switching_value=switching_value,
        minimum_value=0.0,
        default_step=50_000.0,
    )


def _ev_tolerance_grid(current_value: float, switching_value: float | None) -> list[float]:
    """Return a non-negative local grid for EV-tolerance sweeps."""

    return _threshold_grid(
        current_value=current_value,
        switching_value=switching_value,
        minimum_value=0.0,
        default_step=25_000.0,
    )


def _threshold_grid(
    current_value: float,
    switching_value: float | None,
    minimum_value: float | None,
    default_step: float,
) -> list[float]:
    """Return a stable threshold grid around the current and switching values."""

    reference_values = [float(current_value)]
    if switching_value is not None:
        reference_values.append(float(switching_value))
    step = _grid_step(reference_values, default_step)
    lower = min(reference_values) - step
    upper = max(reference_values) + step
    if minimum_value is not None:
        lower = max(lower, minimum_value)
    if minimum_value is not None and lower < minimum_value:
        raise ValueError("Threshold grid crossed an invalid lower domain bound.")
    if upper < lower:
        raise ValueError("Threshold grid bounds are inverted.")

    values = np.arange(lower, upper + (step / 2.0), step, dtype=float).tolist()
    values.extend(reference_values)
    unique_values = sorted({round(float(value), 6) for value in values})
    if minimum_value is not None and any(value < minimum_value for value in unique_values):
        raise ValueError("Threshold grid produced an invalid negative policy threshold.")
    return unique_values


def _grid_step(values: list[float], default_step: float) -> float:
    """Return a readable grid step for one frontier sweep."""

    if len(values) == 1:
        return default_step
    distance = abs(values[0] - values[1])
    if distance <= 1_000.0:
        return 100.0
    if distance <= 5_000.0:
        return 250.0
    if distance <= 20_000.0:
        return 1_000.0
    if distance <= 100_000.0:
        return 5_000.0
    return default_step


def _policy_with_value(
    policy: DecisionPolicyConfig,
    threshold_name: str,
    tested_value: float,
) -> DecisionPolicyConfig:
    """Return one policy copy with a single threshold varied."""

    if threshold_name == "minimum_p05_value_eur":
        return DecisionPolicyConfig(
            name=policy.name,
            minimum_p05_value_eur=float(tested_value),
            maximum_mean_regret_eur=policy.maximum_mean_regret_eur,
            ev_tolerance_eur=policy.ev_tolerance_eur,
        )
    if threshold_name == "maximum_mean_regret_eur":
        return DecisionPolicyConfig(
            name=policy.name,
            minimum_p05_value_eur=policy.minimum_p05_value_eur,
            maximum_mean_regret_eur=float(tested_value),
            ev_tolerance_eur=policy.ev_tolerance_eur,
        )
    return DecisionPolicyConfig(
        name=policy.name,
        minimum_p05_value_eur=policy.minimum_p05_value_eur,
        maximum_mean_regret_eur=policy.maximum_mean_regret_eur,
        ev_tolerance_eur=float(tested_value),
    )


def _policy_value(policy: DecisionPolicyConfig, threshold_name: str) -> float:
    """Return the current numeric value for one policy threshold."""

    if threshold_name == "minimum_p05_value_eur":
        return float(policy.minimum_p05_value_eur)
    if threshold_name == "maximum_mean_regret_eur":
        return float(policy.maximum_mean_regret_eur)
    if threshold_name == "ev_tolerance_eur":
        return float(policy.ev_tolerance_eur)
    raise ValueError(f"Unknown policy threshold: {threshold_name}")


def _switch_direction(
    threshold_name: str,
    current_value: float,
    switching_value: float,
) -> str:
    """Return whether the switching move is more permissive or restrictive."""

    if threshold_name == "minimum_p05_value_eur":
        return "more_permissive" if switching_value < current_value else "more_restrictive"
    return "more_permissive" if switching_value > current_value else "more_restrictive"


def _direction_phrase(threshold_name: str, switch_direction: str) -> str:
    """Return a readable movement phrase for one threshold."""

    if threshold_name == "minimum_p05_value_eur":
        if switch_direction == "more_permissive":
            return "down"
        return "up"
    if switch_direction == "more_permissive":
        return "up"
    return "down"


def _as_optional_reason(value: object) -> str | None:
    """Normalize nullable failure reasons from dataframe values."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)
