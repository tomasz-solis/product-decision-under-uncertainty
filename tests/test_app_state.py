"""Regression tests for app cache payload types."""

from __future__ import annotations

import pickle

import pandas as pd

from simulator.app_state import AppOutputs, PublishedGovernance
from simulator.policy import RecommendationResult


def _recommendation_fixture() -> RecommendationResult:
    """Return a minimal recommendation payload for cache serialization tests."""

    return RecommendationResult(
        policy_name="guardrailed_expected_value",
        selected_option="stabilize_core",
        policy_runner_up=None,
        best_excluded_option="feature_extension",
        selection_reason="Selected for the serialization regression test.",
        binding_constraint="guardrails_relaxed",
        selected_mean_value_eur=373000.0,
        policy_runner_up_mean_value_eur=None,
        best_excluded_mean_value_eur=197000.0,
        selected_p05_value_eur=-360000.0,
        selected_mean_regret_eur=120000.0,
        eligible_option_count=1,
        eligible_options=("stabilize_core",),
        selected_reason_type="highest_ev_eligible",
        best_excluded_failure_reason="misses_downside_floor",
        selected_downside_slack_eur=40000.0,
        selected_regret_slack_eur=80000.0,
        policy_runner_up_downside_slack_eur=None,
        policy_runner_up_regret_slack_eur=None,
        best_excluded_downside_slack_eur=-25000.0,
        best_excluded_regret_slack_eur=5000.0,
    )


def test_app_outputs_is_pickle_serializable() -> None:
    """AppOutputs must stay pickleable for Streamlit cache_data."""

    payload = AppOutputs(
        simulation_settings={"n_worlds": 1000, "scenario": "mid_range_pressure"},
        results=pd.DataFrame({"option": ["stabilize_core"], "value": [1.0]}),
        summary=pd.DataFrame({"option": ["stabilize_core"], "mean_value_eur": [1.0]}),
        diagnostics=pd.DataFrame({"option": ["stabilize_core"], "mean_regret_eur": [0.1]}),
        sensitivity=pd.DataFrame({"option": ["stabilize_core"], "parameter": ["cost"]}),
        driver_analysis=pd.DataFrame({"option": ["stabilize_core"], "parameter": ["cost"]}),
        scenario_results=pd.DataFrame({"scenario": ["mid_range_pressure"]}),
        recommendation=_recommendation_fixture(),
        policy_eligibility=pd.DataFrame({"option": ["stabilize_core"], "eligible": [True]}),
        payoff_delta={
            "selected_option": "stabilize_core",
            "comparison_option": "feature_extension",
            "comparison_option_role": "best_excluded_option",
            "mean_delta_eur": 176000.0,
            "p05_delta_eur": 58000.0,
            "win_rate_vs_comparison": 0.71,
            "delta_rows": [],
        },
        policy_frontier={
            "selected_option": "stabilize_core",
            "baseline_selected_option": "stabilize_core",
            "policy_runner_up": None,
            "best_excluded_option": "feature_extension",
            "comparison_option": "feature_extension",
            "comparison_option_role": "best_excluded_option",
            "frontier_rows": [],
            "secondary_comparison_rows": [],
            "runner_up_comparison_rows": [],
        },
        policy_frontier_grid=pd.DataFrame({"threshold_name": ["minimum_p05_value_eur"]}),
    )

    restored = pickle.loads(pickle.dumps(payload))

    assert isinstance(restored, AppOutputs)
    assert restored.recommendation.selected_option == "stabilize_core"


def test_published_governance_is_pickle_serializable() -> None:
    """PublishedGovernance must stay pickleable for Streamlit cache_data."""

    payload = PublishedGovernance(
        metadata={"scenario": "mid_range_pressure"},
        stability_runs=pd.DataFrame({"seed": [42], "selected_option": ["stabilize_core"]}),
        stability_summary={"selected_option": "stabilize_core"},
        evidence_summary={"note_artifact_path": "artifacts/evidence/public_data_profile.md"},
        manifest_counts={"params": 8},
        freshness_status="fresh",
        freshness_message="Artifacts are in sync.",
        stale_fields=(),
        evidence_note_path="artifacts/evidence/public_data_profile.md",
        frontier_semantics="full-option switch frontier",
    )

    restored = pickle.loads(pickle.dumps(payload))

    assert isinstance(restored, PublishedGovernance)
    assert restored.freshness_status == "fresh"
