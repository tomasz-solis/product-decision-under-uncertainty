"""Tests for robustness helpers and artifact assembly."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import simulator.robustness as robustness


def _minimal_stability_runs() -> pd.DataFrame:
    """Return a compact stability frame for robustness tests."""

    return pd.DataFrame(
        [
            {
                "scenario": "mid_range_pressure",
                "n_worlds": 5000,
                "seed": 42,
                "selected_option": "stabilize_core",
                "comparison_option": "feature_extension",
                "comparison_option_role": "best_excluded_option",
                "ev_leader": "stabilize_core",
                "selected_mean_value_eur": 373000.0,
                "comparison_mean_value_eur": 197000.0,
                "selected_p05_value_eur": -360000.0,
                "comparison_p05_value_eur": -420000.0,
                "binding_constraint": "guardrails_relaxed_highest_ev",
            },
            {
                "scenario": "mid_range_pressure",
                "n_worlds": 5000,
                "seed": 43,
                "selected_option": "stabilize_core",
                "comparison_option": "feature_extension",
                "comparison_option_role": "best_excluded_option",
                "ev_leader": "stabilize_core",
                "selected_mean_value_eur": 371000.0,
                "comparison_mean_value_eur": 195000.0,
                "selected_p05_value_eur": -358000.0,
                "comparison_p05_value_eur": -418000.0,
                "binding_constraint": "guardrails_relaxed_highest_ev",
            },
        ]
    )


def test_convergence_rows_summarize_one_world_count_group() -> None:
    """Convergence rows should summarize rerun consistency and spread."""

    rows = robustness._convergence_rows(_minimal_stability_runs())

    assert len(rows) == 1
    assert rows[0]["world_count"] == 5000
    assert rows[0]["run_count"] == 2
    assert rows[0]["recommendation_consistency"] == 1.0


def test_build_robustness_report_returns_expected_keys(
    monkeypatch,
) -> None:
    """The robustness report should expose the downstream-required sections."""

    monkeypatch.setattr(
        robustness,
        "load_config",
        lambda _path: {
            "simulation": {"n_worlds": 2000, "scenario": "mid_range_pressure"},
            "project": {"seed": 42},
        },
    )
    monkeypatch.setattr(
        robustness,
        "get_analysis_settings",
        lambda _cfg: SimpleNamespace(sensitivity_materiality_threshold_abs_spearman=0.1),
    )
    monkeypatch.setattr(robustness, "_frontier_stability_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(robustness, "_metric_error_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(robustness, "_stress_test_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        robustness,
        "independence_ablation",
        lambda *args, **kwargs: {
            "correlated_selected_option": "stabilize_core",
            "independent_selected_option": "stabilize_core",
            "selected_option_p05_independent_eur": -350000.0,
            "selected_option_p05_correlated_eur": -360000.0,
            "comparison_rows": [],
        },
    )

    result = robustness.build_robustness_report(
        config_path="simulator/config.yaml",
        stability_runs=_minimal_stability_runs(),
        driver_analysis=pd.DataFrame(),
        selected_option="stabilize_core",
    )

    assert result["selected_option"] == "stabilize_core"
    assert "convergence_rows" in result
    assert "frontier_stability_rows" in result
    assert "metric_error_rows" in result
    assert "stress_test_rows" in result
    assert "dependency_ablation" in result


def test_build_robustness_markdown_renders_dependency_ablation() -> None:
    """The markdown summary should render the dependency-ablation section cleanly."""

    payload = {
        "selected_option": "stabilize_core",
        "convergence_rows": [],
        "frontier_stability_rows": [],
        "metric_error_rows": [],
        "stress_test_rows": [],
        "dependency_ablation": {
            "correlated_selected_option": "stabilize_core",
            "independent_selected_option": "feature_extension",
            "selected_option_p05_independent_eur": -330000.0,
            "selected_option_p05_correlated_eur": -360000.0,
            "comparison_rows": [
                {
                    "option_label": "Stabilize Core",
                    "correlated_mean_value_eur": 373000.0,
                    "independent_mean_value_eur": 392000.0,
                    "correlated_p05_value_eur": -360000.0,
                    "independent_p05_value_eur": -330000.0,
                }
            ],
        },
    }

    result = robustness.build_robustness_markdown(payload)

    assert isinstance(result, str)
    assert "Dependency ablation" in result
    assert "Stabilize Core" in result
