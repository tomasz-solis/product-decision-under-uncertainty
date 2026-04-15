"""Tests for analytics helpers."""

from __future__ import annotations

import pandas as pd

from simulator.analytics import _partial_rank_correlations, independence_ablation

CONFIG_PATH = "simulator/config.yaml"


def test_partial_rank_correlations_zero_near_singular_matrix() -> None:
    """Perfect collinearity should zero the partial-rank output instead of exploding."""

    features = pd.DataFrame(
        {
            "baseline_failure_rate": [0.05, 0.10, 0.15, 0.20],
            "regression_event_prob": [0.05, 0.10, 0.15, 0.20],
        }
    )
    target = pd.Series([1.0, 2.0, 4.0, 3.0], name="value")

    result = _partial_rank_correlations(features, target)

    assert result == {
        "baseline_failure_rate": 0.0,
        "regression_event_prob": 0.0,
    }


def test_independence_ablation_returns_expected_shape() -> None:
    """Dependency ablation should compare correlated and independent reruns cleanly."""

    result = independence_ablation(CONFIG_PATH, n_worlds=500, seed=42)

    assert result["selected_option"] in {
        "do_nothing",
        "stabilize_core",
        "feature_extension",
        "new_capability",
    }
    assert isinstance(result["recommendation_changed"], bool)
    comparison_rows = result["comparison_rows"]
    assert isinstance(comparison_rows, list)
    assert len(comparison_rows) == 4
