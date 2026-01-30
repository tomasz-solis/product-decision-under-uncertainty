"""Tests for visualizations.py chart generation functions."""

import pandas as pd
import pytest
from simulator.visualizations import (
    clean_label,
    create_decision_dashboard,
    create_risk_profile_chart,
    create_regret_comparison,
    create_trade_off_matrix,
    create_executive_summary_table,
)


def test_clean_label_snake_case():
    """Test snake_case to Title Case conversion."""
    assert clean_label("feature_extension") == "Feature Extension"
    assert clean_label("do_nothing") == "Do Nothing"
    assert clean_label("new_capability") == "New Capability"


def test_clean_label_camel_case():
    """Test camelCase to Title Case conversion."""
    assert clean_label("featureExtension") == "Feature Extension"
    assert clean_label("doNothing") == "Do Nothing"


def test_clean_label_kebab_case():
    """Test kebab-case to Title Case conversion."""
    assert clean_label("feature-extension") == "Feature Extension"
    assert clean_label("do-nothing") == "Do Nothing"


def test_create_decision_dashboard():
    """Test dashboard creation with valid data."""
    summary = pd.DataFrame({
        "option": ["do_nothing", "stabilize_core"],
        "mean_value_eur": [100000, 200000]
    })
    diagnostics = pd.DataFrame({
        "option": ["do_nothing", "stabilize_core"],
        "win_rate": [0.3, 0.7],
        "mean_regret_eur": [50000, 30000]
    })
    sensitivity = pd.DataFrame({
        "parameter": ["param1", "param2"],
        "spearman_corr": [0.8, 0.6]
    })

    fig = create_decision_dashboard(summary, diagnostics, sensitivity)

    assert fig is not None
    assert hasattr(fig, "data")


def test_create_risk_profile_chart():
    """Test risk profile chart creation."""
    summary = pd.DataFrame({
        "option": ["option_a", "option_b"],
        "p05_value_eur": [50000, 60000],
        "median_value_eur": [100000, 110000],
        "p95_value_eur": [150000, 160000]
    })

    fig = create_risk_profile_chart(summary)

    assert fig is not None
    assert hasattr(fig, "data")


def test_create_regret_comparison():
    """Test regret comparison chart creation."""
    diagnostics = pd.DataFrame({
        "option": ["option_a", "option_b"],
        "mean_regret_eur": [30000, 40000],
        "median_regret_eur": [25000, 35000],
        "p95_regret_eur": [80000, 90000]
    })

    fig = create_regret_comparison(diagnostics)

    assert fig is not None
    assert hasattr(fig, "data")


def test_create_trade_off_matrix():
    """Test trade-off matrix creation."""
    summary = pd.DataFrame({
        "option": ["option_a", "option_b"],
        "mean_value_eur": [100000, 120000]
    })
    diagnostics = pd.DataFrame({
        "option": ["option_a", "option_b"],
        "mean_regret_eur": [30000, 25000]
    })

    fig = create_trade_off_matrix(summary, diagnostics)

    assert fig is not None
    assert hasattr(fig, "data")


def test_create_executive_summary_table():
    """Test executive summary table creation."""
    summary = pd.DataFrame({
        "option": ["option_a", "option_b"],
        "mean_value_eur": [100000, 120000],
        "median_value_eur": [95000, 115000],
        "p05_value_eur": [50000, 60000],
        "p95_value_eur": [150000, 180000]
    })
    diagnostics = pd.DataFrame({
        "option": ["option_a", "option_b"],
        "win_rate": [0.45, 0.55],
        "mean_regret_eur": [30000, 25000]
    })

    df = create_executive_summary_table(summary, diagnostics)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_clean_label_edge_cases():
    """Test edge cases for label cleaning."""
    assert clean_label("") == ""
    assert clean_label("SingleWord") == "Single Word"
    assert clean_label("ALLCAPS") == "Allcaps"
    assert clean_label("multiple___underscores") == "Multiple   Underscores"
