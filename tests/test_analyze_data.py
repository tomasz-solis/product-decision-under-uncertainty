"""Tests for analyze_data.py CSV analysis functions."""

import numpy as np
import pandas as pd
import pytest
from simulator.analyze_data import analyze_column


def test_analyze_column_basic():
    """Test basic column analysis with normal data."""
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 5)
    result = analyze_column(data, "test_column")

    assert "low" in result
    assert "mode" in result
    assert "high" in result
    assert "mean" in result
    assert "std" in result
    assert result["mode"] == pytest.approx(5.5, rel=0.1)


def test_analyze_column_insufficient_data():
    """Test that insufficient data returns error."""
    data = pd.Series([1.0, 2.0, 3.0])
    result = analyze_column(data, "test_column")

    assert "error" in result
    assert "Insufficient data" in result["error"]


def test_analyze_column_empty():
    """Test that empty series returns error."""
    data = pd.Series([])
    result = analyze_column(data, "test_column")

    assert "error" in result
    assert "No valid data" in result["error"]


def test_analyze_column_with_nans():
    """Test that NaN values are properly handled."""
    data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.nan] * 3)
    result = analyze_column(data, "test_column")

    assert "error" not in result
    assert "low" in result
    assert "high" in result


def test_analyze_column_percentiles():
    """Test that percentiles are calculated correctly."""
    data = pd.Series(range(1, 101))  # 1 to 100
    result = analyze_column(data, "test_column")

    assert result["low"] == pytest.approx(5.95, abs=1)  # 5th percentile
    assert result["mode"] == pytest.approx(50.5, abs=1)  # median
    assert result["high"] == pytest.approx(95.05, abs=1)  # 95th percentile


def test_analyze_column_constant_values():
    """Test handling of constant values."""
    data = pd.Series([5.0] * 20)
    result = analyze_column(data, "test_column")

    assert result["low"] == 5.0
    assert result["mode"] == 5.0
    assert result["high"] == 5.0
    assert result["std"] == 0.0
