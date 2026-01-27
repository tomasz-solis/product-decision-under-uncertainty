"""
CSV edge case tests.

Tests handling of problematic CSV inputs that might break the system.
"""

import pandas as pd
import numpy as np
import pytest
import io

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import _analyze_data_quality, _calculate_quality_score, _calculate_correlation_matrix


def test_empty_csv():
    """Empty CSV should be handled gracefully."""
    df = pd.DataFrame()
    corr_matrix, strong_corr = _calculate_correlation_matrix(df)
    assert corr_matrix.empty
    assert len(strong_corr) == 0


def test_single_column():
    """Single column CSV should not crash correlation analysis."""
    df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
    corr_matrix, strong_corr = _calculate_correlation_matrix(df)
    assert corr_matrix.empty or len(corr_matrix) == 1
    assert len(strong_corr) == 0


def test_all_nan_column():
    """Column with all NaN values should be handled."""
    data = pd.Series([np.nan, np.nan, np.nan, np.nan])
    quality = _analyze_data_quality(data, "test_col")
    assert "error" in quality
    assert quality["error"] == "No valid data"


def test_single_value_column():
    """Column with single repeated value should not crash."""
    data = pd.Series([5.0] * 20)
    quality = _analyze_data_quality(data, "test_col")

    # Should handle gracefully
    assert quality.get('std') == 0.0
    assert quality.get('cv') == 0.0  # or inf, both acceptable


def test_very_small_sample():
    """Very small sample should warn."""
    data = pd.Series([1.0, 2.0, 3.0])
    quality = _analyze_data_quality(data, "test_col")

    # Should work but have error or warning
    if 'error' not in quality:
        assert quality.get('n') == 3
        # Normality test might error out
        assert 'normality' in quality


def test_mixed_data_types():
    """CSV with mixed numeric and non-numeric data."""
    df = pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5],
        'text': ['a', 'b', 'c', 'd', 'e'],
        'mixed': [1, 'two', 3, 'four', 5]
    })

    # Only numeric column should be processed
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert len(numeric_cols) == 1
    assert 'numeric' in numeric_cols


def test_extreme_outliers():
    """Data with extreme outliers should be detected."""
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])  # 1000 is extreme outlier
    quality = _analyze_data_quality(data, "test_col")

    assert quality.get('outliers', 0) > 0
    assert quality.get('outlier_pct', 0) > 0


def test_negative_values():
    """Negative values should be handled correctly."""
    data = pd.Series([-10, -5, 0, 5, 10])
    quality = _analyze_data_quality(data, "test_col")

    assert quality.get('mean') == 0.0
    assert quality.get('n') == 5


def test_very_large_values():
    """Very large values should not cause overflow."""
    data = pd.Series([1e6, 2e6, 3e6, 4e6, 5e6])
    quality = _analyze_data_quality(data, "test_col")

    assert quality.get('n') == 5
    assert np.isfinite(quality.get('mean', 0))
    assert np.isfinite(quality.get('std', 0))


def test_all_zeros():
    """Column with all zeros should not crash."""
    data = pd.Series([0.0] * 20)
    quality = _analyze_data_quality(data, "test_col")

    assert quality.get('mean') == 0.0
    assert quality.get('std') == 0.0


def test_quality_score_edge_cases():
    """Quality score should handle edge case inputs."""
    # Minimal data
    quality = {'n': 1, 'cv': 0, 'outlier_pct': 0, 'skewness': 0}
    score = _calculate_quality_score(quality)
    assert 0 <= score <= 100

    # Perfect data
    quality = {'n': 1000, 'cv': 0.01, 'outlier_pct': 0, 'skewness': 0, 'normality': {'is_normal': True}}
    score = _calculate_quality_score(quality)
    assert score >= 90  # Should be high quality

    # Terrible data
    quality = {'n': 5, 'cv': 5.0, 'outlier_pct': 50, 'skewness': 10}
    score = _calculate_quality_score(quality)
    assert score <= 20  # Should be low quality


def test_correlation_with_constant_column():
    """Correlation with constant column should not crash."""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [5, 5, 5, 5, 5]  # Constant
    })

    corr_matrix, strong_corr = _calculate_correlation_matrix(df)
    # Should handle gracefully (correlation will be NaN or undefined)
    assert not corr_matrix.empty


def test_perfectly_correlated_columns():
    """Perfectly correlated columns should be detected."""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [2, 4, 6, 8, 10]  # col2 = 2 * col1
    })

    corr_matrix, strong_corr = _calculate_correlation_matrix(df)
    assert len(strong_corr) > 0  # Should detect perfect correlation
    assert abs(strong_corr[0][2]) > 0.99  # Correlation should be ~1.0


def test_csv_with_spaces_in_numbers():
    """CSV with spaces in numeric strings."""
    csv_data = """col1,col2
1,2
3,4
5, 6
7 ,8"""
    df = pd.read_csv(io.StringIO(csv_data))

    # Pandas should handle this automatically
    assert df['col2'].dtype in [np.int64, np.float64, object]
