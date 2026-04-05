"""Data quality analysis: bootstrap CIs, normality, time series, correlations, scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose


def bootstrap_ci(
    data: pd.Series | np.ndarray[Any, Any],
    percentile: float,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap CI for a percentile. Returns (lower, upper)."""

    values = np.asarray(data, dtype=float)
    rng = np.random.default_rng(42)
    bootstrap_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_stats[i] = np.percentile(sample, percentile)

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def test_normality(data: pd.Series) -> dict[str, Any]:
    """Shapiro-Wilk (n < 5000) or Anderson-Darling normality test."""
    clean = data.dropna()
    if len(clean) < 3:
        return {"error": "Insufficient data"}

    if len(clean) > 5000:
        result = stats.anderson(clean, dist='norm')
        is_normal = result.statistic < result.critical_values[2]
        return {"test": "anderson", "is_normal": is_normal, "statistic": float(result.statistic)}
    else:
        statistic, p_value = stats.shapiro(clean)
        is_normal = p_value > 0.05
        return {"test": "shapiro", "is_normal": is_normal, "p_value": float(p_value)}


def detect_time_series_pattern(
    data: pd.Series,
    dates: pd.Series | None = None,
) -> dict[str, Any] | None:
    """Detect seasonality and trend using STL decomposition."""
    clean = data.dropna()
    if len(clean) < 14:
        return None

    try:
        if dates is not None:
            ts = pd.Series(clean.values, index=pd.to_datetime(dates.dropna()))
        else:
            ts = pd.Series(clean.values)

        result = seasonal_decompose(
            ts, model='additive', period=min(7, len(ts) // 2), extrapolate_trend='freq'
        )

        seasonal_strength = 1.0 - float(np.var(result.resid)) / max(
            float(np.var(result.seasonal + result.resid)), 1e-12
        )
        trend_strength = float(np.var(result.trend.dropna())) / max(
            float(np.var(result.observed.dropna())), 1e-12
        )

        return {
            "has_seasonality": seasonal_strength > 0.1,
            "has_trend": trend_strength > 0.1,
            "seasonal_strength": seasonal_strength,
            "trend_strength": trend_strength,
        }
    except Exception:
        return None


def calculate_correlation_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """Correlation matrix and list of strong correlations (|r| > 0.7)."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame(), []

    corr_matrix = numeric_df.corr()

    strong_corr = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corr.append((cols[i], cols[j], float(corr_val)))

    return corr_matrix, strong_corr


def calculate_quality_score(quality_metrics: dict[str, Any]) -> int:
    """Score data quality 0-100 based on sample size, variance, outliers, skewness."""
    score = 100
    n = quality_metrics.get('n', 0)
    cv = quality_metrics.get('cv', 0)
    outlier_pct = quality_metrics.get('outlier_pct', 0)
    skewness = abs(quality_metrics.get('skewness', 0))

    if n < 10:
        score -= 40
    elif n < 30:
        score -= 30
    elif n < 50:
        score -= 15
    elif n < 100:
        score -= 5

    if cv > 1.0:
        score -= 20
    elif cv > 0.5:
        score -= 10

    if outlier_pct > 10:
        score -= 15
    elif outlier_pct > 5:
        score -= 7

    if skewness > 2:
        score -= 15
    elif skewness > 1:
        score -= 7

    if quality_metrics.get('normality', {}).get('is_normal', False):
        score += 5

    return max(0, min(100, score))


def analyze_data_quality(data: pd.Series, col_name: str) -> dict[str, Any]:
    """Compute stats, outliers, normality, bootstrap CIs, and quality warnings."""

    clean = pd.to_numeric(data.dropna(), errors="coerce").dropna()
    n = len(clean)

    if n == 0:
        return {"error": "No valid data"}

    values = np.asarray(clean, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    median = float(np.median(values))
    cv = abs(std / mean) if mean != 0 else float('inf')

    skewness = float(stats.skew(values, bias=False))
    kurtosis = float(stats.kurtosis(values, bias=False))

    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = int(((values < lower_bound) | (values > upper_bound)).sum())
    outlier_pct = 100 * outliers / n

    numeric_series = pd.Series(values)
    normality = test_normality(numeric_series)

    ci_low: tuple[float, float] | None = None
    ci_mid: tuple[float, float] | None = None
    ci_high: tuple[float, float] | None = None
    if n >= 30:
        ci_low = bootstrap_ci(values, 5, n_bootstrap=500)
        ci_mid = bootstrap_ci(values, 50, n_bootstrap=500)
        ci_high = bootstrap_ci(values, 95, n_bootstrap=500)

    warnings = []
    insights = []

    if n < 30:
        warnings.append(f"Only {n} samples - might be noisy")
    else:
        insights.append(f"Good sample size (n={n})")

    if cv > 0.5:
        warnings.append(f"High variance (CV={cv:.2f})")
    elif cv < 0.1:
        insights.append(f"Low variance (CV={cv:.2f})")

    if outlier_pct > 5:
        warnings.append(f"{outliers} outliers ({outlier_pct:.1f}%)")

    if abs(skewness) > 1:
        warnings.append(f"Skewed (skew={skewness:.2f})")

    if not normality.get("is_normal", True):
        insights.append("Non-normal distribution. Percentile ranges still valid.")

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "cv": cv,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "outliers": outliers,
        "outlier_pct": outlier_pct,
        "normality": normality,
        "ci_p05": ci_low,
        "ci_p50": ci_mid,
        "ci_p95": ci_high,
        "warnings": warnings,
        "insights": insights,
    }


def get_quality_color(score: int) -> str:
    """Map a numeric quality score to a compact display label."""

    if score >= 80:
        return "[OK]"
    elif score >= 60:
        return "[FAIR]"
    else:
        return "[LOW]"
