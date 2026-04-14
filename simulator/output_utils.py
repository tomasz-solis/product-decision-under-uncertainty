"""Shared output helpers used by reporting, presentation, and charts."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from simulator.simulation import OPTION_LABELS


def format_eur(value: float) -> str:
    """Format a euro amount for display."""

    return f"EUR {value:,.0f}"


def format_eur_markdown(value: float) -> str:
    """Format a euro amount for markdown tables."""

    return f"€{value:,.0f}"


def format_pct(value: float) -> str:
    """Format a proportion as a percentage string."""

    return f"{value:.0%}"


def format_number(value: float) -> str:
    """Format a raw numeric value with adaptive precision."""

    if float(value).is_integer():
        return f"{value:,.0f}"
    magnitude = abs(value)
    if magnitude >= 1_000.0:
        return f"{value:,.1f}"
    if magnitude >= 10.0:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def format_threshold_eur(value: float) -> str:
    """Format a threshold value without collapsing small amounts to zero."""

    return f"EUR {format_number(value)}"


def format_threshold_eur_markdown(value: float) -> str:
    """Format a threshold value for markdown without coarse business rounding."""

    return f"€{format_number(value)}"


def clean_label(text: str) -> str:
    """Turn a snake-case-like token into title case."""

    return text.replace("_", " ").replace("-", " ").title()


def confidence_interval_excludes_zero(ci_low: float, ci_high: float) -> bool:
    """Return whether a confidence interval stays on one side of zero."""

    return (ci_low > 0.0 and ci_high > 0.0) or (ci_low < 0.0 and ci_high < 0.0)


def material_sensitivity_rows(
    sensitivity: pd.DataFrame,
    option: str,
    threshold: float,
    limit: int,
) -> pd.DataFrame:
    """Return the material sensitivity rows for one option."""

    rows = (
        sensitivity.loc[sensitivity["option"] == option]
        .assign(abs_spearman=lambda frame: frame["spearman_corr"].abs())
        .sort_values("abs_spearman", ascending=False)
    )
    rows = rows.loc[rows["abs_spearman"] >= threshold].head(limit).copy()
    return rows.drop(columns=["abs_spearman"], errors="ignore").reset_index(drop=True)


def sensitivity_note(
    sensitivity: pd.DataFrame,
    option: str,
    threshold: float,
) -> str | None:
    """Return a note about whether an option has only one or no material drivers."""

    rows = material_sensitivity_rows(
        sensitivity=sensitivity,
        option=option,
        threshold=threshold,
        limit=100,
    )
    if rows.empty:
        return f"No parameter cleared the materiality threshold of |rho| >= {threshold:.2f}."
    if len(rows) == 1:
        parameter = clean_label(str(rows.iloc[0]["parameter"]))
        return f"Only `{parameter}` cleared the materiality threshold of |rho| >= {threshold:.2f}."
    return None


def material_driver_rows(
    driver_analysis: pd.DataFrame,
    option: str,
    threshold: float,
    limit: int,
) -> pd.DataFrame:
    """Return the strongest decision-support drivers for one option."""

    rows = (
        driver_analysis.loc[driver_analysis["option"] == option]
        .assign(abs_partial_rank_corr=lambda frame: frame["partial_rank_corr"].abs())
        .sort_values("abs_partial_rank_corr", ascending=False)
    )
    rows = rows.loc[rows["abs_partial_rank_corr"] >= threshold]
    if {"ci_low", "ci_high"}.issubset(rows.columns):
        rows = rows.loc[
            rows.apply(
                lambda row: confidence_interval_excludes_zero(
                    float(row["ci_low"]),
                    float(row["ci_high"]),
                ),
                axis=1,
            )
        ]
    rows = rows.head(limit).copy()
    return rows.drop(columns=["abs_partial_rank_corr"], errors="ignore").reset_index(drop=True)


def driver_note(
    driver_analysis: pd.DataFrame,
    option: str,
    threshold: float,
) -> str | None:
    """Return a short note about decision-support driver coverage."""

    rows = material_driver_rows(
        driver_analysis=driver_analysis,
        option=option,
        threshold=threshold,
        limit=100,
    )
    if rows.empty:
        return (
            "No decision-support driver cleared the current materiality threshold of "
            f"|partial rho| >= {threshold:.2f}."
        )
    if len(rows) == 1:
        parameter = clean_label(str(rows.iloc[0]["parameter"]))
        return (
            f"Only `{parameter}` cleared the decision-support materiality threshold of "
            f"|partial rho| >= {threshold:.2f}."
        )
    return None


def labeled_option(option: str) -> str:
    """Return the user-facing label for an option key."""

    return OPTION_LABELS.get(option, clean_label(option))


def build_run_context(
    selected_settings: Mapping[str, int | str],
    published_settings: Mapping[str, int | str],
) -> dict[str, str | bool]:
    """Describe the currently selected app run and whether it matches the published case."""

    matches_published = all(
        selected_settings[key] == published_settings[key] for key in published_settings
    )
    selected_label = _scenario_label(selected_settings)
    published_label = _scenario_label(published_settings)
    heading = f"Run summary: {selected_label}"
    detail = (
        f"Scenario `{selected_label}`, seed `{selected_settings['seed']}`, "
        f"`{selected_settings['n_worlds']:,}` worlds."
    )
    note = ""
    if not matches_published:
        note = (
            "Exploratory rerun. The published case uses "
            f"`{published_label}`, seed `{published_settings['seed']}`, and "
            f"`{published_settings['n_worlds']:,}` worlds."
        )
    return {
        "heading": heading,
        "detail": detail,
        "note": note,
        "matches_published": matches_published,
    }


def _scenario_label(settings: Mapping[str, int | str]) -> str:
    """Return the human-readable scenario label for one run context."""

    explicit_label = str(settings.get("scenario_label", "")).strip()
    if explicit_label:
        return explicit_label
    return clean_label(str(settings["scenario"]))
