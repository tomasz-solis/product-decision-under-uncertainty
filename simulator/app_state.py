"""Typed state containers shared between the Streamlit app and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from simulator.policy import PayoffDeltaDiagnostic, PolicyFrontierResult, RecommendationResult


@dataclass(frozen=True)
class AppOutputs:
    """Cached analytical outputs used across the Streamlit app."""

    simulation_settings: dict[str, Any]
    results: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    sensitivity: pd.DataFrame
    driver_analysis: pd.DataFrame
    scenario_results: pd.DataFrame
    recommendation: RecommendationResult
    policy_eligibility: pd.DataFrame
    payoff_delta: PayoffDeltaDiagnostic
    policy_frontier: PolicyFrontierResult
    policy_frontier_grid: pd.DataFrame


@dataclass(frozen=True)
class PublishedGovernance:
    """Published governance bundle shown in the sidebar and reference panels."""

    metadata: dict[str, Any]
    stability_runs: pd.DataFrame
    stability_summary: dict[str, Any]
    evidence_summary: dict[str, Any]
    manifest_counts: dict[str, int]
    freshness_status: str
    freshness_message: str
    stale_fields: tuple[str, ...]
    evidence_note_path: str
    frontier_semantics: str


__all__ = ["AppOutputs", "PublishedGovernance"]
