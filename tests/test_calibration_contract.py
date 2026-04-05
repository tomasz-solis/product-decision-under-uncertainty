"""Tests for the calibration contract and parameter-candidate artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from simulator.calibration import (
    build_parameter_candidates,
    expected_evidence_backed_targets,
    expected_registry_targets,
    load_calibration_contract,
)
from simulator.evidence import profile_public_evidence
from simulator.provenance import load_parameter_registry

CALIBRATION_CONTRACT_PATH = Path("simulator/calibration_contract.yaml")
PARAMETER_REGISTRY_PATH = Path("simulator/parameter_registry.yaml")


def test_calibration_contract_covers_registry_targets_marked_for_future_telemetry() -> None:
    """Every placeholder telemetry target should be covered by the calibration contract."""

    contract = load_calibration_contract(CALIBRATION_CONTRACT_PATH)

    assert expected_registry_targets(PARAMETER_REGISTRY_PATH).issubset(
        expected_evidence_backed_targets(contract)
    )


def test_parameter_candidates_empty_state_has_stable_schema() -> None:
    """The empty evidence state should still emit a stable candidate schema."""

    contract = load_calibration_contract(CALIBRATION_CONTRACT_PATH)
    profile = profile_public_evidence("data/public", "data/public/sources.yaml")
    payload = build_parameter_candidates(profile, contract)

    assert payload["status"] == "awaiting_sources"
    assert payload["target_count"] == len(contract["evidence_backed_targets"])
    assert payload["ready_candidate_count"] == 0
    candidate = payload["candidates"][0]
    assert {
        "target_name",
        "assumption_family",
        "candidate_status",
        "matched_source_ids",
        "required_columns",
        "derived_metric_name",
        "transformation_rule",
        "unit_after_transform",
        "aggregation_window",
        "minimum_quality_checks",
        "expected_output_artifact",
        "candidate_value",
        "notes",
    }.issubset(candidate)


def test_parameter_registry_rejects_malformed_evidence_ids(tmp_path: Path) -> None:
    """Evidence-linked registry rows should reject malformed evidence ids."""

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "baseline_failure_rate:\n"
        "  unit: share_of_volume\n"
        "  business_meaning: example\n"
        "  distribution_type: tri\n"
        "  low: 0.02\n"
        "  mode: 0.05\n"
        "  high: 0.10\n"
        "  source_type: placeholder_for_real_telemetry\n"
        "  source_reference: example\n"
        "  reason_for_range: example\n"
        "  owner: test\n"
        "  last_updated: 2026-04-05\n"
        "  evidence_ids:\n"
        "    - Bad Evidence Id\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lowercase evidence ids"):
        load_parameter_registry(registry_path)
