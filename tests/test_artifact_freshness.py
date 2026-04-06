"""Tests that keep published artifacts and docs in sync with the current model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulator import reporting
from simulator.artifact_freshness import artifact_code_paths
from simulator.calibration import (
    build_parameter_candidates,
    load_calibration_contract,
    write_parameter_candidates_outputs,
)
from simulator.config import get_declared_model_version, load_config
from simulator.evidence import profile_public_evidence, write_public_evidence_outputs
from simulator.reporting import (
    build_case_study_artifacts,
    update_case_study_docs,
    write_case_study_artifacts,
)

ARTIFACTS_DIR = Path("artifacts/case_study")
EVIDENCE_DIR = Path("artifacts/evidence")
CASE_STUDY_PATH = Path("CASE_STUDY.md")
EXECUTIVE_SUMMARY_PATH = Path("EXECUTIVE_SUMMARY.md")
CONFIG_PATH = Path("simulator/config.yaml")
ASSUMPTION_REGISTRY_PATH = Path("simulator/assumption_registry.yaml")
PARAMETER_REGISTRY_PATH = Path("simulator/parameter_registry.yaml")
CALIBRATION_CONTRACT_PATH = Path("simulator/calibration_contract.yaml")


@pytest.fixture(scope="module")
def generated_artifact_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the published artifact set once for freshness checks."""

    output_dir = tmp_path_factory.mktemp("fresh-case-study")
    artifacts = build_case_study_artifacts(CONFIG_PATH)
    write_case_study_artifacts(artifacts, output_dir)
    return output_dir


@pytest.fixture(scope="module")
def generated_evidence_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the evidence-profile outputs once for freshness checks."""

    output_dir = tmp_path_factory.mktemp("fresh-evidence")
    profile = profile_public_evidence("data/public", "data/public/sources.yaml")
    write_public_evidence_outputs(
        profile,
        json_output_path=output_dir / "public_data_profile.json",
        markdown_output_path=output_dir / "public_data_profile.md",
    )
    parameter_candidates = build_parameter_candidates(
        profile,
        load_calibration_contract(CALIBRATION_CONTRACT_PATH),
    )
    write_parameter_candidates_outputs(
        parameter_candidates,
        json_output_path=output_dir / "parameter_candidates.json",
        markdown_output_path=output_dir / "parameter_candidates.md",
    )
    return output_dir


def test_published_json_and_fragments_match_fresh_generation(generated_artifact_dir: Path) -> None:
    """Checked-in artifacts should match a fresh run of the published generator."""

    expected_json_files = {
        "summary.json",
        "diagnostics.json",
        "scenario_results.json",
        "sensitivity.json",
        "parameter_registry.json",
        "policy_eligibility.json",
        "assumption_manifest.json",
        "recommendation.json",
        "payoff_delta_diagnostic.json",
        "policy_frontier.json",
        "policy_frontier_grid.json",
        "stability_runs.json",
        "stability_summary.json",
        "evidence_summary.json",
        "robustness.json",
        "metadata.json",
    }
    expected_text_files = {
        "parameter_registry.csv",
        "driver_analysis.md",
        "recommendation.md",
        "summary_table.md",
        "diagnostics_table.md",
        "policy_eligibility.md",
        "scenario_table.md",
        "payoff_delta_diagnostic.md",
        "policy_frontier.md",
        "stability.md",
        "sensitivity_table.md",
        "robustness.md",
    }

    for filename in expected_json_files:
        fresh = json.loads((generated_artifact_dir / filename).read_text(encoding="utf-8"))
        published = json.loads((ARTIFACTS_DIR / filename).read_text(encoding="utf-8"))
        assert published == fresh

    for filename in expected_text_files:
        fresh = (generated_artifact_dir / filename).read_text(encoding="utf-8")
        published = (ARTIFACTS_DIR / filename).read_text(encoding="utf-8")
        assert published == fresh


def test_published_evidence_artifacts_match_fresh_generation(generated_evidence_dir: Path) -> None:
    """Checked-in evidence artifacts should match a fresh profile run."""

    for filename in [
        "public_data_profile.json",
        "public_data_profile.md",
        "parameter_candidates.json",
        "parameter_candidates.md",
    ]:
        fresh = (generated_evidence_dir / filename).read_text(encoding="utf-8")
        published = (EVIDENCE_DIR / filename).read_text(encoding="utf-8")
        assert published == fresh


def test_published_docs_match_generated_fragments(
    generated_artifact_dir: Path, tmp_path: Path
) -> None:
    """The checked-in markdown docs should already contain the current generated sections."""

    fragments = {
        path.name: path.read_text(encoding="utf-8")
        for path in generated_artifact_dir.iterdir()
        if path.suffix == ".md"
    }
    case_copy = tmp_path / "CASE_STUDY.md"
    exec_copy = tmp_path / "EXECUTIVE_SUMMARY.md"
    case_copy.write_text(CASE_STUDY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    exec_copy.write_text(EXECUTIVE_SUMMARY_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    update_case_study_docs(case_copy, exec_copy, fragments)

    assert case_copy.read_text(encoding="utf-8") == CASE_STUDY_PATH.read_text(encoding="utf-8")
    assert exec_copy.read_text(encoding="utf-8") == EXECUTIVE_SUMMARY_PATH.read_text(
        encoding="utf-8"
    )


def test_metadata_includes_code_and_dependency_fingerprints(
    generated_artifact_dir: Path,
) -> None:
    """Metadata should tie the published outputs to code, config, and dependency state."""

    metadata = json.loads((generated_artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["declared_model_version"] == get_declared_model_version(load_config(CONFIG_PATH))
    assert metadata["discount_rate_annual"] == 0.08
    assert len(metadata["config_sha256"]) == 64
    assert len(metadata["parameter_registry_sha256"]) == 64
    assert len(metadata["assumption_registry_sha256"]) == 64
    assert len(metadata["code_sha256"]) == 64
    assert len(metadata["generator_script_sha256"]) == 64
    assert len(metadata["lockfile_sha256"]) == 64


def test_stability_cache_invalidates_when_same_config_path_changes(tmp_path: Path) -> None:
    """Stability output should refresh when config content changes at the same path."""

    config_path = tmp_path / "config.yaml"
    parameter_registry_path = tmp_path / "parameter_registry.yaml"
    assumption_registry_path = tmp_path / "assumption_registry.yaml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    parameter_registry_path.write_text(
        PARAMETER_REGISTRY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assumption_registry_path.write_text(
        ASSUMPTION_REGISTRY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    reporting._cached_stability_runs.cache_clear()
    initial_cache = reporting._cached_stability_runs.cache_info()
    build_case_study_artifacts(config_path)
    first_run_cache = reporting._cached_stability_runs.cache_info()

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "minimum_p05_value_eur: -300000.0",
            "minimum_p05_value_eur: -1000000.0",
        ),
        encoding="utf-8",
    )
    assumption_registry_path.write_text(
        assumption_registry_path.read_text(encoding="utf-8").replace(
            "value: -300000.0",
            "value: -1000000.0",
            1,
        ),
        encoding="utf-8",
    )

    build_case_study_artifacts(config_path)
    second_run_cache = reporting._cached_stability_runs.cache_info()

    assert first_run_cache.misses == initial_cache.misses + 1
    assert second_run_cache.misses == first_run_cache.misses + 1


def test_artifact_fingerprint_includes_output_utils() -> None:
    """Artifact fingerprints should cover every module that changes published outputs."""

    assert Path("simulator/output_utils.py") in artifact_code_paths()
