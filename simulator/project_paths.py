"""Canonical repo paths shared across scripts, reporting, and the app."""

from __future__ import annotations

from pathlib import Path

CONFIG_PATH = Path("simulator/config.yaml")
PARAMETER_REGISTRY_PATH = Path("simulator/parameter_registry.yaml")
ASSUMPTION_REGISTRY_PATH = Path("simulator/assumption_registry.yaml")
CALIBRATION_CONTRACT_PATH = Path("simulator/calibration_contract.yaml")

PUBLIC_DATA_DIR = Path("data/public")
SOURCE_MANIFEST_PATH = PUBLIC_DATA_DIR / "sources.yaml"
SOURCE_TEMPLATE_PATH = PUBLIC_DATA_DIR / "sources.template.yaml"

CASE_STUDY_ARTIFACTS_DIR = Path("artifacts/case_study")
EVIDENCE_ARTIFACTS_DIR = Path("artifacts/evidence")

PUBLIC_EVIDENCE_PROFILE_JSON = EVIDENCE_ARTIFACTS_DIR / "public_data_profile.json"
PUBLIC_EVIDENCE_PROFILE_MARKDOWN = EVIDENCE_ARTIFACTS_DIR / "public_data_profile.md"
PARAMETER_CANDIDATES_JSON = EVIDENCE_ARTIFACTS_DIR / "parameter_candidates.json"
PARAMETER_CANDIDATES_MARKDOWN = EVIDENCE_ARTIFACTS_DIR / "parameter_candidates.md"

CASE_STUDY_PATH = Path("CASE_STUDY.md")
EXECUTIVE_SUMMARY_PATH = Path("EXECUTIVE_SUMMARY.md")
GENERATOR_SCRIPT_PATH = Path("scripts/generate_case_study_artifacts.py")
