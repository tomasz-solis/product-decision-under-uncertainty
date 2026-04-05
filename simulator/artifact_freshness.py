"""Helpers for content-addressed artifact fingerprints and freshness checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from simulator.config import get_declared_model_version


@dataclass(frozen=True)
class ArtifactFreshness:
    """Freshness verdict for one set of generated governance artifacts."""

    status: str
    mismatched_fields: tuple[str, ...]
    guidance: str


def sha256_file(path: Path) -> str:
    """Hash one file using SHA-256."""

    return sha256(path.read_bytes()).hexdigest()


def sha256_optional_file(path: Path) -> str | None:
    """Hash one optional file if it exists."""

    if not path.exists():
        return None
    return sha256_file(path)


def combined_sha256(paths: list[Path]) -> str:
    """Hash a deterministic ordered list of files into one combined fingerprint."""

    digest = sha256()
    for path in sorted(paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_code_paths() -> list[Path]:
    """Return the code files that define the published case-study artifacts."""

    return [
        Path("simulator/config.py"),
        Path("simulator/simulation.py"),
        Path("simulator/analytics.py"),
        Path("simulator/policy.py"),
        Path("simulator/provenance.py"),
        Path("simulator/reporting.py"),
        Path("simulator/evidence.py"),
        Path("simulator/output_utils.py"),
        Path("simulator/robustness.py"),
        Path("simulator/yaml_utils.py"),
        Path("simulator/artifact_freshness.py"),
        Path("simulator/project_paths.py"),
        Path("simulator/calibration.py"),
        Path("scripts/generate_case_study_artifacts.py"),
        Path("scripts/build_parameter_candidates.py"),
        Path("scripts/profile_public_evidence.py"),
    ]


def build_artifact_metadata(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    parameter_registry_path: Path,
    assumption_registry_path: Path,
    generator_script_path: Path,
    n_worlds: int,
    seed: int,
    annual_volume: int,
    time_horizon_years: int,
    discount_rate_annual: float,
    published_scenario: str,
) -> dict[str, Any]:
    """Return content-addressed metadata for one generated case-study run."""

    return {
        "seed": int(seed),
        "n_worlds": int(n_worlds),
        "annual_volume": int(annual_volume),
        "time_horizon_years": int(time_horizon_years),
        "discount_rate_annual": float(discount_rate_annual),
        "published_scenario": str(published_scenario),
        "scenario_names": list(cfg.get("scenarios", {}).keys()),
        "config_sha256": sha256_file(config_path),
        "parameter_registry_sha256": sha256_file(parameter_registry_path),
        "assumption_registry_sha256": sha256_file(assumption_registry_path),
        "code_sha256": combined_sha256(artifact_code_paths()),
        "lockfile_sha256": sha256_optional_file(Path("uv.lock")),
        "generator_script_sha256": sha256_file(generator_script_path),
        "declared_model_version": get_declared_model_version(cfg),
    }


def compare_artifact_metadata(
    stored_metadata: Mapping[str, Any] | None,
    live_metadata: Mapping[str, Any],
) -> ArtifactFreshness:
    """Compare stored artifact fingerprints with the current repo state."""

    if not stored_metadata:
        return ArtifactFreshness(
            status="unknown",
            mismatched_fields=(),
            guidance="No published metadata was available to compare against the current code.",
        )

    tracked_fields = (
        "config_sha256",
        "parameter_registry_sha256",
        "assumption_registry_sha256",
        "code_sha256",
        "generator_script_sha256",
        "lockfile_sha256",
    )
    mismatched_fields = tuple(
        field
        for field in tracked_fields
        if stored_metadata.get(field) != live_metadata.get(field)
    )
    if not mismatched_fields:
        return ArtifactFreshness(
            status="fresh",
            mismatched_fields=(),
            guidance="Published governance artifacts match the current code and config.",
        )
    return ArtifactFreshness(
        status="stale",
        mismatched_fields=mismatched_fields,
        guidance=(
            "Published governance artifacts do not match the current code or config. "
            "Run `uv run python scripts/generate_case_study_artifacts.py`."
        ),
    )
