"""Calibration-contract helpers that turn profiled evidence into candidate metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simulator.provenance import load_parameter_registry
from simulator.yaml_utils import load_yaml_mapping

REQUIRED_TARGET_FIELDS = {
    "target_name",
    "assumption_family",
    "expected_raw_grain",
    "required_columns",
    "derived_metric_name",
    "transformation_rule",
    "unit_after_transform",
    "aggregation_window",
    "minimum_quality_checks",
    "expected_output_artifact",
}


def load_calibration_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate the calibration contract YAML."""

    payload = load_yaml_mapping(path)
    allowed_keys = {"evidence_backed_targets", "elicited_only_targets", "governance_only_targets"}
    unknown = set(payload) - allowed_keys
    if unknown:
        raise ValueError(f"Unknown field(s) in calibration contract: {sorted(unknown)}.")

    targets = payload.get("evidence_backed_targets", [])
    if not isinstance(targets, list):
        raise ValueError("'evidence_backed_targets' must be a list.")
    normalized_targets: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, target in enumerate(targets):
        context = f"evidence_backed_targets[{index}]"
        if not isinstance(target, dict):
            raise ValueError(f"{context} must be a mapping.")
        missing = REQUIRED_TARGET_FIELDS - set(target)
        if missing:
            raise ValueError(f"{context} is missing fields: {sorted(missing)}.")
        extras = set(target) - REQUIRED_TARGET_FIELDS
        if extras:
            raise ValueError(f"{context} has unknown fields: {sorted(extras)}.")
        name = str(target["target_name"]).strip()
        if not name:
            raise ValueError(f"{context}.target_name must be non-empty.")
        if name in seen_names:
            raise ValueError(f"Duplicate target_name in calibration contract: {name}")
        seen_names.add(name)
        required_columns = [str(column).strip() for column in target["required_columns"]]
        quality_checks = [str(check).strip() for check in target["minimum_quality_checks"]]
        if any(not column for column in required_columns):
            raise ValueError(f"{context}.required_columns must not contain empty values.")
        if any(not check for check in quality_checks):
            raise ValueError(f"{context}.minimum_quality_checks must not contain empty values.")
        normalized_targets.append(
            {
                "target_name": name,
                "assumption_family": str(target["assumption_family"]).strip(),
                "expected_raw_grain": str(target["expected_raw_grain"]).strip(),
                "required_columns": required_columns,
                "derived_metric_name": str(target["derived_metric_name"]).strip(),
                "transformation_rule": str(target["transformation_rule"]).strip(),
                "unit_after_transform": str(target["unit_after_transform"]).strip(),
                "aggregation_window": str(target["aggregation_window"]).strip(),
                "minimum_quality_checks": quality_checks,
                "expected_output_artifact": str(target["expected_output_artifact"]).strip(),
            }
        )

    elicited_only = _normalize_contract_names(
        payload.get("elicited_only_targets", []),
        context="elicited_only_targets",
    )
    governance_only = _normalize_contract_names(
        payload.get("governance_only_targets", []),
        context="governance_only_targets",
    )
    return {
        "evidence_backed_targets": normalized_targets,
        "elicited_only_targets": elicited_only,
        "governance_only_targets": governance_only,
    }


def build_parameter_candidates(
    profile: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Build a stable candidate-metric artifact from profiled evidence plus the contract."""

    sources = profile.get("sources", [])
    candidates: list[dict[str, Any]] = []
    ready_candidate_count = 0
    for target in contract["evidence_backed_targets"]:
        matching_sources = [
            source
            for source in sources
            if target["assumption_family"] in source.get("assumption_families", [])
        ]
        matched_source_ids = [str(source["source_id"]) for source in matching_sources]
        missing_columns = sorted(
            {
                column
                for source in matching_sources
                for column in target["required_columns"]
                if column not in source.get("columns", [])
            }
        )
        if not matching_sources:
            status = "awaiting_source"
            candidate_value = None
            notes = "No profiled source currently targets this assumption family."
        elif missing_columns:
            status = "schema_mismatch"
            candidate_value = None
            notes = f"Matched sources are missing required columns: {', '.join(missing_columns)}."
        else:
            candidate_value = _derive_candidate_value(matching_sources[0], target)
            status = "candidate_ready" if candidate_value is not None else "profile_matched"
            notes = (
                "Profiled evidence matches the declared columns. Review the derived metric "
                "before updating the registry."
            )
        if status == "candidate_ready":
            ready_candidate_count += 1
        candidates.append(
            {
                "target_name": target["target_name"],
                "assumption_family": target["assumption_family"],
                "candidate_status": status,
                "matched_source_ids": matched_source_ids,
                "required_columns": target["required_columns"],
                "derived_metric_name": target["derived_metric_name"],
                "transformation_rule": target["transformation_rule"],
                "unit_after_transform": target["unit_after_transform"],
                "aggregation_window": target["aggregation_window"],
                "minimum_quality_checks": target["minimum_quality_checks"],
                "expected_output_artifact": target["expected_output_artifact"],
                "candidate_value": candidate_value,
                "notes": notes,
            }
        )

    status = "awaiting_sources"
    if ready_candidate_count:
        status = "candidates_ready"
    elif profile.get("file_count", 0):
        status = "profiled_without_candidates"

    return {
        "status": status,
        "profile_status": str(profile.get("status", "unknown")),
        "target_count": len(contract["evidence_backed_targets"]),
        "ready_candidate_count": ready_candidate_count,
        "evidence_backed_families": [
            target["assumption_family"] for target in contract["evidence_backed_targets"]
        ],
        "elicited_only_targets": contract["elicited_only_targets"],
        "governance_only_targets": contract["governance_only_targets"],
        "candidates": candidates,
    }


def build_parameter_candidates_markdown(payload: dict[str, Any]) -> str:
    """Render a compact markdown view of the parameter-candidate artifact."""

    lines = [
        f"- Status: `{payload['status']}`.",
        f"- Evidence-backed targets: `{payload['target_count']}`.",
        f"- Ready candidates: `{payload['ready_candidate_count']}`.",
    ]
    if not payload["candidates"]:
        lines.append("- No evidence-backed targets are defined yet.")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "| Target | Family | Status | Matched sources | Candidate value |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for candidate in payload["candidates"]:
        source_ids = ", ".join(candidate["matched_source_ids"]) or "none"
        candidate_value = "pending"
        if candidate["candidate_value"] is not None:
            candidate_value = f"{float(candidate['candidate_value']):,.6f}"
        lines.append(
            "| "
            f"{candidate['target_name']} | {candidate['assumption_family']} | "
            f"{candidate['candidate_status']} | {source_ids} | {candidate_value} |"
        )
    return "\n".join(lines)


def write_parameter_candidates_outputs(
    payload: dict[str, Any],
    json_output_path: str | Path,
    markdown_output_path: str | Path,
) -> None:
    """Write the parameter-candidate payload as JSON and markdown."""

    json_path = Path(json_output_path)
    markdown_path = Path(markdown_output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path.write_text(build_parameter_candidates_markdown(payload), encoding="utf-8")


def expected_evidence_backed_targets(contract: dict[str, Any]) -> set[str]:
    """Return the evidence-backed assumption-family set from the contract."""

    return {str(target["assumption_family"]) for target in contract["evidence_backed_targets"]}


def expected_registry_targets(parameter_registry_path: str | Path) -> set[str]:
    """Return registry targets that explicitly expect future telemetry or benchmarks."""

    registry = load_parameter_registry(parameter_registry_path)
    rows = registry.loc[registry["source_type"] == "placeholder_for_real_telemetry"]
    return {str(name) for name in rows["parameter_name"].tolist()}


def _normalize_contract_names(value: Any, context: str) -> list[str]:
    """Normalize one simple contract name list."""

    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"'{context}' must be a list.")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"'{context}' must not contain empty entries.")
    return normalized


def _derive_candidate_value(source: dict[str, Any], target: dict[str, Any]) -> float | None:
    """Derive one simple candidate metric from the profiled evidence summary."""

    rule = str(target["transformation_rule"])
    if not rule.startswith("profile_mean(") or not rule.endswith(")"):
        return None
    column = rule.removeprefix("profile_mean(").removesuffix(")")
    numeric_summaries = source.get("numeric_summaries", {})
    summary = numeric_summaries.get(column, {})
    if not isinstance(summary, dict):
        return None
    value = summary.get("mean")
    if value is None:
        return None
    return float(value)
