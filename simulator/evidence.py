"""Helpers for profiling future public evidence sources with a strict manifest."""

from __future__ import annotations

import json
from datetime import date, datetime
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from simulator.project_paths import PUBLIC_EVIDENCE_PROFILE_MARKDOWN
from simulator.yaml_utils import load_yaml_mapping

SUPPORTED_SUFFIXES = {".csv", ".jsonl", ".parquet"}
MANIFEST_TOP_LEVEL_KEYS = {"sources"}
MANIFEST_SOURCE_KEYS = {
    "assumption_families",
    "expected_schema",
    "extraction_date",
    "file_name",
    "grain",
    "license",
    "publication",
    "source_id",
    "source_url",
}
SUPPORTED_SCHEMA_TYPES = {
    "boolean",
    "category",
    "date",
    "datetime",
    "float",
    "integer",
    "number",
    "string",
}
LOW_CARDINALITY_LIMIT = 20


def load_source_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the public-source manifest."""

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Evidence manifest not found: {path}")

    raw = load_yaml_mapping(path)
    unknown = set(raw) - MANIFEST_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown field(s) in evidence manifest: {sorted(unknown)}.")

    sources = raw.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("Evidence manifest field 'sources' must be a list.")

    normalized: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    seen_files: set[str] = set()
    for index, entry in enumerate(sources):
        context = f"sources[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"Evidence manifest entry {context} must be a mapping.")
        _validate_manifest_source_entry(entry, context)

        source_id = str(entry["source_id"]).strip()
        file_name = str(entry["file_name"]).strip()
        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate source_id in evidence manifest: {source_id}")
        if file_name in seen_files:
            raise ValueError(f"Duplicate file_name in evidence manifest: {file_name}")
        seen_source_ids.add(source_id)
        seen_files.add(file_name)

        normalized.append(
            {
                "source_id": source_id,
                "file_name": file_name,
                "source_url": _normalize_optional_text(entry.get("source_url")),
                "publication": _normalize_optional_text(entry.get("publication")),
                "license": str(entry["license"]).strip(),
                "extraction_date": _normalize_iso_date(entry["extraction_date"], context),
                "grain": str(entry["grain"]).strip(),
                "expected_schema": _normalize_expected_schema(
                    entry["expected_schema"],
                    context,
                ),
                "assumption_families": _normalize_string_list(
                    entry["assumption_families"],
                    f"{context}.assumption_families",
                ),
            }
        )
    return normalized


def list_public_data_files(input_dir: str | Path) -> list[Path]:
    """Return the supported raw-data files in the public-data folder."""

    directory = Path(input_dir)
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def profile_public_evidence(
    input_dir: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Profile the declared public-data sources and return one strict JSON payload."""

    directory = Path(input_dir)
    manifest_entries = load_source_manifest(manifest_path)
    files = list_public_data_files(directory)
    _validate_manifest_matches_files(manifest_entries, files)

    if not files:
        return {
            "input_dir": str(directory),
            "manifest_path": str(Path(manifest_path)),
            "status": "ready_for_data",
            "source_count": 0,
            "file_count": 0,
            "sources": [],
            "note": (
                "No public sources are registered yet. Add a manifest entry and a supported "
                "raw file under data/public/, then rerun the profiler."
            ),
        }

    sources: list[dict[str, Any]] = []
    entries_by_file = {entry["file_name"]: entry for entry in manifest_entries}
    for path in files:
        entry = entries_by_file[path.name]
        frame = load_public_frame(path)
        _validate_frame_against_manifest(frame, entry, path)
        sources.append(_profile_source(frame, path, entry))

    return {
        "input_dir": str(directory),
        "manifest_path": str(Path(manifest_path)),
        "status": "profiled",
        "source_count": len(manifest_entries),
        "file_count": len(files),
        "sources": sources,
        "note": "All manifest-declared public sources matched the expected files and schema.",
    }


def write_public_evidence_outputs(
    profile: dict[str, Any],
    json_output_path: str | Path,
    markdown_output_path: str | Path,
) -> None:
    """Write the public-evidence profile as strict JSON plus a short markdown note."""

    json_path = Path(json_output_path)
    markdown_path = Path(markdown_output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    stable_profile = _stable_json_value(profile)
    json_path.write_text(
        json.dumps(stable_profile, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path.write_text(build_public_evidence_markdown(stable_profile), encoding="utf-8")


def build_public_evidence_markdown(profile: dict[str, Any]) -> str:
    """Render a short markdown note from a public-evidence profile payload."""

    if profile.get("status") == "ready_for_data":
        return "\n".join(
            [
                "- Status: `ready_for_data`.",
                "- Manifest: checked in and validated.",
                "- Raw files present: `0`.",
                (
                    "- Next step: add the first public source file under "
                    "`data/public/` and rerun the profiler."
                ),
            ]
        )

    rows = []
    for source in profile.get("sources", []):
        rows.append(
            {
                "Source ID": str(source["source_id"]),
                "File": str(source["file_name"]),
                "Rows": str(source["row_count"]),
                "Duplicates": str(source["duplicate_row_count"]),
                "Columns": str(source["column_count"]),
            }
        )
    return "\n".join(
        [
            f"- Status: `{profile['status']}`.",
            f"- Source count: `{profile['source_count']}`.",
            f"- File count: `{profile['file_count']}`.",
            "",
            _markdown_table(pd.DataFrame(rows)),
        ]
    )


def load_public_evidence_profile(profile_path: str | Path) -> dict[str, Any]:
    """Load one previously written public-evidence profile artifact."""

    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Public evidence profile not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_public_evidence(profile: dict[str, Any]) -> dict[str, Any]:
    """Return a compact evidence summary for the app and reporting layers."""

    assumption_families = sorted(
        {
            family
            for source in profile.get("sources", [])
            for family in source.get("assumption_families", [])
        }
    )
    return {
        "status": profile.get("status", "unknown"),
        "file_count": int(profile.get("file_count", 0)),
        "source_count": int(profile.get("source_count", 0)),
        "assumption_families": assumption_families,
        "source_ids": [str(source["source_id"]) for source in profile.get("sources", [])],
        "note_artifact_path": str(PUBLIC_EVIDENCE_PROFILE_MARKDOWN),
        "note": str(profile.get("note", "")),
    }


def load_public_frame(path: Path) -> pd.DataFrame:
    """Load one supported public-data file into a dataframe."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported public-data file type: {path.suffix}")


def _profile_source(
    frame: pd.DataFrame,
    path: Path,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Build one rich profile payload for a manifest-declared source."""

    return {
        "source_id": entry["source_id"],
        "file_name": path.name,
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "source_url": entry["source_url"],
        "publication": entry["publication"],
        "license": entry["license"],
        "extraction_date": entry["extraction_date"],
        "grain": entry["grain"],
        "assumption_families": entry["assumption_families"],
        "expected_schema": entry["expected_schema"],
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(frame[column].dtype) for column in frame.columns},
        "missingness_by_column": {
            str(column): {
                "missing_count": int(frame[column].isna().sum()),
                "missing_share": float(frame[column].isna().mean()),
            }
            for column in frame.columns
        },
        "duplicate_row_count": int(frame.duplicated().sum()),
        "low_cardinality_distinct_counts": _low_cardinality_counts(frame),
        "date_ranges": _date_ranges(frame, entry["expected_schema"]),
        "numeric_summaries": _numeric_summaries(frame),
    }


def _validate_manifest_source_entry(entry: dict[str, Any], context: str) -> None:
    """Reject missing or extra fields in one evidence-manifest row."""

    missing = {
        "assumption_families",
        "expected_schema",
        "extraction_date",
        "file_name",
        "grain",
        "license",
        "source_id",
    } - set(entry)
    if missing:
        raise ValueError(f"Evidence manifest entry '{context}' is missing keys: {sorted(missing)}.")

    extras = set(entry) - MANIFEST_SOURCE_KEYS
    if extras:
        raise ValueError(f"Evidence manifest entry '{context}' has unknown keys: {sorted(extras)}.")

    source_url = _normalize_optional_text(entry.get("source_url"))
    publication = _normalize_optional_text(entry.get("publication"))
    if not source_url and not publication:
        raise ValueError(
            f"Evidence manifest entry '{context}' must define 'source_url' or 'publication'."
        )

    file_name = str(entry["file_name"]).strip()
    if not file_name:
        raise ValueError(f"Evidence manifest entry '{context}.file_name' must be non-empty.")
    if Path(file_name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Evidence manifest entry '{context}.file_name' must use one of "
            f"{sorted(SUPPORTED_SUFFIXES)}."
        )
    if not str(entry["source_id"]).strip():
        raise ValueError(f"Evidence manifest entry '{context}.source_id' must be non-empty.")
    if not str(entry["license"]).strip():
        raise ValueError(f"Evidence manifest entry '{context}.license' must be non-empty.")
    if not str(entry["grain"]).strip():
        raise ValueError(f"Evidence manifest entry '{context}.grain' must be non-empty.")


def _validate_manifest_matches_files(
    manifest_entries: list[dict[str, Any]],
    files: list[Path],
) -> None:
    """Fail if the manifest and the raw-file directory disagree."""

    manifest_files = {entry["file_name"] for entry in manifest_entries}
    actual_files = {path.name for path in files}
    if manifest_files != actual_files:
        missing = sorted(manifest_files - actual_files)
        extra = sorted(actual_files - manifest_files)
        details = []
        if missing:
            details.append(f"missing files for manifest entries {missing}")
        if extra:
            details.append(f"files without manifest entries {extra}")
        raise ValueError(
            f"Evidence manifest and input directory do not match: {', '.join(details)}."
        )


def _validate_frame_against_manifest(
    frame: pd.DataFrame,
    entry: dict[str, Any],
    path: Path,
) -> None:
    """Validate that one raw dataframe matches the manifest-declared schema."""

    expected_schema = entry["expected_schema"]
    expected_columns = set(expected_schema)
    actual_columns = {str(column) for column in frame.columns}
    if expected_columns != actual_columns:
        missing = sorted(expected_columns - actual_columns)
        extra = sorted(actual_columns - expected_columns)
        details = []
        if missing:
            details.append(f"missing columns {missing}")
        if extra:
            details.append(f"unexpected columns {extra}")
        raise ValueError(
            f"Source '{path.name}' does not match expected schema: {', '.join(details)}."
        )

    for column_name, expected_type in expected_schema.items():
        series = frame[column_name]
        if not _series_matches_schema(series, expected_type):
            actual_type = _describe_series_type(series)
            raise ValueError(
                f"Source '{path.name}' column '{column_name}' expected '{expected_type}' "
                f"but observed '{actual_type}'."
            )


def _normalize_expected_schema(value: Any, context: str) -> dict[str, str]:
    """Normalize one expected-schema mapping and reject unsupported type labels."""

    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"Evidence manifest entry '{context}.expected_schema' must be a non-empty mapping."
        )
    normalized: dict[str, str] = {}
    for column_name, schema_type in value.items():
        column = str(column_name).strip()
        type_name = str(schema_type).strip()
        if not column:
            raise ValueError(
                f"Evidence manifest entry '{context}.expected_schema' has an empty column name."
            )
        if type_name not in SUPPORTED_SCHEMA_TYPES:
            raise ValueError(
                f"Evidence manifest entry '{context}.expected_schema.{column}' must be one of "
                f"{sorted(SUPPORTED_SCHEMA_TYPES)}."
            )
        normalized[column] = type_name
    return normalized


def _normalize_string_list(value: Any, context: str) -> list[str]:
    """Return a non-empty list of stripped strings."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"Field '{context}' must be a non-empty list.")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"Field '{context}' must not contain empty values.")
    return normalized


def _normalize_optional_text(value: Any) -> str | None:
    """Normalize an optional string field."""

    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _normalize_iso_date(value: Any, context: str) -> str:
    """Return an ISO date string and reject malformed values."""

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError(f"Field '{context}.extraction_date' must be an ISO date string.")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:  # pragma: no cover - small guard branch
        raise ValueError(f"Field '{context}.extraction_date' must be an ISO date string.") from exc


def _series_matches_schema(series: pd.Series, expected_type: str) -> bool:
    """Return whether a pandas series satisfies one manifest schema type."""

    if expected_type == "boolean":
        return bool(is_bool_dtype(series))
    if expected_type == "integer":
        return bool(is_integer_dtype(series))
    if expected_type == "float":
        return bool(
            is_numeric_dtype(series) and not is_integer_dtype(series) and not is_bool_dtype(series)
        )
    if expected_type == "number":
        return bool(is_numeric_dtype(series) and not is_bool_dtype(series))
    if expected_type == "category":
        non_null = series.dropna()
        return (
            non_null.nunique(dropna=True) <= LOW_CARDINALITY_LIMIT if not non_null.empty else True
        )
    if expected_type in {"date", "datetime"}:
        if is_datetime64_any_dtype(series):
            return True
        parsed = pd.to_datetime(series.dropna(), errors="coerce")
        return bool(parsed.notna().all()) if not parsed.empty else True
    return not is_numeric_dtype(series) and not is_datetime64_any_dtype(series)


def _describe_series_type(series: pd.Series) -> str:
    """Return a compact logical type label for one pandas series."""

    if is_bool_dtype(series):
        return "boolean"
    if is_integer_dtype(series):
        return "integer"
    if is_numeric_dtype(series):
        return "number"
    if is_datetime64_any_dtype(series):
        return "datetime"
    non_null = series.dropna()
    if not non_null.empty and non_null.nunique(dropna=True) <= LOW_CARDINALITY_LIMIT:
        return "category"
    return "string"


def _low_cardinality_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return distinct counts for columns small enough to inspect quickly."""

    counts: dict[str, int] = {}
    for column in frame.columns:
        distinct = int(frame[column].dropna().nunique())
        if 0 < distinct <= LOW_CARDINALITY_LIMIT:
            counts[str(column)] = distinct
    return counts


def _date_ranges(
    frame: pd.DataFrame, expected_schema: dict[str, str]
) -> dict[str, dict[str, str | None]]:
    """Return min/max date ranges for manifest-declared date-like columns."""

    ranges: dict[str, dict[str, str | None]] = {}
    for column_name, schema_type in expected_schema.items():
        if schema_type not in {"date", "datetime"}:
            continue
        parsed = pd.to_datetime(frame[column_name].dropna(), errors="coerce")
        if parsed.empty or parsed.isna().all():
            ranges[column_name] = {"min": None, "max": None}
            continue
        ranges[column_name] = {
            "min": parsed.min().isoformat(),
            "max": parsed.max().isoformat(),
        }
    return ranges


def _numeric_summaries(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Return strict numeric summaries with nulls instead of NaN."""

    summaries: dict[str, dict[str, float | None]] = {}
    numeric = frame.select_dtypes(include=["number"])
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            summaries[str(column)] = {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
            }
            continue
        summaries[str(column)] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
        }
    return summaries


def _stable_json_value(value: Any) -> Any:
    """Normalize floats and datelike values before JSON serialization."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if pd.isna(number):
            return None
        return round(number, 6)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _stable_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_json_value(item) for item in value]
    return value


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a tiny markdown table without optional dependencies."""

    headers = [str(column) for column in frame.columns]
    rows = frame.astype(str).values.tolist()
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
