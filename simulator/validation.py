"""Structural validation helpers shared by config and registry loading.

The simulator config and the provenance registries both parse user-authored YAML and
need the same guards with the same readable error messages. These helpers live in a
dependency-free leaf module so ``config`` and ``provenance`` can share one
implementation without creating an import cycle.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def validate_allowed_keys(
    payload: dict[str, Any],
    allowed_keys: set[str],
    context: str,
) -> None:
    """Reject unknown keys in one mapping with a sorted, readable error."""

    unknown = sorted(str(key) for key in set(payload) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown field(s) in '{context}': {', '.join(unknown)}.")


def normalize_iso_date(value: Any, context: str) -> str:
    """Return an ISO date string and reject malformed values.

    ``date`` and ``datetime`` values are emitted as ISO date strings; strings are
    validated through ``date.fromisoformat``. Any other type raises a ``ValueError``
    naming ``context`` so the offending field is obvious to the caller.
    """

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError(f"Field '{context}' must be an ISO date string.")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:  # pragma: no cover - small guard branch
        raise ValueError(f"Field '{context}' must be an ISO date string.") from exc
