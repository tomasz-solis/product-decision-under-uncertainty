"""Strict YAML loading helpers shared across config and provenance files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class StrictMappingLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load one YAML file and reject duplicate keys or non-mapping roots."""

    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            payload = yaml.load(handle, Loader=StrictMappingLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse YAML file '{yaml_path}'.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"YAML file '{yaml_path}' must have a mapping at the root.")
    return payload


def _construct_mapping(
    loader: StrictMappingLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """Construct one mapping node while rejecting duplicate keys."""

    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key detected: {key!r}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictMappingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)
