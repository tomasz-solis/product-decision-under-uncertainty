"""Portability and link-integrity tests for public markdown docs."""

from __future__ import annotations

import re
from pathlib import Path

ABSOLUTE_PATH_PATTERNS = ("/Users/", "C:\\", "file://")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def test_markdown_docs_reject_local_absolute_paths() -> None:
    """Public docs should not contain local-machine paths."""

    for path in _markdown_files():
        content = path.read_text(encoding="utf-8")
        for pattern in ABSOLUTE_PATH_PATTERNS:
            assert pattern not in content, (
                f"Found local absolute path pattern {pattern!r} in {path}"
            )


def test_markdown_relative_links_resolve_inside_repo() -> None:
    """Relative markdown links should point to files that exist in the repo."""

    for path in _markdown_files():
        content = path.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK_RE.findall(content):
            clean_target = target.split("#", maxsplit=1)[0]
            if not clean_target or "://" in clean_target or clean_target.startswith("mailto:"):
                continue
            resolved = (path.parent / clean_target).resolve()
            assert resolved.exists(), f"Broken relative link {target!r} in {path}"


def _markdown_files() -> list[Path]:
    """Return the markdown files that represent the public repo surface."""

    roots = [Path("."), Path("simulator"), Path("artifacts"), Path("data")]
    files: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root == Path("."):
            files.update(root.glob("*.md"))
        else:
            files.update(root.rglob("*.md"))
    return sorted(path for path in files if not path.name.startswith("FIXME"))
