"""Resolve SQL query files stored in `queries/`."""

from __future__ import annotations

from pathlib import Path

from . import QUERIES_DIR


def query_path(name: str) -> Path:
    """Return the path to queries/<name>.sql (the .sql suffix is optional)."""
    if not name.endswith(".sql"):
        name = f"{name}.sql"
    return QUERIES_DIR / name


def load_query(name: str) -> str:
    return query_path(name).read_text()
