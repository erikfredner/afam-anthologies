"""Shared utilities for the afam-anthologies analysis scripts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
QUERIES_DIR = REPO_ROOT / "queries"
OUTPUT_DIR = REPO_ROOT / "output"
# Tracked in git and served as the GitHub Pages root, unlike the gitignored
# OUTPUT_DIR — so nothing here is created on import.
DOCS_DIR = REPO_ROOT / "docs"
ENV_FILE = REPO_ROOT / ".env"

__all__ = ["DATA_DIR", "DOCS_DIR", "ENV_FILE", "OUTPUT_DIR", "QUERIES_DIR", "REPO_ROOT"]
