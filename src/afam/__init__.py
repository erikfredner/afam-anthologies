"""Shared utilities for the afam-anthologies analysis scripts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
QUERIES_DIR = REPO_ROOT / "queries"
OUTPUT_DIR = REPO_ROOT / "output"
ENV_FILE = REPO_ROOT / ".env"

__all__ = ["REPO_ROOT", "DATA_DIR", "QUERIES_DIR", "OUTPUT_DIR", "ENV_FILE"]
