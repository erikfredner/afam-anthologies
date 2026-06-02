"""Shared matplotlib defaults and the canonical output directory.

Importing this module ensures `output/` exists so figure-producing scripts
can write to `OUTPUT_DIR / "<name>.png"` without each one repeating the
mkdir boilerplate.
"""

from __future__ import annotations

from . import OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
FIGSIZE = (10, 6)
WORK_COLOR = "#1f77b4"
AUTHOR_COLOR = "#d62728"

__all__ = ["OUTPUT_DIR", "DPI", "FIGSIZE", "WORK_COLOR", "AUTHOR_COLOR"]
