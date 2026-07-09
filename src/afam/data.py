"""CSV input loader for the one remaining hand-maintained dataset.

Analysis and viz scripts read live from PostgreSQL; the only CSV still loaded is
the hand-maintained `data/vernacular_pages.csv` (via `afam.vernacular`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import DATA_DIR


def load_csv(name: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file from `data/` with the project's standard options."""
    defaults = {"dtype": str, "na_filter": False}
    defaults.update(kwargs)
    path = name if Path(name).is_absolute() else DATA_DIR / name
    return pd.read_csv(path, **defaults)
