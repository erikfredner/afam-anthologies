"""CSV input loaders and edition-key construction.

Two `edition_key` formats are used in this project:

  * **Pipe format** (`{series}|{edition}`) — used by 2026-era CSV/DB scripts.
    Falls back to `{strip_volume(title)}|{edition}` when `series` is empty
    but a `volume` field is present, else the bare `anthology_id`.

  * **Underscore format** (`{series_id}_{anthology_edition}`) — used by the
    legacy `202505121539 authors works.csv` dataset.

Both helpers operate on the wide DataFrames produced by the data exports
listed in CLAUDE.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import DATA_DIR


def strip_volume(title: str) -> str:
    """Drop a trailing `Vol. N` suffix from an anthology title."""
    return re.sub(r",?\s+[Vv]ol\.?\s+\d+\s*$", "", title).strip()


def assign_edition_key_pipe(df: pd.DataFrame) -> pd.DataFrame:
    """Add an `edition_key` column to a 2026-era CSV DataFrame.

    Expects columns: anthology_id, anthology_title, series, edition_number, volume.
    """
    meta_rows: list[dict] = []
    for _, r in (
        df[["anthology_id", "anthology_title", "series", "edition_number", "volume"]]
        .drop_duplicates()
        .iterrows()
    ):
        series = r["series"].strip()
        edition = r["edition_number"].strip()
        volume = r["volume"].strip()
        title = r["anthology_title"].strip()
        aid = r["anthology_id"]

        if series:
            key = f"{series}|{edition}"
        elif volume:
            key = f"{strip_volume(title)}|{edition}"
        else:
            key = aid

        meta_rows.append({"anthology_id": aid, "edition_key": key})

    return df.merge(pd.DataFrame(meta_rows), on="anthology_id")


def assign_edition_key_underscore(df: pd.DataFrame) -> pd.DataFrame:
    """Add an `edition_key` column to a 202505-era CSV DataFrame.

    Expects columns: series_id, anthology_edition, anthology_id.
    Format: `{series_id}_{anthology_edition}` when series_id is set, else anthology_id.
    """
    out = df.copy()
    out["edition_key"] = [
        f"{sid}_{ed}" if str(sid).strip() else aid
        for sid, ed, aid in zip(
            out["series_id"],
            out["anthology_edition"],
            out["anthology_id"],
            strict=False,
        )
    ]
    return out


def load_csv(name: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file from `data/` with the project's standard options."""
    defaults = {"dtype": str, "na_filter": False}
    defaults.update(kwargs)
    path = name if Path(name).is_absolute() else DATA_DIR / name
    return pd.read_csv(path, **defaults)
