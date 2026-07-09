#!/usr/bin/env python3
"""
edition_stats.py

Summarise, for each edition of *The Norton Anthology of African American
Literature* (NAAAL, series_id=3), read live from the database:

  • Edition          – edition number
  • Total            – distinct authors in that edition
  • Reselected       – authors repeated from the *previous* edition
  • Newly Selected   – authors who never appeared in any earlier NAAAL edition

The resulting file is saved to **data/naaal_edition_stats.csv**.

Usage:
    uv run python analysis/summaries/edition_stats.py
"""

import argparse
from pathlib import Path

import pandas as pd

from afam import DATA_DIR
from afam.db import query as query_db
from afam.sql import query_path

NAAAL_SERIES_ID = 3
DEFAULT_OUT = DATA_DIR / "naaal_edition_stats.csv"


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """`df` is the wide (work × author × edition) frame."""
    naaal = df[df["series_id"] == NAAAL_SERIES_ID].dropna(subset=["author_id"])
    editions = (
        naaal[["edition_id", "edition_number", "anthology_publication_year"]]
        .drop_duplicates()
        .sort_values("anthology_publication_year")
    )

    rows = []
    cumulative_authors: set = set()
    prev_authors: set | None = None
    for _, ed in editions.iterrows():
        eid = int(ed["edition_id"])
        authors = set(naaal.loc[naaal["edition_id"] == eid, "author_id"])
        rows.append(
            {
                "Edition": ed["edition_number"],
                "Year": int(ed["anthology_publication_year"]),
                "Total": len(authors),
                "Reselected": len(authors & prev_authors) if prev_authors else 0,
                "Newly Selected": len(authors - cumulative_authors),
            }
        )
        cumulative_authors |= authors
        prev_authors = authors

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output CSV (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    df = query_db(query_path("works-authors-per-afam-edition"))
    stats_df = compute_stats(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_path, index=False)
    print(stats_df.to_string(index=False))
    print(f"\n[INFO] Saved edition statistics → {out_path}")


if __name__ == "__main__":
    main()
