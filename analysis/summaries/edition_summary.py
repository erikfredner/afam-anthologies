#!/usr/bin/env python3
"""
edition_summary.py

For each AFAM-tagged anthology edition (read live from the database, ordered by
year), computes:
  - total works,
  - works reselected from the previous edition (chronologically),
  - percentage reselected.

Outputs a summary CSV to the `data/` directory and prints it.

Usage:
    uv run python analysis/summaries/edition_summary.py
"""

import argparse

import pandas as pd

from afam import DATA_DIR
from afam.db import query as query_db
from afam.editions import EDITION_LABELS
from afam.sql import query_path

OUT_CSV = DATA_DIR / "edition_summary.csv"


def compute_edition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """`df` is the wide (work × author × edition) frame."""
    editions = (
        df[["edition_id", "anthology_publication_year"]]
        .drop_duplicates()
        .sort_values(["anthology_publication_year", "edition_id"])
    )

    summary_rows = []
    prev_ids: set = set()
    for _, ed in editions.iterrows():
        eid = int(ed["edition_id"])
        current_ids = set(df.loc[df["edition_id"] == eid, "work_id"])
        total = len(current_ids)
        reselected = len(current_ids & prev_ids)
        pct = (reselected / total * 100) if total else 0.0

        summary_rows.append(
            {
                "edition_id": eid,
                "Edition": EDITION_LABELS.get(eid, str(eid)),
                "Year": int(ed["anthology_publication_year"]),
                "Works": total,
                "Works reselected": reselected,
                "Works reselected %": round(pct, 2),
            }
        )
        prev_ids = current_ids

    return pd.DataFrame(summary_rows)


def main():
    argparse.ArgumentParser(
        description="Summarize anthology editions from the live database."
    ).parse_args()

    df = query_db(query_path("works-authors-per-afam-edition"))
    summary_df = compute_edition_summary(df)

    summary_df.to_csv(OUT_CSV, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSummary written to {OUT_CSV}")


if __name__ == "__main__":
    main()
