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
    """`df` is the wide (work × author × edition) frame.

    Several AFAM editions share a publication year (e.g. three in 1968, four in
    1971) with no genuine editorial "earlier -> later" relationship among
    themselves. Comparing against only the immediately-preceding row in an
    edition_id-broken sort would (a) compare same-year siblings against each
    other as if one preceded the other, and (b) compare the edition following a
    same-year cluster against just one arbitrary cluster member instead of the
    cluster's full prior selections. Each edition is instead compared against
    the union of works from the previous *distinct year*'s edition(s).
    """
    editions = (
        df[["edition_id", "anthology_publication_year"]]
        .drop_duplicates()
        .sort_values(["anthology_publication_year", "edition_id"])
    )
    edition_works = {
        int(eid): set(grp["work_id"]) for eid, grp in df.groupby("edition_id")
    }

    summary_rows = []
    prev_year_ids: set = set()
    for year, year_editions in editions.groupby("anthology_publication_year"):
        for eid in year_editions["edition_id"]:
            eid = int(eid)
            current_ids = edition_works.get(eid, set())
            total = len(current_ids)
            reselected = len(current_ids & prev_year_ids)
            pct = (reselected / total * 100) if total else 0.0

            summary_rows.append(
                {
                    "edition_id": eid,
                    "Edition": EDITION_LABELS.get(eid, str(eid)),
                    "Year": int(year),
                    "Works": total,
                    "Works reselected": reselected,
                    "Works reselected %": round(pct, 2),
                }
            )
        prev_year_ids = set().union(
            *(edition_works.get(int(eid), set()) for eid in year_editions["edition_id"])
        )

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
