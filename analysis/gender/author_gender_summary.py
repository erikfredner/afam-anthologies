#!/usr/bin/env python3
"""
author_gender_summary.py

Computes, for each AFAM-tagged anthology edition (read live from the database):
  - total distinct authors,
  - total distinct women authors,
  - percentage of women authors.

Outputs a summary CSV to the `data/` directory and prints it.

Usage:
    uv run python analysis/gender/author_gender_summary.py
"""

import argparse

import pandas as pd

from afam import DATA_DIR
from afam.db import query as query_db
from afam.editions import EDITION_LABELS
from afam.sql import query_path

OUT_CSV = DATA_DIR / "author_gender_summary.csv"


def compute_gender_summary(df: pd.DataFrame) -> pd.DataFrame:
    """`df` is the gender-work-consistency frame (author_id, gender, edition_id,
    edition_year)."""
    df = df.dropna(subset=["author_id"]).copy()

    summary = []
    for ed, ed_df in df.groupby("edition_id"):
        total_authors = ed_df["author_id"].nunique()
        women_count = ed_df.loc[
            ed_df["gender"].astype(str).str.lower() == "female", "author_id"
        ].nunique()
        pct = (women_count / total_authors * 100) if total_authors else 0.0
        summary.append(
            {
                "edition_id": int(ed),
                "Edition": EDITION_LABELS.get(int(ed), str(int(ed))),
                "Year": int(ed_df["edition_year"].min()),
                "Authors": total_authors,
                "Women authors": women_count,
                "Women authors %": round(pct, 2),
            }
        )

    return (
        pd.DataFrame(summary).sort_values("Year").reset_index(drop=True)
    )


def main():
    argparse.ArgumentParser(
        description="Summarize authors by gender for each anthology edition."
    ).parse_args()

    df = query_db(query_path("gender-work-consistency"))
    summary_df = compute_gender_summary(df)

    summary_df.to_csv(OUT_CSV, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSummary written to {OUT_CSV}")


if __name__ == "__main__":
    main()
