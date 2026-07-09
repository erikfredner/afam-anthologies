#!/usr/bin/env python3
"""
series_pair_reselection.py

Compares the two paired editions the Norton American Literature series
(series_id=1) publishes for each edition number — the full edition and its
shorter/package companion — read live from the database. For each edition
number it reports, for each role (Full / Shorter):
  - distinct authors,
  - authors carried over from the previous edition of the same role,
  - that carry-over as a percentage of current authors.

The two same-edition-number editions are distinguished by work count (the
edition with more works is treated as Full), matching the project convention in
queries/works-without-authors-naal-naaal.sql, because edition titles are not
stored.

Outputs a summary CSV to the `data/` directory.

Usage:
    uv run python viz/reselection/series_pair_reselection.py
"""

import argparse

import pandas as pd

from afam import DATA_DIR
from afam.db import query as query_db
from afam.sql import query_path

NAFAM_SERIES_ID = 1
OUT_CSV = DATA_DIR / "series_pair_reselection_summary.csv"


def classify_editions(s1: pd.DataFrame) -> pd.DataFrame:
    """Tag each series-1 edition as 'Full' or 'Shorter' by work count within its
    edition number (more works → Full)."""
    work_counts = s1.groupby("edition_id")["work_id"].nunique()
    meta = (
        s1[["edition_id", "edition_number", "anthology_publication_year"]]
        .drop_duplicates()
        .copy()
    )
    meta["work_count"] = meta["edition_id"].map(work_counts)
    meta = meta.sort_values(["edition_number", "work_count"], ascending=[True, False])
    meta["role"] = (
        meta.groupby("edition_number").cumcount().map({0: "Full", 1: "Shorter"})
    )
    return meta


def compute_reselection_by_role(
    s1: pd.DataFrame, meta: pd.DataFrame, role: str
) -> dict:
    """For one role, map edition_number → {authors, reselected, reselected_pct}."""
    eds = meta[meta["role"] == role].sort_values("anthology_publication_year")
    authors_by_ednum = {
        row["edition_number"]: set(
            s1.loc[s1["edition_id"] == int(row["edition_id"]), "author_id"].dropna()
        )
        for _, row in eds.iterrows()
    }

    order = list(eds["edition_number"])
    summary = {}
    for idx, ednum in enumerate(order):
        current = authors_by_ednum[ednum]
        count = len(current)
        if idx == 0:
            reselected = 0
            pct = 0.0
        else:
            reselected = len(current & authors_by_ednum[order[idx - 1]])
            pct = (reselected / count * 100) if count else 0.0
        summary[ednum] = {
            "authors": count,
            "reselected": reselected,
            "reselected_pct": round(pct, 2),
        }
    return summary


def main():
    argparse.ArgumentParser(
        description="Reselection metrics for the paired Norton American editions."
    ).parse_args()

    df = query_db(query_path("naal-american-authors-works"))
    s1 = df[df["series_id"] == NAFAM_SERIES_ID]
    meta = classify_editions(s1)

    full_summary = compute_reselection_by_role(s1, meta, "Full")
    shorter_summary = compute_reselection_by_role(s1, meta, "Shorter")

    all_editions = sorted(set(full_summary) | set(shorter_summary), key=int)
    rows = []
    for ed in all_editions:
        rows.append(
            {
                "Edition": ed,
                "Full authors": full_summary.get(ed, {}).get("authors", 0),
                "Shorter authors": shorter_summary.get(ed, {}).get("authors", 0),
                "Full reselected": full_summary.get(ed, {}).get("reselected", 0),
                "Full reselected %": full_summary.get(ed, {}).get(
                    "reselected_pct", 0.0
                ),
                "Shorter reselected": shorter_summary.get(ed, {}).get("reselected", 0),
                "Shorter reselected %": shorter_summary.get(ed, {}).get(
                    "reselected_pct", 0.0
                ),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_CSV, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSummary written to {OUT_CSV}")


if __name__ == "__main__":
    main()
