#!/usr/bin/env python3
"""
work_reselection_probability.py

Reads an input CSV of anthology works by edition and computes, for each “from‑edition”:
  - Edition: the earlier edition number
  - Works: total distinct works in that edition
  - Works reselected: how many of those appear in the next edition
  - Probability of reselection: (reselected / Works)
  - Odds of reselection: p / (1 – p)
  - Standard error: sqrt(p * (1 – p) / Works)

Also prints to stdout the overall probability and odds of reselection across all edition‑pairs.

Sample CLI usage:
    python work_reselection_probability.py path/to/input.csv
"""

import argparse
import math

import pandas as pd

from afam import DATA_DIR
from afam.db import query as query_db
from afam.editions import EDITION_LABELS
from afam.sql import query_path

OUT_CSV = DATA_DIR / "work_reselection_summary.csv"


def compute_reselection_summary(df: pd.DataFrame):
    """`df` is the wide (work × author × edition) frame. Editions are ordered
    chronologically by year (ties broken by edition_id)."""
    editions = (
        df[["edition_id", "anthology_publication_year"]]
        .drop_duplicates()
        .sort_values(["anthology_publication_year", "edition_id"])
    )
    ed_list = [int(e) for e in editions["edition_id"]]
    rows = []

    total_at_risk = 0
    total_reselected = 0

    # iterate over consecutive edition pairs
    for ed_current, ed_next in zip(ed_list, ed_list[1:]):
        works_current = set(df.loc[df["edition_id"] == ed_current, "work_id"])
        works_next = set(df.loc[df["edition_id"] == ed_next, "work_id"])

        n_current = len(works_current)
        reselected = len(works_current & works_next)
        p = reselected / n_current if n_current else 0.0
        odds = p / (1 - p) if 0 < p < 1 else float("inf") if p == 1 else 0.0
        se = math.sqrt(p * (1 - p) / n_current) if n_current else 0.0

        rows.append(
            {
                "Edition": EDITION_LABELS.get(ed_current, str(ed_current)),
                "Works": n_current,
                "Works reselected": reselected,
                "Probability of reselection": round(p, 4),
                "Odds of reselection": round(odds, 4),
                "Standard error": round(se, 4),
            }
        )

        total_at_risk += n_current
        total_reselected += reselected

    overall_p = total_reselected / total_at_risk if total_at_risk else 0.0
    overall_odds = (
        overall_p / (1 - overall_p)
        if 0 < overall_p < 1
        else float("inf")
        if overall_p == 1
        else 0.0
    )

    return pd.DataFrame(rows), overall_p, overall_odds


def main():
    argparse.ArgumentParser(
        description="Compute work reselection probabilities between AFAM editions."
    ).parse_args()

    df = query_db(query_path("works-authors-per-afam-edition"))
    summary_df, overall_p, overall_odds = compute_reselection_summary(df)
    summary_df.to_csv(OUT_CSV, index=False)
    print(f"Summary CSV written to {OUT_CSV}")

    # print overall stats
    print(
        f"Overall probability of work reselection across all editions: {overall_p:.4f}"
    )
    print(
        f"Overall odds of work reselection across all editions:       {overall_odds:.4f}"
    )


if __name__ == "__main__":
    main()
