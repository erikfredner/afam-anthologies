#!/usr/bin/env python3
"""
author_never_cut.py

Reads AFAM anthology authors by edition from the live database and computes the
overall probability, odds, and standard error that an author is never cut in any
subsequent edition (i.e., once they debut they remain in every later edition,
through the final edition). Editions are ordered chronologically by year.

Writes the results to stdout.

Usage:
    uv run python analysis/summaries/author_never_cut.py
"""

import argparse
import math

import pandas as pd

from afam.db import query as query_db
from afam.sql import query_path


def compute_never_cut_stats(df: pd.DataFrame):
    """`df` is the wide (work × author × edition) frame. Editions are positioned
    chronologically by year; editions sharing a year (e.g. three in 1968, four
    in 1971) have no genuine editorial order among themselves, so they collapse
    to a single rank instead of each getting its own position via an arbitrary
    edition_id tiebreak -- otherwise an author present in every year except one
    same-year sibling would show a spurious "gap" and be misclassified as cut."""
    editions = df[["edition_id", "anthology_publication_year"]].drop_duplicates()
    years = sorted(editions["anthology_publication_year"].unique())
    year_rank = {int(y): i for i, y in enumerate(years)}
    edition_rank = {
        int(row["edition_id"]): year_rank[int(row["anthology_publication_year"])]
        for _, row in editions.iterrows()
    }
    max_rank = len(years) - 1

    author_groups = (
        df.dropna(subset=["author_id"]).groupby("author_id")["edition_id"].unique()
    )

    at_risk = 0
    never_cut = 0
    for eds in author_groups:
        ranks = sorted({edition_rank[int(e)] for e in eds})
        min_rank = ranks[0]
        # only authors who have at least one subsequent edition (year)
        if min_rank < max_rank:
            at_risk += 1
            # require presence in every year-rank from debut through the final
            expected_count = max_rank - min_rank + 1
            if ranks[-1] == max_rank and len(ranks) == expected_count:
                never_cut += 1

    # compute probability, odds, and standard error
    p = never_cut / at_risk if at_risk else 0.0
    odds = p / (1 - p) if 0 < p < 1 else (float("inf") if p == 1 else 0.0)
    se = math.sqrt(p * (1 - p) / at_risk) if at_risk else 0.0

    return at_risk, never_cut, p, odds, se


def main():
    argparse.ArgumentParser(
        description="Probability that an author is never cut in subsequent editions."
    ).parse_args()

    df = query_db(query_path("works-authors-per-afam-edition"))
    at_risk, never_cut, p, odds, se = compute_never_cut_stats(df)

    # output to stdout
    print(f"Authors at risk of being cut:       {at_risk}")
    print(f"Authors never cut across editions:  {never_cut}")
    print(f"Probability of never being cut:     {p:.4f}")
    print(f"Odds of never being cut:            {odds:.4f}")
    print(f"Standard error:                     {se:.4f}")


if __name__ == "__main__":
    main()
