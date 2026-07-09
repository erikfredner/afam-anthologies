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
    chronologically by year (ties broken by edition_id)."""
    editions = (
        df[["edition_id", "anthology_publication_year"]]
        .drop_duplicates()
        .sort_values(["anthology_publication_year", "edition_id"])
        .reset_index(drop=True)
    )
    position = {int(e): i for i, e in enumerate(editions["edition_id"])}
    max_position = len(editions) - 1

    author_groups = (
        df.dropna(subset=["author_id"]).groupby("author_id")["edition_id"].unique()
    )

    at_risk = 0
    never_cut = 0
    for eds in author_groups:
        positions = sorted({position[int(e)] for e in eds})
        min_pos = positions[0]
        # only authors who have at least one subsequent edition
        if min_pos < max_position:
            at_risk += 1
            # require presence in every edition from debut through the final
            expected_count = max_position - min_pos + 1
            if positions[-1] == max_position and len(positions) == expected_count:
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
