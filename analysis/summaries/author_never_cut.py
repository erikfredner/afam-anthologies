#!/usr/bin/env python3
"""
author_never_cut.py

Reads an input CSV of anthology authors by edition and computes the overall probability,
odds, and standard error that an author will never be cut in any subsequent edition
(i.e., once they appear, they remain in every later edition).

Writes the results to stdout.

Sample CLI usage:
    python author_never_cut.py path/to/input.csv
"""

import argparse
import pandas as pd
import math


def compute_never_cut_stats(df: pd.DataFrame):
    # ensure edition is integer
    df["anthology_edition"] = df["anthology_edition"].astype(int)
    all_editions = sorted(df["anthology_edition"].unique())
    max_edition = all_editions[-1]

    # group by author
    author_groups = df.groupby("author_id")["anthology_edition"].unique()

    at_risk = 0
    never_cut = 0

    for editions in author_groups:
        min_ed = editions.min()
        # only authors who have at least one subsequent edition
        if min_ed < max_edition:
            at_risk += 1
            # require author appears in every edition from their first through the final
            expected_count = max_edition - min_ed + 1
            actual_count = len(editions)
            last_ed = editions.max()
            if last_ed == max_edition and actual_count == expected_count:
                never_cut += 1

    # compute probability, odds, and standard error
    p = never_cut / at_risk if at_risk else 0.0
    odds = p / (1 - p) if 0 < p < 1 else (float("inf") if p == 1 else 0.0)
    se = math.sqrt(p * (1 - p) / at_risk) if at_risk else 0.0

    return at_risk, never_cut, p, odds, se


def main():
    parser = argparse.ArgumentParser(
        description="Compute probability that an author is never cut in subsequent editions."
    )
    parser.add_argument(
        "input_csv",
        help='Path to the input CSV (must include "author_id" and "anthology_edition" columns)',
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    at_risk, never_cut, p, odds, se = compute_never_cut_stats(df)

    # output to stdout
    print(f"Authors at risk of being cut:       {at_risk}")
    print(f"Authors never cut across editions:  {never_cut}")
    print(f"Probability of never being cut:     {p:.4f}")
    print(f"Odds of never being cut:            {odds:.4f}")
    print(f"Standard error:                     {se:.4f}")


if __name__ == "__main__":
    main()
