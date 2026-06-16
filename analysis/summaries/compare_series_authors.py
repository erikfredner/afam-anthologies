#!/usr/bin/env python3
"""
compare_series_authors.py

Reads an input CSV of anthology authors and prints to stdout two lists:
  1. Authors who appear in any edition of series A but never in series B
  2. Authors who appear in any edition of series B but never in series A

Assumes exactly two distinct values in the "anthology_series" column.

Sample CLI usage:
    python compare_series_authors.py path/to/input.csv
"""

import sys
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="List authors unique to each of two anthology series."
    )
    parser.add_argument(
        "input_csv",
        help='Path to the input CSV (must include "author_id", "author_name", and "anthology_series" columns)',
    )
    args = parser.parse_args()

    # Load and dedupe on author + series
    df = pd.read_csv(args.input_csv, dtype={"author_id": str})
    df = df[["author_id", "author_name", "anthology_series"]].drop_duplicates()

    # Identify the two series
    series = df["anthology_series"].unique()
    if len(series) != 2:
        sys.exit(
            f"ERROR: Expected exactly two anthology_series values, found {len(series)}: {series.tolist()}"
        )

    s1, s2 = series

    # Build sets of (id, name)
    s1_authors = set(
        map(
            tuple, df[df["anthology_series"] == s1][["author_id", "author_name"]].values
        )
    )
    s2_authors = set(
        map(
            tuple, df[df["anthology_series"] == s2][["author_id", "author_name"]].values
        )
    )

    only_s1 = sorted(s1_authors - s2_authors, key=lambda x: x[1])
    only_s2 = sorted(s2_authors - s1_authors, key=lambda x: x[1])

    # Print results
    print(f"Authors in “{s1}” but not in “{s2}”:")
    if only_s1:
        for author_id, author_name in only_s1:
            print(f"{author_id}, {author_name}")
    else:
        print("  (none)")

    print(f"\nAuthors in “{s2}” but not in “{s1}”:")
    if only_s2:
        for author_id, author_name in only_s2:
            print(f"{author_id}, {author_name}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
