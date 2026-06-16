#!/usr/bin/env python3
"""
list_universally_selected.py

Reads an input CSV of works and authors across multiple anthology series and prints to stdout:

1. A list of authors who appear in every edition of every series.
2. A list of works that appear in every edition of every series.

Authors are printed as:
    - Author Name

Works are printed as:
    "Title" by Author Name

Both lists are sorted alphabetically (authors by name; works by author then title).

Sample CLI usage:
    python list_universally_selected.py path/to/input.csv
"""

import sys
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="List authors and works selected in every edition of every anthology series."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV (must include columns: "
        '"work_id","work_title","author_id","author_name",'
        '"series_name","anthology_edition")',
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv, dtype=str)
    required = {
        "work_id",
        "work_title",
        "author_id",
        "author_name",
        "series_name",
        "anthology_edition",
    }
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        sys.exit(f"ERROR: Missing required columns: {', '.join(missing)}")

    # Normalize edition to int
    df["anthology_edition"] = df["anthology_edition"].astype(int)

    # Determine full set of editions per series
    series_names = sorted(df["series_name"].unique())
    series_editions = {
        s: set(df.loc[df["series_name"] == s, "anthology_edition"])
        for s in series_names
    }

    # --- Authors ---
    # Deduplicate on author + series + edition
    auth_df = df[
        ["author_id", "author_name", "series_name", "anthology_edition"]
    ].drop_duplicates()
    universal_authors = []
    for author_id, sub in auth_df.groupby("author_id"):
        name = sub["author_name"].iloc[0]
        # Editions by series for this author
        ed_by_series = {
            s: set(sub.loc[sub["series_name"] == s, "anthology_edition"])
            for s in series_names
        }
        # Check full coverage
        if all(ed_by_series[s] == series_editions[s] for s in series_names):
            universal_authors.append(name)
    universal_authors.sort(key=lambda n: n.lower())

    # --- Works ---
    # Deduplicate on work + series + edition
    work_df = df[
        ["work_id", "work_title", "author_name", "series_name", "anthology_edition"]
    ].drop_duplicates()
    universal_works = []
    for work_id, sub in work_df.groupby("work_id"):
        title = sub["work_title"].iloc[0]
        author = sub["author_name"].iloc[0]
        ed_by_series = {
            s: set(sub.loc[sub["series_name"] == s, "anthology_edition"])
            for s in series_names
        }
        if all(ed_by_series[s] == series_editions[s] for s in series_names):
            universal_works.append((author, title))
    universal_works.sort(key=lambda at: (at[0].lower(), at[1].lower()))

    # Output authors
    print("Universally selected authors:")
    if universal_authors:
        for name in universal_authors:
            print(f"- {name}")
    else:
        print("(none)")

    print("\nUniversally selected works:")
    if universal_works:
        for author, title in universal_works:
            print(f'"{title}" by {author}')
    else:
        print("(none)")


if __name__ == "__main__":
    main()
