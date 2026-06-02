#!/usr/bin/env python3
"""
edition_stats_csv.py

Generate a CSV file summarising, for each edition of *The Norton Anthology
of African American Literature* (NAFAL):

  • Edition          – edition number
  • Total            – distinct authors in that edition
  • Reselected       – authors repeated from the *previous* edition
  • Newly Selected   – authors who never appeared in any earlier edition

The resulting file is saved to **data/nafal_edition_stats.csv**.

Usage
-----
python edition_stats_csv.py path/to/nafam_authors.csv
"""

import argparse
import os
import pathlib

import pandas as pd


def compute_stats(csv_path: str) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Read & filter
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    nafam = df[
        df["anthology_series"] == "The Norton Anthology of African American Literature"
    ].copy()

    # drop duplicate author‑edition rows
    nafam = nafam.drop_duplicates(subset=["anthology_edition", "author_id"])

    # ensure numeric, sorted editions
    nafam["anthology_edition"] = nafam["anthology_edition"].astype(int)
    nafam = nafam.sort_values("anthology_edition")

    # ------------------------------------------------------------------
    # 2. Compute stats per edition
    # ------------------------------------------------------------------
    rows = []
    cumulative_authors: set[int] = set()
    prev_authors: set[int] | None = None

    for edition, sub in nafam.groupby("anthology_edition"):
        authors = set(sub["author_id"])
        rows.append(
            {
                "Edition": edition,
                "Total": len(authors),
                "Reselected": len(authors & prev_authors) if prev_authors else 0,
                "Newly Selected": len(authors - cumulative_authors),
            }
        )
        cumulative_authors |= authors
        prev_authors = authors

    return pd.DataFrame(rows).sort_values("Edition")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Input CSV with author / anthology data")
    parser.add_argument(
        "--out",
        default="data/nafal_edition_stats.csv",
        help="Output CSV (default: data/nafal_edition_stats.csv)",
    )
    args = parser.parse_args()

    stats_df = compute_stats(args.csv)

    # ensure data/ directory exists
    out_path = pathlib.Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)

    stats_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved edition statistics → {out_path}")


if __name__ == "__main__":
    main()
