#!/usr/bin/env python3
"""
author_gender_summary.py

Reads an input CSV of anthology authors (with edition and gender) and computes, for each edition:
  - total distinct authors,
  - total distinct women authors,
  - percentage of women authors.

Outputs a summary CSV to the `data/` directory.

Sample CLI usage:
    python author_gender_summary.py path/to/input.csv
"""

import os
import argparse
import pandas as pd


def compute_gender_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure edition is integer
    df["anthology_edition"] = df["anthology_edition"].astype(int)

    editions = sorted(df["anthology_edition"].unique())
    summary = []

    for ed in editions:
        ed_df = df[df["anthology_edition"] == ed]
        total_authors = ed_df["author_id"].nunique()
        # Identify women by gender_identity_name (case‑insensitive)
        women_ids = ed_df.loc[
            ed_df["gender_identity_name"].str.lower() == "female", "author_id"
        ].unique()
        women_count = len(women_ids)
        pct = (women_count / total_authors * 100) if total_authors else 0.0

        summary.append(
            {
                "Edition": ed,
                "Authors": total_authors,
                "Women authors": women_count,
                "Women authors %": round(pct, 2),
            }
        )

    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize authors by gender for each anthology edition."
    )
    parser.add_argument(
        "input_csv",
        help='Path to the input CSV file (must include an "anthology_edition" column)',
    )
    args = parser.parse_args()

    # Read input
    df = pd.read_csv(args.input_csv)

    # Compute summary
    summary_df = compute_gender_summary(df)

    # Prepare output path
    input_base = os.path.splitext(os.path.basename(args.input_csv))[0]
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{input_base}_gender_summary.csv")

    # Write CSV
    summary_df.to_csv(output_path, index=False)
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
