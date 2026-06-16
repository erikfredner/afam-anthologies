#!/usr/bin/env python3
"""
series_pair_reselection.py

Reads an input CSV of authors across two paired anthology series editions
(e.g., NAAL and NAALSE) and computes, for each numeric edition:
  - Edition
  - NAAL authors: unique authors in the NAAL series
  - NAALSE authors: unique authors in the NAALSE (Shorter Edition) series
  - NAAL reselected: count of NAAL authors carried over from the previous NAAL edition
  - NAAL reselected %: that count as a percentage of current‑edition NAAL authors
  - NAALSE reselected: count of NAALSE authors carried over from the previous NAALSE edition
  - NAALSE reselected %: that count as a percentage of current‑edition NAALSE authors

Outputs a summary CSV to the `data/` directory.

Sample CLI usage:
    python series_pair_reselection.py path/to/input.csv
"""

import os
import argparse
import pandas as pd


def compute_reselection_by_series(df: pd.DataFrame, series_name: str):
    """
    For a given series, returns a dict mapping edition → {
      'authors': int,
      'reselected': int,
      'reselected_pct': float
    }
    """
    subset = df[df["anthology_series"] == series_name].copy()
    subset["anthology_edition"] = subset["anthology_edition"].astype(int)

    editions = sorted(subset["anthology_edition"].unique())
    authors_by_edition = {
        ed: set(subset.loc[subset["anthology_edition"] == ed, "author_id"])
        for ed in editions
    }

    summary = {}
    for idx, ed in enumerate(editions):
        current = authors_by_edition[ed]
        count = len(current)
        if idx == 0:
            reselected = 0
            pct = 0.0
        else:
            prev = editions[idx - 1]
            reselected = len(current & authors_by_edition[prev])
            pct = (reselected / count * 100) if count else 0.0

        summary[ed] = {
            "authors": count,
            "reselected": reselected,
            "reselected_pct": round(pct, 2),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compute reselection metrics for two paired anthology series."
    )
    parser.add_argument(
        "input_csv",
        help='Path to the input CSV (must include "anthology_series", "author_id", and "anthology_edition" columns)',
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Identify the two series: assume one contains "Shorter" in its name
    series_names = df["anthology_series"].unique()
    try:
        shorter = next(s for s in series_names if "Shorter" in s)
        full = next(s for s in series_names if "Shorter" not in s)
    except StopIteration:
        raise ValueError(
            "Could not uniquely identify full vs. Shorter Edition series names."
        )

    full_summary = compute_reselection_by_series(df, full)
    shorter_summary = compute_reselection_by_series(df, shorter)

    # Combine editions
    all_editions = sorted(set(full_summary) | set(shorter_summary))
    rows = []
    for ed in all_editions:
        rows.append(
            {
                "Edition": ed,
                "NAAL authors": full_summary.get(ed, {}).get("authors", 0),
                "NAALSE authors": shorter_summary.get(ed, {}).get("authors", 0),
                "NAAL reselected": full_summary.get(ed, {}).get("reselected", 0),
                "NAAL reselected %": full_summary.get(ed, {}).get(
                    "reselected_pct", 0.0
                ),
                "NAALSE reselected": shorter_summary.get(ed, {}).get("reselected", 0),
                "NAALSE reselected %": shorter_summary.get(ed, {}).get(
                    "reselected_pct", 0.0
                ),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("Edition")

    # Write output CSV
    base = os.path.splitext(os.path.basename(args.input_csv))[0]
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_series_reselection_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
