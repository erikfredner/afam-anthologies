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

import os
import argparse
import pandas as pd
import math


def compute_reselection_summary(df: pd.DataFrame):
    # ensure edition is integer
    df["anthology_edition"] = df["anthology_edition"].astype(int)
    editions = sorted(df["anthology_edition"].unique())
    rows = []

    total_at_risk = 0
    total_reselected = 0

    # iterate over consecutive edition pairs
    for i in range(len(editions) - 1):
        ed_current = editions[i]
        ed_next = editions[i + 1]

        works_current = set(df.loc[df["anthology_edition"] == ed_current, "work_id"])
        works_next = set(df.loc[df["anthology_edition"] == ed_next, "work_id"])

        n_current = len(works_current)
        reselected = len(works_current & works_next)
        p = reselected / n_current if n_current else 0.0
        odds = p / (1 - p) if 0 < p < 1 else float("inf") if p == 1 else 0.0
        se = math.sqrt(p * (1 - p) / n_current) if n_current else 0.0

        rows.append(
            {
                "Edition": ed_current,
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
    parser = argparse.ArgumentParser(
        description="Compute work reselection probabilities between anthology editions."
    )
    parser.add_argument(
        "input_csv",
        help='Path to the input CSV (must include "anthology_edition" and "work_id" columns)',
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    summary_df, overall_p, overall_odds = compute_reselection_summary(df)

    # write summary CSV
    base = os.path.splitext(os.path.basename(args.input_csv))[0]
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_work_reselection_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"Summary CSV written to {out_path}")

    # print overall stats
    print(
        f"Overall probability of work reselection across all editions: {overall_p:.4f}"
    )
    print(
        f"Overall odds of work reselection across all editions:       {overall_odds:.4f}"
    )


if __name__ == "__main__":
    main()
