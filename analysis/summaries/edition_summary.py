#!/usr/bin/env python3
"""
edition_summary.py

Reads an input CSV of anthology works and computes, for each edition:
  - total works,
  - works reselected from the previous edition,
  - percentage reselected.

Outputs a summary CSV to the `data/` directory.

Sample CLI usage:
    python edition_summary.py path/to/input.csv
"""

import os
import argparse
import pandas as pd

def compute_edition_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure correct dtypes
    df['anthology_edition'] = df['anthology_edition'].astype(int)
    df = df.dropna(subset=['work_id', 'anthology_edition'])

    editions = sorted(df['anthology_edition'].unique())
    summary_rows = []

    prev_ids = set()
    for ed in editions:
        current_ids = set(df.loc[df['anthology_edition'] == ed, 'work_id'])
        total = len(current_ids)
        reselected = len(current_ids & prev_ids)
        pct = (reselected / total * 100) if total else 0.0

        summary_rows.append({
            'Edition': ed,
            'Works': total,
            'Works reselected': reselected,
            'Works reselected %': round(pct, 2)
        })

        prev_ids = current_ids

    return pd.DataFrame(summary_rows)

def main():
    parser = argparse.ArgumentParser(
        description="Summarize anthology editions from a works CSV."
    )
    parser.add_argument(
        'input_csv',
        help='Path to the input CSV file'
    )
    args = parser.parse_args()

    # Read the input CSV
    df = pd.read_csv(args.input_csv)

    # Compute the edition summary
    summary_df = compute_edition_summary(df)

    # Prepare output path
    input_basename = os.path.splitext(os.path.basename(args.input_csv))[0]
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{input_basename}_edition_summary.csv")

    # Write to CSV
    summary_df.to_csv(output_path, index=False)
    print(f"Summary written to {output_path}")

if __name__ == '__main__':
    main()
