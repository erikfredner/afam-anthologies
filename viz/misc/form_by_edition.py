#!/usr/bin/env python3
"""
form_by_edition.py

Reads an input CSV of works with their literary form and edition, then:

1. Computes, for each edition:
     - number of works per literary form
     - proportion of works per form
   and writes this table to `data/<input_basename>_form_summary.csv`.

2. Creates a stacked‐bar chart showing the percentage distribution of forms
   across editions and saves it to `viz/<input_basename>_form_distribution.png`.

Sample CLI usage:
    python form_by_edition.py path/to/input.csv
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from afam import DATA_DIR
from afam.viz_style import OUTPUT_DIR

def load_and_clean(path):
    df = pd.read_csv(path, dtype=str)
    # normalize empty or missing forms
    df['literary_form'] = df['literary_form'].fillna('Unknown')
    df.loc[df['literary_form'].str.strip() == '', 'literary_form'] = 'Unknown'
    df['anthology_edition'] = df['anthology_edition'].astype(int)
    # ensure each work only counted once per edition
    return df.drop_duplicates(subset=['work_id','anthology_edition'])

def compute_counts(df):
    # pivot to get counts of unique work_id per edition×form
    counts = (
        df
        .groupby(['anthology_edition','literary_form'])['work_id']
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    return counts

def compute_proportions(counts):
    # row‐normalize so each edition sums to 1.0
    return counts.div(counts.sum(axis=1), axis=0)

def write_summary(counts, props, out_csv):
    # flatten back into long form
    summary = (
        counts
        .reset_index()
        .melt(id_vars='anthology_edition', var_name='literary_form', value_name='count')
    )
    # grab matching proportions
    summary['proportion'] = summary.apply(
        lambda r: props.loc[r['anthology_edition'], r['literary_form']],
        axis=1
    )
    # rename for readability
    summary = summary.rename(columns={
        'anthology_edition': 'Edition',
        'count': 'Count',
        'proportion': 'Proportion'
    })
    summary.to_csv(out_csv, index=False)

def plot_distribution(props, out_png):
    # convert to percentages
    pct = props * 100
    ax = pct.plot(
        kind='bar',
        stacked=True,
        figsize=(8, 5),
        legend=True
    )
    ax.set_xlabel('Edition')
    ax.set_ylabel('Percentage of works')
    ax.set_title('Literary‐form distribution by edition')
    ax.legend(title='Form', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    p = argparse.ArgumentParser(
        description="Compute and plot literary‐form breakdown by edition."
    )
    p.add_argument('input_csv', help='CSV with work_id, literary_form, anthology_edition columns')
    args = p.parse_args()

    df = load_and_clean(args.input_csv)
    counts = compute_counts(df)
    props  = compute_proportions(counts)

    base = os.path.splitext(os.path.basename(args.input_csv))[0]

    summary_csv = DATA_DIR / f"{base}_form_summary.csv"
    write_summary(counts, props, summary_csv)
    print(f"Form summary written to {summary_csv}")

    plot_png = OUTPUT_DIR / f"{base}_form_distribution.png"
    plot_distribution(props, plot_png)
    print(f"Form distribution plot saved to {plot_png}")

if __name__ == '__main__':
    main()
