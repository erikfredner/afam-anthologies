#!/usr/bin/env python3
"""
form_by_edition.py

Reads works with their literary form and edition from the live database, then:

1. Computes, for each edition:
     - number of works per literary form
     - proportion of works per form
   and writes this table to `data/form_by_edition_summary.csv`.

2. Creates a stacked‐bar chart showing the percentage distribution of forms
   across editions and saves it to `output/form_by_edition_distribution.png`.

Usage:
    uv run python viz/misc/form_by_edition.py
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt

from afam import DATA_DIR
from afam.db import query as query_db
from afam.editions import EDITION_LABELS
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR


def load_from_db() -> pd.DataFrame:
    """One row per (work, edition) over AFAM editions, with the work's literary
    form. Multi-form works collapse to the lowest form_id (work-forms-afam.sql);
    excerpts inherit their parent work's form. Each edition is labelled
    "<year> <name>" so chronological order falls out of sort_index()."""
    wide = query_db(query_path("works-authors-per-afam-edition"))
    forms = query_db(query_path("work-forms-afam"))
    form_map = dict(zip(forms["work_id"].astype(int), forms["form_name"]))

    base = (
        wide[["work_id", "parent_id", "edition_id", "anthology_publication_year"]]
        .drop_duplicates(["work_id", "edition_id"])
        .copy()
    )

    def resolve_form(row) -> str:
        wid = int(row["work_id"])
        if wid in form_map:
            return form_map[wid]
        pid = row["parent_id"]
        if pd.notna(pid) and int(pid) in form_map:
            return form_map[int(pid)]
        return "Unknown"

    base["literary_form"] = base.apply(resolve_form, axis=1)
    base["anthology_edition"] = [
        f"{int(y)} {EDITION_LABELS.get(int(e), e)}"
        for e, y in zip(base["edition_id"], base["anthology_publication_year"])
    ]
    return base[["work_id", "anthology_edition", "literary_form"]]


def compute_counts(df):
    # pivot to get counts of unique work_id per edition×form
    counts = (
        df.groupby(["anthology_edition", "literary_form"])["work_id"]
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
    summary = counts.reset_index().melt(
        id_vars="anthology_edition", var_name="literary_form", value_name="count"
    )
    # grab matching proportions
    summary["proportion"] = summary.apply(
        lambda r: props.loc[r["anthology_edition"], r["literary_form"]], axis=1
    )
    # rename for readability
    summary = summary.rename(
        columns={
            "anthology_edition": "Edition",
            "count": "Count",
            "proportion": "Proportion",
        }
    )
    summary.to_csv(out_csv, index=False)


def plot_distribution(props, out_png):
    # convert to percentages
    pct = props * 100
    ax = pct.plot(kind="bar", stacked=True, figsize=(8, 5), legend=True)
    ax.set_xlabel("Edition")
    ax.set_ylabel("Percentage of works")
    ax.set_title("Literary‐form distribution by edition")
    ax.legend(title="Form", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    argparse.ArgumentParser(
        description="Compute and plot literary‐form breakdown by edition."
    ).parse_args()

    df = load_from_db()
    counts = compute_counts(df)
    props = compute_proportions(counts)

    summary_csv = DATA_DIR / "form_by_edition_summary.csv"
    write_summary(counts, props, summary_csv)
    print(f"Form summary written to {summary_csv}")

    plot_png = OUTPUT_DIR / "form_by_edition_distribution.png"
    plot_distribution(props, plot_png)
    print(f"Form distribution plot saved to {plot_png}")


if __name__ == "__main__":
    main()
