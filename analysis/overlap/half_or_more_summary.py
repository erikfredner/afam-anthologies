"""
half_or_more_summary.py
-----------------------
Summary table of unique authors/works and how many of each appear in half or
more of all African American literary anthology editions, computed three ways:

  All works  — every work row counts, excerpts included
  Root works — excerpt rows (parent_id set) are dropped before counting
  Coalesced  — each work is grouped with its family via COALESCE(parent_id, id),
               so a root work and its excerpts count as one work whose edition
               count covers appearances by any family member

Author counts under "Coalesced" equal "All works" by construction: grouping
works into families does not change which editions an author appears in.

Output rows:
  Unique authors    — distinct authors appearing in any tagged edition
  Authors in ≥ half — authors in at least half of the tagged editions
  Unique works      — distinct works (or families) appearing in any edition
  Works in ≥ half   — works (or families) in at least half of the editions
"""

from __future__ import annotations

import argparse

import pandas as pd

from afam import DATA_DIR
from afam.cli import add_save_csv_flag
from afam.db import query
from afam.sql import query_path

OUT_CSV = DATA_DIR / "half_or_more_summary.csv"

VARIANT_LABELS = {
    "all": "All works (incl. excerpts)",
    "root": "Root works only",
    "coalesced": "Coalesced (excerpts → root)",
}


def variant_stats(
    df: pd.DataFrame, work_key: pd.Series, total_editions: int
) -> dict[str, int]:
    """Unique/half-or-more counts for one variant.

    `work_key` is an index-aligned Series identifying the work (or work
    family) each appearance row belongs to.
    """
    work_editions = df.groupby(work_key)["edition_id"].nunique()
    author_editions = (
        df.dropna(subset=["author_id"]).groupby("author_id")["edition_id"].nunique()
    )
    return {
        "unique_authors": len(author_editions),
        "authors_half": int((author_editions * 2 >= total_editions).sum()),
        "unique_works": len(work_editions),
        "works_half": int((work_editions * 2 >= total_editions).sum()),
    }


def compute_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return (summary, total_editions); summary has one row per metric and
    one integer column per variant."""
    total_editions = df["edition_id"].nunique()

    root = df[df["parent_id"].isna()]
    coalesced_key = df["parent_id"].fillna(df["work_id"])

    stats = {
        "all": variant_stats(df, df["work_id"], total_editions),
        "root": variant_stats(root, root["work_id"], total_editions),
        "coalesced": variant_stats(df, coalesced_key, total_editions),
    }
    summary = pd.DataFrame(stats).rename(columns=VARIANT_LABELS)
    summary.index = pd.Index(
        ["Unique authors", "Authors in ≥ half", "Unique works", "Works in ≥ half"],
        name="Metric",
    )
    return summary, total_editions


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Render counts as strings, with percentages on the '≥ half' rows."""
    formatted = summary.astype(object).copy()
    for col in summary.columns:
        formatted.loc["Unique authors", col] = f"{summary.loc['Unique authors', col]:,}"
        formatted.loc["Unique works", col] = f"{summary.loc['Unique works', col]:,}"
        formatted.loc["Authors in ≥ half", col] = (
            f"{summary.loc['Authors in ≥ half', col]:,} "
            f"({summary.loc['Authors in ≥ half', col] / summary.loc['Unique authors', col]:.1%})"
        )
        formatted.loc["Works in ≥ half", col] = (
            f"{summary.loc['Works in ≥ half', col]:,} "
            f"({summary.loc['Works in ≥ half', col] / summary.loc['Unique works', col]:.1%})"
        )
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_save_csv_flag(parser)
    args = parser.parse_args()

    raw = query(query_path("works-per-afam-edition"))
    summary, total_editions = compute_summary(raw)
    table = format_summary(summary)

    threshold = -(-total_editions // 2)  # ceil(total / 2)
    print(f'{total_editions} tagged editions; "half or more" = ≥{threshold}\n')
    print(table.to_string())

    if args.save_csv:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_CSV)
        print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
