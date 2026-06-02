"""
works_without_authors_naal_naaal.py
------------------------------------
Counts work selections without any associated author across all editions of:
  - The Norton Anthology of American Literature (full variant only)
  - The Norton Anthology of African American Literature

Output columns (per-edition):
  Anthology  — NAAL or NAAAL
  Year       — publication year
  Edition    — edition number
  Total      — distinct works selected in that edition (root works by default)
  No Author  — works with no entry in data_work_authors
  %          — percentage without author

Followed by a summary row per anthology across all editions.
"""

from __future__ import annotations

import argparse

import pandas as pd

from afam import DATA_DIR
from afam.db import query as run_query
from afam.sql import query_path

OUT_CSV = DATA_DIR / "works_without_authors_naal_naaal.csv"

ROOT_FILTER = "WHERE w.parent_id IS NULL"


def print_edition_table(group: pd.DataFrame, anthology: str) -> None:
    display = pd.DataFrame({
        "Year":      group["year"],
        "Edition":   group["edition_number"],
        "Total":     group["total_works"],
        "No Author": group["works_without_author"],
        "%":         group["pct_without_author"],
    })
    print(f"\n=== {anthology} ===")
    print(display.to_string(index=False))


def print_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("anthology", sort=True)
        .agg(
            editions=("year", "count"),
            total_works=("total_works", "sum"),
            works_without_author=("works_without_author", "sum"),
        )
        .reset_index()
    )
    summary["pct_without_author"] = (
        summary["works_without_author"] / summary["total_works"] * 100
    ).round(2)
    display = pd.DataFrame({
        "Anthology":   summary["anthology"],
        "Editions":    summary["editions"],
        "Total":       summary["total_works"],
        "No Author":   summary["works_without_author"],
        "%":           summary["pct_without_author"],
    })
    print("\n=== Summary (all editions) ===")
    print(display.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-excerpts",
        action="store_true",
        help="Include excerpt works (default: root works only)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help=f"Write per-edition results to {OUT_CSV}",
    )
    args = parser.parse_args()

    sql = query_path("works-without-authors-naal-naaal").read_text()
    if args.include_excerpts:
        sql = sql.replace(ROOT_FILTER, "")

    df = run_query(sql)

    for anthology, group in df.groupby("anthology", sort=True):
        print_edition_table(group, anthology)

    print_summary(df)

    if args.save_csv:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
