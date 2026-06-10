"""
authors_in_half_or_more_afam_eds.py
------------------------------------
Lists authors appearing in at least half of all African American literary
anthology editions, queried live from the database.

Output columns:
  Author       — author name
  Selections   — number of distinct editions that selected the author
  Reselections — later selections / later opportunities
"""

from __future__ import annotations

import argparse

import pandas as pd

from afam import DATA_DIR
from afam.db import query
from afam.names import author_sort_key
from afam.sql import query_path

OUT_CSV = DATA_DIR / "authors_in_half_or_more_afam_eds.csv"


def format_reselections(row: pd.Series) -> str:
    selections = row["reselection_count"]
    opportunities = row["opportunities"]
    return f"{int(selections)}/{int(opportunities)}"


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.assign(author_sort_key=df["author_name"].map(author_sort_key))
        .sort_values(
            ["edition_count", "author_sort_key"],
            ascending=[False, True],
        )
    )
    result = pd.DataFrame({
        "Author":       df["author_name"],
        "Selections":   df["edition_count"],
        "Reselections": df.apply(format_reselections, axis=1),
    })
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help=f"Write results to {OUT_CSV}",
    )
    args = parser.parse_args()

    raw = query(query_path("authors-in-half-or-more-afam-eds"))
    table = build_table(raw)

    print(table.to_string(index=False))
    print(f"\n{len(table)} authors")

    if args.save_csv:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_CSV, index=False)
        print(f"Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
