"""
works_in_half_or_more_afam_eds.py
----------------------------------
Lists works appearing in at least half of all African American literary
anthology editions, queried live from the database.

Output columns:
  Work        — '"Title" by Author'
  Anthologies — number of distinct editions that selected the work
"""

from __future__ import annotations

import argparse

import pandas as pd

from afam import DATA_DIR
from afam.db import query
from afam.sql import query_path

OUT_CSV = DATA_DIR / "works_in_half_or_more_afam_eds.csv"


def join_authors(names: pd.Series) -> str:
    unique = list(dict.fromkeys(names))
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return ", ".join(unique[:-1]) + f", and {unique[-1]}"


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["work_id", "work_title", "edition_count"], sort=False)
        .agg(author_name=("author_name", join_authors))
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["edition_count", "author_name", "work_title"],
        ascending=[False, True, True],
    )
    result = pd.DataFrame({
        "Work":        grouped.apply(lambda r: f'"{r["work_title"]}" by {r["author_name"]}', axis=1),
        "Anthologies": grouped["edition_count"],
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

    raw = query(query_path("works-in-half-or-more-afam-eds"))
    table = build_table(raw)

    print(table.to_string(index=False))
    print(f"\n{len(table)} works")

    if args.save_csv:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_CSV, index=False)
        print(f"Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
