"""
authors_in_half_or_more_afam_eds.py
------------------------------------
Lists authors appearing in at least half of all African American literary
anthology editions, queried live from the database.

Output columns:
  Author      — "Name (b. YYYY)" or "Name" when birth year is unknown
  Anthologies — number of distinct editions that selected the author
"""

from __future__ import annotations

import argparse

import pandas as pd

from afam import DATA_DIR
from afam.db import query
from afam.sql import query_path

OUT_CSV = DATA_DIR / "authors_in_half_or_more_afam_eds.csv"


def format_author(name: str, birth_year) -> str:
    if birth_year and str(birth_year).strip() and not (
        isinstance(birth_year, float) and birth_year != birth_year  # NaN check
    ):
        return f"{name} (b. {int(birth_year)})"
    return name


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["edition_count", "birth_year"], ascending=[False, True], na_position="last")
    result = pd.DataFrame({
        "Author":      df.apply(lambda r: format_author(r["author_name"], r["birth_year"]), axis=1),
        "Anthologies": df["edition_count"],
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
