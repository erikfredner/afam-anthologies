"""women_author_gaps.py
----------------------
Table of women authors in the gender-annotated anthology dataset, showing
how many anthology editions each was selected for and the largest temporal
gap between selections (trailing gap to 2026 always included).

Usage:
    uv run python analysis/women_author_gaps.py
    uv run python analysis/women_author_gaps.py --threshold 5
"""

from __future__ import annotations

import argparse
import pandas as pd

from afam import DATA_DIR
from afam.db import query as query_db
from afam.sql import query_path

OUT_CSV = "women_author_gaps.csv"

PRESENT_YEAR = 2026


# ── Gap computation ───────────────────────────────────────────────────────────


def largest_gap(years: list[int]) -> int:
    """Return the largest gap between consecutive selections, including the
    trailing gap from the last selection year to PRESENT_YEAR."""
    years = sorted(set(years))
    gaps = [years[i + 1] - years[i] for i in range(len(years) - 1)]
    gaps.append(PRESENT_YEAR - years[-1])
    return max(gaps)


# ── Aggregation ───────────────────────────────────────────────────────────────


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """`df` is the gender-work-consistency frame (author_id, author_name,
    gender, edition_id, edition_year)."""
    df = df.dropna(subset=["author_id"]).copy()
    df = df[df["gender"].astype(str).str.lower() == "female"]

    records = []
    for (author_id, author_name), grp in df.groupby(
        ["author_id", "author_name"], sort=False
    ):
        # One year per edition (min year for that edition)
        ek_year = grp.groupby("edition_id")["edition_year"].min()
        years = sorted(int(y) for y in ek_year.values)

        records.append(
            {
                "author_id": author_id,
                "Author": author_name,
                "Gender": "Female",
                "Anthologies": grp["edition_id"].nunique(),
                "First selected": years[0],
                "Last selected": years[-1],
                "Largest gap (yrs)": largest_gap(years),
            }
        )

    result = (
        pd.DataFrame(records)
        .sort_values(["Anthologies", "Author"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return result


# ── Output table ──────────────────────────────────────────────────────────────


def print_table(table: pd.DataFrame) -> None:
    cols = [
        "Author",
        "Gender",
        "Anthologies",
        "First selected",
        "Last selected",
        "Largest gap (yrs)",
    ]
    widths = [40, 6, 11, 14, 13, 18]
    align = ["left", "left", "right", "right", "right", "right"]

    def fmt(val: object, width: int, a: str) -> str:
        s = str(val)
        return s.ljust(width) if a == "left" else s.rjust(width)

    header = " | ".join(fmt(c, w, a) for c, w, a in zip(cols, widths, align))
    sep = "-" * len(header)
    print(header)
    print(sep)

    for _, row in table.iterrows():
        print(" | ".join(fmt(row[c], w, a) for c, w, a in zip(cols, widths, align)))


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Women authors: anthology count and largest selection gap."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        metavar="N",
        help="Only show authors appearing in at least N anthologies.",
    )
    args = parser.parse_args()

    df = query_db(query_path("gender-work-consistency"))
    table = build_table(df)

    if args.threshold is not None:
        table = table[table["Anthologies"] >= args.threshold].reset_index(drop=True)

    out_path = DATA_DIR / OUT_CSV
    table.drop(columns=["author_id"]).to_csv(out_path, index=False)
    print(f"Saved → {out_path}  ({len(table):,} authors)")
    print()
    print_table(table)


if __name__ == "__main__":
    main()
