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
from pathlib import Path

import pandas as pd


DATA_FILE = (
    Path(__file__).parent.parent
    / "data"
    / "2026-03-17 af am anthology authors with genders.csv"
)
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_CSV = "women_author_gaps.csv"

PRESENT_YEAR = 2026


# ── Edition-key logic ────────────────────────────────────────────────────────


def assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["edition_key"] = df.apply(
        lambda r: (
            f"{r['series_id']}|{r['edition']}"
            if r["series_id"].strip()
            else r["anthology_id"]
        ),
        axis=1,
    )
    return df


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
    df = df[df["gender"].str.strip().str.lower() == "female"].copy()
    df = assign_edition_key(df)
    df["publication_year"] = df["publication_year"].astype(int)

    records = []
    for (author_id, author_name, gender), grp in df.groupby(
        ["author_id", "author_name", "gender"], sort=False
    ):
        edition_keys = grp["edition_key"].unique()
        # One year per edition_key (min year for that edition)
        ek_year = grp.groupby("edition_key")["publication_year"].min()
        years = sorted(ek_year.values.tolist())

        records.append(
            {
                "author_id": author_id,
                "Author": author_name,
                "Gender": gender,
                "Anthologies": len(edition_keys),
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

    df = pd.read_csv(DATA_FILE, dtype=str, na_filter=False)
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
