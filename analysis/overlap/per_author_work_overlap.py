"""
per_author_work_overlap.py
---------------------------
Per-author analysis of work overlap across anthology editions.

For each author who appears in two or more editions from *different* series,
report:
  - author metadata (id, name, birth/death year)
  - number of editions in which the author appears
  - mean/median number of works per anthologization
  - min/mean/median/max # works overlapping between cross-series edition pairs
  - the author's work appearing in the most cross-series edition pairs

Within-series pairs (e.g. NAAAL 1997 vs NAAAL 2004) are excluded — successive
editions of the same series tend to inherit prior selections, so within-series
overlap reflects path-dependence rather than independent agreement. Editions
without a series (e.g. Call and Response 1998) are treated as their own
singleton series, so they are always cross-series with everything else.

Authors with zero cross-series edition pairs are excluded from the output.

By default, writes two CSV variants to data/:
  - per_author_work_overlap_all.csv        (root works + excerpts)
  - per_author_work_overlap_root_only.csv  (works with no parent only)
Pass --no-csv to skip both writes.

Usage:
    uv run python analysis/overlap/per_author_work_overlap.py
    uv run python analysis/overlap/per_author_work_overlap.py --no-csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from statistics import mean, median

import pandas as pd

from afam import DATA_DIR
from afam.db import query as query_db
from afam.sql import query_path

OUT_CSV_ALL = DATA_DIR / "per_author_work_overlap_all.csv"
OUT_CSV_ROOT = DATA_DIR / "per_author_work_overlap_root_only.csv"


def _series_identity(series_id, edition_id) -> str:
    if pd.notna(series_id):
        return f"series_{int(series_id)}"
    return f"standalone_{int(edition_id)}"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-author cross-series overlap statistics.

    Expected columns: author_id, author_name, author_birth_year,
    author_death_year, work_id, work_title, edition_id, series_id.
    Rows with null author_id must already be removed.
    """
    df = df.copy()
    df["series_identity"] = [
        _series_identity(s, e) for s, e in zip(df["series_id"], df["edition_id"])
    ]

    work_title: dict[int, str] = dict(
        zip(df["work_id"].astype(int), df["work_title"])
    )

    rows: list[dict] = []
    for aid, grp in df.groupby("author_id"):
        ed_to_works: dict[int, set[int]] = defaultdict(set)
        ed_to_series: dict[int, str] = {}
        for _, r in grp.iterrows():
            eid = int(r["edition_id"])
            ed_to_works[eid].add(int(r["work_id"]))
            ed_to_series[eid] = r["series_identity"]

        per_ed_counts = [len(s) for s in ed_to_works.values()]
        n_editions = len(ed_to_works)

        overlaps: list[int] = []
        pair_counts: dict[int, int] = defaultdict(int)
        for ei, ej in combinations(ed_to_works.keys(), 2):
            if ed_to_series[ei] == ed_to_series[ej]:
                continue
            shared = ed_to_works[ei] & ed_to_works[ej]
            overlaps.append(len(shared))
            for w in shared:
                pair_counts[w] += 1

        if not overlaps:
            continue

        if pair_counts:
            # Highest pair count; ties broken alphabetically by title.
            best = min(
                pair_counts.keys(),
                key=lambda w: (-pair_counts[w], work_title[w]),
            )
            top_work_id = best
            top_work_title = work_title[best]
            top_work_pair_count = pair_counts[best]
        else:
            top_work_id = None
            top_work_title = None
            top_work_pair_count = 0

        first_row = grp.iloc[0]
        rows.append(
            {
                "author_id": int(aid),
                "author_name": first_row["author_name"],
                "author_birth_year": (
                    int(first_row["author_birth_year"])
                    if pd.notna(first_row["author_birth_year"])
                    else None
                ),
                "author_death_year": (
                    int(first_row["author_death_year"])
                    if pd.notna(first_row["author_death_year"])
                    else None
                ),
                "n_editions": n_editions,
                "mean_works_per_edition": round(mean(per_ed_counts), 3),
                "median_works_per_edition": median(per_ed_counts),
                "n_cross_series_pairs": len(overlaps),
                "min_overlap": min(overlaps),
                "mean_overlap": round(mean(overlaps), 3),
                "median_overlap": median(overlaps),
                "max_overlap": max(overlaps),
                "top_work_id": top_work_id,
                "top_work_title": top_work_title,
                "top_work_pair_count": top_work_pair_count,
            }
        )

    out = pd.DataFrame(rows)
    for col in ("author_birth_year", "author_death_year", "top_work_id"):
        if col in out.columns:
            out[col] = out[col].astype("Int64")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="skip writing CSVs; print the tables to stdout only",
    )
    args = parser.parse_args()

    raw = query_db(query_path("work-selection-divergence"))
    raw = raw[raw["author_id"].notna()].copy()

    variants = [
        ("all works (root + excerpts)", OUT_CSV_ALL, raw),
        ("root works only", OUT_CSV_ROOT, raw[raw["parent_id"].isna()].copy()),
    ]

    for label, out_csv, df in variants:
        table = compute(df).sort_values(
            ["n_editions", "n_cross_series_pairs", "max_overlap", "author_name"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

        print(f"\n=== {label} ===")
        display = table.drop(columns=["top_work_id"])
        with pd.option_context("display.max_rows", None, "display.width", 240):
            print(display.to_string(index=False))
        print(f"\n{len(table):,} authors with at least one cross-series edition pair")

        if not args.no_csv:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(out_csv, index=False)
            print(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()
