"""Identify pre-Norton selections absent from contemporary anthologies.

Ranks authors and works with the strongest records before the first edition of
The Norton Anthology of African American Literature, limited to entities with
zero selections in AFAM anthologies from 2010 onward.

Usage:
    uv run python analysis/reselection/early_selection_dropouts.py
    uv run python analysis/reselection/early_selection_dropouts.py --save-csv
    uv run python analysis/reselection/early_selection_dropouts.py --include-excerpts
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import pandas as pd

from afam import DATA_DIR
from afam.cli import add_root_works_flag, add_save_csv_flag
from afam.db import query
from afam.sql import query_path

NORTON_SERIES_ID = "3"
CONTEMPORARY_FROM = 2010
TOP_N = 25
OUT_AUTHORS = DATA_DIR / "early_selection_dropouts_authors.csv"
OUT_WORKS = DATA_DIR / "early_selection_dropouts_works.csv"


def _clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _is_root_parent(value: object) -> bool:
    return _clean_id(value) == ""


def _join_years(years: Iterable[object]) -> str:
    vals = sorted({int(y) for y in years if not pd.isna(y)})
    return ", ".join(str(y) for y in vals)


def _first_nonempty(values: pd.Series) -> object:
    for value in values:
        if _clean_id(value):
            return value
    return values.iloc[0] if len(values) else ""


def _join_names(values: pd.Series) -> str:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        name = str(value).strip()
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return "; ".join(names)


def load_data() -> pd.DataFrame:
    df = query(query_path("work-selection-divergence"))
    df["anthology_publication_year"] = df["anthology_publication_year"].astype(int)
    return df


def filter_root_works(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["parent_id"].map(_is_root_parent)].copy()


def detect_norton_first_year(
    df: pd.DataFrame, norton_series_id: str = NORTON_SERIES_ID
) -> int:
    mask = df["series_id"].map(_clean_id) == _clean_id(norton_series_id)
    years = df.loc[mask, "anthology_publication_year"]
    if years.empty:
        raise ValueError(f"No Norton/NAAAL rows found for series_id={norton_series_id}")
    return int(years.min())


def count_period_editions(
    df: pd.DataFrame, early_before: int, contemporary_from: int
) -> dict[str, int]:
    editions = (
        df[["edition_id", "anthology_publication_year"]]
        .dropna(subset=["edition_id"])
        .drop_duplicates("edition_id")
        .copy()
    )
    return {
        "early": int((editions["anthology_publication_year"] < early_before).sum()),
        "post_early_pre_contemporary": int(
            (
                (editions["anthology_publication_year"] >= early_before)
                & (editions["anthology_publication_year"] < contemporary_from)
            ).sum()
        ),
        "contemporary": int(
            (editions["anthology_publication_year"] >= contemporary_from).sum()
        ),
        "total": len(editions),
    }


def compute_selection_stats(
    pairs: pd.DataFrame,
    id_col: str,
    early_before: int,
    contemporary_from: int,
    early_edition_count: int,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                id_col,
                "early_selection_count",
                "early_selection_rate",
                "early_years",
                "first_early_year",
                "last_early_year",
                "total_selection_count",
                "post_early_pre_contemporary_count",
                "contemporary_selection_count",
            ]
        )

    pairs = (
        pairs[[id_col, "edition_id", "anthology_publication_year"]]
        .dropna(subset=[id_col, "edition_id"])
        .drop_duplicates([id_col, "edition_id"])
        .copy()
    )
    pairs["anthology_publication_year"] = pairs["anthology_publication_year"].astype(int)

    rows: list[dict] = []
    for entity_id, grp in pairs.groupby(id_col, sort=False):
        early = grp[grp["anthology_publication_year"] < early_before]
        middle = grp[
            (grp["anthology_publication_year"] >= early_before)
            & (grp["anthology_publication_year"] < contemporary_from)
        ]
        contemporary = grp[grp["anthology_publication_year"] >= contemporary_from]

        early_count = len(early)
        rows.append(
            {
                id_col: entity_id,
                "early_selection_count": early_count,
                "early_selection_rate": (
                    early_count / early_edition_count if early_edition_count else 0.0
                ),
                "early_years": _join_years(early["anthology_publication_year"]),
                "first_early_year": (
                    int(early["anthology_publication_year"].min())
                    if early_count
                    else float("nan")
                ),
                "last_early_year": (
                    int(early["anthology_publication_year"].max())
                    if early_count
                    else float("nan")
                ),
                "total_selection_count": len(grp),
                "post_early_pre_contemporary_count": len(middle),
                "contemporary_selection_count": len(contemporary),
            }
        )

    return pd.DataFrame(rows)


def _filter_and_rank(
    df: pd.DataFrame,
    *,
    min_early_count: int,
    label_col: str,
) -> pd.DataFrame:
    out = df[
        (df["early_selection_count"] >= min_early_count)
        & (df["contemporary_selection_count"] == 0)
    ].copy()
    return out.sort_values(
        ["early_selection_count", "early_selection_rate", "last_early_year", label_col],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_author_dropouts(
    df: pd.DataFrame,
    early_before: int,
    contemporary_from: int,
    early_edition_count: int,
    min_early_count: int,
) -> pd.DataFrame:
    pairs = (
        df[["author_id", "edition_id", "anthology_publication_year"]]
        .dropna(subset=["author_id", "edition_id"])
        .drop_duplicates(["author_id", "edition_id"])
    )
    stats = compute_selection_stats(
        pairs, "author_id", early_before, contemporary_from, early_edition_count
    )
    if stats.empty:
        return stats

    names = (
        df[["author_id", "author_name"]]
        .dropna(subset=["author_id"])
        .groupby("author_id", sort=False)["author_name"]
        .agg(_first_nonempty)
        .reset_index()
    )
    out = names.merge(stats, on="author_id", how="inner")
    cols = [
        "author_id",
        "author_name",
        "early_selection_count",
        "early_selection_rate",
        "early_years",
        "first_early_year",
        "last_early_year",
        "total_selection_count",
        "post_early_pre_contemporary_count",
        "contemporary_selection_count",
    ]
    return _filter_and_rank(out[cols], min_early_count=min_early_count, label_col="author_name")


def build_work_dropouts(
    df: pd.DataFrame,
    early_before: int,
    contemporary_from: int,
    early_edition_count: int,
    min_early_count: int,
) -> pd.DataFrame:
    pairs = (
        df[["work_id", "edition_id", "anthology_publication_year"]]
        .dropna(subset=["work_id", "edition_id"])
        .drop_duplicates(["work_id", "edition_id"])
    )
    stats = compute_selection_stats(
        pairs, "work_id", early_before, contemporary_from, early_edition_count
    )
    if stats.empty:
        return stats

    meta = (
        df[["work_id", "work_title", "parent_id", "author_name"]]
        .dropna(subset=["work_id"])
        .groupby("work_id", sort=False)
        .agg(
            work_title=("work_title", _first_nonempty),
            parent_id=("parent_id", _first_nonempty),
            author_names=("author_name", _join_names),
        )
        .reset_index()
    )
    out = meta.merge(stats, on="work_id", how="inner")
    cols = [
        "work_id",
        "work_title",
        "author_names",
        "parent_id",
        "early_selection_count",
        "early_selection_rate",
        "early_years",
        "first_early_year",
        "last_early_year",
        "total_selection_count",
        "post_early_pre_contemporary_count",
        "contemporary_selection_count",
    ]
    return _filter_and_rank(out[cols], min_early_count=min_early_count, label_col="work_title")


def _print_table(title: str, df: pd.DataFrame, top: int, columns: list[str]) -> None:
    print(f"\n{title} ({len(df):,} total)")
    if df.empty:
        print("  No matching rows.")
        return
    shown = df.head(top).copy()
    if "early_selection_rate" in shown:
        shown["early_selection_rate"] = shown["early_selection_rate"].map(
            lambda x: f"{x:.3f}"
        )
    print(shown[columns].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contemporary-from",
        type=int,
        default=CONTEMPORARY_FROM,
        help=f"Year at or after which selections count as contemporary (default: {CONTEMPORARY_FROM})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=TOP_N,
        help=f"Number of rows to print per table (default: {TOP_N})",
    )
    parser.add_argument(
        "--min-early-count",
        type=int,
        default=1,
        help="Minimum pre-Norton edition count required for inclusion (default: 1)",
    )
    add_root_works_flag(parser)
    add_save_csv_flag(parser)
    args = parser.parse_args()

    raw = load_data()
    norton_first_year = detect_norton_first_year(raw)
    period_counts = count_period_editions(
        raw, early_before=norton_first_year, contemporary_from=args.contemporary_from
    )

    df = filter_root_works(raw) if args.only_root_works else raw.copy()
    authors = build_author_dropouts(
        df,
        early_before=norton_first_year,
        contemporary_from=args.contemporary_from,
        early_edition_count=period_counts["early"],
        min_early_count=args.min_early_count,
    )
    works = build_work_dropouts(
        df,
        early_before=norton_first_year,
        contemporary_from=args.contemporary_from,
        early_edition_count=period_counts["early"],
        min_early_count=args.min_early_count,
    )

    mode = "root works only" if args.only_root_works else "including excerpts"
    print("-- Early Selection Dropouts ------------------------------------------")
    print(f"First Norton/NAAAL year       : {norton_first_year}")
    print(f"Early period                  : < {norton_first_year}")
    print(f"Contemporary period           : >= {args.contemporary_from}")
    print(f"Mode                          : {mode}")
    print(f"Early editions                : {period_counts['early']}")
    print(f"Contemporary editions         : {period_counts['contemporary']}")
    print(f"Minimum early selections      : {args.min_early_count}")

    _print_table(
        "Authors absent from contemporary anthologies",
        authors,
        args.top,
        [
            "author_name",
            "early_selection_count",
            "early_selection_rate",
            "early_years",
            "total_selection_count",
            "post_early_pre_contemporary_count",
        ],
    )
    _print_table(
        "Works absent from contemporary anthologies",
        works,
        args.top,
        [
            "work_title",
            "author_names",
            "early_selection_count",
            "early_selection_rate",
            "early_years",
            "total_selection_count",
            "post_early_pre_contemporary_count",
        ],
    )

    if args.save_csv:
        authors.to_csv(OUT_AUTHORS, index=False)
        works.to_csv(OUT_WORKS, index=False)
        print(f"\nSaved -> {OUT_AUTHORS}")
        print(f"Saved -> {OUT_WORKS}")


if __name__ == "__main__":
    main()
