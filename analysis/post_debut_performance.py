"""
post_debut_performance.py
--------------------------
Analyzes the relative performance of authors and works after their initial
chronological selection in any African American literary anthology.

For every author and work, computes:
  - first_year:        year of initial selection
  - opportunities:     count of anthologies published in year >= first_year
  - selections:        count of those anthologies in which they appeared
  - selection_rate:    selections / opportunities
  - expected_rate:     mean selection probability across opportunity anthologies
                       (= included_count(A) / eligible_count(A) per anthology A,
                        where eligible = entities with first_year <= year(A))
  - obs_over_expected: selection_rate / expected_rate
  - p_value:           binomial test (alternative: greater)

Outputs:
  data/post_debut_performance_works.csv
  data/post_debut_performance_authors.csv

And prints to stdout top performers (significant p values, sorted by
obs_over_expected desc).

Usage:
    uv run python analysis/post_debut_performance.py
    uv run python analysis/post_debut_performance.py --only-root-works
    uv run python analysis/post_debut_performance.py --alpha 0.01
"""

from __future__ import annotations

import argparse
import re
from itertools import zip_longest
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


DATA_FILE = (
    Path(__file__).parent.parent / "data" / "2026-03-13 works per afam anthology.csv"
)
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_WORKS = "post_debut_performance_works.csv"
OUT_AUTHORS = "post_debut_performance_authors.csv"
DEFAULT_ALPHA = 0.05


# ── Edition-key logic (mirrors most_anthologized.py) ─────────────────────────


def _strip_volume(title: str) -> str:
    return re.sub(r",?\s+[Vv]ol\.?\s+\d+\s*$", "", title).strip()


def assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    meta_rows: list[dict] = []
    for _, r in (
        df[
            [
                "anthology_id",
                "anthology_title",
                "series_id",
                "anthology_edition",
                "anthology_volume",
            ]
        ]
        .drop_duplicates()
        .iterrows()
    ):
        series_id = r["series_id"].strip()
        edition = r["anthology_edition"].strip()
        volume = r["anthology_volume"].strip()
        title = r["anthology_title"].strip()
        aid = r["anthology_id"]
        if series_id:
            key = f"{series_id}|{edition}"
        elif volume:
            key = f"{_strip_volume(title)}|{edition}"
        else:
            key = aid
        meta_rows.append({"anthology_id": aid, "edition_key": key})
    return df.merge(pd.DataFrame(meta_rows), on="anthology_id")


# ── Formatting helpers ────────────────────────────────────────────────────────


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(only_root_works: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_full, df) where df has the optional root-work filter applied.

    Both DataFrames have an edition_key column added.
    """
    df_full = pd.read_csv(DATA_FILE, dtype=str, na_filter=False)
    df_full["anthology_publication_year"] = df_full[
        "anthology_publication_year"
    ].astype(int)
    df_full = assign_edition_key(df_full)
    if only_root_works:
        df = df_full[
            (df_full["parent_work_id"].str.strip() == "")
            & (df_full["parent_work_title"].str.strip() == "")
        ].copy()
    else:
        df = df_full
    return df_full, df


# ── Author expansion ──────────────────────────────────────────────────────────


def expand_author_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Expand multi-author rows into one row per (edition_key, author_id).

    author_ids split on "," ; author_names split on ";" ; author_birth_years on ",".
    """
    records: list[dict] = []
    for _, row in df[df["author_ids"] != ""].iterrows():
        ids = [i.strip() for i in row["author_ids"].split(",") if i.strip()]
        names = [n.strip() for n in row["author_names"].split(";")]
        bys = [b.strip() for b in row["author_birth_years"].split(",")]
        for aid, name, by in zip_longest(ids, names, bys, fillvalue=""):
            if not aid:
                continue
            records.append(
                {
                    "author_id": aid,
                    "author_name": name,
                    "author_birth_year": by,
                    "edition_key": row["edition_key"],
                    "anthology_publication_year": row["anthology_publication_year"],
                }
            )
    return pd.DataFrame(records)


# ── Anthology universe ────────────────────────────────────────────────────────


def build_anthology_universe(df_full: pd.DataFrame) -> dict[str, int]:
    """Return {edition_key: year} for all editions in the full dataset."""
    return df_full.groupby("edition_key")["anthology_publication_year"].min().to_dict()


# ── First-year per entity ─────────────────────────────────────────────────────


def work_first_years(df: pd.DataFrame) -> dict[str, int]:
    return df.groupby("work_id")["anthology_publication_year"].min().to_dict()


def author_first_years(expanded: pd.DataFrame) -> dict[str, int]:
    return expanded.groupby("author_id")["anthology_publication_year"].min().to_dict()


# ── Per-anthology pool sizes ──────────────────────────────────────────────────


def compute_anthology_pools(
    df: pd.DataFrame,
    expanded: pd.DataFrame,
    anthology_universe: dict[str, int],
    wfy: dict[str, int],
    afy: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Return {edition_key: {'p_work': float, 'p_author': float}}."""
    # Included sets per edition
    edition_works: dict[str, set[str]] = (
        df.groupby("edition_key")["work_id"].apply(set).to_dict()
    )
    edition_authors: dict[str, set[str]] = (
        expanded.groupby("edition_key")["author_id"].apply(set).to_dict()
    )

    pools: dict[str, dict[str, float]] = {}
    for ek, year in anthology_universe.items():
        eligible_w = {wid for wid, fy in wfy.items() if fy <= year}
        included_w = edition_works.get(ek, set())
        p_w = len(included_w) / len(eligible_w) if eligible_w else 0.0

        eligible_a = {au for au, fy in afy.items() if fy <= year}
        included_a = edition_authors.get(ek, set())
        p_a = len(included_a) / len(eligible_a) if eligible_a else 0.0

        pools[ek] = {"p_work": p_w, "p_author": p_a}

    return pools


# ── Per-entity stats ──────────────────────────────────────────────────────────


def compute_entity_stats(
    entity_ids: list[str],
    entity_first_years: dict[str, int],
    entity_anthology_sets: dict[str, set[str]],
    anthology_universe: dict[str, int],
    anthology_p: dict[str, float],
) -> list[dict]:
    rows: list[dict] = []
    for eid in entity_ids:
        first_year = entity_first_years[eid]
        opps = [aid for aid, yr in anthology_universe.items() if yr >= first_year]
        opportunities = len(opps)
        entity_set = entity_anthology_sets.get(eid, set())
        selections = sum(1 for aid in opps if aid in entity_set)
        selection_rate = selections / opportunities  # opportunities >= 1 always

        p_vals = [anthology_p[aid] for aid in opps]
        expected_rate = sum(p_vals) / len(p_vals)

        if expected_rate == 0.0:
            obs_over_expected = float("inf") if selections > 0 else float("nan")
            p_value = float("nan")
        else:
            obs_over_expected = selection_rate / expected_rate
            p_value = binomtest(
                selections, opportunities, p=expected_rate, alternative="greater"
            ).pvalue

        rows.append(
            {
                "entity_id": eid,
                "first_year": first_year,
                "opportunities": opportunities,
                "selections": selections,
                "selection_rate": selection_rate,
                "expected_rate": expected_rate,
                "obs_over_expected": obs_over_expected,
                "p_value": p_value,
            }
        )
    return rows


# ── Output DataFrames ─────────────────────────────────────────────────────────


def build_works_df(df: pd.DataFrame, stats_rows: list[dict]) -> pd.DataFrame:
    meta = (
        df[
            [
                "work_id",
                "work_title",
                "parent_work_id",
                "parent_work_title",
                "author_ids",
                "author_names",
            ]
        ]
        .drop_duplicates("work_id")
        .set_index("work_id")
    )
    stats = (
        pd.DataFrame(stats_rows)
        .rename(columns={"entity_id": "work_id"})
        .set_index("work_id")
    )
    result = meta.join(stats).reset_index()
    return result[
        [
            "work_id",
            "work_title",
            "parent_work_id",
            "parent_work_title",
            "author_ids",
            "author_names",
            "first_year",
            "opportunities",
            "selections",
            "selection_rate",
            "expected_rate",
            "obs_over_expected",
            "p_value",
        ]
    ]


def build_authors_df(expanded: pd.DataFrame, stats_rows: list[dict]) -> pd.DataFrame:
    meta = (
        expanded[["author_id", "author_name", "author_birth_year"]]
        .drop_duplicates("author_id")
        .set_index("author_id")
    )
    stats = (
        pd.DataFrame(stats_rows)
        .rename(columns={"entity_id": "author_id"})
        .set_index("author_id")
    )
    result = meta.join(stats).reset_index()
    return result[
        [
            "author_id",
            "author_name",
            "author_birth_year",
            "first_year",
            "opportunities",
            "selections",
            "selection_rate",
            "expected_rate",
            "obs_over_expected",
            "p_value",
        ]
    ]


# ── Stdout tables ─────────────────────────────────────────────────────────────


def print_works_table(works_df: pd.DataFrame, alpha: float) -> None:
    sig = works_df[works_df["p_value"] <= alpha].sort_values(
        "obs_over_expected", ascending=False, na_position="last"
    )
    print(f"\n===== TOP WORKS (p \u2264 {alpha}) =====")
    if sig.empty:
        print("  (no significant results)")
        return

    cols = [
        "Work Title",
        "Author(s)",
        "1st Yr",
        "Opps",
        "Sel",
        "Sel%",
        "Exp%",
        "Obs/Exp",
        "p-value",
    ]
    widths = [40, 25, 6, 4, 3, 6, 6, 7, 9]
    aligns = ["l", "l", "r", "r", "r", "r", "r", "r", "r"]

    header_parts = []
    for col, w, a in zip(cols, widths, aligns):
        header_parts.append(col.ljust(w) if a == "l" else col.rjust(w))
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for _, row in sig.iterrows():
        title = str(row["work_title"])[:40]
        authors = str(row["author_names"])[:25]
        vals = [
            title.ljust(40),
            authors.ljust(25),
            str(int(row["first_year"])).rjust(6),
            str(int(row["opportunities"])).rjust(4),
            str(int(row["selections"])).rjust(3),
            fmt_pct(row["selection_rate"]).rjust(6),
            fmt_pct(row["expected_rate"]).rjust(6),
            f"{row['obs_over_expected']:.2f}".rjust(7),
            fmt_p(row["p_value"]).rjust(9),
        ]
        print(" | ".join(vals))


def print_authors_table(authors_df: pd.DataFrame, alpha: float) -> None:
    sig = authors_df[authors_df["p_value"] <= alpha].sort_values(
        "obs_over_expected", ascending=False, na_position="last"
    )
    print(f"\n===== TOP AUTHORS (p \u2264 {alpha}) =====")
    if sig.empty:
        print("  (no significant results)")
        return

    cols = [
        "Author",
        "Birth Yr",
        "1st Yr",
        "Opps",
        "Sel",
        "Sel%",
        "Exp%",
        "Obs/Exp",
        "p-value",
    ]
    widths = [35, 8, 6, 4, 3, 6, 6, 7, 9]
    aligns = ["l", "r", "r", "r", "r", "r", "r", "r", "r"]

    header_parts = []
    for col, w, a in zip(cols, widths, aligns):
        header_parts.append(col.ljust(w) if a == "l" else col.rjust(w))
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for _, row in sig.iterrows():
        name = str(row["author_name"])[:35]
        by = str(row["author_birth_year"]) if row["author_birth_year"] else ""
        vals = [
            name.ljust(35),
            by.rjust(8),
            str(int(row["first_year"])).rjust(6),
            str(int(row["opportunities"])).rjust(4),
            str(int(row["selections"])).rjust(3),
            fmt_pct(row["selection_rate"]).rjust(6),
            fmt_pct(row["expected_rate"]).rjust(6),
            f"{row['obs_over_expected']:.2f}".rjust(7),
            fmt_p(row["p_value"]).rjust(9),
        ]
        print(" | ".join(vals))


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-root-works",
        action="store_true",
        help="Exclude works that are excerpts/selections (parent_work_id or parent_work_title non-empty).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        metavar="FLOAT",
        help=f"Significance threshold for stdout tables (default: {DEFAULT_ALPHA}).",
    )
    args = parser.parse_args()

    df_full, df = load_data(args.only_root_works)
    expanded = expand_author_rows(df)

    anthology_universe = build_anthology_universe(df_full)

    wfy = work_first_years(df)
    afy = author_first_years(expanded)

    pools = compute_anthology_pools(df, expanded, anthology_universe, wfy, afy)

    # Entity → edition sets
    work_anthology_sets: dict[str, set[str]] = (
        df.groupby("work_id")["edition_key"].apply(set).to_dict()
    )
    author_anthology_sets: dict[str, set[str]] = (
        expanded.groupby("author_id")["edition_key"].apply(set).to_dict()
    )

    work_stats = compute_entity_stats(
        list(wfy.keys()),
        wfy,
        work_anthology_sets,
        anthology_universe,
        {ek: p["p_work"] for ek, p in pools.items()},
    )
    author_stats = compute_entity_stats(
        list(afy.keys()),
        afy,
        author_anthology_sets,
        anthology_universe,
        {ek: p["p_author"] for ek, p in pools.items()},
    )

    works_df = build_works_df(df, work_stats)
    authors_df = build_authors_df(expanded, author_stats)

    works_path = DATA_DIR / OUT_WORKS
    authors_path = DATA_DIR / OUT_AUTHORS
    works_df.to_csv(works_path, index=False)
    authors_df.to_csv(authors_path, index=False)
    print(f"Saved \u2192 {works_path}  ({len(works_df):,} works)")
    print(f"Saved \u2192 {authors_path}  ({len(authors_df):,} authors)")

    print_works_table(works_df, args.alpha)
    print_authors_table(authors_df, args.alpha)


if __name__ == "__main__":
    main()
