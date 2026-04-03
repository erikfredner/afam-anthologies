"""
author_vs_work_debut_reselection.py
-------------------------------------
Compares how likely author debuts are to be reselected across subsequent
anthologies vs. work debuts.

A "debut" is the first anthology appearance for an author or work
(by anthology_publication_year). A debut is "reselected" if the entity
appears in at least one subsequent anthology edition with a strictly later
publication year. Entities debuting in the latest edition year are excluded
(they have no subsequent opportunities).

Two reselection definitions are reported side-by-side:
  all-series:    any later appearance, including further editions of the
                 same anthology series as the debut (e.g. NAAL 1st → NAAL 2nd)
  cross-series:  only later appearances in a *different* series (or a different
                 standalone anthology) — same-series continuations are not counted

Reports aggregate statistics (likelihood ratio, odds ratio, chi-square test)
for both definitions, plus a time-series plot of reselection rates by debut year.

Outputs:
  viz/author_vs_work_debut_reselection.png  (always)
  data/author_vs_work_debut_reselection.csv (with --save-csv)

Usage:
    uv run python analysis/author_vs_work_debut_reselection.py
    uv run python analysis/author_vs_work_debut_reselection.py --only-root-works
    uv run python analysis/author_vs_work_debut_reselection.py --save-csv
    uv run python analysis/author_vs_work_debut_reselection.py --out my_plot.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


DATA_FILE = (
    Path(__file__).parent.parent / "data" / "2026-03-13 works per afam anthology.csv"
)
DATA_DIR = Path(__file__).parent.parent / "data"
VIZ_DIR = Path(__file__).parent.parent / "viz"
OUT_PNG = VIZ_DIR / "author_vs_work_debut_reselection.png"
OUT_CSV = DATA_DIR / "author_vs_work_debut_reselection.csv"

C_AUTHOR = "#1f77b4"
C_WORK = "#ff7f0e"
GRID_KW = dict(alpha=0.25, linestyle=":")


# ── Edition-key logic (mirrors post_debut_performance.py) ────────────────────


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


# ── Entry-group (series identity) ────────────────────────────────────────────


def add_entry_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add entry_group: series_id when non-empty, else anthology_id.

    Two appearances share a group iff they belong to the same anthology series
    (or are the same standalone anthology). Used to distinguish same-series
    continuations from cross-series reselections.
    """
    df = df.copy()
    df["entry_group"] = df.apply(
        lambda r: (
            r["series_id"].strip() if r["series_id"].strip() else r["anthology_id"]
        ),
        axis=1,
    )
    return df


# ── Formatting helpers ────────────────────────────────────────────────────────


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(only_root_works: bool) -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, dtype=str, na_filter=False)
    df["anthology_publication_year"] = df["anthology_publication_year"].astype(int)
    df = assign_edition_key(df)
    df = add_entry_group(df)
    if only_root_works:
        df = df[
            (df["parent_work_id"].str.strip() == "")
            & (df["parent_work_title"].str.strip() == "")
        ].copy()
    return df


# ── Author expansion ──────────────────────────────────────────────────────────


def expand_author_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (author_id, edition_key); deduped within edition."""
    records: list[dict] = []
    for _, row in df[df["author_ids"] != ""].iterrows():
        ids = [i.strip() for i in row["author_ids"].split(",") if i.strip()]
        for aid in ids:
            records.append(
                {
                    "author_id": aid,
                    "anthology_publication_year": row["anthology_publication_year"],
                    "edition_key": row["edition_key"],
                    "entry_group": row["entry_group"],
                }
            )
    expanded = pd.DataFrame(records)
    return expanded.drop_duplicates(subset=["author_id", "edition_key"])


# ── Debut + reselection computation ──────────────────────────────────────────


def _compute_records(entity_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """One row per entity: debut_year, is_reselected, is_reselected_cross_series.

    is_reselected:             appeared in any later edition (any series)
    is_reselected_cross_series: appeared in any later edition from a *different*
                                series/anthology than the debut
    """
    debut_years = (
        entity_df.groupby(id_col)["anthology_publication_year"]
        .min()
        .rename("debut_year")
        .reset_index()
    )
    merged = entity_df.merge(debut_years, on=id_col)

    # debut_group: entry_group of the earliest appearance
    debut_rows = merged[merged["anthology_publication_year"] == merged["debut_year"]]
    debut_groups = (
        debut_rows.groupby(id_col)["entry_group"]
        .first()
        .rename("debut_group")
        .reset_index()
    )
    merged = merged.merge(debut_groups, on=id_col)

    later = merged[merged["anthology_publication_year"] > merged["debut_year"]]
    is_resel = later.groupby(id_col).size().gt(0).rename("is_reselected").reset_index()
    cross_later = later[later["entry_group"] != later["debut_group"]]
    is_cross = (
        cross_later.groupby(id_col)
        .size()
        .gt(0)
        .rename("is_reselected_cross_series")
        .reset_index()
    )

    result = debut_years.merge(is_resel, on=id_col, how="left").merge(
        is_cross, on=id_col, how="left"
    )
    result["is_reselected"] = result["is_reselected"].fillna(False)
    result["is_reselected_cross_series"] = result["is_reselected_cross_series"].fillna(
        False
    )
    return result[[id_col, "debut_year", "is_reselected", "is_reselected_cross_series"]]


def compute_work_records(df: pd.DataFrame) -> pd.DataFrame:
    return _compute_records(df, "work_id")


def compute_author_records(expanded: pd.DataFrame) -> pd.DataFrame:
    return _compute_records(expanded, "author_id")


# ── Aggregate stats ───────────────────────────────────────────────────────────


def print_aggregate(
    works: pd.DataFrame,
    authors: pd.DataFrame,
    max_year: int,
) -> None:
    w = works[works["debut_year"] < max_year]
    a = authors[authors["debut_year"] < max_year]

    n_w_excl = len(works) - len(w)
    n_a_excl = len(authors) - len(a)

    n_w = len(w)
    n_a = len(a)
    r_w = int(w["is_reselected"].sum())
    r_a = int(a["is_reselected"].sum())
    rate_w = r_w / n_w
    rate_a = r_a / n_a

    lr = rate_a / rate_w
    odds_w = rate_w / (1 - rate_w)
    odds_a = rate_a / (1 - rate_a)
    or_ = odds_a / odds_w

    table = [[r_a, n_a - r_a], [r_w, n_w - r_w]]
    chi2, p_val, _, _ = chi2_contingency(table)

    # Cross-series stats
    r_w_cs = int(w["is_reselected_cross_series"].sum())
    r_a_cs = int(a["is_reselected_cross_series"].sum())
    rate_w_cs = r_w_cs / n_w
    rate_a_cs = r_a_cs / n_a
    lr_cs = rate_a_cs / rate_w_cs
    odds_w_cs = rate_w_cs / (1 - rate_w_cs)
    odds_a_cs = rate_a_cs / (1 - rate_a_cs)
    or_cs = odds_a_cs / odds_w_cs
    table_cs = [[r_a_cs, n_a - r_a_cs], [r_w_cs, n_w - r_w_cs]]
    chi2_cs, p_cs, _, _ = chi2_contingency(table_cs)

    print(
        "\n── Debut Reselection: Author vs Work ──────────────────────────────────────"
    )
    print(
        f"  (Excluded {n_a_excl} author debut(s) and {n_w_excl} work debut(s)"
        f" debuting in final edition year {max_year} — no subsequent opportunities)\n"
    )
    print("  All-series (any later appearance counts as reselection):")
    print(
        f"    Author debuts:  {n_a:5d}  |  reselected: {r_a:4d}  |  rate: {fmt_pct(rate_a)}"
    )
    print(
        f"    Work debuts:    {n_w:5d}  |  reselected: {r_w:4d}  |  rate: {fmt_pct(rate_w)}"
    )
    print(f"    Likelihood ratio (author/work): {lr:.2f}x")
    print(f"    Odds ratio       (author/work): {or_:.2f}x")
    print(f"    Chi-square 2×2:  χ²={chi2:.2f}  p={fmt_p(p_val)}")
    print()
    print("  Cross-series only (same-series continuations excluded):")
    print(
        f"    Author debuts:  {n_a:5d}  |  reselected: {r_a_cs:4d}  |  rate: {fmt_pct(rate_a_cs)}"
    )
    print(
        f"    Work debuts:    {n_w:5d}  |  reselected: {r_w_cs:4d}  |  rate: {fmt_pct(rate_w_cs)}"
    )
    print(f"    Likelihood ratio (author/work): {lr_cs:.2f}x")
    print(f"    Odds ratio       (author/work): {or_cs:.2f}x")
    print(f"    Chi-square 2×2:  χ²={chi2_cs:.2f}  p={fmt_p(p_cs)}")
    print()


# ── Time-series data ──────────────────────────────────────────────────────────


def build_timeseries(
    works: pd.DataFrame,
    authors: pd.DataFrame,
    max_year: int,
) -> pd.DataFrame:
    w = works[works["debut_year"] < max_year]
    a = authors[authors["debut_year"] < max_year]

    w_ts = (
        w.groupby("debut_year")
        .agg(
            work_debuts=("is_reselected", "count"),
            work_reselected=("is_reselected", "sum"),
            work_reselected_cs=("is_reselected_cross_series", "sum"),
        )
        .reset_index()
        .rename(columns={"debut_year": "year"})
    )
    w_ts["work_rate"] = w_ts["work_reselected"] / w_ts["work_debuts"]
    w_ts["work_rate_cs"] = w_ts["work_reselected_cs"] / w_ts["work_debuts"]

    a_ts = (
        a.groupby("debut_year")
        .agg(
            author_debuts=("is_reselected", "count"),
            author_reselected=("is_reselected", "sum"),
            author_reselected_cs=("is_reselected_cross_series", "sum"),
        )
        .reset_index()
        .rename(columns={"debut_year": "year"})
    )
    a_ts["author_rate"] = a_ts["author_reselected"] / a_ts["author_debuts"]
    a_ts["author_rate_cs"] = a_ts["author_reselected_cs"] / a_ts["author_debuts"]

    ts = (
        pd.merge(w_ts, a_ts, on="year", how="outer")
        .sort_values("year")
        .reset_index(drop=True)
    )
    return ts


# ── Plot ──────────────────────────────────────────────────────────────────────


def make_plot(
    ts: pd.DataFrame,
    works: pd.DataFrame,
    authors: pd.DataFrame,
    max_year: int,
    out_path: Path,
) -> None:
    w = works[works["debut_year"] < max_year]
    a = authors[authors["debut_year"] < max_year]
    agg_rate_w = float(w["is_reselected"].mean())
    agg_rate_a = float(a["is_reselected"].mean())

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    agg_rate_w_cs = float(w["is_reselected_cross_series"].mean())
    agg_rate_a_cs = float(a["is_reselected_cross_series"].mean())

    # ── Top panel: reselection rates ─────────────────────────────────────────
    mask_w = ts["work_rate"].notna()
    mask_a = ts["author_rate"].notna()
    mask_w_cs = ts["work_rate_cs"].notna()
    mask_a_cs = ts["author_rate_cs"].notna()

    ax_top.plot(
        ts.loc[mask_w, "year"],
        ts.loc[mask_w, "work_rate"] * 100,
        color=C_WORK,
        marker="o",
        markersize=5,
        linewidth=1.5,
        label=f"Work — all series ({fmt_pct(agg_rate_w)} overall)",
    )
    ax_top.plot(
        ts.loc[mask_a, "year"],
        ts.loc[mask_a, "author_rate"] * 100,
        color=C_AUTHOR,
        marker="s",
        markersize=5,
        linewidth=1.5,
        label=f"Author — all series ({fmt_pct(agg_rate_a)} overall)",
    )
    ax_top.plot(
        ts.loc[mask_w_cs, "year"],
        ts.loc[mask_w_cs, "work_rate_cs"] * 100,
        color=C_WORK,
        marker="o",
        markersize=4,
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
        label=f"Work — cross-series ({fmt_pct(agg_rate_w_cs)} overall)",
    )
    ax_top.plot(
        ts.loc[mask_a_cs, "year"],
        ts.loc[mask_a_cs, "author_rate_cs"] * 100,
        color=C_AUTHOR,
        marker="s",
        markersize=4,
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
        label=f"Author — cross-series ({fmt_pct(agg_rate_a_cs)} overall)",
    )

    ax_top.set_ylabel("Reselection rate (%)", fontsize=11)
    ax_top.set_ylim(-5, 110)
    ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_top.grid(**GRID_KW)
    ax_top.legend(loc="upper right", fontsize=8.5)
    ax_top.set_title(
        "Reselection Rate by Debut Year: Author vs Work Debuts\n"
        "(solid = any later appearance; dashed = cross-series only)",
        fontsize=12,
    )

    # ── Bottom panel: debut counts ───────────────────────────────────────────
    years_all = np.array(sorted(ts["year"].dropna().astype(int).unique()), dtype=float)
    ts_idx = ts.set_index("year")
    w_counts = ts_idx["work_debuts"].reindex(years_all).fillna(0).values
    a_counts = ts_idx["author_debuts"].reindex(years_all).fillna(0).values
    bar_w = 0.4

    ax_bot.bar(
        years_all - bar_w / 2,
        w_counts,
        width=bar_w,
        color=C_WORK,
        alpha=0.75,
        label="Work debuts",
    )
    ax_bot.bar(
        years_all + bar_w / 2,
        a_counts,
        width=bar_w,
        color=C_AUTHOR,
        alpha=0.75,
        label="Author debuts",
    )
    ax_bot.set_ylabel("# Debuts", fontsize=10)
    ax_bot.set_xlabel("Debut year (anthology publication year)", fontsize=11)
    ax_bot.grid(**GRID_KW)
    ax_bot.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reselection rates of author debuts vs work debuts."
    )
    parser.add_argument(
        "--only-root-works",
        action="store_true",
        help="Exclude excerpts/selections (works with a parent work).",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Write time-series table to data/author_vs_work_debut_reselection.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PNG,
        help="Output PNG path (default: viz/author_vs_work_debut_reselection.png).",
    )
    args = parser.parse_args()

    df = load_data(args.only_root_works)
    expanded = expand_author_rows(df)

    max_year = int(df["anthology_publication_year"].max())

    works = compute_work_records(df)
    authors = compute_author_records(expanded)

    print_aggregate(works, authors, max_year)

    ts = build_timeseries(works, authors, max_year)

    if args.save_csv:
        ts.to_csv(OUT_CSV, index=False)
        print(f"  Time-series CSV saved → {OUT_CSV}")

    make_plot(ts, works, authors, max_year, args.out)


if __name__ == "__main__":
    main()
