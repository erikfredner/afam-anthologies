"""
author_vs_work_debut_reselection.py
-------------------------------------
Compare how likely author debuts and work debuts are to be selected again by
subsequent African-American Literature anthology editions.

A debut is the first edition, ordered by (publication year, edition_id), in
which an author or work appears. A reselection is any later edition appearance.
The script reports:

  * ever reselected after debut
  * per-opportunity reselection rate across later editions
  * cross-series sensitivity, excluding later editions from the debut series
  * time to first reselection
  * whether debut works and their debut authors return together or separately

Outputs:
  output/author_vs_work_debut_reselection.png
  data/author_vs_work_debut_reselection.csv          (with --save-csv)
  data/author_vs_work_debut_reselection_summary.csv  (with --save-csv)
  data/debut_work_author_reselection_outcomes.csv    (with --save-csv)

Usage:
    uv run python analysis/reselection/author_vs_work_debut_reselection.py
    uv run python analysis/reselection/author_vs_work_debut_reselection.py --only-root-works
    uv run python analysis/reselection/author_vs_work_debut_reselection.py --save-csv
    uv run python analysis/reselection/author_vs_work_debut_reselection.py --out my_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from afam import DATA_DIR
from afam.db import query
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

OUT_PNG = OUTPUT_DIR / "author_vs_work_debut_reselection.png"
OUT_TS_CSV = DATA_DIR / "author_vs_work_debut_reselection.csv"
OUT_SUMMARY_CSV = DATA_DIR / "author_vs_work_debut_reselection_summary.csv"
OUT_PAIRED_CSV = DATA_DIR / "debut_work_author_reselection_outcomes.csv"

C_AUTHOR = "#1f77b4"
C_WORK = "#ff7f0e"
GRID_KW = dict(alpha=0.25, linestyle=":")


# -- Formatting -----------------------------------------------------------------


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.1f}%"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


# -- Data loading and normalization --------------------------------------------


def add_entry_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable anthology-series group.

    Editions in a series share ``series:<id>``. Standalone anthologies use their
    edition id, so every standalone edition is its own group.
    """
    out = df.copy()
    out["entry_group"] = [
        f"series:{series_id}" if series_id else f"edition:{edition_id}"
        for series_id, edition_id in zip(
            out.get("series_id", pd.Series(index=out.index)).map(_clean_id),
            out["edition_id"].map(_clean_id),
            strict=False,
        )
    ]
    return out


def build_edition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per edition with chronological order and entry group."""
    editions = (
        df[["edition_id", "anthology_publication_year", "entry_group"]]
        .drop_duplicates("edition_id")
        .copy()
    )
    editions["anthology_publication_year"] = editions[
        "anthology_publication_year"
    ].astype(int)
    editions = editions.sort_values(
        ["anthology_publication_year", "edition_id"], kind="mergesort"
    ).reset_index(drop=True)
    editions["edition_order"] = np.arange(len(editions), dtype=int)
    return editions


def load_data(only_root_works: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load AFAM work/author appearances from PostgreSQL.

    The all-edition table is built before the optional root-work filter so that
    opportunity denominators still reflect the full AFAM anthology sequence.
    """
    raw = query(query_path("works-per-afam-edition"))
    raw["anthology_publication_year"] = raw["anthology_publication_year"].astype(int)
    raw = add_entry_group(raw)
    all_editions = build_edition_table(raw)

    if only_root_works:
        raw = raw[raw["parent_id"].isna()].copy()

    return raw, all_editions


def build_work_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (work, edition)."""
    return (
        df[["work_id", "edition_id", "entry_group"]]
        .dropna(subset=["work_id", "edition_id"])
        .drop_duplicates(["work_id", "edition_id"])
        .copy()
    )


def build_author_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (author, edition)."""
    return (
        df[["author_id", "edition_id", "entry_group"]]
        .dropna(subset=["author_id", "edition_id"])
        .drop_duplicates(["author_id", "edition_id"])
        .copy()
    )


# -- Debut and reselection computation -----------------------------------------


def _empty_records(id_col: str) -> pd.DataFrame:
    cols = [
        id_col,
        "debut_year",
        "debut_edition_id",
        "debut_order",
        "debut_group",
        "subsequent_count",
        "cross_series_subsequent_count",
        "reselection_count",
        "cross_series_reselection_count",
        "reselection_prob",
        "cross_series_reselection_prob",
        "is_reselected",
        "is_reselected_cross_series",
        "n_appearances",
        "first_reselection_year",
        "years_to_first_reselection",
    ]
    return pd.DataFrame(columns=cols)


def compute_entity_records(
    entity_df: pd.DataFrame, id_col: str, all_editions: pd.DataFrame
) -> pd.DataFrame:
    """Return one row per entity with all-series and cross-series metrics."""
    if entity_df.empty:
        return _empty_records(id_col)

    ed_lookup = all_editions[
        ["edition_id", "anthology_publication_year", "edition_order", "entry_group"]
    ].rename(columns={"entry_group": "edition_group"})
    appearances = (
        entity_df[[id_col, "edition_id", "entry_group"]]
        .dropna(subset=[id_col, "edition_id"])
        .drop_duplicates([id_col, "edition_id"])
        .merge(ed_lookup, on="edition_id", how="inner")
        .sort_values([id_col, "edition_order", "edition_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if appearances.empty:
        return _empty_records(id_col)

    debut = (
        appearances.groupby(id_col, sort=False)
        .first()
        .reset_index()
        .rename(
            columns={
                "anthology_publication_year": "debut_year",
                "edition_id": "debut_edition_id",
                "edition_order": "debut_order",
                "entry_group": "debut_group",
            }
        )[[id_col, "debut_year", "debut_edition_id", "debut_order", "debut_group"]]
    )

    appearance_orders = appearances.groupby(id_col)["edition_order"].apply(list).to_dict()
    appearance_years = (
        appearances.groupby(id_col)["anthology_publication_year"].apply(list).to_dict()
    )
    appearance_groups = appearances.groupby(id_col)["entry_group"].apply(list).to_dict()
    all_orders = all_editions["edition_order"].to_numpy(dtype=int)
    all_groups_by_order = all_editions.set_index("edition_order")["entry_group"].to_dict()

    records: list[dict] = []
    for _, row in debut.iterrows():
        entity_id = row[id_col]
        debut_order = int(row["debut_order"])
        debut_year = int(row["debut_year"])
        debut_group = row["debut_group"]

        later_orders = [int(o) for o in all_orders if int(o) > debut_order]
        cross_later_orders = [
            o for o in later_orders if all_groups_by_order.get(o) != debut_group
        ]

        orders = [int(o) for o in appearance_orders.get(entity_id, [])]
        years = [int(y) for y in appearance_years.get(entity_id, [])]
        groups = appearance_groups.get(entity_id, [])
        later_appearances = [
            (order, year, group)
            for order, year, group in zip(orders, years, groups, strict=False)
            if order > debut_order
        ]
        cross_later_appearances = [
            (order, year, group)
            for order, year, group in later_appearances
            if group != debut_group
        ]

        reselection_count = len({order for order, _, _ in later_appearances})
        cross_reselection_count = len({order for order, _, _ in cross_later_appearances})
        first_year = (
            min(year for _, year, _ in later_appearances)
            if later_appearances
            else float("nan")
        )

        records.append(
            {
                id_col: entity_id,
                "debut_year": debut_year,
                "debut_edition_id": row["debut_edition_id"],
                "debut_order": debut_order,
                "debut_group": debut_group,
                "subsequent_count": len(later_orders),
                "cross_series_subsequent_count": len(cross_later_orders),
                "reselection_count": reselection_count,
                "cross_series_reselection_count": cross_reselection_count,
                "reselection_prob": _safe_div(reselection_count, len(later_orders)),
                "cross_series_reselection_prob": _safe_div(
                    cross_reselection_count, len(cross_later_orders)
                ),
                "is_reselected": reselection_count > 0,
                "is_reselected_cross_series": cross_reselection_count > 0,
                "n_appearances": len(set(orders)),
                "first_reselection_year": first_year,
                "years_to_first_reselection": (
                    first_year - debut_year if not pd.isna(first_year) else float("nan")
                ),
            }
        )

    return pd.DataFrame(records).sort_values([id_col]).reset_index(drop=True)


def compute_work_records(df: pd.DataFrame, all_editions: pd.DataFrame) -> pd.DataFrame:
    return compute_entity_records(build_work_rows(df), "work_id", all_editions)


def compute_author_records(df: pd.DataFrame, all_editions: pd.DataFrame) -> pd.DataFrame:
    return compute_entity_records(build_author_rows(df), "author_id", all_editions)


# -- Aggregate statistics -------------------------------------------------------


def _rate_stats(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return k / n, float(lo), float(hi)


def _comparison_row(
    metric: str,
    description: str,
    author_k: int,
    author_n: int,
    work_k: int,
    work_n: int,
) -> dict:
    author_rate, author_lo, author_hi = _rate_stats(author_k, author_n)
    work_rate, work_lo, work_hi = _rate_stats(work_k, work_n)
    table = [[author_k, author_n - author_k], [work_k, work_n - work_k]]

    try:
        chi2, chi2_p, _, _ = chi2_contingency(table, correction=False)
    except ValueError:
        chi2, chi2_p = float("nan"), float("nan")

    try:
        _, z_p = proportions_ztest([author_k, work_k], [author_n, work_n])
    except (ValueError, ZeroDivisionError):
        z_p = float("nan")

    author_odds = _safe_div(author_rate, 1 - author_rate)
    work_odds = _safe_div(work_rate, 1 - work_rate)

    return {
        "metric": metric,
        "description": description,
        "author_n": author_n,
        "author_k": author_k,
        "author_rate": author_rate,
        "author_ci_lo": author_lo,
        "author_ci_hi": author_hi,
        "work_n": work_n,
        "work_k": work_k,
        "work_rate": work_rate,
        "work_ci_lo": work_lo,
        "work_ci_hi": work_hi,
        "rate_difference_author_minus_work": author_rate - work_rate,
        "risk_ratio_author_over_work": _safe_div(author_rate, work_rate),
        "odds_ratio_author_over_work": _safe_div(author_odds, work_odds),
        "chi2": chi2,
        "chi2_p": chi2_p,
        "two_proportion_z_p": z_p,
    }


def build_summary(works: pd.DataFrame, authors: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate author/work comparisons for core estimands."""
    w_eligible = works[works["subsequent_count"] > 0]
    a_eligible = authors[authors["subsequent_count"] > 0]
    w_cross = works[works["cross_series_subsequent_count"] > 0]
    a_cross = authors[authors["cross_series_subsequent_count"] > 0]

    rows = [
        _comparison_row(
            "ever_all",
            "Entity was reselected at least once in any later edition",
            int(a_eligible["is_reselected"].sum()),
            len(a_eligible),
            int(w_eligible["is_reselected"].sum()),
            len(w_eligible),
        ),
        _comparison_row(
            "ever_cross_series",
            "Entity was reselected at least once in a later different series/standalone group",
            int(a_cross["is_reselected_cross_series"].sum()),
            len(a_cross),
            int(w_cross["is_reselected_cross_series"].sum()),
            len(w_cross),
        ),
        _comparison_row(
            "opportunity_all",
            "Later edition opportunities that reselect the debut entity",
            int(a_eligible["reselection_count"].sum()),
            int(a_eligible["subsequent_count"].sum()),
            int(w_eligible["reselection_count"].sum()),
            int(w_eligible["subsequent_count"].sum()),
        ),
        _comparison_row(
            "opportunity_cross_series",
            "Later different-series opportunities that reselect the debut entity",
            int(a_cross["cross_series_reselection_count"].sum()),
            int(a_cross["cross_series_subsequent_count"].sum()),
            int(w_cross["cross_series_reselection_count"].sum()),
            int(w_cross["cross_series_subsequent_count"].sum()),
        ),
    ]
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame, works: pd.DataFrame, authors: pd.DataFrame) -> None:
    print("\n-- Debut Reselection: Author vs Work --------------------------------")
    for _, row in summary.iterrows():
        print(f"\n  {row['metric']}: {row['description']}")
        print(
            f"    Authors: n={int(row['author_n']):5d}  k={int(row['author_k']):5d}  "
            f"rate={fmt_pct(row['author_rate'])}  "
            f"95% CI [{fmt_pct(row['author_ci_lo'])}, {fmt_pct(row['author_ci_hi'])}]"
        )
        print(
            f"    Works:   n={int(row['work_n']):5d}  k={int(row['work_k']):5d}  "
            f"rate={fmt_pct(row['work_rate'])}  "
            f"95% CI [{fmt_pct(row['work_ci_lo'])}, {fmt_pct(row['work_ci_hi'])}]"
        )
        print(
            f"    Diff={fmt_pct(row['rate_difference_author_minus_work'])}  "
            f"RR={row['risk_ratio_author_over_work']:.2f}x  "
            f"OR={row['odds_ratio_author_over_work']:.2f}x  "
            f"chi2 p={fmt_p(row['chi2_p'])}  z-test p={fmt_p(row['two_proportion_z_p'])}"
        )

    for label, frame in [("Authors", authors), ("Works", works)]:
        reselected = frame[frame["is_reselected"]]
        median_years = reselected["years_to_first_reselection"].median()
        mean_prob = frame.loc[frame["subsequent_count"] > 0, "reselection_prob"].mean()
        print(
            f"\n  {label}: mean per-debut later-edition reselection probability "
            f"{fmt_pct(mean_prob)}; median years to first reselection "
            f"{median_years:.1f}" if not pd.isna(median_years) else
            f"\n  {label}: mean per-debut later-edition reselection probability "
            f"{fmt_pct(mean_prob)}; median years to first reselection N/A"
        )
    print()


# -- Time-series data -----------------------------------------------------------


def _timeseries_for(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    eligible = frame[frame["subsequent_count"] > 0]
    cross = frame[frame["cross_series_subsequent_count"] > 0]

    all_ts = (
        eligible.groupby("debut_year")
        .agg(
            **{
                f"{prefix}_debuts": ("is_reselected", "count"),
                f"{prefix}_reselected": ("is_reselected", "sum"),
            }
        )
        .reset_index()
        .rename(columns={"debut_year": "year"})
    )
    all_ts[f"{prefix}_rate"] = (
        all_ts[f"{prefix}_reselected"] / all_ts[f"{prefix}_debuts"]
    )

    cross_ts = (
        cross.groupby("debut_year")
        .agg(
            **{
                f"{prefix}_cross_debuts": ("is_reselected_cross_series", "count"),
                f"{prefix}_reselected_cs": ("is_reselected_cross_series", "sum"),
            }
        )
        .reset_index()
        .rename(columns={"debut_year": "year"})
    )
    cross_ts[f"{prefix}_rate_cs"] = (
        cross_ts[f"{prefix}_reselected_cs"] / cross_ts[f"{prefix}_cross_debuts"]
    )
    return all_ts.merge(cross_ts, on="year", how="outer")


def build_timeseries(works: pd.DataFrame, authors: pd.DataFrame) -> pd.DataFrame:
    ts = (
        _timeseries_for(works, "work")
        .merge(_timeseries_for(authors, "author"), on="year", how="outer")
        .sort_values("year")
        .reset_index(drop=True)
    )
    return ts


# -- Debut work / author paired outcomes ---------------------------------------


def build_debut_work_author_outcomes(
    raw: pd.DataFrame, works: pd.DataFrame, authors: pd.DataFrame
) -> pd.DataFrame:
    """Classify whether debut works and their authors return together."""
    work_authors = (
        raw[["work_id", "author_id"]]
        .dropna(subset=["work_id", "author_id"])
        .drop_duplicates()
    )
    authors_by_work = work_authors.groupby("work_id")["author_id"].apply(set).to_dict()
    author_reselected = authors.set_index("author_id")["is_reselected"].to_dict()

    rows = []
    for _, work in works[works["subsequent_count"] > 0].iterrows():
        work_id = work["work_id"]
        author_ids = authors_by_work.get(work_id, set())
        author_returned = any(bool(author_reselected.get(aid, False)) for aid in author_ids)
        work_returned = bool(work["is_reselected"])

        if work_returned and author_returned:
            outcome = "author_and_work"
        elif author_returned:
            outcome = "author_only"
        elif work_returned:
            outcome = "work_only"
        else:
            outcome = "neither"

        rows.append(
            {
                "work_id": work_id,
                "debut_year": work["debut_year"],
                "n_authors": len(author_ids),
                "work_reselected": work_returned,
                "any_author_reselected": author_returned,
                "outcome": outcome,
            }
        )

    return pd.DataFrame(rows)


def print_paired_outcomes(outcomes: pd.DataFrame) -> None:
    if outcomes.empty:
        print("  No eligible debut work/author paired outcomes.")
        return
    counts = outcomes["outcome"].value_counts().reindex(
        ["author_and_work", "author_only", "work_only", "neither"], fill_value=0
    )
    total = int(counts.sum())
    print("  Debut work / debut author paired outcomes:")
    for outcome, count in counts.items():
        print(f"    {outcome:<16} {int(count):5d}  {fmt_pct(count / total)}")
    print()


# -- Plot -----------------------------------------------------------------------


def make_plot(
    ts: pd.DataFrame,
    works: pd.DataFrame,
    authors: pd.DataFrame,
    out_path: Path,
) -> None:
    w = works[works["subsequent_count"] > 0]
    a = authors[authors["subsequent_count"] > 0]
    w_cross = works[works["cross_series_subsequent_count"] > 0]
    a_cross = authors[authors["cross_series_subsequent_count"] > 0]

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    series_specs = [
        ("work_rate", C_WORK, "o", "-", f"Work - all ({fmt_pct(w['is_reselected'].mean())})"),
        (
            "author_rate",
            C_AUTHOR,
            "s",
            "-",
            f"Author - all ({fmt_pct(a['is_reselected'].mean())})",
        ),
        (
            "work_rate_cs",
            C_WORK,
            "o",
            "--",
            f"Work - cross-series ({fmt_pct(w_cross['is_reselected_cross_series'].mean())})",
        ),
        (
            "author_rate_cs",
            C_AUTHOR,
            "s",
            "--",
            f"Author - cross-series ({fmt_pct(a_cross['is_reselected_cross_series'].mean())})",
        ),
    ]
    for col, color, marker, linestyle, label in series_specs:
        mask = ts[col].notna()
        ax_top.plot(
            ts.loc[mask, "year"],
            ts.loc[mask, col] * 100,
            color=color,
            marker=marker,
            markersize=5 if linestyle == "-" else 4,
            linewidth=1.5 if linestyle == "-" else 1.2,
            linestyle=linestyle,
            alpha=1.0 if linestyle == "-" else 0.8,
            label=label,
        )

    ax_top.set_ylabel("Ever reselected (%)", fontsize=11)
    ax_top.set_ylim(-5, 110)
    ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_top.grid(**GRID_KW)
    ax_top.legend(loc="upper right", fontsize=8.5)
    ax_top.set_title(
        "Debut Reselection by Debut Year: Author vs Work",
        fontsize=12,
    )

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
    ax_bot.set_xlabel("Debut year", fontsize=11)
    ax_bot.grid(**GRID_KW)
    ax_bot.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {out_path}")


# -- Main -----------------------------------------------------------------------


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
        help="Write time-series, summary, and paired-outcome CSVs to data/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PNG,
        help=f"Output PNG path (default: {OUT_PNG}).",
    )
    args = parser.parse_args()

    raw, all_editions = load_data(args.only_root_works)
    works = compute_work_records(raw, all_editions)
    authors = compute_author_records(raw, all_editions)
    summary = build_summary(works, authors)
    ts = build_timeseries(works, authors)
    paired = build_debut_work_author_outcomes(raw, works, authors)

    print_summary(summary, works, authors)
    print_paired_outcomes(paired)

    if args.save_csv:
        ts.to_csv(OUT_TS_CSV, index=False)
        summary.to_csv(OUT_SUMMARY_CSV, index=False)
        paired.to_csv(OUT_PAIRED_CSV, index=False)
        print(f"  Time-series CSV saved -> {OUT_TS_CSV}")
        print(f"  Summary CSV saved -> {OUT_SUMMARY_CSV}")
        print(f"  Paired outcome CSV saved -> {OUT_PAIRED_CSV}")

    make_plot(ts, works, authors, args.out)


if __name__ == "__main__":
    main()
