"""
new_selection_reselection_probability.py
-----------------------------------------
For any work or author making its debut in an AFAM-tagged anthology, what is
the probability that it will be selected again by at least one subsequent
AFAM anthology?

Definitions
-----------
debut    : first appearance across all AFAM editions, ordered by
           (publication year, edition_id)
reselected: appears in at least one later AFAM edition
excluded : items debuting in the most recent AFAM edition have no subsequent
           opportunity and are excluded from the denominator

Two scopes are reported side by side:

all          : reselection by any later AFAM edition, including later editions
               of the debut's own series (e.g. NAAAL 1997 -> NAAAL 2004)
cross-series : reselection only by later editions outside the debut's series
               (standalone anthologies count as their own one-edition group);
               the denominator is likewise limited to debuts with at least one
               later out-of-series opportunity

Outputs overall probability + 95% Wilson score confidence interval for both
works and authors under both scopes, then the same statistics broken down by
debut decade.

Usage
-----
    uv run python analysis/new_selection_reselection_probability.py
    uv run python analysis/new_selection_reselection_probability.py --include-excerpts
    uv run python analysis/new_selection_reselection_probability.py --save-csv
"""

from __future__ import annotations

import argparse

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from afam import DATA_DIR
from afam.cli import add_root_works_flag

from author_vs_work_debut_reselection import (
    compute_author_records,
    compute_work_records,
    load_data,
)

OUT_CSV = DATA_DIR / "new_selection_reselection_probability.csv"


# ── Statistics ────────────────────────────────────────────────────────────────


def reselection_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    k = int(df["is_reselected"].sum())
    p = k / n if n else float("nan")
    lo, hi = (
        proportion_confint(k, n, alpha=0.05, method="wilson")
        if n
        else (float("nan"), float("nan"))
    )
    return {
        "n": n,
        "k": k,
        "p": round(p, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
    }


def decade_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decade"] = (df["debut_year"] // 10 * 10).astype(int)
    rows = []
    for decade, grp in df.groupby("decade"):
        s = reselection_stats(grp)
        rows.append({"decade": decade, **s})
    return pd.DataFrame(rows).set_index("decade")


def build_reselection_frames(
    only_root_works: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-debut work and author records, unfiltered for eligibility.

    Use scope_frame() to apply the scope-specific opportunity denominator.
    """
    raw, all_editions = load_data(only_root_works)
    works = compute_work_records(raw, all_editions)
    authors = compute_author_records(raw, all_editions)
    cols = [
        "debut_year",
        "subsequent_count",
        "is_reselected",
        "cross_series_subsequent_count",
        "is_reselected_cross_series",
    ]
    return works[["work_id", *cols]].copy(), authors[["author_id", *cols]].copy()


def scope_frame(df: pd.DataFrame, cross_series: bool) -> pd.DataFrame:
    """Filter to debuts with later opportunities in scope; normalize the flag.

    In cross-series scope only later editions outside the debut's series count,
    both as opportunities (denominator) and as reselections.
    """
    if cross_series:
        out = df.loc[df["cross_series_subsequent_count"] > 0].copy()
        out["is_reselected"] = out["is_reselected_cross_series"]
        return out
    return df.loc[df["subsequent_count"] > 0].copy()


# ── Reporting ─────────────────────────────────────────────────────────────────


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_summary(label: str, stats: dict) -> None:
    print(
        f"  {label:<26}  n={stats['n']:>5}  k={stats['k']:>5}  "
        f"p={_fmt_pct(stats['p'])}  "
        f"95% CI [{_fmt_pct(stats['ci_lo'])}, {_fmt_pct(stats['ci_hi'])}]"
    )


def print_decade_table(
    label: str, bkdn_all: pd.DataFrame, bkdn_cross: pd.DataFrame
) -> None:
    merged = bkdn_all.join(bkdn_cross, how="outer", rsuffix="_cs")
    print(f"\n  {label} by debut decade (all vs cross-series):")
    print(
        f"  {'Decade':<8}  {'n':>5}  {'k':>5}  {'p':>7}  |"
        f"  {'n_cs':>5}  {'k_cs':>5}  {'p_cs':>7}"
    )

    def _cell(v: float, fmt) -> str:
        return "—" if pd.isna(v) else fmt(v)

    for decade, row in merged.iterrows():
        print(
            f"  {decade:<8}  {_cell(row['n'], lambda v: str(int(v))):>5}  "
            f"{_cell(row['k'], lambda v: str(int(v))):>5}  "
            f"{_cell(row['p'], _fmt_pct):>7}  |"
            f"  {_cell(row['n_cs'], lambda v: str(int(v))):>5}  "
            f"{_cell(row['k_cs'], lambda v: str(int(v))):>5}  "
            f"{_cell(row['p_cs'], _fmt_pct):>7}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_root_works_flag(parser)
    parser.add_argument(
        "--save-csv", action="store_true", help="write results to data/"
    )
    args = parser.parse_args()

    works_records, authors_records = build_reselection_frames(args.only_root_works)

    works_label = "Works (root only)" if args.only_root_works else "Works"

    frames = {
        (works_label, "all"): scope_frame(works_records, cross_series=False),
        (works_label, "cross-series"): scope_frame(works_records, cross_series=True),
        ("Authors", "all"): scope_frame(authors_records, cross_series=False),
        ("Authors", "cross-series"): scope_frame(authors_records, cross_series=True),
    }
    stats = {key: reselection_stats(df) for key, df in frames.items()}
    breakdowns = {key: decade_breakdown(df) for key, df in frames.items()}

    print(
        "\nProbability that a debut selection is reselected by any subsequent AFAM anthology"
    )
    print("=" * 80)
    print(
        "Scopes: 'all' counts later editions of the debut's own series as reselection\n"
        "opportunities; 'cross-series' excludes them from both numerator and denominator."
    )
    for entity in (works_label, "Authors"):
        for scope in ("all", "cross-series"):
            print_summary(f"{entity} ({scope})", stats[(entity, scope)])

    for entity in (works_label, "Authors"):
        print_decade_table(
            entity, breakdowns[(entity, "all")], breakdowns[(entity, "cross-series")]
        )
    print()

    if args.save_csv:
        rows = []
        for (entity, scope), s in stats.items():
            rows.append({"entity": entity, "scope": scope, "decade": "overall", **s})
            for decade, row in breakdowns[(entity, scope)].iterrows():
                rows.append(
                    {
                        "entity": entity,
                        "scope": scope,
                        "decade": decade,
                        **row.to_dict(),
                    }
                )
        out = pd.DataFrame(rows)
        out.to_csv(OUT_CSV, index=False)
        print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
