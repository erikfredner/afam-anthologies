"""
edge_case_sensitivity.py
-------------------------
Robustness check: re-run the project's four headline author-vs-work claims
with three edge-case anthologies added to the AFAM corpus, to confirm the
published findings hold regardless of their exclusion.

Excluded anthologies (no row in data_edition_literary_traditions linking to
the 'African-American Literature' tradition -- the only filter any script in
this repo uses to build its AFAM edition set):

  - Black Culture: Reading and Writing Black (1972)      edition_id=79
  - Crossing the Danger Water (1993)                      edition_id=62
  - African American Literature series, 1st ed. (1993)    edition_id=22

There is no other blocklist anywhere in the codebase; the exclusion is
purely the absence of that tag. This script widens the tag filter in-memory
(never touching the database or the checked-in query file) to include these
three ids alongside the normally-tagged set, then prints each claim's
statistic baseline (26 tagged editions) vs. with the edge cases (29).

Usage:
    uv run python analysis/robustness/edge_case_sensitivity.py
    uv run python analysis/robustness/edge_case_sensitivity.py --include-excerpts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from afam.cli import add_root_works_flag
from afam.db import query
from afam.sql import load_query

sys.path.insert(0, str(Path(__file__).parents[1] / "reselection"))
sys.path.insert(0, str(Path(__file__).parents[1] / "overlap"))
sys.path.insert(0, str(Path(__file__).parents[2] / "viz" / "reselection"))

from author_vs_work_debut_reselection import (  # noqa: E402
    add_entry_group,
    build_edition_table,
    build_summary,
    compute_author_records,
    compute_work_records,
)
from edition_pair_retention_scatter import (  # noqa: E402
    _above_share,
    compute_pair_retention,
)
from half_or_more_summary import compute_summary as half_or_more_compute_summary  # noqa: E402
from new_selection_reselection_probability import (  # noqa: E402
    reselection_stats,
    scope_frame,
)

EDGE_CASE_EDITION_IDS = (
    79,
    62,
    22,
)  # Black Culture, Crossing the Danger Water, AAL ed.1
CTE_OPEN = "WITH tagged_editions AS ("
BASE_TAG_CLAUSE = "lt.\"name\" = 'African-American Literature'"


def _find_cte_close(sql: str, body_start: int) -> int:
    """Index of the ')' that closes the CTE whose body starts at `body_start`.

    `body_start` must point just past the CTE's own opening '('. Scans paren
    depth from there rather than jumping to "the next literal ')'" -- that
    naive approach would target the wrong paren if the CTE body ever gains a
    nested subquery, function call, or a stray ')' inside a comment before its
    real close.
    """
    depth = 1
    for i, ch in enumerate(sql[body_start:], start=body_start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(
        "Unbalanced parentheses while scanning the tagged_editions CTE in "
        "queries/works-per-afam-edition.sql; update _augmented_sql() to match."
    )


def _augmented_sql() -> str:
    """The tagged_editions CTE widened to also admit the three edge-case ids.

    Editions with zero rows in data_edition_literary_traditions are dropped by
    that CTE's inner JOIN before its WHERE clause ever runs, so relaxing the
    WHERE alone (e.g. "OR e.id IN (...)") would not surface them -- the three
    edge-case editions have no tradition-tag row at all, tagged or otherwise.
    A UNION of a plain id lookup is required instead.
    """
    sql = load_query("works-per-afam-edition")
    cte_start = sql.find(CTE_OPEN)
    if cte_start == -1:
        raise ValueError(
            "queries/works-per-afam-edition.sql no longer starts with the "
            "expected 'WITH tagged_editions AS (' CTE; update _augmented_sql()."
        )
    body_start = cte_start + len(CTE_OPEN)  # just past the CTE's opening "("
    close_idx = _find_cte_close(sql, body_start)
    if BASE_TAG_CLAUSE not in sql[body_start:close_idx]:
        raise ValueError(
            "tagged_editions CTE no longer contains the expected tradition-name "
            "filter; update _augmented_sql() to match works-per-afam-edition.sql."
        )
    ids = ", ".join(str(i) for i in EDGE_CASE_EDITION_IDS)
    insertion = f"\n    UNION\n    SELECT id FROM data_edition WHERE id IN ({ids})\n"
    return sql[:close_idx] + insertion + sql[close_idx:]


def _load(sql: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror load_data()'s pre-filter state: full raw rows + edition table."""
    raw = query(sql)
    raw["anthology_publication_year"] = raw["anthology_publication_year"].astype(int)
    raw = add_entry_group(raw)
    all_editions = build_edition_table(raw)
    return raw, all_editions


def _root_filter(raw: pd.DataFrame, only_root_works: bool) -> pd.DataFrame:
    return raw[raw["parent_id"].isna()].copy() if only_root_works else raw


def load_variants() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    return {
        "baseline": _load(load_query("works-per-afam-edition")),
        "with_edge_cases": _load(_augmented_sql()),
    }


# -- Formatting ------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x * 100:.1f}%"


# -- Per-claim comparisons ---------------------------------------------------


def claim_1_never_repeated(
    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]], only_root_works: bool
) -> None:
    print("\nClaim 1 -- share of debuts never repeated by any subsequent anthology")
    print("-" * 78)
    for label, (raw, editions) in variants.items():
        raw = _root_filter(raw, only_root_works)
        works = compute_work_records(raw, editions)
        authors = compute_author_records(raw, editions)
        w = reselection_stats(scope_frame(works, cross_series=False))
        a = reselection_stats(scope_frame(authors, cross_series=False))
        print(
            f"  {label:<16} works: n={w['n']:>4} never-repeated={_fmt_pct(1 - w['p']):>6}   "
            f"authors: n={a['n']:>4} never-repeated={_fmt_pct(1 - a['p']):>6}"
        )


def claim_2_half_or_more(
    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    # half_or_more_summary.py always computes from the unfiltered raw table
    # (it derives all/root/coalesced variants itself) -- no root-works flag.
    print("\nClaim 2 -- more authors than works selected >= half the anthologies")
    print("-" * 78)
    for label, (raw_full, _editions) in variants.items():
        summary, total_editions = half_or_more_compute_summary(raw_full)
        print(f"  {label} (total_editions={total_editions}):")
        for variant_col in summary.columns:
            authors_half = int(summary.loc["Authors in ≥ half", variant_col])
            works_half = int(summary.loc["Works in ≥ half", variant_col])
            print(
                f"    {variant_col:<30} authors_half={authors_half:>3}  "
                f"works_half={works_half:>3}  authors > works: {authors_half > works_half}"
            )


def claim_3_debut_author_vs_work(
    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]], only_root_works: bool
) -> None:
    print("\nClaim 3 -- debut authors more likely reselected than debut works")
    print("-" * 78)
    for label, (raw, editions) in variants.items():
        raw = _root_filter(raw, only_root_works)
        works = compute_work_records(raw, editions)
        authors = compute_author_records(raw, editions)
        row = build_summary(works, authors).set_index("metric").loc["ever_all"]
        print(
            f"  {label:<16} author_rate={_fmt_pct(row['author_rate']):>6}  "
            f"work_rate={_fmt_pct(row['work_rate']):>6}  "
            f"RR={row['risk_ratio_author_over_work']:.2f}x  "
            f"OR={row['odds_ratio_author_over_work']:.2f}x  "
            f"chi2 p={row['chi2_p']:.2e}"
        )


def claim_4_pair_retention(
    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]], only_root_works: bool
) -> None:
    print("\nClaim 4 -- author retention > work retention across paired comparisons")
    print("-" * 78)
    for label, (raw, editions) in variants.items():
        raw = _root_filter(raw, only_root_works)
        pairs = compute_pair_retention(raw, editions)
        chrono = pairs[~pairs["same_year"]]
        above, n = _above_share(chrono)
        share = _fmt_pct(above / n) if n else "N/A"
        print(
            f"  {label:<16} chronological pairs={n:>4}  "
            f"author > work in {above:>4} ({share})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_root_works_flag(parser)
    args = parser.parse_args()

    variants = load_variants()

    print("=" * 78)
    print(
        "Robustness check: Black Culture (1972, id=79), Crossing the Danger Water "
        "(1993, id=62), and African American Literature series ed.1 (1993, id=22) "
        "added to the AFAM corpus alongside the baseline tagged set."
    )
    print("=" * 78)

    claim_1_never_repeated(variants, args.only_root_works)
    claim_2_half_or_more(variants)
    claim_3_debut_author_vs_work(variants, args.only_root_works)
    claim_4_pair_retention(variants, args.only_root_works)
    print()


if __name__ == "__main__":
    main()
