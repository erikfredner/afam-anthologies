"""
poet_signature_concentration.py
-------------------------------
Rank widely-anthologized poets by how thoroughly editorial selection collapses
onto a single "signature" poem -- the inverse of poet_work_dispersal.py.

poet_work_dispersal surfaces the most *dispersed* poets (Sterling Brown is in
nearly every edition, yet no single Brown poem reaches even half of them). This
script asks the opposite question: which poet combines

  1. high selection volume -- total_selections, the count of work-edition
     selections across the corpus;
  2. concentration into few works -- a low distinct_poems count; and
  3. a dominant signature poem -- top_poem_coverage, the share of the poet's own
     poetry editions carried by their single most-anthologized poem.

The motivating case is Margaret Walker, carried into edition after edition by
essentially one poem, "For My People". A poet like that sits near
top_poem_coverage = 1.0 with a small distinct_poems count, while a broad poet
like Langston Hughes spreads across many poems and scores low here.

The three factors are presented side by side without a composite score: the
table is sorted to put the most concentrated high-volume poets on top
(top_poem_coverage desc, then total_selections desc, then distinct_poems asc),
and the reader weighs the tradeoff. The --min-editions threshold keeps a
low-volume poet who happens to sit at coverage 1.0 off the top of the list.

Like poet_work_dispersal, the unit of analysis is the leaf work (individual
poem): collection containers are dropped so each poem is counted once regardless
of how it was catalogued.

Usage:
    uv run python analysis/concentration/poet_signature_concentration.py
    uv run python analysis/concentration/poet_signature_concentration.py --min-editions 13
    uv run python analysis/concentration/poet_signature_concentration.py --save-csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "concentration"))

from author_form_concentration import _fmt, compute  # noqa: E402
from poet_work_dispersal import (  # noqa: E402
    build_poet_dispersal,
    fetch_container_ids,
    filter_leaf_works,
)

from afam import DATA_DIR  # noqa: E402
from afam.cli import add_save_csv_flag  # noqa: E402
from afam.db import query as query_db  # noqa: E402
from afam.names import author_sort_key  # noqa: E402
from afam.sql import query_path  # noqa: E402

OUT_CSV = DATA_DIR / "poet_signature_concentration.csv"
POETRY = "poetry"
FOCAL_AUTHOR = "Margaret Walker"

CSV_COLS = [
    "author_id",
    "author_name",
    "author_last_name",
    "total_selections",
    "poetry_editions",
    "distinct_poems",
    "top_poem",
    "top_poem_editions",
    "top_poem_coverage",
    "signature_work_share_count",
    "effective_poems_count",
]

DISPLAY_COLS = [
    "author_name",
    "total_selections",
    "poetry_editions",
    "distinct_poems",
    "top_poem",
    "top_poem_editions",
    "top_poem_coverage",
    "signature_work_share_count",
]


def build_signature_concentration(
    outputs: dict[str, pd.DataFrame], raw: pd.DataFrame, *, min_editions: int = 10
) -> pd.DataFrame:
    """Per-poet signature-concentration table, most concentrated first.

    Reuses build_poet_dispersal for the per-poet poetry columns (poetry_editions,
    distinct_poems, top_poem, top_poem_editions, top_poem_coverage), then merges
    in the two columns it drops: total_selections (selection volume) and
    signature_work_share_count (share of the poet's poetry selections in their
    single most-selected poem).
    """
    dispersal = build_poet_dispersal(outputs, raw, min_editions=min_editions)

    poetry_conc = outputs["concentration"]
    poetry_conc = poetry_conc[poetry_conc["form_name"].str.casefold() == POETRY][
        ["author_id", "total_selections", "signature_work_share_count"]
    ]

    table = dispersal.merge(poetry_conc, on="author_id", how="left")

    table["_sort"] = table["author_name"].map(author_sort_key)
    table = table.sort_values(
        ["top_poem_coverage", "total_selections", "distinct_poems", "_sort"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).drop(columns="_sort")

    return table[CSV_COLS].reset_index(drop=True)


def print_table(table: pd.DataFrame, *, min_editions: int) -> None:
    if table.empty:
        print(f"No poets with poetry_editions >= {min_editions}.")
        return
    print(
        f"\nPoet signature concentration (poetry; poets in >= {min_editions} editions)"
    )
    print(
        "Ranked by top_poem_coverage (signature poem editions / poetry editions), "
        "then total_selections, then distinct_poems."
    )
    _print_rows(table)


def print_focal(table: pd.DataFrame) -> None:
    print(f"\n{FOCAL_AUTHOR} (signature-poem poet)")
    focal = table[table["author_name"] == FOCAL_AUTHOR]
    if focal.empty:
        print(f"No row for {FOCAL_AUTHOR} at the current --min-editions threshold.")
        return
    _print_rows(focal)


def _print_rows(table: pd.DataFrame) -> None:
    fmt = {
        "top_poem_coverage": lambda v: _fmt(v, 3),
        "signature_work_share_count": lambda v: _fmt(v, 3),
    }
    shown = table[DISPLAY_COLS].copy()
    for col, func in fmt.items():
        shown[col] = shown[col].map(func)
    print(shown.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_save_csv_flag(parser)
    parser.add_argument(
        "--min-editions",
        type=int,
        default=10,
        help="minimum poetry editions for a poet to appear in the table",
    )
    args = parser.parse_args()

    raw = query_db(query_path("author-page-share-reselection"))
    raw = filter_leaf_works(raw, fetch_container_ids())

    outputs = compute(raw)
    table = build_signature_concentration(outputs, raw, min_editions=args.min_editions)

    if args.save_csv:
        table.to_csv(OUT_CSV, index=False)
        print(f"Wrote {OUT_CSV}")
    else:
        print_table(table, min_editions=args.min_editions)
        print_focal(table)


if __name__ == "__main__":
    main()
