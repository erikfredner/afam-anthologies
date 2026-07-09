"""
poet_work_dispersal.py
----------------------
Compare concentration vs. dispersal of editorial selection at the level of
individual poems, among widely-anthologized poets.

An author can be universally canonized (selected as a poet across the corpus)
while editors disagree sharply about *which* of that author's poems belong in
the anthology. This script surfaces that gap with two side-by-side senses of
"dispersal":

  - top_poem_coverage: the poet's single most-anthologized poem, expressed as a
    fraction of the editions that poet's poetry appears in. This is the direct
    test of the Sterling Brown case -- selected for nearly every edition, yet no
    single Brown poem reaches even half of them (coverage < 0.5).
  - effective_poems_count / effective_poems_page: 1 / HHI of the poet's poems,
    i.e. the equivalent number of equally-weighted poems. This rewards breadth:
    Langston Hughes spreads across ~29 effective poems, yet still anchors a
    near-canonical "Mother to Son" (coverage > 0.5).

The two metrics disagree on who is "most dispersed", which is the point: Brown
tops top_poem_coverage (most contested canon) while Hughes tops effective_poems
(widest oeuvre).

The top poem by edition coverage may differ from the page-weighted "signature"
poem reported by author_form_concentration -- e.g. Countee Cullen's "Yet Do I
Marvel" leads by edition count while the longer "Heritage" leads by pages.

Unit of analysis -- leaf works (individual poems). The source data catalogues a
single poem inconsistently: sometimes as a standalone root work, sometimes as a
child of the collection volume it appeared in (e.g. "If We Must Die" is a child
of "Harlem Shadows <volume of poems>"). The repo-wide root-only convention --
meant to collapse novel excerpts into whole works -- therefore *deletes* the
canonical poems for poetry. This script instead keeps every leaf work (any work
that is not a parent of another work) and drops the collection containers, so
each individual poem is counted once regardless of how it was catalogued.

Usage:
    uv run python analysis/concentration/poet_work_dispersal.py
    uv run python analysis/concentration/poet_work_dispersal.py --min-editions 13
    uv run python analysis/concentration/poet_work_dispersal.py --save-csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "concentration"))

from author_form_concentration import _fmt, compute  # noqa: E402

from afam import DATA_DIR  # noqa: E402
from afam.cli import add_save_csv_flag  # noqa: E402
from afam.db import query as query_db  # noqa: E402
from afam.names import author_sort_key  # noqa: E402
from afam.sql import query_path  # noqa: E402

OUT_CSV = DATA_DIR / "poet_work_dispersal.csv"
POETRY = "poetry"

CSV_COLS = [
    "author_id",
    "author_name",
    "author_last_name",
    "poetry_editions",
    "author_editions_all_forms",
    "distinct_works_all_forms",
    "distinct_poems",
    "top_poem",
    "top_poem_editions",
    "top_poem_coverage",
    "effective_poems_count",
    "effective_poems_page",
    "hhi_count",
    "norm_entropy_count",
    "gini_count",
]

DISPLAY_COLS = [
    "author_name",
    "poetry_editions",
    "distinct_poems",
    "top_poem",
    "top_poem_editions",
    "top_poem_coverage",
    "effective_poems_count",
    "effective_poems_page",
    "norm_entropy_count",
]


def fetch_container_ids() -> set[int]:
    """Work ids that are the parent of another work (collection volumes, novels)."""
    df = query_db(query_path("work-parent-ids"))
    return {int(x) for x in df["work_id"].dropna()}


def filter_leaf_works(raw: pd.DataFrame, container_ids: set[int]) -> pd.DataFrame:
    """Keep only leaf works -- individual poems, not collection containers.

    A "container" is any work that is the parent of another work (a poetry volume
    like *Harlem Shadows*, or a parent novel). Its child leaves are the
    meaningful selection units; the container itself is dropped. This replaces
    the repo-wide root-only convention, which for poetry wrongly deletes the
    canonical poems catalogued as children of their collection.
    """
    container_ids = set(container_ids)
    return raw[~raw["work_id"].isin(container_ids)].reset_index(drop=True)


def build_poet_dispersal(
    outputs: dict[str, pd.DataFrame],
    raw: pd.DataFrame,
    raw_all: pd.DataFrame | None = None,
    *,
    min_editions: int = 10,
) -> pd.DataFrame:
    """Per-poet poetry dispersal table, ranked for the Brown/Hughes comparison.

    `raw` is the leaf-filtered frame (individual poems); `raw_all` is the
    unfiltered frame (every work, all forms) used for the all-forms
    distinct-works count (e.g. Hughes 141 vs McKay 58, draft L121). When
    `raw_all` is omitted it defaults to `raw`.
    """
    if raw_all is None:
        raw_all = raw
    concentration = outputs["concentration"]
    per_work = outputs["per_work"]

    poetry_conc = concentration[
        concentration["form_name"].str.casefold() == POETRY
    ].copy()
    poetry_works = per_work[per_work["form_name"].str.casefold() == POETRY].copy()
    poetry_works = poetry_works.dropna(subset=["author_id"])

    # Top poem per poet by the number of editions selecting it. per_work is
    # pre-sorted by page_p desc then work_title, so idxmax breaks ties toward the
    # higher page-share poem, then alphabetically -- a deterministic choice.
    top_idx = poetry_works.groupby("author_id")["n_editions_selecting_work"].idxmax()
    top = poetry_works.loc[
        top_idx, ["author_id", "work_title", "n_editions_selecting_work"]
    ].rename(
        columns={
            "work_title": "top_poem",
            "n_editions_selecting_work": "top_poem_editions",
        }
    )

    # Editions the author appears in across all forms (Brown is in ~26 overall,
    # vs. 24 as a poet) -- context for the poetry-only denominator.
    all_forms = (
        raw.dropna(subset=["author_id"])
        .groupby("author_id")["edition_id"]
        .nunique()
        .rename("author_editions_all_forms")
        .reset_index()
    )

    # Distinct anthologized works across ALL forms, counted from the unfiltered
    # frame (every work, incl. excerpts/containers). Reproduces the draft's
    # all-forms tally (Hughes 141 vs McKay 58, L121).
    distinct_all = (
        raw_all.dropna(subset=["author_id"])
        .groupby("author_id")["work_id"]
        .nunique()
        .rename("distinct_works_all_forms")
        .reset_index()
    )

    table = (
        poetry_conc.merge(top, on="author_id", how="left")
        .merge(all_forms, on="author_id", how="left")
        .merge(distinct_all, on="author_id", how="left")
    )
    table = table.rename(
        columns={
            "distinct_editions": "poetry_editions",
            "distinct_works": "distinct_poems",
            "effective_works_count": "effective_poems_count",
            "effective_works_page": "effective_poems_page",
            "normalized_entropy_count": "norm_entropy_count",
        }
    )
    table["top_poem_coverage"] = table["top_poem_editions"] / table["poetry_editions"]
    table = table[table["poetry_editions"] >= min_editions].copy()

    table["_sort"] = table["author_name"].map(author_sort_key)
    table = table.sort_values(
        ["poetry_editions", "top_poem_coverage", "_sort"],
        ascending=[False, True, True],
        kind="mergesort",
    ).drop(columns="_sort")

    return table[CSV_COLS].reset_index(drop=True)


def print_table(table: pd.DataFrame, *, min_editions: int) -> None:
    if table.empty:
        print(f"No poets with poetry_editions >= {min_editions}.")
        return
    print(f"\nPoet work-level dispersal (poetry; poets in >= {min_editions} editions)")
    print(
        "top_poem_coverage = editions with the poet's most-anthologized poem / "
        "poetry editions"
    )
    fmt = {
        "top_poem_coverage": lambda v: _fmt(v, 3),
        "effective_poems_count": lambda v: _fmt(v, 1),
        "effective_poems_page": lambda v: _fmt(v, 1),
        "norm_entropy_count": lambda v: _fmt(v, 3),
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

    raw_all = query_db(query_path("author-page-share-reselection"))
    raw = filter_leaf_works(raw_all, fetch_container_ids())

    outputs = compute(raw)
    table = build_poet_dispersal(
        outputs, raw, raw_all, min_editions=args.min_editions
    )

    if args.save_csv:
        table.to_csv(OUT_CSV, index=False)
        print(f"Wrote {OUT_CSV}")
    else:
        print_table(table, min_editions=args.min_editions)


if __name__ == "__main__":
    main()
