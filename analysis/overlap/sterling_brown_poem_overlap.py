"""
sterling_brown_poem_overlap.py
------------------------------
Fill in the values for the sentence:

    "The median anthology selects X Sterling Brown poems. Across every possible
     pairing of anthologies, on average, those anthologies select Y of the same
     poems by Brown."

X = the median number of distinct Brown poems selected per anthology.
Y = the mean number of *shared* Brown poems across every pair of anthologies.

Scope and definitions:
  - "Anthology" = an edition tagged with the African-American Literature
    tradition (26 editions, 1929–2025) — the same universe used elsewhere in
    this repo.
  - "Poem" = a work by Brown whose literary form is ``poetry`` and which is not
    itself a container/section for other Brown works. Brown's collection
    *Southern Road* is recorded as a nested hierarchy (the volume and its
    sections "Part One: Road So Rocky" … "Part Four: Vestiges" are tagged
    poetry but are parents of the actual leaf poems); counting those containers
    as poems would double-count, so any work that is the ``parent_id`` of
    another Brown work is dropped. "Same poem" = same ``work_id``.

The ``--universe`` flag controls which anthologies enter both statistics:
  - ``all`` (default): all 26 AFAM editions, with zero-filled counts for the
    few that select no Brown poem — selecting zero poems is itself a choice and
    is counted.
  - ``brown``: only anthologies that select at least one Brown poem (24 of 26).

Note on the universe: Brown has at least one work in *all 26* AFAM editions,
but in 2 of them (edition 64, *Dark Symphony*, 1968; edition 57, *Black
Insights*, 1971) he is represented only by nonfiction essays, no poem. So at
the poem level the count is 24/26, and under ``--universe all`` those two
editions enter with zero poems.

Usage:
    uv run python analysis/overlap/sterling_brown_poem_overlap.py
    uv run python analysis/overlap/sterling_brown_poem_overlap.py --universe all
    uv run python analysis/overlap/sterling_brown_poem_overlap.py --author-id 477
"""

from __future__ import annotations

import argparse
from itertools import combinations
from statistics import mean, median

import pandas as pd

from afam.db import query as query_db

# Sterling Brown's author id in the `anthologies` database.
STERLING_BROWN_ID = 477

POEMS_SQL = """
WITH afam AS (
    SELECT DISTINCT e.id
    FROM data_edition e
    JOIN data_edition_literary_traditions elt ON elt.edition_id = e.id
    JOIN data_literarytradition lt ON lt.id = elt.literarytradition_id
    WHERE lt."name" = 'African-American Literature'
),
work_min_form AS (
    SELECT work_id, MIN(form_id) AS form_id
    FROM data_work_form
    GROUP BY work_id
)
SELECT DISTINCT
    w.id        AS work_id,
    w.title     AS work_title,
    w.parent_id AS parent_id,
    e.id        AS edition_id,
    e."year"    AS year,
    f."name"    AS form
FROM data_work w
JOIN data_work_authors wa     ON wa.work_id = w.id
JOIN data_workinanthology wia ON wia.work_id = w.id
JOIN data_volume v            ON v.id = wia.volume_id
JOIN data_edition e           ON e.id = v.edition_id
JOIN afam                     ON afam.id = e.id
LEFT JOIN work_min_form wmf    ON wmf.work_id = w.id
LEFT JOIN data_form f          ON f.id = wmf.form_id
WHERE wa.author_id = %(author_id)s;
"""

AFAM_EDITION_COUNT_SQL = """
SELECT COUNT(DISTINCT e.id) AS n
FROM data_edition e
JOIN data_edition_literary_traditions elt ON elt.edition_id = e.id
JOIN data_literarytradition lt ON lt.id = elt.literarytradition_id
WHERE lt."name" = 'African-American Literature';
"""


def compute(df: pd.DataFrame, n_afam_editions: int) -> dict:
    """Compute per-anthology poem counts and pairwise overlap.

    `df` holds one row per (work, edition) for the target author, with columns
    work_id, parent_id, edition_id, form. Returns a dict of statistics for both
    the "brown" (anthologies-with-the-author) and "all" (zero-filled) universes.
    """
    # Containers/sections: any work that is the parent of another of the
    # author's works. These are dropped so only leaf poems are counted.
    parents = set(df["parent_id"].dropna().astype(int))
    poems = df[
        (df["form"] == "poetry") & (~df["work_id"].astype(int).isin(parents))
    ]

    edition_poems: dict[int, set[int]] = {
        int(eid): set(grp["work_id"].astype(int))
        for eid, grp in poems.groupby("edition_id")
    }

    brown_sets = list(edition_poems.values())
    # Zero-filled universe: every AFAM edition, with empty sets for those that
    # select no poem by the author.
    all_sets = brown_sets + [
        set() for _ in range(n_afam_editions - len(brown_sets))
    ]

    def stats(sets: list[set[int]]) -> dict:
        counts = [len(s) for s in sets]
        overlaps = [len(a & b) for a, b in combinations(sets, 2)]
        return {
            "n_anthologies": len(sets),
            "median_poems": median(counts) if counts else 0,
            "n_pairs": len(overlaps),
            "mean_overlap": round(mean(overlaps), 3) if overlaps else 0.0,
        }

    return {
        "n_distinct_poems": poems["work_id"].nunique(),
        "brown": stats(brown_sets),
        "all": stats(all_sets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--author-id",
        type=int,
        default=STERLING_BROWN_ID,
        help="author id to analyze (default: Sterling Brown, 477)",
    )
    parser.add_argument(
        "--universe",
        choices=["brown", "all"],
        default="all",
        help=(
            "which anthologies enter the sentence: 'all' = every AFAM edition, "
            "zero-filled (default); 'brown' = only those that select a Brown "
            "poem"
        ),
    )
    args = parser.parse_args()

    df = query_db(POEMS_SQL, {"author_id": args.author_id})
    if df.empty:
        raise SystemExit(f"No works found for author id {args.author_id}.")

    n_afam_editions = int(query_db(AFAM_EDITION_COUNT_SQL).iloc[0]["n"])
    result = compute(df, n_afam_editions)

    print(f"Distinct Brown poems anthologized: {result['n_distinct_poems']}")
    print(f"Total AFAM anthologies (editions):  {n_afam_editions}\n")

    for name, label in [
        ("brown", "anthologies that select Brown"),
        ("all", "all AFAM anthologies (zero-filled)"),
    ]:
        s = result[name]
        marker = "  <-- chosen" if name == args.universe else ""
        print(f"[{label}]{marker}")
        print(f"    anthologies            = {s['n_anthologies']}")
        print(f"    median poems / anthology (X) = {s['median_poems']}")
        print(f"    pairs                  = {s['n_pairs']}")
        print(f"    mean shared poems (Y)  = {s['mean_overlap']}\n")

    chosen = result[args.universe]
    x = chosen["median_poems"]
    y = chosen["mean_overlap"]
    x_str = f"{x:g}"
    y_str = f"{y:g}"
    print("Filled-in sentence:")
    print(
        f'    "The median anthology selects {x_str} Sterling Brown poems. '
        f"Across every possible pairing of anthologies, on average, those "
        f'anthologies select {y_str} of the same poems by Brown."'
    )


if __name__ == "__main__":
    main()
