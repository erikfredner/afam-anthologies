"""
Identify authors above an edition_count threshold whose presence is not
explained by authorship of a frequently anthologized work.
"""

from __future__ import annotations

import argparse

import pandas as pd

from afam.db import query


AUTHOR_WORK_COUNTS_SQL = """
WITH tagged_editions AS (
    SELECT DISTINCT e.id
    FROM data_edition e
    JOIN data_edition_literary_traditions elt
      ON elt.edition_id = e.id
    JOIN data_literarytradition lt
      ON lt.id = elt.literarytradition_id
    WHERE lt."name" = 'African-American Literature'
),
author_counts AS (
    SELECT
        a.id AS author_id,
        a."name" AS author_name,
        COUNT(DISTINCT e.id)::int AS author_edition_count
    FROM data_author a
    JOIN data_work_authors wa
      ON wa.author_id = a.id
    JOIN data_workinanthology wia
      ON wia.work_id = wa.work_id
    JOIN data_volume v
      ON v.id = wia.volume_id
    JOIN data_edition e
      ON e.id = v.edition_id
    JOIN tagged_editions te
      ON te.id = e.id
    GROUP BY a.id, a."name"
),
work_author_counts AS (
    SELECT
        wa.author_id,
        w.id AS work_id,
        w.title AS work_title,
        COUNT(DISTINCT e.id)::int AS work_edition_count
    FROM data_work w
    JOIN data_work_authors wa
      ON wa.work_id = w.id
    JOIN data_workinanthology wia
      ON wia.work_id = w.id
    JOIN data_volume v
      ON v.id = wia.volume_id
    JOIN data_edition e
      ON e.id = v.edition_id
    JOIN tagged_editions te
      ON te.id = e.id
    GROUP BY wa.author_id, w.id, w.title
),
best_work_by_author AS (
    SELECT DISTINCT ON (author_id)
        author_id,
        work_id,
        work_title,
        work_edition_count AS best_work_edition_count
    FROM work_author_counts
    ORDER BY author_id, work_edition_count DESC, work_title
)
SELECT
    ac.author_id,
    ac.author_name,
    ac.author_edition_count,
    bw.work_id,
    bw.work_title,
    COALESCE(bw.best_work_edition_count, 0)::int AS best_work_edition_count
FROM author_counts ac
LEFT JOIN best_work_by_author bw
  ON bw.author_id = ac.author_id
ORDER BY ac.author_edition_count DESC, ac.author_name;
"""


def classify_authors(df: pd.DataFrame, threshold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return frequent authors with and without a work meeting the same threshold."""
    frequent = df[df["author_edition_count"] >= threshold].copy()
    explained = frequent[frequent["best_work_edition_count"] >= threshold].copy()
    unexplained = frequent[frequent["best_work_edition_count"] < threshold].copy()
    return explained, unexplained


def print_results(explained: pd.DataFrame, unexplained: pd.DataFrame, threshold: int) -> None:
    print(f"Threshold: edition_count >= {threshold}\n")

    print(
        f"Frequently anthologized authors WITH a frequently anthologized work "
        f"({len(explained)}):"
    )
    for row in explained.itertuples(index=False):
        print(
            f"  {row.author_name:40s}  "
            f"author editions: {row.author_edition_count:3d}  "
            f"best work editions: {row.best_work_edition_count:3d}"
        )

    print()
    print(
        f"Frequently anthologized authors WITHOUT a frequently anthologized work "
        f"({len(unexplained)}):"
    )
    for row in unexplained.itertuples(index=False):
        print(f"  {row.author_name:40s}  author editions: {row.author_edition_count:3d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=int,
        default=13,
        help="Minimum edition_count to include (default: 13)",
    )
    args = parser.parse_args()

    counts = query(AUTHOR_WORK_COUNTS_SQL)
    explained, unexplained = classify_authors(counts, args.threshold)
    print_results(explained, unexplained, args.threshold)


if __name__ == "__main__":
    main()
