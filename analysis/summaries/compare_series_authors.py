#!/usr/bin/env python3
"""
compare_series_authors.py

Prints two lists, read live from the database:
  1. Authors who appear in any edition of The Norton Anthology of American
     Literature (series_id=1) but never in the African American one.
  2. Authors who appear in any edition of The Norton Anthology of African
     American Literature (series_id=3) but never in the American one.

Usage:
    uv run python analysis/summaries/compare_series_authors.py
"""

import argparse

from afam.db import query as query_db
from afam.sql import query_path

SERIES_NAMES = {
    1: "The Norton Anthology of American Literature",
    3: "The Norton Anthology of African American Literature",
}


def main():
    argparse.ArgumentParser(
        description="List authors unique to each of the two Norton series."
    ).parse_args()

    df = query_db(query_path("naal-american-authors-works")).dropna(
        subset=["author_id"]
    )

    def authors_in(series_id: int) -> set:
        sub = df[df["series_id"] == series_id]
        return set(zip(sub["author_id"], sub["author_name"]))

    s1, s3 = 1, 3
    s1_authors = authors_in(s1)
    s3_authors = authors_in(s3)

    only_s1 = sorted(s1_authors - s3_authors, key=lambda x: str(x[1]))
    only_s3 = sorted(s3_authors - s1_authors, key=lambda x: str(x[1]))

    print(f"Authors in “{SERIES_NAMES[s1]}” but not in “{SERIES_NAMES[s3]}”:")
    for author_id, author_name in only_s1:
        print(f"{int(author_id)}, {author_name}")
    if not only_s1:
        print("  (none)")

    print(f"\nAuthors in “{SERIES_NAMES[s3]}” but not in “{SERIES_NAMES[s1]}”:")
    for author_id, author_name in only_s3:
        print(f"{int(author_id)}, {author_name}")
    if not only_s3:
        print("  (none)")


if __name__ == "__main__":
    main()
