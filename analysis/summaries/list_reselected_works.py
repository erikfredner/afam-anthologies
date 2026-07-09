#!/usr/bin/env python3
"""
list_reselected_works.py

Reads works and authors across the AFAM anthology corpus from the live database
and prints to stdout:

1. Authors who appear in every AFAM edition.
2. Works that appear in every AFAM edition.

Authors are printed as:
    - Author Name

Works are printed as:
    "Title" by Author Name

Both lists are sorted alphabetically (authors by name; works by author then title).

Usage:
    uv run python analysis/summaries/list_reselected_works.py
"""

import argparse

from afam.db import query as query_db
from afam.sql import query_path


def main():
    argparse.ArgumentParser(
        description="List authors and works selected in every AFAM edition."
    ).parse_args()

    authors = query_db(query_path("authors-in-all-afam-eds"))
    works = query_db(query_path("works-in-all-afam-eds"))
    wide = query_db(query_path("works-authors-per-afam-edition"))

    # One author name per work (first listed author) for the "by Author" label.
    work_author = (
        wide.dropna(subset=["author_id"])
        .drop_duplicates("work_id")
        .set_index("work_id")["author_name"]
    )

    universal_authors = sorted(authors["name"], key=lambda n: str(n).lower())
    universal_works = sorted(
        (
            (work_author.get(int(row["id"]), ""), row["title"])
            for _, row in works.iterrows()
        ),
        key=lambda at: (str(at[0]).lower(), str(at[1]).lower()),
    )

    print("Universally selected authors:")
    if universal_authors:
        for name in universal_authors:
            print(f"- {name}")
    else:
        print("(none)")

    print("\nUniversally selected works:")
    if universal_works:
        for author, title in universal_works:
            print(f'"{title}" by {author}')
    else:
        print("(none)")


if __name__ == "__main__":
    main()
