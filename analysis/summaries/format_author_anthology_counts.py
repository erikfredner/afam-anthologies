#!/usr/bin/env python3
"""
Generate a comma-separated list of author names with anthology counts for authors
selected in 10 or more anthologies. The list is sorted by anthology_count descending;
authors with the same count are alphabetized. Output format:
"Author Name (10), Next Author (9), ..."
"""

import argparse

from afam.db import query as query_db
from afam.sql import query_path


def load_counts():
    """
    Return a list of (author_name, anthology_count) for authors appearing in 10
    or more AFAM anthology editions, read live from the database.
    """
    df = query_db(query_path("author-edition-counts-afam"))
    return [
        (name, int(count))
        for name, count in zip(df["author_name"], df["edition_count"])
        if int(count) >= 10
    ]


def format_authors(authors):
    """
    Given a list of (author_name, anthology_count), return a
    comma-separated formatted string sorted by count desc, then name asc.
    """
    # Sort by count descending, then author_name ascending
    sorted_authors = sorted(authors, key=lambda x: (-x[1], x[0]))
    return ", ".join(f"{name} ({count})" for name, count in sorted_authors)


def main():
    argparse.ArgumentParser(
        description="Generate formatted list of authors with >=10 anthologies."
    ).parse_args()
    authors = load_counts()
    output = format_authors(authors)
    print(output)


if __name__ == "__main__":
    main()
