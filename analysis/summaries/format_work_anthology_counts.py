#!/usr/bin/env python3
"""
Generate a comma-separated list of works with anthology counts for works
selected in 10 or more anthologies. The list is sorted by anthology_count descending;
works with the same count are alphabetized by work name. Output format:
"Work Name" by Author Name (10), "Next Work" by Next Author (9), ...
"""

import argparse

from afam.db import query as query_db
from afam.sql import query_path


def load_counts():
    """
    Return a list of (work_name, author_name, anthology_count) for works
    appearing in 10 or more AFAM anthology editions, read live from the database.
    """
    counts = query_db(query_path("work-edition-counts-afam"))
    wide = query_db(query_path("works-authors-per-afam-edition"))
    authors_by_work = (
        wide.dropna(subset=["author_id"])
        .drop_duplicates(["work_id", "author_name"])
        .groupby("work_id")["author_name"]
        .apply(lambda names: " & ".join(sorted(names)))
    )

    records = []
    for _, row in counts.iterrows():
        count = int(row["edition_count"])
        if count < 10:
            continue
        author = authors_by_work.get(int(row["work_id"]), "")
        records.append((row["work_title"], author, count))
    return records


def format_entries(entries):
    """
    Given a list of (work_name, author_name, count), return a
    comma-separated formatted string sorted by count desc, then work_name asc.
    """
    sorted_entries = sorted(entries, key=lambda x: (-x[2], x[0]))
    return ", ".join(
        f'"{work}" by {author} ({count})' for work, author, count in sorted_entries
    )


def main():
    argparse.ArgumentParser(
        description="Generate formatted list of works with >=10 anthologies."
    ).parse_args()
    entries = load_counts()
    output = format_entries(entries)
    print(output)


if __name__ == "__main__":
    main()
