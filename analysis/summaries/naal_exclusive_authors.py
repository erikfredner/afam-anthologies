#!/usr/bin/env python3
"""
naal_exclusive_authors.py

Authors who appear in any edition of The Norton Anthology of African American
Literature (NAAAL, series_id=3) but in no edition of The Norton Anthology of
American Literature (Norton American, series_id=1), read live from the database.
For each, count the number of NAAAL editions they appear in, and print a single
line "Author Name (count), …" ordered by edition count descending then name.

Usage:
    uv run python analysis/summaries/naal_exclusive_authors.py
"""

import argparse

from afam.db import query as query_db
from afam.sql import query_path

NAAAL_SERIES_ID = 3
NAFAM_SERIES_ID = 1


def main():
    argparse.ArgumentParser(
        description="List authors exclusive to NAAAL with counts of NAAAL editions."
    ).parse_args()

    df = query_db(query_path("naal-american-authors-works")).dropna(
        subset=["author_id"]
    )
    naal = df[df["series_id"] == NAAAL_SERIES_ID]
    nafam_author_ids = set(df.loc[df["series_id"] == NAFAM_SERIES_ID, "author_id"])

    naal_counts = naal.groupby("author_id")["edition_id"].nunique()
    name_map = naal.drop_duplicates("author_id").set_index("author_id")["author_name"]

    exclusive = {
        name_map[aid]: int(cnt)
        for aid, cnt in naal_counts.items()
        if aid not in nafam_author_ids
    }

    if not exclusive:
        print("")
        return

    count_groups: dict[int, list[str]] = {}
    for author, cnt in exclusive.items():
        count_groups.setdefault(cnt, []).append(author)

    entries = []
    for cnt in sorted(count_groups, reverse=True):
        for author in sorted(count_groups[cnt]):
            entries.append(f"{author} ({cnt})")

    print(", ".join(entries))


if __name__ == "__main__":
    main()
