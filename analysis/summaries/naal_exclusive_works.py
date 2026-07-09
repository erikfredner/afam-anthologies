#!/usr/bin/env python3
"""
naal_exclusive_works.py

Works that appear in every edition of The Norton Anthology of African American
Literature (NAAAL, series_id=3) but never in any edition of The Norton Anthology
of American Literature (Norton American, series_id=1), read live from the
database. Print a single line grouping works by author, e.g.:
    Author A: "Title 1", "Title 2"; Author B: "Title 3"

Usage:
    uv run python analysis/summaries/naal_exclusive_works.py
"""

import argparse

import pandas as pd

from afam.db import query as query_db
from afam.sql import query_path

NAAAL_SERIES_ID = 3
NAFAM_SERIES_ID = 1


def main():
    argparse.ArgumentParser(
        description=(
            "List works in every NAAAL edition but never in Norton American, "
            "grouped by author."
        )
    ).parse_args()

    df = query_db(query_path("naal-american-authors-works"))
    naal = df[df["series_id"] == NAAAL_SERIES_ID]
    nafam_work_ids = set(df.loc[df["series_id"] == NAFAM_SERIES_ID, "work_id"])

    n_all_editions = naal["edition_id"].nunique()
    work_edition_counts = naal.groupby("work_id")["edition_id"].nunique()
    complete_ids = {
        wid
        for wid, cnt in work_edition_counts.items()
        if cnt == n_all_editions and wid not in nafam_work_ids
    }

    if not complete_ids:
        print("")
        return

    meta = naal.loc[
        naal["work_id"].isin(complete_ids),
        ["work_id", "work_title", "author_name"],
    ].drop_duplicates(["work_id", "author_name"])
    works_by_author: dict[str, list[str]] = {}
    for _, work in meta.iterrows():
        author = work["author_name"]
        author = "(no author)" if pd.isna(author) else author
        title = work["work_title"]
        works_by_author.setdefault(author, []).append(title)

    groups = []
    for author in sorted(works_by_author, key=lambda x: x.lower()):
        titles = sorted(works_by_author[author], key=lambda t: t.lower())
        quoted = ", ".join(f'"{t}"' for t in titles)
        groups.append(f"{author}: {quoted}")

    print("; ".join(groups))


if __name__ == "__main__":
    main()
