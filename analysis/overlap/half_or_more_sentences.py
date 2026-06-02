"""
half_or_more_sentences.py
--------------------------
Prints two summary sentences with accurate counts of authors and works
that appear in at least half of all African American literary anthology editions.
"""

from __future__ import annotations

from afam.db import query
from afam.sql import query_path


def main() -> None:
    authors_all = query(query_path("author-edition-counts-afam"))
    authors_half = query(query_path("authors-in-half-or-more-afam-eds"))
    works_all = query(query_path("work-edition-counts-afam"))
    works_half = query(query_path("works-in-half-or-more-afam-eds"))

    total_authors = len(authors_all)
    half_authors = len(authors_half)
    pct_authors = round(half_authors / total_authors * 100)

    total_works = len(works_all)
    half_works = works_half["work_id"].nunique()
    pct_works = round(half_works / total_works * 100)

    print(
        f"Of the {total_authors:,} unique authors who appeared in any anthology, "
        f"{half_authors:,} ({pct_authors}%) appear in half or more of the anthologies."
    )
    print(
        f"Of the {total_works:,} unique works, "
        f"{half_works:,} ({pct_works}%) appear in half or more of the anthologies."
    )


if __name__ == "__main__":
    main()
