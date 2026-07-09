"""
Compute straight-up overlap between the first edition of The Norton Anthology of African American Literature (1996)
and all anthologies published before 1996.
"""

import argparse

from afam.db import query as query_db
from afam.sql import query_path

# NAAAL first edition: series_id=3, edition_number="1". Prior pool = AFAM
# editions published before 1996 (matching the original CSV analysis cutoff).
NAAAL_SERIES_ID = 3
NAAAL_EDITION_NUMBER = "1"
PRIOR_CUTOFF_YEAR = 1996


def main() -> None:
    """
    Load anthology data, compute overlap metrics, and print results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-root-works",
        action="store_true",
        help="Limit work-level overlap analysis to works without a parent work.",
    )
    args = parser.parse_args()
    df = query_db(query_path("works-authors-per-afam-edition"))

    # Identify NAAAL first edition
    naaal_1996 = df[
        (df["series_id"] == NAAAL_SERIES_ID)
        & (df["edition_number"] == NAAAL_EDITION_NUMBER)
    ]
    # Optionally restrict to root works only
    if args.only_root_works:
        naaal_works = naaal_1996[naaal_1996["parent_id"].isna()]
    else:
        naaal_works = naaal_1996
    works_naaal_1996: set = set(naaal_works["work_id"])

    # Identify all pre-1996 anthologies
    pre1996 = df[df["anthology_publication_year"] < PRIOR_CUTOFF_YEAR]
    if args.only_root_works:
        pre1996_works = pre1996[pre1996["parent_id"].isna()]
    else:
        pre1996_works = pre1996
    works_pre1996: set = set(pre1996_works["work_id"])

    # Guard clauses
    if not works_naaal_1996:
        raise ValueError("No works found for NAAAL 1996 in the data.")
    if not works_pre1996:
        raise ValueError("No works found in pre-1996 anthologies.")

    # Compute overlap metrics
    overlap = works_naaal_1996 & works_pre1996
    total = len(works_naaal_1996)
    previous_count = len(overlap)
    coverage_ratio = previous_count / total
    novelty_ratio = 1 - coverage_ratio

    coverage_pct = coverage_ratio * 100
    novelty_pct = novelty_ratio * 100

    # Print results
    print(f"Total NAAAL 1996 works: {total}")
    print(f"Previously anthologised: {previous_count}")
    print(f"Coverage: {coverage_pct:.1f}%")
    print(f"Novelty: {novelty_pct:.1f}%")
    summary = (
        f"Out of {total} works in NAAAL 1996, {previous_count} ({coverage_pct:.1f}%) "
        f"had been printed in an earlier anthology, leaving ({novelty_pct:.1f})% "
        f"that were new to textbook audiences."
    )
    print(f"Summary: {summary}")
    # Compute author overlap metrics
    # Authors of NAAAL 1996 works
    authors_naaal_1996: set = set(naaal_1996["author_id"].dropna())
    # Authors in pre-1996 anthologies
    authors_pre1996: set = set(pre1996["author_id"].dropna())
    # Compute author overlap
    overlap_authors = authors_naaal_1996 & authors_pre1996
    total_authors = len(authors_naaal_1996)
    previous_authors_count = len(overlap_authors)
    coverage_authors_ratio = previous_authors_count / total_authors
    novelty_authors_ratio = 1 - coverage_authors_ratio
    coverage_authors_pct = coverage_authors_ratio * 100
    novelty_authors_pct = novelty_authors_ratio * 100
    # Print author results
    print(f"Total NAAAL 1996 authors: {total_authors}")
    print(f"Previously anthologised authors: {previous_authors_count}")
    print(f"Author Coverage: {coverage_authors_pct:.1f}%")
    print(f"Author Novelty: {novelty_authors_pct:.1f}%")
    summary_authors = (
        f"Out of {total_authors} authors in NAAAL 1996, {previous_authors_count} ({coverage_authors_pct:.1f}%) "
        f"had been printed in an earlier anthology, leaving ({novelty_authors_pct:.1f})% "
        f"that were new to textbook audiences."
    )
    print(f"Author Summary: {summary_authors}")


if __name__ == "__main__":
    main()
