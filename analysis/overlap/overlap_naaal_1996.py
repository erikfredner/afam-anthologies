"""
Compute straight-up overlap between the first edition of The Norton Anthology of African American Literature (1996)
and all anthologies published before 1996.
"""
import argparse
from typing import Set

from afam.data import load_csv

def main() -> None:
    """
    Load anthology data, compute overlap metrics, and print results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-root-works", action="store_true",
                        help="Limit work-level overlap analysis to works without parent works (parent_work_title empty).")
    args = parser.parse_args()
    df = load_csv("202505121539 authors works.csv")
    # Ensure series_id column exists for edition_key logic
    if "series_id" not in df.columns:
        df["series_id"] = ""

    # Create a helper column for unique edition identification
    df["edition_key"] = df.apply(
        lambda row: f"{row['series_id']}_{row['anthology_edition']}"
        if row["series_id"] else row["anthology_id"],
        axis=1,
    )

    # Identify NAAAL 1996 first edition
    mask_naaal_1996 = (
        (df["anthology_series"] == "The Norton Anthology of African American Literature")
        & (df["anthology_edition"] == "1")
        & (df["anthology_year"] == "1996")
    )
    naaal_1996 = df.loc[mask_naaal_1996]
    # Optionally restrict to root works only
    if args.only_root_works:
        naaal_works = naaal_1996[naaal_1996["parent_work_title"] == ""]
    else:
        naaal_works = naaal_1996
    works_naaal_1996: Set[str] = set(naaal_works["work_id"])

    # Identify all pre-1996 anthologies
    pre1996 = df[df["anthology_year"].astype(int) < 1996]
    if args.only_root_works:
        pre1996_works = pre1996[pre1996["parent_work_title"] == ""]
    else:
        pre1996_works = pre1996
    works_pre1996: Set[str] = set(pre1996_works["work_id"])

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
    authors_naaal_1996: Set[str] = set(naaal_1996["work_author"])
    # Authors in pre-1996 anthologies
    authors_pre1996: Set[str] = set(pre1996["work_author"])
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