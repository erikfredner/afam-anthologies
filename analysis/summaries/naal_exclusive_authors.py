#!/usr/bin/env python3
"""
naal_exclusive_authors.py

Select authors who appear in any edition of The Norton Anthology of African American Literature (NAAL)
but in no edition of The Norton Anthology of American Literature (Norton American). For each such author,
count the number of NAAL editions they appear in. Print a single line listing authors as
"Author Name (count), Next Author (count), …", ordered by edition count descending (4→1) and
alphabetically within each count group.

Sample CLI usage:
    python viz/naal_exclusive_authors.py data/202504211417_naal_nafam_authors.csv
"""

import argparse
import csv
import sys


def main():
    parser = argparse.ArgumentParser(
        description="List authors exclusive to NAAL with counts of NAAL editions."
    )
    parser.add_argument(
        "input_csv",
        help='Path to input CSV (must include "author_name", "anthology_series", and "anthology_edition" columns)',
    )
    args = parser.parse_args()

    NAAL = "The Norton Anthology of African American Literature"
    NAFAM = "The Norton Anthology of American Literature"

    naal_editions = {}
    nafam_authors = set()
    try:
        with open(args.input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for col in ("author_name", "anthology_series", "anthology_edition"):
                if col not in reader.fieldnames:
                    sys.exit(f"ERROR: Expected column '{col}' in {args.input_csv}")
            for row in reader:
                author = row["author_name"].strip()
                series = row["anthology_series"].strip()
                edition = row["anthology_edition"].strip()
                if series == NAAL:
                    naal_editions.setdefault(author, set()).add(edition)
                elif series == NAFAM:
                    nafam_authors.add(author)
    except FileNotFoundError:
        sys.exit(f"ERROR: File not found: {args.input_csv}")

    # Filter authors exclusive to NAAL and count editions
    exclusive = {
        author: len(editions)
        for author, editions in naal_editions.items()
        if author not in nafam_authors
    }

    if not exclusive:
        print("")
        return

    # Group authors by edition count and prepare output
    count_groups = {}
    for author, cnt in exclusive.items():
        count_groups.setdefault(cnt, []).append(author)

    entries = []
    for cnt in sorted(count_groups.keys(), reverse=True):
        for author in sorted(count_groups[cnt]):
            entries.append(f"{author} ({cnt})")

    print(", ".join(entries))


if __name__ == "__main__":
    main()
