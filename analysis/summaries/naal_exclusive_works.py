#!/usr/bin/env python3
"""
naal_exclusive_works.py

Select works that appear in every edition of The Norton Anthology of African American Literature (NAAL)
but never appear in any edition of The Norton Anthology of American Literature (Norton American).
Retain only those works that appear in all NAAL editions (four total).
Print a single line grouping works by author, for example:
    Author A: "Title 1", "Title 2"; Author B: "Title 3"; Author C: "Title 4", "Title 5"

Sample CLI usage:
    python viz/naal_exclusive_works.py data/202504211659_works.csv
"""

import argparse
import csv
import sys

def main():
    parser = argparse.ArgumentParser(
        description=("List works appearing in all editions of the Norton Anthology of African American Literature "
                     "but never in the Norton Anthology of American Literature, grouped by author.")
    )
    parser.add_argument(
        'input_csv', nargs='?', default='data/202504211659_works.csv',
        help=('Path to input CSV (default: data/202504211659_works.csv; '
              'must include "work_title", "author_name", "series_name", '
              'and "anthology_edition" columns)')
    )
    args = parser.parse_args()

    NAAL = "The Norton Anthology of African American Literature"
    NAFAM = "The Norton Anthology of American Literature"

    naal_editions = {}
    nafam_works = set()
    naal_all_editions = set()
    try:
        with open(args.input_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Validate required columns
            for col in ('work_title', 'author_name', 'series_name', 'anthology_edition'):
                if col not in reader.fieldnames:
                    sys.exit(f"ERROR: Expected column '{col}' in {args.input_csv}")
            # Collect editions per work for NAAL and track any appearances in NAFAM
            for row in reader:
                title = row['work_title'].strip()
                author = row['author_name'].strip()
                series = row['series_name'].strip()
                edition = row['anthology_edition'].strip()
                if series == NAAL:
                    naal_editions.setdefault((title, author), set()).add(edition)
                    naal_all_editions.add(edition)
                elif series == NAFAM:
                    nafam_works.add((title, author))
    except FileNotFoundError:
        sys.exit(f"ERROR: File not found: {args.input_csv}")

    # Determine works exclusive to NAAL that appear in all NAAL editions
    complete_works = {
        (title, author): editions
        for (title, author), editions in naal_editions.items()
        if (title, author) not in nafam_works and editions == naal_all_editions
    }

    if not complete_works:
        print("")
        return

    # Group works by author, sorting authors and titles alphabetically
    works_by_author = {}
    for (title, author) in complete_works:
        works_by_author.setdefault(author, []).append(title)

    groups = []
    for author in sorted(works_by_author, key=lambda x: x.lower()):
        titles = sorted(works_by_author[author], key=lambda t: t.lower())
        quoted = ", ".join(f"\"{t}\"" for t in titles)
        groups.append(f"{author}: {quoted}")

    # Print a single line with semicolon-separated author groups
    print("; ".join(groups))

if __name__ == '__main__':
    main()