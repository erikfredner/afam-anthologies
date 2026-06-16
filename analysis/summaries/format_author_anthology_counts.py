#!/usr/bin/env python3
"""
Generate a comma-separated list of author names with anthology counts for authors
selected in 10 or more anthologies. The list is sorted by anthology_count descending;
authors with the same count are alphabetized. Output format:
"Author Name (10), Next Author (9), ..."
"""

import csv
import argparse
import sys


def load_counts(input_csv):
    """
    Read the input CSV and return a list of (author_name, anthology_count)
    for authors appearing in 10 or more anthologies.
    """
    records = []
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Validate required columns
        for col in ("author_name", "anthology_count"):
            if col not in reader.fieldnames:
                sys.exit(f"Error: Column '{col}' not found in input CSV")
        for row in reader:
            name = row["author_name"].strip()
            try:
                count = int(row["anthology_count"])
            except ValueError:
                continue
            if count >= 10:
                records.append((name, count))
    return records


def format_authors(authors):
    """
    Given a list of (author_name, anthology_count), return a
    comma-separated formatted string sorted by count desc, then name asc.
    """
    # Sort by count descending, then author_name ascending
    sorted_authors = sorted(authors, key=lambda x: (-x[1], x[0]))
    return ", ".join(f"{name} ({count})" for name, count in sorted_authors)


def main():
    parser = argparse.ArgumentParser(
        description="Generate formatted list of authors with >=10 anthologies."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="data/author_anthology_counts.csv",
        help="Path to author_anthology_counts.csv",
    )
    args = parser.parse_args()
    authors = load_counts(args.input_csv)
    output = format_authors(authors)
    print(output)


if __name__ == "__main__":
    main()
