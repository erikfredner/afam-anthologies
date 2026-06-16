#!/usr/bin/env python3
"""
Count the number of distinct anthologies each author appears in.
Distinct anthologies are defined by unique combinations of
series_name, anthology_name, anthology_year, and anthology_edition_number.
"""

import csv
import argparse


def count_anthologies(input_csv):
    """
    Read the input CSV and return a dict mapping
    (author_id, author_name) or (None, author_name) to count of distinct anthologies.
    """
    anth_sets = {}
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Determine if author_id is available
        use_id = "author_id" in reader.fieldnames
        # Ensure required anthology columns exist
        for col in (
            "series_name",
            "anthology_name",
            "anthology_year",
            "anthology_edition_number",
        ):
            if col not in reader.fieldnames:
                raise ValueError(f"Column '{col}' not found in input file")
        if "author_name" not in reader.fieldnames:
            raise ValueError("Column 'author_name' not found in input file")
        # Process rows
        for row in reader:
            author_name = row.get("author_name", "").strip()
            author_key = (
                (row.get("author_id", "").strip(), author_name)
                if use_id
                else (None, author_name)
            )

            series = row.get("series_name", "").strip()
            anth_name = row.get("anthology_name", "").strip()
            year = row.get("anthology_year", "").strip()
            edition = row.get("anthology_edition_number", "").strip()
            anth_key = (series, anth_name, year, edition)

            anth_sets.setdefault(author_key, set()).add(anth_key)
    # Convert sets to counts
    return {author: len(anths) for author, anths in anth_sets.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Count distinct anthologies per author."
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_csv", help="Path to output CSV file")
    args = parser.parse_args()

    counts = count_anthologies(args.input_csv)
    # Prepare header
    has_id = any(author_key[0] for author_key in counts)
    header = (
        ["author_id", "author_name", "anthology_count"]
        if has_id
        else ["author_name", "anthology_count"]
    )
    # Write output CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # Sort by author_name
        for (author_id, author_name), count in sorted(
            counts.items(), key=lambda x: x[0][1]
        ):
            if has_id:
                writer.writerow([author_id, author_name, count])
            else:
                writer.writerow([author_name, count])


if __name__ == "__main__":  # noqa: C901
    main()
