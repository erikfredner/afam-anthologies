#!/usr/bin/env python3
"""
Generate a comma-separated list of works with anthology counts for works
selected in 10 or more anthologies. The list is sorted by anthology_count descending;
works with the same count are alphabetized by work name. Output format:
"Work Name" by Author Name (10), "Next Work" by Next Author (9), ...
"""
import csv
import argparse
import sys

def load_counts(input_csv):
    """
    Read the input CSV and return a list of (work_name, author_name, anthology_count)
    for works appearing in 10 or more anthologies.
    """
    records = []
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Validate required columns
        for col in ('work_name', 'author_name', 'anthology_count'):
            if col not in reader.fieldnames:
                sys.exit(f"Error: Column '{col}' not found in input CSV")
        for row in reader:
            work = row['work_name'].strip()
            author = row['author_name'].strip()
            try:
                count = int(row['anthology_count'])
            except ValueError:
                continue
            if count >= 10:
                records.append((work, author, count))
    return records

def format_entries(entries):
    """
    Given a list of (work_name, author_name, count), return a
    comma-separated formatted string sorted by count desc, then work_name asc.
    """
    sorted_entries = sorted(entries, key=lambda x: (-x[2], x[0]))
    return ", ".join(f'"{work}" by {author} ({count})' for work, author, count in sorted_entries)

def main():
    parser = argparse.ArgumentParser(
        description="Generate formatted list of works with >=10 anthologies."
    )
    parser.add_argument(
        'input_csv', nargs='?', default='data/work_anthology_counts.csv',
        help='Path to work_anthology_counts.csv'
    )
    args = parser.parse_args()
    entries = load_counts(args.input_csv)
    output = format_entries(entries)
    print(output)

if __name__ == '__main__':
    main()