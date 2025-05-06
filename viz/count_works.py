#!/usr/bin/env python3
"""
Count the number of distinct anthologies in which each work appears.
Distinct anthologies are defined by unique combinations of
anthology_series, anthology_name, anthology_year, and anthology_edition_number.
"""
import csv
import argparse

def count_works(input_csv):
    """
    Read the input CSV and return a dict mapping
    (author_id, author_name) or (None, author_name) to count of distinct works.
    """
    # Map each work to the set of anthologies it appears in
    work_anthologies = {}
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Ensure required columns exist
        required = (
            'work_id',
            'work_name',
            'parent_work_name',
            'author_name',
            'anthology_series',
            'anthology_name',
            'anthology_year',
            'anthology_edition_number'
        )
        for col in required:
            if col not in reader.fieldnames:
                raise ValueError(f"Column '{col}' not found in input file")
        # Process rows
        for row in reader:
            author = row.get('author_name', '').strip()
            if not author:
                continue
            work_id = row.get('work_id', '').strip()
            work_name = row.get('work_name', '').strip()
            parent = row.get('parent_work_name', '').strip()
            work_key = (work_id, work_name, parent, author)

            series = row.get('anthology_series', '').strip()
            anth_name = row.get('anthology_name', '').strip()
            year = row.get('anthology_year', '').strip()
            edition = row.get('anthology_edition_number', '').strip()
            anth_key = (series, anth_name, year, edition)

            work_anthologies.setdefault(work_key, set()).add(anth_key)
    # Convert sets to counts
    return {work: len(anths) for work, anths in work_anthologies.items()}

def main():
    parser = argparse.ArgumentParser(
        description='Count distinct anthologies per work.'
    )
    parser.add_argument('input_csv', help='Path to input works CSV file')
    parser.add_argument('output_csv', help='Path to output CSV file')
    args = parser.parse_args()

    counts = count_works(args.input_csv)
    # Output columns: work_name, parent_work_name, author_name, anthology_count
    header = ['work_name', 'parent_work_name', 'author_name', 'anthology_count']
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # Sort by work_name
        for (work_id, work_name, parent, author), count in sorted(
            counts.items(), key=lambda item: item[0][1]
        ):
            writer.writerow([work_name, parent, author, count])

if __name__ == '__main__':
    main()