"""
Identify authors above an edition_count threshold whose presence is not
explained by authorship of a frequently anthologized work.
"""

import argparse
import csv
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=int,
        default=13,
        help="Minimum edition_count to include (default: 13)",
    )
    parser.add_argument(
        "--authors-csv",
        default="data/most_anthologized_authors.csv",
        help="Path to most_anthologized_authors.csv",
    )
    parser.add_argument(
        "--works-csv",
        default="data/most_anthologized_works.csv",
        help="Path to most_anthologized_works.csv",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent
    authors = load_csv(data_dir / args.authors_csv)
    works = load_csv(data_dir / args.works_csv)

    threshold = args.threshold

    # Authors above threshold
    frequent_authors: dict[str, int] = {
        row["author_name"]: int(row["edition_count"])
        for row in authors
        if int(row["edition_count"]) >= threshold
    }

    # Authors whose works appear above threshold (split multi-author entries)
    authors_with_frequent_works: dict[str, int] = {}
    for row in works:
        if int(row["edition_count"]) < threshold:
            continue
        for name in row["author_names"].split(";"):
            name = name.strip()
            if not name:
                continue
            best = authors_with_frequent_works.get(name, 0)
            authors_with_frequent_works[name] = max(best, int(row["edition_count"]))

    explained = {
        name: count
        for name, count in frequent_authors.items()
        if name in authors_with_frequent_works
    }
    unexplained = {
        name: count
        for name, count in frequent_authors.items()
        if name not in authors_with_frequent_works
    }

    print(f"Threshold: edition_count >= {threshold}\n")

    print(
        f"Frequently anthologized authors WITH a frequently anthologized work "
        f"({len(explained)}):"
    )
    for name, count in sorted(explained.items(), key=lambda x: -x[1]):
        work_count = authors_with_frequent_works[name]
        print(f"  {name:40s}  author editions: {count:3d}  best work editions: {work_count:3d}")

    print()
    print(
        f"Frequently anthologized authors WITHOUT a frequently anthologized work "
        f"({len(unexplained)}):"
    )
    for name, count in sorted(unexplained.items(), key=lambda x: -x[1]):
        print(f"  {name:40s}  author editions: {count:3d}")


if __name__ == "__main__":
    main()
