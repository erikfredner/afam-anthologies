#!/usr/bin/env python3
"""
Author selection records across the African American anthology corpus.

Default output: a comma-separated list of author names with anthology counts for
authors selected in 10 or more anthologies, sorted by anthology_count descending;
authors with the same count are alphabetized. Output format:
"Author Name (10), Next Author (9), ..."

With --percentiles, print each author's rank performance relative to peers: the
percentile rank of their selection count among all authors ever selected in an
AFAM anthology, plus the "top X%" share that count places them in. --author NAME
reports that sentence for a single author.
"""

import argparse

import pandas as pd

from afam.db import query as query_db
from afam.sql import query_path

MIN_ANTHOLOGIES = 10


def load_all_counts():
    """
    Return a list of (author_name, anthology_count) for every author appearing in
    at least one AFAM anthology edition, read live from the database.
    """
    df = query_db(query_path("author-edition-counts-afam"))
    return [
        (name, int(count))
        for name, count in zip(df["author_name"], df["edition_count"])
    ]


def load_counts(min_anthologies=MIN_ANTHOLOGIES):
    """
    Return a list of (author_name, anthology_count) for authors appearing in
    `min_anthologies` or more AFAM anthology editions.
    """
    return [
        (name, count) for name, count in load_all_counts() if count >= min_anthologies
    ]


def total_editions():
    """Number of AFAM-tagged editions — the denominator of a selection record."""
    return int(query_db(query_path("afam-edition-count"))["n"].iloc[0])


def format_authors(authors):
    """
    Given a list of (author_name, anthology_count), return a
    comma-separated formatted string sorted by count desc, then name asc.
    """
    # Sort by count descending, then author_name ascending
    sorted_authors = sorted(authors, key=lambda x: (-x[1], x[0]))
    return ", ".join(f"{name} ({count})" for name, count in sorted_authors)


def compute_percentiles(authors, n_editions=None):
    """
    Rank authors by selection count against the full population of selected authors.

    `authors` is a list of (author_name, anthology_count) covering every author to
    rank against — percentiles are only meaningful over the whole population, so
    pass the unfiltered list from load_all_counts().

    Returns a DataFrame sorted by count desc, then name asc, with columns:
      author, selections, selection_rate (if n_editions given), percentile, top_pct

    percentile — share of all authors this author is selected strictly more often
                 than (0-100); ties do not count toward it.
    top_pct    — share of all authors selected at least this often, i.e. the
                 author is in the "top top_pct%" by selection record. Tied authors
                 share the same value.
    """
    df = pd.DataFrame(authors, columns=["author", "selections"])
    if df.empty:
        return df.assign(percentile=[], top_pct=[])

    n = len(df)
    counts = df["selections"].to_numpy()
    # Compare every author's count against the whole population (n is small).
    df["percentile"] = [100.0 * (counts < c).sum() / n for c in counts]
    df["top_pct"] = [100.0 * (counts >= c).sum() / n for c in counts]
    if n_editions:
        df.insert(2, "selection_rate", df["selections"] / n_editions)
    return df.sort_values(
        ["selections", "author"], ascending=[False, True]
    ).reset_index(drop=True)


def describe_author(table, author, n_editions=None, n_authors=None):
    """Return a one-sentence summary of `author`'s percentile standing, or None."""
    matches = table[table["author"].str.casefold() == author.casefold()]
    if matches.empty:
        matches = table[table["author"].str.contains(author, case=False, regex=False)]
    if matches.empty:
        return None

    row = matches.iloc[0]
    record = f"{int(row['selections'])}"
    if n_editions:
        record += f" of {n_editions}"
    population = f"{n_authors} authors" if n_authors else "authors"
    return (
        f"{row['author']} is in the top {row['top_pct']:.1f}% of {population} by "
        f"selection record ({record} anthologies; percentile rank "
        f"{row['percentile']:.1f})."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Author anthology-selection counts, optionally as percentiles."
    )
    parser.add_argument(
        "--percentiles",
        action="store_true",
        help="Print percentile ranks instead of the formatted name (count) list.",
    )
    parser.add_argument(
        "--author",
        help="Report one author's percentile standing (implies --percentiles).",
    )
    parser.add_argument(
        "--min-anthologies",
        type=int,
        default=MIN_ANTHOLOGIES,
        help=f"Minimum anthology count to display (default {MIN_ANTHOLOGIES}). "
        "Percentiles are always computed over all authors.",
    )
    args = parser.parse_args()

    all_authors = load_all_counts()

    if not (args.percentiles or args.author):
        print(format_authors([a for a in all_authors if a[1] >= args.min_anthologies]))
        return

    n_editions = total_editions()
    table = compute_percentiles(all_authors, n_editions=n_editions)

    if args.author:
        sentence = describe_author(
            table, args.author, n_editions=n_editions, n_authors=len(table)
        )
        print(sentence or f"No author matching {args.author!r}.")
        return

    shown = table[table["selections"] >= args.min_anthologies]
    print(
        shown.to_string(
            index=False,
            formatters={
                "selection_rate": "{:.3f}".format,
                "percentile": "{:.1f}".format,
                "top_pct": "{:.1f}".format,
            },
        )
    )
    print(
        f"\n{len(shown)} of {len(table)} authors shown "
        f"(>= {args.min_anthologies} of {n_editions} anthologies); "
        "percentiles computed over all authors ever selected."
    )


if __name__ == "__main__":
    main()
