from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "analysis" / "summaries"))

import format_author_anthology_counts as facs

AUTHORS = [
    ("Top", 10),
    ("Tied A", 6),
    ("Tied B", 6),
    ("Middle", 3),
    ("Bottom", 1),
]


def test_percentiles_rank_against_whole_population():
    table = facs.compute_percentiles(AUTHORS)
    by_author = table.set_index("author")

    # Sorted by selections desc, then name asc.
    assert list(table["author"]) == ["Top", "Tied A", "Tied B", "Middle", "Bottom"]

    # Top author beats the other 4 of 5 and is alone in the top 20%.
    assert by_author.loc["Top", "percentile"] == 80.0
    assert by_author.loc["Top", "top_pct"] == 20.0

    # Ties share both values; neither counts toward the other's percentile.
    assert (
        by_author.loc["Tied A", "percentile"] == by_author.loc["Tied B", "percentile"]
    )
    assert by_author.loc["Tied A", "percentile"] == 40.0
    assert by_author.loc["Tied A", "top_pct"] == 60.0

    # Lowest author is above nobody but still inside the top 100%.
    assert by_author.loc["Bottom", "percentile"] == 0.0
    assert by_author.loc["Bottom", "top_pct"] == 100.0


def test_selection_rate_added_when_edition_count_given():
    table = facs.compute_percentiles(AUTHORS, n_editions=20)
    assert table.loc[0, "selection_rate"] == 0.5

    assert "selection_rate" not in facs.compute_percentiles(AUTHORS).columns


def test_describe_author_matches_case_insensitively_and_partially():
    table = facs.compute_percentiles(AUTHORS, n_editions=20)

    sentence = facs.describe_author(
        table, "tied a", n_editions=20, n_authors=len(table)
    )
    assert sentence.startswith("Tied A is in the top 60.0% of 5 authors")
    assert "6 of 20 anthologies" in sentence

    assert facs.describe_author(table, "Midd") is not None
    assert facs.describe_author(table, "Nobody") is None


def test_load_counts_filters_by_minimum(monkeypatch):
    monkeypatch.setattr(facs, "load_all_counts", lambda: AUTHORS)
    assert facs.load_counts(min_anthologies=6) == [
        ("Top", 10),
        ("Tied A", 6),
        ("Tied B", 6),
    ]
