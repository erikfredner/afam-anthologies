from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "overlap"))

from half_or_more_summary import compute_summary, format_summary  # noqa: E402

ALL = "All works (incl. excerpts)"
ROOT = "Root works only"
COALESCED = "Coalesced (excerpts → root)"


def make_appearances() -> pd.DataFrame:
    """Four editions; half-or-more threshold is therefore ≥2.

    Work 1 is a root work in editions 1–3 (author a1).
    Work 2 is an excerpt of work 1, in edition 4 only (author a1).
    Work 3 is an authorless root work in edition 1 only.
    Work 4 is a root work in edition 2 only (author a2).
    """
    return pd.DataFrame(
        [
            {"work_id": 1, "parent_id": None, "edition_id": 1, "author_id": "a1"},
            {"work_id": 1, "parent_id": None, "edition_id": 2, "author_id": "a1"},
            {"work_id": 1, "parent_id": None, "edition_id": 3, "author_id": "a1"},
            {"work_id": 2, "parent_id": 1,    "edition_id": 4, "author_id": "a1"},
            {"work_id": 3, "parent_id": None, "edition_id": 1, "author_id": None},
            {"work_id": 4, "parent_id": None, "edition_id": 2, "author_id": "a2"},
        ]
    )


def test_compute_summary_counts_all_variants() -> None:
    summary, total_editions = compute_summary(make_appearances())

    assert total_editions == 4

    # All works: works 1–4 distinct; only work 1 (3 editions) meets ≥2.
    assert summary.loc["Unique works", ALL] == 4
    assert summary.loc["Works in ≥ half", ALL] == 1

    # Root only: excerpt work 2 dropped.
    assert summary.loc["Unique works", ROOT] == 3
    assert summary.loc["Works in ≥ half", ROOT] == 1

    # Coalesced: works 1+2 merge into one family covering all 4 editions.
    assert summary.loc["Unique works", COALESCED] == 3
    assert summary.loc["Works in ≥ half", COALESCED] == 1

    # Authors: a1 in 4 editions, a2 in 1; authorless rows excluded.
    assert summary.loc["Unique authors", ALL] == 2
    assert summary.loc["Authors in ≥ half", ALL] == 1

    # Root only drops a1's excerpt appearance (edition 4) but a1 still has 3.
    assert summary.loc["Unique authors", ROOT] == 2
    assert summary.loc["Authors in ≥ half", ROOT] == 1

    # Coalescing works never changes author counts.
    assert (summary[COALESCED][:2] == summary[ALL][:2]).all()


def test_compute_summary_root_variant_loses_excerpt_only_authors() -> None:
    df = pd.DataFrame(
        [
            {"work_id": 1, "parent_id": None, "edition_id": 1, "author_id": "a1"},
            {"work_id": 2, "parent_id": 1,    "edition_id": 2, "author_id": "a2"},
        ]
    )
    summary, _ = compute_summary(df)

    assert summary.loc["Unique authors", ALL] == 2
    assert summary.loc["Unique authors", ROOT] == 1


def test_format_summary_adds_percentages() -> None:
    summary, _ = compute_summary(make_appearances())
    table = format_summary(summary)

    assert table.loc["Unique works", ALL] == "4"
    assert table.loc["Works in ≥ half", ALL] == "1 (25.0%)"
    assert table.loc["Authors in ≥ half", ALL] == "1 (50.0%)"
