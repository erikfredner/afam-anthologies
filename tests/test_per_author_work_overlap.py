from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "overlap"))

from per_author_work_overlap import compute  # noqa: E402


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with the column schema compute() expects."""
    defaults = {
        "author_birth_year": None,
        "author_death_year": None,
        "series_id": None,
    }
    df = pd.DataFrame(rows)
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


def test_cross_series_excludes_within_series_pair():
    """
    Author A appears in NAAAL ed.1, NAAAL ed.2, and a standalone.
    - Within-series pair (NAAAL ed.1 vs NAAAL ed.2) is excluded.
    - Two cross-series pairs are kept (NAAAL ed.1 vs standalone, NAAAL ed.2 vs standalone).
    - Work W1 appears in NAAAL ed.1 and standalone → 1 cross-series pair.
    - Work W2 appears in NAAAL ed.2 and standalone → 1 cross-series pair.
    - Work W3 appears in NAAAL ed.1 and NAAAL ed.2 only → 0 cross-series pairs (within-series).
    - Both W1 and W2 tie at 1 pair → tie-break alphabetically.
    """
    rows = [
        # NAAAL ed.1 (series_id=3, edition_id=16): A has W1, W3
        {
            "author_id": 1,
            "author_name": "Author A",
            "work_id": 1,
            "work_title": "B work",
            "edition_id": 16,
            "series_id": 3,
        },
        {
            "author_id": 1,
            "author_name": "Author A",
            "work_id": 3,
            "work_title": "shared in series",
            "edition_id": 16,
            "series_id": 3,
        },
        # NAAAL ed.2 (series_id=3, edition_id=17): A has W2, W3
        {
            "author_id": 1,
            "author_name": "Author A",
            "work_id": 2,
            "work_title": "A work",
            "edition_id": 17,
            "series_id": 3,
        },
        {
            "author_id": 1,
            "author_name": "Author A",
            "work_id": 3,
            "work_title": "shared in series",
            "edition_id": 17,
            "series_id": 3,
        },
        # Standalone (no series, edition_id=60): A has W1, W2
        {
            "author_id": 1,
            "author_name": "Author A",
            "work_id": 1,
            "work_title": "B work",
            "edition_id": 60,
            "series_id": None,
        },
        {
            "author_id": 1,
            "author_name": "Author A",
            "work_id": 2,
            "work_title": "A work",
            "edition_id": 60,
            "series_id": None,
        },
    ]
    out = compute(make_df(rows))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["author_id"] == 1
    assert row["n_editions"] == 3
    # mean works per edition: (2 + 2 + 2) / 3 = 2.0
    assert row["mean_works_per_edition"] == 2.0
    assert row["median_works_per_edition"] == 2
    # cross-series pairs: (NAAAL1, standalone) and (NAAAL2, standalone) = 2
    assert row["n_cross_series_pairs"] == 2
    # Overlaps: NAAAL1 ∩ standalone = {W1} → 1; NAAAL2 ∩ standalone = {W2} → 1
    assert row["min_overlap"] == 1
    assert row["max_overlap"] == 1
    assert row["mean_overlap"] == 1.0
    assert row["median_overlap"] == 1
    # Top work: W1 and W2 tie at count=1. Tie-break alphabetically: "A work" < "B work".
    assert row["top_work_title"] == "A work"
    assert row["top_work_pair_count"] == 1


def test_same_series_only_author_excluded():
    """Author appearing only in two same-series editions has no cross-series pair."""
    rows = [
        {
            "author_id": 2,
            "author_name": "Author B",
            "work_id": 10,
            "work_title": "X",
            "edition_id": 16,
            "series_id": 3,
        },
        {
            "author_id": 2,
            "author_name": "Author B",
            "work_id": 11,
            "work_title": "Y",
            "edition_id": 17,
            "series_id": 3,
        },
    ]
    out = compute(make_df(rows))
    assert len(out) == 0


def test_single_edition_author_excluded():
    """Author appearing in only one edition has no pairs at all."""
    rows = [
        {
            "author_id": 3,
            "author_name": "Author C",
            "work_id": 20,
            "work_title": "Z",
            "edition_id": 16,
            "series_id": 3,
        },
    ]
    out = compute(make_df(rows))
    assert len(out) == 0


def test_two_standalones_are_cross_series():
    """Two standalone editions (both with series_id=None) are treated as different
    singleton series, so the pair is kept."""
    rows = [
        {
            "author_id": 4,
            "author_name": "Author D",
            "work_id": 30,
            "work_title": "shared",
            "edition_id": 60,
            "series_id": None,
        },
        {
            "author_id": 4,
            "author_name": "Author D",
            "work_id": 30,
            "work_title": "shared",
            "edition_id": 61,
            "series_id": None,
        },
    ]
    out = compute(make_df(rows))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_cross_series_pairs"] == 1
    assert row["max_overlap"] == 1
    assert row["top_work_title"] == "shared"


def test_top_work_counts_only_cross_series_pairs():
    """
    Author appears in 4 editions: NAAAL ed.1, ed.2, ed.3, and one standalone.
    Work W appears in all 4 editions.
      - Within-series pairs (3 NAAAL × NAAAL pairs) — excluded.
      - Cross-series pairs: each NAAAL × standalone = 3.
    So W's pair_count should be 3.
    """
    rows = []
    for eid in (16, 17, 13):  # NAAAL editions
        rows.append(
            {
                "author_id": 5,
                "author_name": "Author E",
                "work_id": 100,
                "work_title": "Everywhere",
                "edition_id": eid,
                "series_id": 3,
            }
        )
    rows.append(
        {
            "author_id": 5,
            "author_name": "Author E",
            "work_id": 100,
            "work_title": "Everywhere",
            "edition_id": 60,
            "series_id": None,
        }
    )
    out = compute(make_df(rows))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_editions"] == 4
    assert row["n_cross_series_pairs"] == 3
    assert row["top_work_pair_count"] == 3
    assert row["top_work_title"] == "Everywhere"


def test_birth_and_death_year_passthrough():
    """Birth and death year are surfaced as ints when present, None when null."""
    rows = [
        {
            "author_id": 6,
            "author_name": "Known Years",
            "author_birth_year": 1900,
            "author_death_year": 1980,
            "work_id": 200,
            "work_title": "x",
            "edition_id": 16,
            "series_id": 3,
        },
        {
            "author_id": 6,
            "author_name": "Known Years",
            "author_birth_year": 1900,
            "author_death_year": 1980,
            "work_id": 200,
            "work_title": "x",
            "edition_id": 60,
            "series_id": None,
        },
    ]
    out = compute(make_df(rows))
    row = out.iloc[0]
    assert row["author_birth_year"] == 1900
    assert row["author_death_year"] == 1980


def test_overlap_can_be_zero():
    """If two cross-series editions share no works, overlap=0 is recorded."""
    rows = [
        {
            "author_id": 7,
            "author_name": "No Overlap",
            "work_id": 300,
            "work_title": "only here",
            "edition_id": 16,
            "series_id": 3,
        },
        {
            "author_id": 7,
            "author_name": "No Overlap",
            "work_id": 301,
            "work_title": "only there",
            "edition_id": 60,
            "series_id": None,
        },
    ]
    out = compute(make_df(rows))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_cross_series_pairs"] == 1
    assert row["min_overlap"] == 0
    assert row["max_overlap"] == 0
    # No work appeared in any cross-series pair, so top_work fields are empty.
    assert pd.isna(row["top_work_id"])
    assert row["top_work_pair_count"] == 0
