from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "reselection"))
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "reselection"))

from author_vs_work_debut_reselection import (  # noqa: E402
    add_entry_group,
    build_edition_table,
)
from edition_pair_retention_scatter import (  # noqa: E402
    callout_pair_index,
    compute_pair_retention,
)


def make_raw(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    defaults = {
        "work_id": None,
        "edition_id": None,
        "anthology_publication_year": None,
        "series_id": None,
        "edition_number": None,
        "parent_id": None,
        "author_id": None,
        "author_birth_year": None,
    }
    df = pd.DataFrame([{**defaults, **row} for row in rows])
    df = add_entry_group(df)
    return df, build_edition_table(df)


@pytest.fixture
def pairs() -> pd.DataFrame:
    raw, editions = make_raw(
        [
            {
                "work_id": "w1",
                "author_id": "a1",
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "series_id": 1,
            },
            {
                "work_id": "w2",
                "author_id": "a2",
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "series_id": 1,
            },
            {
                "work_id": "w1",
                "author_id": "a1",
                "edition_id": 2,
                "anthology_publication_year": 2005,
                "series_id": 1,
            },
            {
                "work_id": "w3",
                "author_id": "a3",
                "edition_id": 2,
                "anthology_publication_year": 2005,
                "series_id": 1,
            },
            {
                "work_id": "w4",
                "author_id": "a1",
                "edition_id": 3,
                "anthology_publication_year": 2010,
            },
        ]
    )
    return compute_pair_retention(raw, editions)


def test_one_row_per_ordered_pair(pairs: pd.DataFrame):
    assert len(pairs) == 3
    assert pairs[["earlier_edition_id", "later_edition_id"]].apply(
        tuple, axis=1
    ).tolist() == [
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_retention_fractions(pairs: pd.DataFrame):
    indexed = pairs.set_index(["earlier_edition_id", "later_edition_id"])

    # Edition 1 -> 2: w1 of {w1, w2} returns; a1 of {a1, a2} returns.
    assert indexed.loc[(1, 2), "work_retention"] == pytest.approx(0.5)
    assert indexed.loc[(1, 2), "author_retention"] == pytest.approx(0.5)

    # Edition 1 -> 3: no works return, but a1 does.
    assert indexed.loc[(1, 3), "work_retention"] == pytest.approx(0.0)
    assert indexed.loc[(1, 3), "author_retention"] == pytest.approx(0.5)


def test_same_series_and_year_gap_flags(pairs: pd.DataFrame):
    indexed = pairs.set_index(["earlier_edition_id", "later_edition_id"])

    assert bool(indexed.loc[(1, 2), "same_series"]) is True
    assert bool(indexed.loc[(1, 3), "same_series"]) is False
    assert bool(indexed.loc[(2, 3), "same_series"]) is False
    assert indexed.loc[(1, 3), "year_gap"] == 10


def test_callout_pair_index_matches_requested_direction(pairs: pd.DataFrame):
    idx = callout_pair_index(pairs, (1, 2))
    row = pairs.loc[idx]
    assert (row["earlier_edition_id"], row["later_edition_id"]) == (1, 2)


def test_callout_pair_index_falls_back_to_reverse_direction(pairs: pd.DataFrame):
    # Only (1, 2) exists, not (2, 1); the lookup should still find it.
    idx = callout_pair_index(pairs, (2, 1))
    row = pairs.loc[idx]
    assert (row["earlier_edition_id"], row["later_edition_id"]) == (1, 2)


def test_callout_pair_index_raises_for_missing_pair(pairs: pd.DataFrame):
    with pytest.raises(ValueError):
        callout_pair_index(pairs, (1, 99))


def test_all_distinct_years_have_no_same_year_rows(pairs: pd.DataFrame):
    assert not pairs["same_year"].any()


@pytest.fixture
def same_year_pairs() -> pd.DataFrame:
    raw, editions = make_raw(
        [
            {
                "work_id": "w1",
                "author_id": "a1",
                "edition_id": 10,
                "anthology_publication_year": 1968,
                "series_id": 1,
            },
            {
                "work_id": "w2",
                "author_id": "a2",
                "edition_id": 10,
                "anthology_publication_year": 1968,
                "series_id": 1,
            },
            {
                "work_id": "w1",
                "author_id": "a1",
                "edition_id": 11,
                "anthology_publication_year": 1968,
                "series_id": 2,
            },
            {
                "work_id": "w3",
                "author_id": "a3",
                "edition_id": 11,
                "anthology_publication_year": 1968,
                "series_id": 2,
            },
            {
                "work_id": "w4",
                "author_id": "a4",
                "edition_id": 11,
                "anthology_publication_year": 1968,
                "series_id": 2,
            },
        ]
    )
    return compute_pair_retention(raw, editions)


def test_same_year_pair_emits_both_permutations(same_year_pairs: pd.DataFrame):
    assert len(same_year_pairs) == 2
    assert same_year_pairs["same_year"].all()

    indexed = same_year_pairs.set_index(["earlier_edition_id", "later_edition_id"])

    # Edition 10 as "earlier": w1 of {w1, w2} returns (0.5); a1 of {a1, a2}
    # returns (0.5).
    assert indexed.loc[(10, 11), "work_retention"] == pytest.approx(0.5)
    assert indexed.loc[(10, 11), "author_retention"] == pytest.approx(0.5)
    assert indexed.loc[(10, 11), "year_gap"] == 0

    # Edition 11 as "earlier": w1 of {w1, w3, w4} returns (1/3); a1 of
    # {a1, a3, a4} returns (1/3) — a different value than the reverse
    # direction, confirming both permutations are independently normalized.
    assert indexed.loc[(11, 10), "work_retention"] == pytest.approx(1 / 3)
    assert indexed.loc[(11, 10), "author_retention"] == pytest.approx(1 / 3)
    assert indexed.loc[(11, 10), "year_gap"] == 0
