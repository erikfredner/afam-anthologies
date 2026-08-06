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
    _html_label,
    callout_pair_index,
    compute_pair_retention,
    drop_unauthored_works,
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


def test_retention_counts_reproduce_the_fractions(pairs: pd.DataFrame):
    indexed = pairs.set_index(["earlier_edition_id", "later_edition_id"])

    # Edition 1 -> 2: 1 of 2 works and 1 of 2 authors return.
    assert indexed.loc[(1, 2), "works_retained"] == 1
    assert indexed.loc[(1, 2), "earlier_n_works"] == 2
    assert indexed.loc[(1, 2), "authors_retained"] == 1
    assert indexed.loc[(1, 2), "earlier_n_authors"] == 2

    ratio = pairs["works_retained"] / pairs["earlier_n_works"]
    assert ratio.tolist() == pytest.approx(pairs["work_retention"].tolist())
    ratio = pairs["authors_retained"] / pairs["earlier_n_authors"]
    assert ratio.tolist() == pytest.approx(pairs["author_retention"].tolist())


def test_html_label_escapes_ampersands():
    # EDITION_LABELS[60] is "Call & Response"; a bare "&" would be invalid
    # markup in the plotly tooltip.
    label = _html_label("Call & Response")
    assert label == "<i>Call &amp; Response</i>"


def test_html_label_keeps_edition_suffix_outside_the_italics():
    assert _html_label("NAAAL ed.3") == "<i>NAAAL</i> ed.3"


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
def unauthored_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    # w2 is anonymous (no author row) and appears in both editions; every
    # other work carries an author. Edition 1 holds {w1, w2}, edition 2 holds
    # {w2, w3}, so keeping w2 makes work retention 1/2 while author retention
    # is 0/1 — dropping it moves work retention to 0/1.
    return make_raw(
        [
            {
                "work_id": "w1",
                "author_id": "a1",
                "edition_id": 1,
                "anthology_publication_year": 2000,
            },
            {
                "work_id": "w2",
                "author_id": None,
                "edition_id": 1,
                "anthology_publication_year": 2000,
            },
            {
                "work_id": "w2",
                "author_id": None,
                "edition_id": 2,
                "anthology_publication_year": 2005,
            },
            {
                "work_id": "w3",
                "author_id": "a3",
                "edition_id": 2,
                "anthology_publication_year": 2005,
            },
        ]
    )


def test_drop_unauthored_works_removes_only_null_author_rows(
    unauthored_raw: tuple[pd.DataFrame, pd.DataFrame],
):
    raw, _ = unauthored_raw
    filtered = drop_unauthored_works(raw)

    assert set(filtered["work_id"]) == {"w1", "w3"}
    assert filtered["author_id"].notna().all()
    # The original frame is left untouched.
    assert set(raw["work_id"]) == {"w1", "w2", "w3"}


def test_unauthored_works_inflate_work_retention_only(
    unauthored_raw: tuple[pd.DataFrame, pd.DataFrame],
):
    raw, editions = unauthored_raw

    with_unauthored = compute_pair_retention(raw, editions).set_index(
        ["earlier_edition_id", "later_edition_id"]
    )
    assert with_unauthored.loc[(1, 2), "work_retention"] == pytest.approx(0.5)
    assert with_unauthored.loc[(1, 2), "author_retention"] == pytest.approx(0.0)

    authored_only = compute_pair_retention(
        drop_unauthored_works(raw), editions
    ).set_index(["earlier_edition_id", "later_edition_id"])
    assert authored_only.loc[(1, 2), "work_retention"] == pytest.approx(0.0)
    # Author retention is unchanged: unauthored works contribute no authors.
    assert authored_only.loc[(1, 2), "author_retention"] == pytest.approx(0.0)


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
