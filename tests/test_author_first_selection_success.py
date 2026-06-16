from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "reselection"))

from author_first_selection_success import compute  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with the column schema compute() expects."""
    defaults = {
        "anthology_id": "",
        "anthology_title": "",
        "series": "",
        "edition_number": "",
        "volume": "",
        "publication_year": "",
        "author_id": "",
    }
    df = pd.DataFrame(rows)
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)
    return df.astype(str)


# ── Test 1: 2-anthology dataset ───────────────────────────────────────────────


def test_two_anthologies_basic():
    """
    Author A: debuts in ant1, not in ant2 → prob=0.0, ever=0
    Author B: debuts in ant1, also in ant2 → prob=1.0, ever=1
    Author C: only in ant2 → subsequent_count=0, prob=NaN, ever=NaN
    """
    rows = [
        {
            "anthology_id": "ant1",
            "anthology_title": "First Anth",
            "publication_year": "2000",
            "author_id": "A",
        },
        {
            "anthology_id": "ant1",
            "anthology_title": "First Anth",
            "publication_year": "2000",
            "author_id": "B",
        },
        {
            "anthology_id": "ant2",
            "anthology_title": "Second Anth",
            "publication_year": "2010",
            "author_id": "B",
        },
        {
            "anthology_id": "ant2",
            "anthology_title": "Second Anth",
            "publication_year": "2010",
            "author_id": "C",
        },
    ]
    detail, edition = compute(make_df(rows))

    by_author = detail.set_index("author_id")

    # Author A: subsequent_count=1, reselection=0
    assert by_author.loc["A", "debut_edition_key"] == "ant1"
    assert by_author.loc["A", "subsequent_count"] == 1
    assert by_author.loc["A", "reselection_count"] == 0
    assert by_author.loc["A", "reselection_prob"] == pytest.approx(0.0)
    assert by_author.loc["A", "ever_reselected"] == 0.0

    # Author B: subsequent_count=1, reselection=1
    assert by_author.loc["B", "debut_edition_key"] == "ant1"
    assert by_author.loc["B", "reselection_prob"] == pytest.approx(1.0)
    assert by_author.loc["B", "ever_reselected"] == 1.0

    # Author C: no subsequent
    assert by_author.loc["C", "debut_edition_key"] == "ant2"
    assert by_author.loc["C", "subsequent_count"] == 0
    assert math.isnan(by_author.loc["C", "reselection_prob"])
    assert math.isnan(by_author.loc["C", "ever_reselected"])


# ── Test 2: 3-anthology dataset ───────────────────────────────────────────────


def test_three_anthologies_varied_probs():
    """
    Author P: debuts in ant1 (2000), subsequent=[ant2, ant3], in ant3 only → prob=0.5
    Author Q: debuts in ant2 (2005), subsequent=[ant3], not in ant3 → prob=0.0
    Author R: debuts in ant3 (2010), no subsequent → NaN
    """
    rows = [
        {
            "anthology_id": "ant1",
            "anthology_title": "Anth 1",
            "publication_year": "2000",
            "author_id": "P",
        },
        {
            "anthology_id": "ant2",
            "anthology_title": "Anth 2",
            "publication_year": "2005",
            "author_id": "Q",
        },
        {
            "anthology_id": "ant3",
            "anthology_title": "Anth 3",
            "publication_year": "2010",
            "author_id": "P",
        },
        {
            "anthology_id": "ant3",
            "anthology_title": "Anth 3",
            "publication_year": "2010",
            "author_id": "R",
        },
    ]
    detail, _ = compute(make_df(rows))
    by_author = detail.set_index("author_id")

    assert by_author.loc["P", "subsequent_count"] == 2
    assert by_author.loc["P", "reselection_count"] == 1
    assert by_author.loc["P", "reselection_prob"] == pytest.approx(0.5)
    assert by_author.loc["P", "ever_reselected"] == 1.0

    assert by_author.loc["Q", "subsequent_count"] == 1
    assert by_author.loc["Q", "reselection_count"] == 0
    assert by_author.loc["Q", "reselection_prob"] == pytest.approx(0.0)
    assert by_author.loc["Q", "ever_reselected"] == 0.0

    assert by_author.loc["R", "subsequent_count"] == 0
    assert math.isnan(by_author.loc["R", "reselection_prob"])


# ── Test 3: Multi-volume dedup ────────────────────────────────────────────────


def test_multivol_dedup():
    """
    Same author in both volumes of edition 1 (same series+edition → same edition_key)
    → one row in detail output per author.
    """
    rows = [
        {
            "anthology_id": "vol1",
            "anthology_title": "Big Anth, Vol. 1",
            "series": "BigSeries",
            "edition_number": "1",
            "publication_year": "2000",
            "author_id": "X",
        },
        {
            "anthology_id": "vol2",
            "anthology_title": "Big Anth, Vol. 2",
            "series": "BigSeries",
            "edition_number": "1",
            "publication_year": "2000",
            "author_id": "X",
        },
        {
            "anthology_id": "ant2",
            "anthology_title": "Later Anth",
            "publication_year": "2010",
            "author_id": "X",
        },
    ]
    detail, _ = compute(make_df(rows))

    # Only one row for author X (deduped across volumes)
    assert len(detail[detail["author_id"] == "X"]) == 1
    row = detail[detail["author_id"] == "X"].iloc[0]
    assert row["debut_edition_key"] == "BigSeries|1"
    assert row["subsequent_count"] == 1
    assert row["reselection_count"] == 1


# ── Test 4: Tie-breaking ──────────────────────────────────────────────────────


def test_tiebreak_smallest_edition_key_alphabetically():
    """
    Two editions in the same year — author in both gets debut at smaller edition_key.
    """
    rows = [
        {
            "anthology_id": "aaa",
            "anthology_title": "Anth AAA",
            "publication_year": "1990",
            "author_id": "Z",
        },
        {
            "anthology_id": "zzz",
            "anthology_title": "Anth ZZZ",
            "publication_year": "1990",
            "author_id": "Z",
        },
        {
            "anthology_id": "later",
            "anthology_title": "Later Anth",
            "publication_year": "2000",
            "author_id": "Z",
        },
    ]
    detail, _ = compute(make_df(rows))
    row = detail[detail["author_id"] == "Z"].iloc[0]
    # "aaa" < "zzz" alphabetically
    assert row["debut_edition_key"] == "aaa"
    assert row["debut_year"] == 1990
    assert row["subsequent_count"] == 2
    assert row["reselection_count"] == 2


def test_tiebreak_shared_and_unique_authors():
    """
    Authors unique to each of two same-year editions each debut there.
    Author in both gets debut at smaller edition_key.
    """
    rows = [
        {
            "anthology_id": "ant_a",
            "anthology_title": "Anth A",
            "publication_year": "2000",
            "author_id": "shared",
        },
        {
            "anthology_id": "ant_a",
            "anthology_title": "Anth A",
            "publication_year": "2000",
            "author_id": "only_a",
        },
        {
            "anthology_id": "ant_b",
            "anthology_title": "Anth B",
            "publication_year": "2000",
            "author_id": "shared",
        },
        {
            "anthology_id": "ant_b",
            "anthology_title": "Anth B",
            "publication_year": "2000",
            "author_id": "only_b",
        },
        # later edition to provide subsequent opportunities
        {
            "anthology_id": "later",
            "anthology_title": "Later",
            "publication_year": "2010",
            "author_id": "only_a",
        },
    ]
    detail, _ = compute(make_df(rows))
    by_author = detail.set_index("author_id")

    # "ant_a" < "ant_b"
    assert by_author.loc["shared", "debut_edition_key"] == "ant_a"
    assert by_author.loc["only_a", "debut_edition_key"] == "ant_a"
    assert by_author.loc["only_b", "debut_edition_key"] == "ant_b"


# ── Test 5: Most-recent anthology exclusion ───────────────────────────────────


def test_most_recent_has_no_subsequent():
    """Authors debuting in the last anthology have subsequent_count=0 and prob=NaN."""
    rows = [
        {
            "anthology_id": "early",
            "anthology_title": "Early",
            "publication_year": "1950",
            "author_id": "old",
        },
        {
            "anthology_id": "recent",
            "anthology_title": "Recent",
            "publication_year": "2020",
            "author_id": "new",
        },
    ]
    detail, edition = compute(make_df(rows))

    new_row = detail[detail["author_id"] == "new"].iloc[0]
    assert new_row["subsequent_count"] == 0
    assert math.isnan(new_row["reselection_prob"])
    assert math.isnan(new_row["ever_reselected"])

    # Edition aggregate for "recent" should have n_with_subsequent=0
    recent_agg = edition[edition["debut_edition_key"] == "recent"].iloc[0]
    assert recent_agg["n_with_subsequent"] == 0
    assert math.isnan(recent_agg["mean_reselection_prob"])
    assert math.isnan(recent_agg["ever_reselected_rate"])


# ── Test 6: Aggregate correctness ────────────────────────────────────────────


def test_aggregate_correctness():
    """
    ant1 (2000): Authors A, B
    ant2 (2005): Authors A, C
    ant3 (2010): Authors A, B, C, D

    Author debuts:
    - A: ant1, subsequent=[ant2, ant3], reselected both → prob=1.0, ever=1
    - B: ant1, subsequent=[ant2, ant3], reselected only in ant3 → prob=0.5, ever=1
    - C: ant2, subsequent=[ant3], reselected → prob=1.0, ever=1
    - D: ant3, subsequent=[] → NaN

    Aggregate for ant1: n=2, n_with_subsequent=2, mean_prob=0.75, ever_rate=1.0
    Aggregate for ant2: n=1, n_with_subsequent=1, mean_prob=1.0, ever_rate=1.0
    Aggregate for ant3: n=1, n_with_subsequent=0, mean_prob=NaN, ever_rate=NaN
    """
    rows = [
        {
            "anthology_id": "ant1",
            "anthology_title": "Anth 1",
            "publication_year": "2000",
            "author_id": "A",
        },
        {
            "anthology_id": "ant1",
            "anthology_title": "Anth 1",
            "publication_year": "2000",
            "author_id": "B",
        },
        {
            "anthology_id": "ant2",
            "anthology_title": "Anth 2",
            "publication_year": "2005",
            "author_id": "A",
        },
        {
            "anthology_id": "ant2",
            "anthology_title": "Anth 2",
            "publication_year": "2005",
            "author_id": "C",
        },
        {
            "anthology_id": "ant3",
            "anthology_title": "Anth 3",
            "publication_year": "2010",
            "author_id": "A",
        },
        {
            "anthology_id": "ant3",
            "anthology_title": "Anth 3",
            "publication_year": "2010",
            "author_id": "B",
        },
        {
            "anthology_id": "ant3",
            "anthology_title": "Anth 3",
            "publication_year": "2010",
            "author_id": "C",
        },
        {
            "anthology_id": "ant3",
            "anthology_title": "Anth 3",
            "publication_year": "2010",
            "author_id": "D",
        },
    ]
    detail, edition = compute(make_df(rows))

    by_ek = edition.set_index("debut_edition_key")

    ant1 = by_ek.loc["ant1"]
    assert ant1["n_debut_authors"] == 2
    assert ant1["n_with_subsequent"] == 2
    assert ant1["mean_reselection_prob"] == pytest.approx(0.75)
    assert ant1["ever_reselected_rate"] == pytest.approx(1.0)

    ant2 = by_ek.loc["ant2"]
    assert ant2["n_debut_authors"] == 1
    assert ant2["n_with_subsequent"] == 1
    assert ant2["mean_reselection_prob"] == pytest.approx(1.0)
    assert ant2["ever_reselected_rate"] == pytest.approx(1.0)

    ant3 = by_ek.loc["ant3"]
    assert ant3["n_debut_authors"] == 1
    assert ant3["n_with_subsequent"] == 0
    assert math.isnan(ant3["mean_reselection_prob"])
    assert math.isnan(ant3["ever_reselected_rate"])


# ── Test 7: All-anthologies scope (standalone + series) ───────────────────────


def test_standalone_and_series_both_included():
    """Standalone (no series) and series-based anthologies are both counted."""
    rows = [
        # Series-based
        {
            "anthology_id": "s1",
            "anthology_title": "Series Anth",
            "series": "MySeries",
            "edition_number": "1",
            "publication_year": "1960",
            "author_id": "alpha",
        },
        # Standalone
        {
            "anthology_id": "standalone",
            "anthology_title": "Solo Anth",
            "publication_year": "1970",
            "author_id": "alpha",
        },
        {
            "anthology_id": "standalone",
            "anthology_title": "Solo Anth",
            "publication_year": "1970",
            "author_id": "beta",
        },
    ]
    detail, edition = compute(make_df(rows))

    # Both edition keys present
    all_eks = set(edition["debut_edition_key"])
    assert "MySeries|1" in all_eks
    assert "standalone" in all_eks

    # alpha debuts in series edition (earlier year)
    alpha_row = detail[detail["author_id"] == "alpha"].iloc[0]
    assert alpha_row["debut_edition_key"] == "MySeries|1"
    assert alpha_row["subsequent_count"] == 1  # "standalone" is subsequent
    assert alpha_row["reselection_count"] == 1  # alpha is in "standalone"

    # beta debuts in standalone (no prior)
    beta_row = detail[detail["author_id"] == "beta"].iloc[0]
    assert beta_row["debut_edition_key"] == "standalone"
    assert beta_row["subsequent_count"] == 0
