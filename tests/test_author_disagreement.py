from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "overlap"))

from author_disagreement import (  # noqa: E402
    cohen_kappa,
    compute_author_inclusion,
    compute_pairwise,
    compute_single_series_share,
    edition_level_sign_test,
    prepare_long,
    summarize_pairs,
)


def make_raw(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with the column schema prepare_long() expects."""
    defaults = {"series_id": None, "author_name": "Anon"}
    df = pd.DataFrame(rows)
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


def make_long(rows: list[dict]) -> pd.DataFrame:
    """Build a prepared long frame (one row per edition-author)."""
    df = pd.DataFrame(rows)
    if "author_name" not in df.columns:
        df["author_name"] = "Anon"
    return df


# ── prepare_long ──────────────────────────────────────────────────────────────


def test_prepare_long_dedupes_and_builds_series_identity():
    raw = make_raw(
        [
            # Author 1 has two works in edition 16 → one long row
            {
                "author_id": 1,
                "edition_id": 16,
                "series_id": 3,
                "anthology_publication_year": 1997,
            },
            {
                "author_id": 1,
                "edition_id": 16,
                "series_id": 3,
                "anthology_publication_year": 1997,
            },
            # Standalone edition (no series)
            {
                "author_id": 1,
                "edition_id": 60,
                "series_id": None,
                "anthology_publication_year": 1998,
            },
            # Null author dropped
            {
                "author_id": None,
                "edition_id": 16,
                "series_id": 3,
                "anthology_publication_year": 1997,
            },
        ]
    )
    long = prepare_long(raw)
    assert len(long) == 2
    by_edition = long.set_index("edition_id")
    assert by_edition.loc[16, "series_identity"] == "series_3"
    assert by_edition.loc[60, "series_identity"] == "standalone_60"
    assert by_edition.loc[60, "year"] == 1998


# ── cohen_kappa ───────────────────────────────────────────────────────────────


def test_cohen_kappa_chance_level():
    # A={1,2}, B={2,3}, pool={1..4}: p_o=0.5, p_e=0.5 → kappa=0
    kappa, p_o, p_e = cohen_kappa(
        frozenset({1, 2}), frozenset({2, 3}), frozenset({1, 2, 3, 4})
    )
    assert p_o == 0.5
    assert p_e == 0.5
    assert kappa == 0.0


def test_cohen_kappa_perfect_agreement():
    kappa, p_o, _ = cohen_kappa(
        frozenset({1, 2}), frozenset({1, 2}), frozenset({1, 2, 3, 4})
    )
    assert p_o == 1.0
    assert kappa == 1.0


def test_cohen_kappa_empty_pool_and_no_variation():
    kappa, _, _ = cohen_kappa(frozenset({1}), frozenset({1}), frozenset())
    assert math.isnan(kappa)
    # Both raters say yes to everyone: p_e == 1 → nan
    kappa, _, p_e = cohen_kappa(frozenset({1, 2}), frozenset({1, 2}), frozenset({1, 2}))
    assert p_e == 1.0
    assert math.isnan(kappa)


# ── compute_pairwise ──────────────────────────────────────────────────────────


@pytest.fixture
def three_edition_long() -> pd.DataFrame:
    """Editions 16/17 share series_3; 60 is a standalone.

    Rosters: 16 (1997) = {1,2,3}; 60 (1998) = {2,3,4,5}; 17 (2004) = {1,2}.
    """
    rows = []
    for eid, series, year, authors in [
        (16, "series_3", 1997, [1, 2, 3]),
        (60, "standalone_60", 1998, [2, 3, 4, 5]),
        (17, "series_3", 2004, [1, 2]),
    ]:
        for aid in authors:
            rows.append(
                {
                    "edition_id": eid,
                    "series_identity": series,
                    "year": year,
                    "author_id": aid,
                }
            )
    return make_long(rows)


def test_compute_pairwise_cross_series_flags_and_metrics(three_edition_long):
    pairs = compute_pairwise(three_edition_long, min_pool=1)
    assert len(pairs) == 3
    indexed = pairs.set_index(["edition_a", "edition_b"])

    within = indexed.loc[(16, 17)]
    assert not within["cross_series"]

    p_16_60 = indexed.loc[(16, 60)]
    assert p_16_60["cross_series"]
    assert p_16_60["n_intersection"] == 2
    assert p_16_60["jaccard"] == pytest.approx(2 / 5)
    assert p_16_60["overlap_coef"] == pytest.approx(2 / 3)

    p_60_17 = indexed.loc[(60, 17)]
    assert p_60_17["cross_series"]
    assert p_60_17["jaccard"] == pytest.approx(1 / 5)
    assert p_60_17["overlap_coef"] == pytest.approx(1 / 2)


def test_compute_pairwise_kappa_pool_uses_earlier_year(three_edition_long):
    pairs = compute_pairwise(three_edition_long, min_pool=1)
    indexed = pairs.set_index(["edition_a", "edition_b"])
    # Pair (16, 60): earlier year 1997 → pool = debuts <= 1997 = {1,2,3}
    assert indexed.loc[(16, 60), "pool_size"] == 3
    # Pair (60, 17): earlier year 1998 → pool = {1,2,3,4,5}
    assert indexed.loc[(60, 17), "pool_size"] == 5


def test_compute_pairwise_min_pool_guard(three_edition_long):
    pairs = compute_pairwise(three_edition_long, min_pool=10)
    assert pairs["kappa"].isna().all()
    assert pairs["jaccard"].notna().all()


# ── compute_author_inclusion ──────────────────────────────────────────────────


@pytest.fixture
def inclusion_long() -> pd.DataFrame:
    """Author 1 in NAAAL eds 16 (1997) + 17 (2004); standalone 60 (2000)
    excludes author 1. Author 2 debuts in the final edition only."""
    rows = [
        {"edition_id": 16, "series_identity": "series_3", "year": 1997, "author_id": 1},
        {"edition_id": 17, "series_identity": "series_3", "year": 2004, "author_id": 1},
        {
            "edition_id": 60,
            "series_identity": "standalone_60",
            "year": 2000,
            "author_id": 9,
        },
        {"edition_id": 17, "series_identity": "series_3", "year": 2004, "author_id": 2},
    ]
    return make_long(rows)


def test_inclusion_series_level_collapses_editions(inclusion_long):
    table = compute_author_inclusion(inclusion_long, level="series").set_index(
        "author_id"
    )
    # Author 1: eligible units = series_3 (max 2004) + standalone_60 (max 2000)
    # both >= debut 1997; included in series_3 only → share 1/2
    assert table.loc[1, "n_eligible"] == 2
    assert table.loc[1, "n_included"] == 1
    assert table.loc[1, "inclusion_share"] == pytest.approx(0.5)
    assert not table.loc[1, "majority_included"]


def test_inclusion_edition_level(inclusion_long):
    table = compute_author_inclusion(inclusion_long, level="edition").set_index(
        "author_id"
    )
    # Author 1: editions with year >= 1997 = all three; included in 2 → 2/3
    assert table.loc[1, "n_eligible"] == 3
    assert table.loc[1, "n_included"] == 2
    assert table.loc[1, "inclusion_share"] == pytest.approx(2 / 3)


def test_inclusion_final_edition_debut_is_trivially_perfect(inclusion_long):
    table = compute_author_inclusion(inclusion_long, level="edition").set_index(
        "author_id"
    )
    # Author 2 debuts in 2004, the latest edition → 1 eligible, 1 included
    assert table.loc[2, "n_eligible"] == 1
    assert table.loc[2, "inclusion_share"] == 1.0
    # The headline filter (n_eligible >= 2) would exclude this author
    filt = table[table["n_eligible"] >= 2]
    assert 2 not in filt.index


def test_inclusion_same_year_edition_counts_as_eligible():
    rows = [
        {
            "edition_id": 1,
            "series_identity": "standalone_1",
            "year": 2000,
            "author_id": 1,
        },
        {
            "edition_id": 2,
            "series_identity": "standalone_2",
            "year": 2000,
            "author_id": 9,
        },
    ]
    table = compute_author_inclusion(make_long(rows), level="edition").set_index(
        "author_id"
    )
    assert table.loc[1, "n_eligible"] == 2


def test_inclusion_rejects_unknown_level(inclusion_long):
    with pytest.raises(ValueError):
        compute_author_inclusion(inclusion_long, level="volume")


# ── compute_single_series_share ───────────────────────────────────────────────


def test_single_series_share():
    rows = [
        # Author 1: two editions of one series → single-unit
        {"edition_id": 16, "series_identity": "series_3", "year": 1997, "author_id": 1},
        {"edition_id": 17, "series_identity": "series_3", "year": 2004, "author_id": 1},
        # Author 2: two different units → not single-unit
        {"edition_id": 16, "series_identity": "series_3", "year": 1997, "author_id": 2},
        {
            "edition_id": 60,
            "series_identity": "standalone_60",
            "year": 1998,
            "author_id": 2,
        },
    ]
    result = compute_single_series_share(make_long(rows))
    assert result["n_authors"] == 2
    assert result["n_single_series"] == 1
    assert result["share"] == pytest.approx(0.5)


# ── summarize_pairs / edition_level_sign_test ─────────────────────────────────


def test_summarize_pairs_known_vector():
    s = summarize_pairs(pd.Series([0.2, 0.4, 0.6, math.nan]))
    assert s["n"] == 3
    assert s["median"] == pytest.approx(0.4)
    assert s["n_below"] == 2
    assert s["pct_below"] == pytest.approx(2 / 3)
    # binomtest(2, 3, 0.5, 'greater') = P(X >= 2) = 0.5
    assert s["sign_p"] == pytest.approx(0.5)


def test_edition_level_sign_test_uses_per_edition_medians():
    pairs = pd.DataFrame(
        [
            # Edition 1 pairs: medians 0.2; edition 2: median of {0.2, 0.8} = 0.5
            {"edition_a": 1, "edition_b": 2, "jaccard": 0.2, "cross_series": True},
            {"edition_a": 2, "edition_b": 3, "jaccard": 0.8, "cross_series": True},
            # Within-series pair must be ignored
            {"edition_a": 1, "edition_b": 3, "jaccard": 0.9, "cross_series": False},
        ]
    )
    result = edition_level_sign_test(pairs, "jaccard")
    # Medians: ed1 → 0.2 (below), ed2 → 0.5 (not below), ed3 → 0.8 (not below)
    assert result["n_editions"] == 3
    assert result["n_below"] == 1
