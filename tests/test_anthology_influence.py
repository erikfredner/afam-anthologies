from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "influence"))

from anthology_influence import (  # noqa: E402
    compute_all_variants,
    compute_debut_sets,
    compute_expected_rates,
    compute_influence_table,
    compute_prior_counts,
    compute_stratified_influence_table,
    compute_stratum_rates,
    edition_entry_groups,
    edition_years,
    sorted_edition_ids,
)


def make_raw(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "edition_id": None,
        "anthology_publication_year": None,
        "work_id": None,
        "author_id": None,
        "series_id": None,
        "parent_id": None,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_compute_debut_sets_first_edition_owns_everything_and_ties_go_to_lower_id():
    ed_items = {
        1: {"w1", "w2"},
        2: {"w2", "w3"},  # same year as 3, lower id
        3: {"w3", "w4"},
    }
    ed_year = {1: 2000, 2: 2010, 3: 2010}
    raw = make_raw(
        [
            {
                "edition_id": eid,
                "anthology_publication_year": ed_year[eid],
                "work_id": w,
            }
            for eid, items in ed_items.items()
            for w in items
        ]
    )
    sorted_eds = sorted_edition_ids(raw)
    assert sorted_eds == [1, 2, 3]

    debuts = compute_debut_sets(ed_items, sorted_eds)
    assert debuts[1] == {"w1", "w2"}
    assert debuts[2] == {"w3"}  # same-year co-debut attributed to lower edition_id
    assert debuts[3] == {"w4"}


def test_compute_expected_rates_uses_strictly_earlier_years():
    ed_items = {
        1: {"w1", "w2"},
        2: {"w2", "w3"},
        3: {"w1", "w4"},  # same year as 2: must not see 2's items in its pool
        4: {"w3", "w5"},
    }
    ed_year = {1: 2000, 2: 2010, 3: 2010, 4: 2020}
    sorted_eds = [1, 2, 3, 4]

    expected = compute_expected_rates(ed_items, ed_year, sorted_eds)

    assert expected[1] == (0, 0)  # first edition: empty pool
    assert expected[2] == (1, 2)  # pool = {w1, w2}; B ∩ pool = {w2}
    assert expected[3] == (1, 2)  # same pool as 2; B ∩ pool = {w1}
    assert expected[4] == (1, 4)  # pool = {w1..w4}; B ∩ pool = {w3}


def test_compute_influence_table_hand_example():
    ed_items = {
        1: {"w1", "w2"},
        2: {"w1", "w3"},
        3: {"w2", "w3"},
    }
    ed_year = {1: 2000, 2: 2010, 3: 2020}
    sorted_eds = [1, 2, 3]
    expected = compute_expected_rates(ed_items, ed_year, sorted_eds)

    table = compute_influence_table(
        ed_items, ed_items, ed_year, sorted_eds, expected
    ).set_index("edition_id")

    # Edition 1: S_A = {w1, w2}; B=2 picks w1, B=3 picks w2 → 2/4 observed.
    row = table.loc[1]
    assert row["n_subsequent"] == 2
    assert row["trials"] == 4
    assert row["successes"] == 2
    assert row["obs_rate"] == pytest.approx(0.5)
    # p_2 = |{w1}|/|{w1,w2}| = 0.5; p_3 = |{w2,w3}|/|{w1,w2,w3}| = 2/3
    assert row["exp_rate"] == pytest.approx((2 * 0.5 + 2 * 2 / 3) / 4)
    assert row["lift"] == pytest.approx(0.5 / ((2 * 0.5 + 2 * 2 / 3) / 4))

    # Last edition: no subsequent editions → NaN stats.
    last = table.loc[3]
    assert last["n_subsequent"] == 0
    assert last["trials"] == 0
    assert math.isnan(last["lift"])
    assert math.isnan(last["p_value"])


def test_same_year_edition_not_counted_as_subsequent():
    ed_items = {1: {"w1"}, 2: {"w1"}, 3: {"w1"}}
    ed_year = {1: 2000, 2: 2010, 3: 2010}
    sorted_eds = [1, 2, 3]
    expected = compute_expected_rates(ed_items, ed_year, sorted_eds)

    table = compute_influence_table(
        ed_items, ed_items, ed_year, sorted_eds, expected
    ).set_index("edition_id")

    assert table.loc[1, "n_subsequent"] == 2
    assert table.loc[2, "n_subsequent"] == 0
    assert table.loc[3, "n_subsequent"] == 0


def test_exclude_within_series_skips_pairs_and_shrinks_pools():
    ed_items = {
        1: {"w1", "w2"},  # series 9
        2: {"w1", "w3"},  # series 9 — reprints w1 from its own series
        3: {"w4"},  # standalone
    }
    ed_year = {1: 2000, 2: 2010, 3: 2020}
    sorted_eds = [1, 2, 3]
    entry_group = {1: "series:9", 2: "series:9", 3: "edition:3"}

    expected = compute_expected_rates(
        ed_items,
        ed_year,
        sorted_eds,
        entry_group=entry_group,
        exclude_within_series=True,
    )
    # Edition 2's pool excludes its own series → empty pool.
    assert expected[2] == (0, 0)
    # Edition 3's pool keeps both series-9 editions.
    assert expected[3] == (0, 3)

    table = compute_influence_table(
        ed_items,
        ed_items,
        ed_year,
        sorted_eds,
        expected,
        entry_group=entry_group,
        exclude_within_series=True,
    ).set_index("edition_id")

    # Edition 1's only counted target is edition 3 (edition 2 shares its series).
    row = table.loc[1]
    assert row["n_subsequent"] == 1
    assert row["trials"] == 2
    assert row["successes"] == 0


def test_p_value_matches_direct_binomtest():
    ed_items = {
        1: {"w1", "w2"},
        2: {"w1", "w3"},
        3: {"w2", "w3"},
    }
    ed_year = {1: 2000, 2: 2010, 3: 2020}
    sorted_eds = [1, 2, 3]
    expected = compute_expected_rates(ed_items, ed_year, sorted_eds)

    table = compute_influence_table(
        ed_items, ed_items, ed_year, sorted_eds, expected
    ).set_index("edition_id")

    row = table.loc[1]
    direct = binomtest(
        int(row["successes"]),
        int(row["trials"]),
        p=float(row["exp_rate"]),
        alternative="two-sided",
    ).pvalue
    assert row["p_value"] == pytest.approx(direct)


def test_compute_all_variants_blocks_and_first_edition_all_equals_debut():
    raw = make_raw(
        [
            {
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "work_id": "w1",
                "author_id": "a1",
            },
            {
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "work_id": "w2",
                "author_id": "a2",
            },
            {
                "edition_id": 2,
                "anthology_publication_year": 2010,
                "work_id": "w1",
                "author_id": "a1",
            },
            {
                "edition_id": 2,
                "anthology_publication_year": 2010,
                "work_id": "w3",
                "author_id": "a3",
            },
            {
                "edition_id": 3,
                "anthology_publication_year": 2020,
                "work_id": "w2",
                "author_id": "a2",
            },
        ]
    )

    results = compute_all_variants(raw)

    assert set(results["entity"]) == {"works", "authors"}
    assert set(results["variant"]) == {"all", "debut"}
    assert len(results) == 4 * 3  # 4 blocks × 3 editions

    cols = ["n_items", "trials", "successes", "obs_rate", "exp_rate", "lift"]
    for entity in ["works", "authors"]:
        block = results[(results["entity"] == entity) & (results["edition_id"] == 1)]
        all_row = block[block["variant"] == "all"][cols].reset_index(drop=True)
        deb_row = block[block["variant"] == "debut"][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(all_row, deb_row)


def test_edition_helpers():
    raw = make_raw(
        [
            {
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "work_id": "w1",
                "author_id": "a1",
                "series_id": 3,
            },
            {"edition_id": 2, "anthology_publication_year": 2010, "work_id": "w2"},
        ]
    )
    assert edition_years(raw) == {1: 2000, 2: 2010}
    assert edition_entry_groups(raw) == {1: "series:3", 2: "edition:2"}


def test_compute_prior_counts_and_stratum_rates():
    ed_items = {
        1: {"w1", "w2"},
        2: {"w1", "w3"},
        3: {"w1", "w4"},
    }
    ed_year = {1: 2000, 2: 2010, 3: 2020}
    sorted_eds = [1, 2, 3]

    prior_counts = compute_prior_counts(ed_items, ed_year, sorted_eds)
    assert prior_counts[1] == {}
    assert prior_counts[2] == {"w1": 1, "w2": 1}
    assert prior_counts[3] == {"w1": 2, "w2": 1, "w3": 1}

    rates = compute_stratum_rates(prior_counts[3], ed_items[3])
    # Stratum c=2: {w1}, picked. Stratum c=1: {w2, w3}, neither picked.
    assert rates == {2: (1, 1), 1: (0, 2)}


def test_stratified_table_neutralizes_popularity_skew():
    # A's items are w1 (picked by everyone) and w2 (never repicked). Under
    # stratification, w1 is judged against the high-prior-count stratum it
    # occupies, so A's lift collapses to 1; the uniform baseline gives >1.
    ed_items = {
        1: {"w1", "w2"},
        2: {"w1", "w3"},
        3: {"w1", "w4"},
    }
    ed_year = {1: 2000, 2: 2010, 3: 2020}
    sorted_eds = [1, 2, 3]
    prior_counts = compute_prior_counts(ed_items, ed_year, sorted_eds)

    strat = compute_stratified_influence_table(
        ed_items, ed_items, ed_year, sorted_eds, prior_counts
    ).set_index("edition_id")

    row = strat.loc[1]
    # B=2: stratum c=1 = {w1,w2}, rate 1/2 → expected 0.5 per item.
    # B=3: w1 in c=2 (rate 1), w2 in c=1 (rate 0).
    assert row["trials"] == 4
    assert row["successes"] == 2
    assert row["exp_rate"] == pytest.approx(2.0 / 4)
    assert row["lift"] == pytest.approx(1.0)

    uniform = compute_influence_table(
        ed_items,
        ed_items,
        ed_year,
        sorted_eds,
        compute_expected_rates(ed_items, ed_year, sorted_eds),
    ).set_index("edition_id")
    assert uniform.loc[1, "lift"] > strat.loc[1, "lift"]
    # Observed counts are identical across modes; only the baseline moves.
    assert uniform.loc[1, "successes"] == strat.loc[1, "successes"]
    assert uniform.loc[1, "trials"] == strat.loc[1, "trials"]


def test_stratified_table_respects_exclude_within_series():
    ed_items = {
        1: {"w1", "w2"},  # series 9
        2: {"w1", "w3"},  # series 9
        3: {"w1", "w4"},  # standalone
    }
    ed_year = {1: 2000, 2: 2010, 3: 2020}
    sorted_eds = [1, 2, 3]
    entry_group = {1: "series:9", 2: "series:9", 3: "edition:3"}

    prior_counts = compute_prior_counts(
        ed_items,
        ed_year,
        sorted_eds,
        entry_group=entry_group,
        exclude_within_series=True,
    )
    # Edition 2's pool excludes its own series entirely.
    assert prior_counts[2] == {}
    # Edition 3 keeps both series-9 editions in its prior counts.
    assert prior_counts[3] == {"w1": 2, "w2": 1, "w3": 1}

    table = compute_stratified_influence_table(
        ed_items,
        ed_items,
        ed_year,
        sorted_eds,
        prior_counts,
        entry_group=entry_group,
        exclude_within_series=True,
    ).set_index("edition_id")
    # Edition 1's only counted target is edition 3 (edition 2 shares its series).
    assert table.loc[1, "n_subsequent"] == 1
    assert table.loc[1, "trials"] == 2


def test_compute_all_variants_stratified_mode():
    raw = make_raw(
        [
            {"edition_id": 1, "anthology_publication_year": 2000, "work_id": "w1"},
            {"edition_id": 1, "anthology_publication_year": 2000, "work_id": "w2"},
            {"edition_id": 2, "anthology_publication_year": 2010, "work_id": "w1"},
            {"edition_id": 3, "anthology_publication_year": 2020, "work_id": "w1"},
        ]
    )
    results = compute_all_variants(raw, stratify_prior_count=True)
    works_all = results[
        (results["entity"] == "works") & (results["variant"] == "all")
    ].set_index("edition_id")
    assert works_all.loc[1, "lift"] == pytest.approx(1.0)
