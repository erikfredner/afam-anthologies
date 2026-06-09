from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "heatmaps"))

from author_work_shared_scatter import (  # noqa: E402
    build_pair_records,
    permutation_p_value,
    summarize_outcomes,
    summarize_proximity_bands,
)


def fixture_data():
    keys = ["1|1", "1|2", "100", "200"]
    year_map = {
        "1|1": 1900,
        "1|2": 1905,
        "100": 1903,
        "200": 1930,
    }
    author_sets = {
        "1|1": {"a"},
        "1|2": {"a", "b"},
        "100": {"a"},
        "200": set(),
    }
    work_sets = {
        "1|1": {"w1"},
        "1|2": {"w2"},
        "100": {"w3"},
        "200": set(),
    }
    return keys, work_sets, author_sets, year_map


def test_zero_overlap_cross_series_pairs_are_in_denominator():
    keys, work_sets, author_sets, year_map = fixture_data()

    pairs = build_pair_records(keys, work_sets, author_sets, year_map)
    summary = summarize_outcomes(pairs)

    assert summary.n == 5
    assert summary.authors_gt_works == 2
    assert summary.works_gt_authors == 0
    assert summary.equal == 3
    assert math.isclose(summary.author_gt_rate, 2 / 5)


def test_within_series_pairs_excluded_by_default_and_includable():
    keys, work_sets, author_sets, year_map = fixture_data()

    default_pairs = build_pair_records(keys, work_sets, author_sets, year_map)
    all_pairs = build_pair_records(
        keys, work_sets, author_sets, year_map, cross_series_only=False
    )

    assert {("1|1", "1|2")} == {
        (p.key_i, p.key_j) for p in all_pairs
    } - {(p.key_i, p.key_j) for p in default_pairs}
    assert len(default_pairs) == 5
    assert len(all_pairs) == 6


def test_two_standalones_are_retained_as_cross_series_pair():
    keys, work_sets, author_sets, year_map = fixture_data()

    pairs = build_pair_records(keys, work_sets, author_sets, year_map)

    assert any((p.key_i, p.key_j) == ("100", "200") for p in pairs)


def test_proximity_band_splits_by_absolute_publication_year_gap():
    keys, work_sets, author_sets, year_map = fixture_data()
    pairs = build_pair_records(keys, work_sets, author_sets, year_map)

    summaries = summarize_proximity_bands(
        pairs, year_map, bands=[5], permutations=25, seed=42
    )
    row = summaries[0]

    assert row.band == 5
    assert row.close_n == 2
    assert row.close_k == 2
    assert row.far_n == 3
    assert row.far_k == 0
    assert math.isclose(row.close_rate, 1.0)
    assert math.isclose(row.far_rate, 0.0)
    assert math.isclose(row.rate_diff, 1.0)
    assert 0.0 <= row.z_p <= 1.0
    assert 0.0 <= row.permutation_p <= 1.0


def test_permutation_p_value_is_seed_stable():
    keys, work_sets, author_sets, year_map = fixture_data()
    pairs = build_pair_records(keys, work_sets, author_sets, year_map)
    reversed_year_map = dict(reversed(list(year_map.items())))

    p1 = permutation_p_value(pairs, year_map, band=5, permutations=50, seed=7)
    p2 = permutation_p_value(pairs, year_map, band=5, permutations=50, seed=7)
    p3 = permutation_p_value(pairs, reversed_year_map, band=5, permutations=50, seed=7)

    assert p1 == p2
    assert p1 == p3
    assert 0.0 <= p1 <= 1.0
