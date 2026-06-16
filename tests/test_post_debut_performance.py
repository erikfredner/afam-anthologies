from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "reselection"))

from post_debut_performance import (  # noqa: E402
    build_edition_table,
    build_entity_pairs,
    compute_entity_stats,
    filter_root_works,
    poisson_binomial_upper_tail,
)


def make_raw(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "edition_id": None,
        "anthology_publication_year": None,
        "work_id": None,
        "work_title": "Work",
        "parent_id": None,
        "parent_work_title": None,
        "author_id": None,
        "author_name": "Author",
        "author_birth_year": None,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def sample_work_rows() -> pd.DataFrame:
    return make_raw(
        [
            {"edition_id": 1, "anthology_publication_year": 2000, "work_id": "w1"},
            {"edition_id": 1, "anthology_publication_year": 2000, "work_id": "w2"},
            {"edition_id": 2, "anthology_publication_year": 2000, "work_id": "w1"},
            {"edition_id": 2, "anthology_publication_year": 2000, "work_id": "w3"},
            {"edition_id": 3, "anthology_publication_year": 2010, "work_id": "w1"},
            {"edition_id": 3, "anthology_publication_year": 2010, "work_id": "w2"},
            {"edition_id": 3, "anthology_publication_year": 2010, "work_id": "w4"},
            {"edition_id": 4, "anthology_publication_year": 2020, "work_id": "w2"},
            {"edition_id": 4, "anthology_publication_year": 2020, "work_id": "w4"},
            {"edition_id": 4, "anthology_publication_year": 2020, "work_id": "w5"},
        ]
    )


def test_poisson_binomial_upper_tail_known_values():
    # P(X >= 2) for probabilities [0.2, 0.5, 0.8]
    # = P(2 successes) + P(3 successes) = 0.42 + 0.08
    assert poisson_binomial_upper_tail(2, [0.2, 0.5, 0.8]) == pytest.approx(0.5)
    assert poisson_binomial_upper_tail(0, [0.2, 0.5]) == pytest.approx(1.0)
    assert poisson_binomial_upper_tail(3, [0.2, 0.5]) == pytest.approx(0.0)


def test_same_year_later_edition_counts_as_post_debut_opportunity():
    raw = sample_work_rows()
    editions = build_edition_table(raw)
    pairs = build_entity_pairs(raw, "work_id", editions)

    stats = pd.DataFrame(compute_entity_stats(pairs, "work_id", editions)).set_index(
        "entity_id"
    )

    assert stats.loc["w1", "opportunities"] == 3
    assert stats.loc["w1", "selections"] == 2
    assert stats.loc["w1", "first_year"] == 2000


def test_expected_count_uses_later_editions_and_leave_one_out_probability():
    raw = sample_work_rows()
    editions = build_edition_table(raw)
    pairs = build_entity_pairs(raw, "work_id", editions)

    stats = pd.DataFrame(compute_entity_stats(pairs, "work_id", editions)).set_index(
        "entity_id"
    )

    # w1's post-debut edition probabilities:
    # ed2: selected; other eligible selected share = 0/1
    # ed3: selected; other eligible selected share = 1/2
    # ed4: not selected; selected eligible share = 2/3
    assert stats.loc["w1", "expected_count"] == pytest.approx(0 + 0.5 + 2 / 3)
    assert stats.loc["w1", "selection_rate"] == pytest.approx(2 / 3)
    assert stats.loc["w1", "obs_over_expected"] == pytest.approx(2 / (7 / 6))
    assert stats.loc["w1", "p_value"] == pytest.approx(1 / 3)


def test_debut_edition_is_excluded_from_opportunities():
    raw = sample_work_rows()
    editions = build_edition_table(raw)
    pairs = build_entity_pairs(raw, "work_id", editions)

    stats = pd.DataFrame(compute_entity_stats(pairs, "work_id", editions)).set_index(
        "entity_id"
    )

    assert stats.loc["w5", "opportunities"] == 0
    assert stats.loc["w5", "selections"] == 0
    assert pd.isna(stats.loc["w5", "selection_rate"])
    assert pd.isna(stats.loc["w5", "p_value"])


def test_root_only_opportunities_include_excerpt_only_editions():
    raw = make_raw(
        [
            {"edition_id": 1, "anthology_publication_year": 2000, "work_id": "w1"},
            {"edition_id": 1, "anthology_publication_year": 2000, "work_id": "w2"},
            {
                "edition_id": 2,
                "anthology_publication_year": 2010,
                "work_id": "excerpt",
                "parent_id": "w1",
            },
            {"edition_id": 3, "anthology_publication_year": 2020, "work_id": "w2"},
        ]
    )

    editions = build_edition_table(raw)
    root_pairs = build_entity_pairs(filter_root_works(raw), "work_id", editions)

    stats = pd.DataFrame(
        compute_entity_stats(root_pairs, "work_id", editions)
    ).set_index("entity_id")

    assert stats.loc["w1", "opportunities"] == 2
    assert stats.loc["w1", "selections"] == 0


def test_author_stats_deduplicate_multiple_works_in_same_edition():
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
                "author_id": "a1",
            },
            {
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "work_id": "w3",
                "author_id": "a2",
            },
            {
                "edition_id": 2,
                "anthology_publication_year": 2010,
                "work_id": "w4",
                "author_id": "a1",
            },
            {
                "edition_id": 2,
                "anthology_publication_year": 2010,
                "work_id": "w5",
                "author_id": "a2",
            },
        ]
    )
    editions = build_edition_table(raw)
    pairs = build_entity_pairs(raw, "author_id", editions)

    stats = pd.DataFrame(compute_entity_stats(pairs, "author_id", editions)).set_index(
        "entity_id"
    )

    assert len(pairs[pairs["author_id"] == "a1"]) == 2
    assert stats.loc["a1", "opportunities"] == 1
    assert stats.loc["a1", "selections"] == 1


def test_filter_root_works_removes_rows_with_parent_id():
    raw = make_raw(
        [
            {"work_id": "root_none", "parent_id": None},
            {"work_id": "root_empty", "parent_id": ""},
            {"work_id": "child", "parent_id": "root_none"},
        ]
    )

    filtered = filter_root_works(raw)

    assert set(filtered["work_id"]) == {"root_none", "root_empty"}
