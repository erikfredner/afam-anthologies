from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "reselection"))

from author_vs_work_debut_reselection import add_entry_group  # noqa: E402
from work_pool_dilution import (  # noqa: E402
    FocalWork,
    build_units,
    compute_focal_record,
    poisson_binomial_tail,
    prepare_appearances,
    summarize_author_portfolio,
    weighted_subset_inclusion_probability,
)


def make_raw(rows: list[dict]) -> pd.DataFrame:
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
    return add_entry_group(pd.DataFrame([{**defaults, **row} for row in rows]))


def test_focal_record_tracks_prior_pool_repeat_slots_and_selection_probability():
    raw = make_raw(
        [
            {
                "work_id": "f",
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "author_id": "a",
            },
            {
                "work_id": "x",
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "author_id": "a",
            },
            {
                "work_id": "y",
                "edition_id": 2,
                "anthology_publication_year": 2010,
                "author_id": "a",
            },
            {
                "work_id": "f",
                "edition_id": 3,
                "anthology_publication_year": 2020,
                "author_id": "a",
            },
            {
                "work_id": "y",
                "edition_id": 3,
                "anthology_publication_year": 2020,
                "author_id": "a",
            },
            {
                "work_id": "x",
                "edition_id": 4,
                "anthology_publication_year": 2030,
                "author_id": "a",
            },
            {
                "work_id": "z",
                "edition_id": 4,
                "anthology_publication_year": 2030,
                "author_id": "a",
            },
        ]
    )
    units = build_units(raw)
    appearances = prepare_appearances(raw, units)
    record = compute_focal_record(
        appearances,
        units,
        FocalWork("Focal", "Author", "f", "a"),
        null_model="uniform",
    )

    assert record["prior_pool"].tolist() == [2, 3, 3]
    assert record["repeat_slots"].tolist() == [0, 2, 1]
    assert record["selected"].tolist() == [False, True, False]
    assert record["p_null"].tolist() == pytest.approx([0.0, 2 / 3, 1 / 3])


def test_poisson_binomial_tail_is_exact_for_small_distribution():
    probs = [0.5, 0.25]

    assert poisson_binomial_tail(probs, 0) == pytest.approx(1.0)
    assert poisson_binomial_tail(probs, 1) == pytest.approx(0.625)
    assert poisson_binomial_tail(probs, 2) == pytest.approx(0.125)
    assert poisson_binomial_tail(probs, 3) == pytest.approx(0.0)


def test_popularity_weighted_null_matches_weighted_subset_probability():
    weights = {"f": 3, "a": 1, "b": 1}

    assert weighted_subset_inclusion_probability(weights, "f", 1) == pytest.approx(
        3 / 5
    )
    assert weighted_subset_inclusion_probability(weights, "f", 2) == pytest.approx(
        6 / 7
    )
    assert weighted_subset_inclusion_probability(weights, "f", 3) == pytest.approx(1.0)


def test_popularity_weighted_null_reduces_to_uniform_when_weights_equal():
    weights = {"f": 1, "a": 1, "b": 1, "c": 1}

    assert weighted_subset_inclusion_probability(weights, "f", 2) == pytest.approx(0.5)


def test_summarize_author_portfolio_counts_frequency_and_lift_candidates():
    portfolio = pd.DataFrame(
        [
            {
                "author_name": "A",
                "total_selections": 12,
                "post_debut_opportunities": 20,
                "expected": 2.0,
                "lift": 5.5,
            },
            {
                "author_name": "A",
                "total_selections": 5,
                "post_debut_opportunities": 20,
                "expected": 2.0,
                "lift": 2.5,
            },
            {
                "author_name": "B",
                "total_selections": 13,
                "post_debut_opportunities": 9,
                "expected": 1.0,
                "lift": 8.0,
            },
            {
                "author_name": "B",
                "total_selections": 3,
                "post_debut_opportunities": 12,
                "expected": 1.0,
                "lift": 4.1,
            },
        ]
    )

    summary = summarize_author_portfolio(portfolio).set_index("author")

    assert summary.loc["A", "work_pool"] == 2
    assert summary.loc["A", "works_ge_10_selections"] == 1
    assert summary.loc["A", "works_ge_4x_lift_opp_ge_10"] == 1
    assert summary.loc["B", "works_ge_13_selections"] == 1
    assert summary.loc["B", "works_ge_4x_lift_opp_ge_10"] == 1
