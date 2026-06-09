from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "reselection"))

from author_vs_work_debut_reselection import (  # noqa: E402
    add_entry_group,
    build_debut_work_author_outcomes,
    build_paired_summary,
    build_edition_table,
    build_summary,
    compute_author_records,
    compute_work_records,
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


def test_same_year_later_edition_counts_as_subsequent_reselection():
    raw, editions = make_raw(
        [
            {"work_id": "w1", "edition_id": 1, "anthology_publication_year": 2000, "author_id": "a1"},
            {"work_id": "w1", "edition_id": 2, "anthology_publication_year": 2000, "author_id": "a1"},
            {"work_id": "w2", "edition_id": 3, "anthology_publication_year": 2010, "author_id": "a2"},
        ]
    )

    works = compute_work_records(raw, editions).set_index("work_id")

    assert works.loc["w1", "debut_edition_id"] == 1
    assert works.loc["w1", "subsequent_count"] == 2
    assert works.loc["w1", "reselection_count"] == 1
    assert works.loc["w1", "reselection_prob"] == pytest.approx(0.5)
    assert works.loc["w1", "years_to_first_reselection"] == 0


def test_cross_series_uses_debut_edition_group_not_arbitrary_debut_year_group():
    raw, editions = make_raw(
        [
            {
                "work_id": "w1",
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "series_id": 10,
                "author_id": "a1",
            },
            {
                "work_id": "w1",
                "edition_id": 2,
                "anthology_publication_year": 2000,
                "series_id": 20,
                "author_id": "a1",
            },
            {
                "work_id": "w1",
                "edition_id": 3,
                "anthology_publication_year": 2010,
                "series_id": 10,
                "author_id": "a1",
            },
        ]
    )

    works = compute_work_records(raw, editions).set_index("work_id")

    assert works.loc["w1", "debut_group"] == "series:10"
    assert works.loc["w1", "subsequent_count"] == 2
    assert works.loc["w1", "reselection_count"] == 2
    assert works.loc["w1", "cross_series_subsequent_count"] == 1
    assert works.loc["w1", "cross_series_reselection_count"] == 1
    assert works.loc["w1", "is_reselected_cross_series"]


def test_summary_reports_ever_and_opportunity_estimands():
    raw, editions = make_raw(
        [
            {"work_id": "w1", "edition_id": 1, "anthology_publication_year": 2000, "author_id": "a1"},
            {"work_id": "w2", "edition_id": 1, "anthology_publication_year": 2000, "author_id": "a2"},
            {"work_id": "w3", "edition_id": 2, "anthology_publication_year": 2010, "author_id": "a1"},
            {"work_id": "w4", "edition_id": 3, "anthology_publication_year": 2020, "author_id": "a3"},
        ]
    )
    works = compute_work_records(raw, editions)
    authors = compute_author_records(raw, editions)

    summary = build_summary(works, authors).set_index("metric")

    assert set(summary.index) == {
        "ever_all",
        "ever_cross_series",
        "opportunity_all",
        "opportunity_cross_series",
    }
    assert summary.loc["ever_all", "author_k"] == 1
    assert summary.loc["ever_all", "author_n"] == 2
    assert summary.loc["ever_all", "work_k"] == 0
    assert summary.loc["ever_all", "work_n"] == 3
    assert summary.loc["opportunity_all", "author_rate"] == pytest.approx(1 / 4)
    assert summary.loc["opportunity_all", "work_rate"] == pytest.approx(0.0)


def test_debut_work_author_outcomes_detect_author_only_return():
    raw, editions = make_raw(
        [
            {"work_id": "w1", "edition_id": 1, "anthology_publication_year": 2000, "author_id": "a1"},
            {"work_id": "w2", "edition_id": 2, "anthology_publication_year": 2010, "author_id": "a1"},
            {"work_id": "w3", "edition_id": 2, "anthology_publication_year": 2010, "author_id": "a2"},
        ]
    )
    works = compute_work_records(raw, editions)
    authors = compute_author_records(raw, editions)

    outcomes = build_debut_work_author_outcomes(raw, works, authors).set_index("work_id")

    assert not bool(outcomes.loc["w1", "work_reselected"])
    assert bool(outcomes.loc["w1", "any_author_reselected"])
    assert outcomes.loc["w1", "outcome"] == "author_only"


def test_work_debut_pair_uses_work_debut_not_author_debut():
    raw, editions = make_raw(
        [
            {"work_id": "old", "edition_id": 1, "anthology_publication_year": 1990, "author_id": "a1"},
            {"work_id": "new", "edition_id": 2, "anthology_publication_year": 2000, "author_id": "a1"},
            {"work_id": "other", "edition_id": 3, "anthology_publication_year": 2010, "author_id": "a2"},
        ]
    )
    works = compute_work_records(raw, editions)
    authors = compute_author_records(raw, editions)

    outcomes = build_debut_work_author_outcomes(raw, works, authors).set_index("work_id")

    assert "new" in outcomes.index
    assert not bool(outcomes.loc["new", "work_reselected"])
    assert not bool(outcomes.loc["new", "any_author_reselected"])
    assert outcomes.loc["new", "outcome"] == "neither"


def test_paired_summary_counts_discordant_outcomes_for_mcnemar():
    outcomes = pd.DataFrame(
        [
            {
                "work_reselected": False,
                "any_author_reselected": True,
                "work_reselected_cross_series": False,
                "any_author_reselected_cross_series": True,
            },
            {
                "work_reselected": True,
                "any_author_reselected": False,
                "work_reselected_cross_series": False,
                "any_author_reselected_cross_series": False,
            },
            {
                "work_reselected": True,
                "any_author_reselected": True,
                "work_reselected_cross_series": True,
                "any_author_reselected_cross_series": True,
            },
            {
                "work_reselected": False,
                "any_author_reselected": False,
                "work_reselected_cross_series": False,
                "any_author_reselected_cross_series": False,
            },
        ]
    )

    summary = build_paired_summary(outcomes).set_index("metric")

    assert summary.loc["paired_all", "paired_author_and_work"] == 1
    assert summary.loc["paired_all", "paired_author_only"] == 1
    assert summary.loc["paired_all", "paired_work_only"] == 1
    assert summary.loc["paired_all", "paired_neither"] == 1
    assert 0.0 <= summary.loc["paired_all", "mcnemar_p"] <= 1.0
