from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "reselection"))

from frequent_author_work_selection_scatter import (  # noqa: E402
    HOVER_TEMPLATE,
    add_hover_columns,
    build_author_work_rows,
    build_edition_table,
    build_entity_pairs,
    build_plot_frames,
    compute_year_based_records,
    filter_root_works,
    qualifying_author_ids,
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


def test_qualifying_author_ids_uses_half_or_more_editions():
    raw = make_raw(
        [
            {"edition_id": 1, "anthology_publication_year": 1900, "author_id": "a1", "work_id": "w1"},
            {"edition_id": 2, "anthology_publication_year": 1910, "author_id": "a1", "work_id": "w2"},
            {"edition_id": 3, "anthology_publication_year": 1920, "author_id": "a1", "work_id": "w3"},
            {"edition_id": 1, "anthology_publication_year": 1900, "author_id": "a2", "work_id": "w4"},
            {"edition_id": 2, "anthology_publication_year": 1910, "author_id": "a2", "work_id": "w5"},
            {"edition_id": 4, "anthology_publication_year": 1930, "work_id": "w0"},
            {"edition_id": 5, "anthology_publication_year": 1940, "author_id": "a3", "work_id": "w6"},
        ]
    )
    editions = build_edition_table(raw)
    author_pairs = build_entity_pairs(raw, "author_id", editions)

    assert len(editions) == 5
    assert qualifying_author_ids(author_pairs, len(editions)) == {"a1"}


def test_year_based_records_exclude_same_year_after_debut():
    raw = make_raw(
        [
            {"edition_id": 1, "anthology_publication_year": 2000, "work_id": "w1"},
            {"edition_id": 2, "anthology_publication_year": 2000, "work_id": "w1"},
            {"edition_id": 3, "anthology_publication_year": 2010, "work_id": "w1"},
            {"edition_id": 4, "anthology_publication_year": 2020, "work_id": "w2"},
        ]
    )
    editions = build_edition_table(raw)
    pairs = build_entity_pairs(raw, "work_id", editions)

    records = compute_year_based_records(pairs, "work_id", editions).set_index("work_id")

    assert records.loc["w1", "debut_year"] == 2000
    assert records.loc["w1", "opportunities"] == 2
    assert records.loc["w1", "selections"] == 1
    assert records.loc["w1", "selection_rate"] == pytest.approx(0.5)


def test_author_records_deduplicate_multiple_works_in_same_edition():
    raw = make_raw(
        [
            {"edition_id": 1, "anthology_publication_year": 2000, "author_id": "a1", "work_id": "w1"},
            {"edition_id": 1, "anthology_publication_year": 2000, "author_id": "a1", "work_id": "w2"},
            {"edition_id": 2, "anthology_publication_year": 2010, "author_id": "a1", "work_id": "w3"},
        ]
    )
    editions = build_edition_table(raw)
    pairs = build_entity_pairs(raw, "author_id", editions)
    records = compute_year_based_records(pairs, "author_id", editions).set_index("author_id")

    assert len(pairs[pairs["author_id"] == "a1"]) == 2
    assert records.loc["a1", "total_selection_count"] == 2
    assert records.loc["a1", "opportunities"] == 1
    assert records.loc["a1", "selections"] == 1


def test_build_author_work_rows_filters_root_works_and_computes_spread():
    raw = make_raw(
        [
            {"edition_id": 1, "anthology_publication_year": 1900, "author_id": "a1", "author_name": "Author One", "work_id": "w1", "work_title": "One"},
            {"edition_id": 2, "anthology_publication_year": 1910, "author_id": "a1", "author_name": "Author One", "work_id": "w1", "work_title": "One"},
            {"edition_id": 1, "anthology_publication_year": 1900, "author_id": "a1", "author_name": "Author One", "work_id": "w2", "work_title": "Two"},
            {"edition_id": 1, "anthology_publication_year": 1900, "author_id": "a1", "author_name": "Author One", "work_id": "child", "work_title": "Child", "parent_id": "w1"},
            {"edition_id": 2, "anthology_publication_year": 1910, "author_id": "a2", "author_name": "Author Two", "work_id": "w3", "work_title": "Three"},
        ]
    )
    editions = build_edition_table(raw)
    root = filter_root_works(raw)

    rows = build_author_work_rows(raw, root, editions).set_index(["author_id", "work_id"])

    assert set(rows.index) == {("a1", "w1"), ("a1", "w2"), ("a2", "w3")}
    assert rows.loc[("a1", "w1"), "work_selection_rate"] == pytest.approx(1.0)
    assert rows.loc[("a1", "w2"), "work_selection_rate"] == pytest.approx(0.0)
    assert rows.loc[("a1", "w1"), "work_rate_spread"] == pytest.approx(0.5)
    assert rows.loc[("a1", "w2"), "work_rate_spread"] == pytest.approx(0.5)
    assert rows.loc[("a2", "w3"), "work_rate_spread"] == pytest.approx(0.0)


def test_plot_frames_exclude_works_with_one_or_fewer_opportunities():
    rows = pd.DataFrame(
        [
            {
                "author_id": "a1",
                "work_id": "w1",
                "work_opportunities": 1,
                "author_selections": 1,
                "work_selections": 0,
                "author_selection_rate": 0.5,
                "work_selection_rate": 0.0,
                "work_rate_spread": 0.0,
            },
            {
                "author_id": "a1",
                "work_id": "w2",
                "work_opportunities": 2,
                "author_selections": 1,
                "work_selections": 1,
                "author_selection_rate": 0.5,
                "work_selection_rate": 0.5,
                "work_rate_spread": 0.5,
            }
        ]
    )

    counts, rates = build_plot_frames(rows, seed=7)

    assert len(counts) == 1
    assert len(rates) == 1
    assert counts.iloc[0]["work_id"] == "w2"
    assert rates.iloc[0]["work_id"] == "w2"


def test_hover_columns_and_template_use_compact_opportunity_format():
    rows = pd.DataFrame(
        [
            {
                "work_id": "w1",
                "work_title": "A Work",
                "author_id": "a1",
                "author_name": "An Author",
            },
            {
                "work_id": "w2",
                "work_title": None,
                "author_id": "a2",
                "author_name": None,
            },
        ]
    )

    out = add_hover_columns(rows)

    assert out.loc[0, "hover_work"] == "A Work"
    assert out.loc[0, "hover_author"] == "An Author"
    assert out.loc[1, "hover_work"] == "w2"
    assert out.loc[1, "hover_author"] == "a2"
    assert HOVER_TEMPLATE == (
        "<b>%{customdata[0]} by %{customdata[1]}</b><br>"
        "Work: %{customdata[2]} of %{customdata[3]} opps.<br>"
        "Author: %{customdata[4]} of %{customdata[5]} opps."
        "<extra></extra>"
    )
