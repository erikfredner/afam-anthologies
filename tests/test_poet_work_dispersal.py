from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "concentration"))

from poet_work_dispersal import build_poet_dispersal, filter_leaf_works  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────────────


def make_raw(rows):
    """Minimal author-page-share-reselection-shaped frame.

    Columns mirror queries/author-page-share-reselection.sql; page columns are
    filled so author_form_concentration.compute can build spans without error.
    """
    cols = [
        "work_id",
        "work_title",
        "parent_id",
        "volume_id",
        "edition_id",
        "edition_year",
        "author_id",
        "author_name",
        "form_id",
        "form_name",
        "toc_page",
        "toc_next",
        "first_toc_page",
        "last_toc_page",
    ]
    return pd.DataFrame(rows, columns=cols)


def poetry_rows():
    """Two poets with contrasting work-level dispersal.

    Brown: 4 editions, each picks a different poem (max poem in 1/4 editions).
    Hughes: 4 editions, the same poem in 3 of them (3/4 editions).
    Each work is one page (toc_page n .. n+1) in its own single-work volume.
    """
    rows = []
    # Brown -- fully dispersed: poem k selected only in edition k.
    for ed in range(1, 5):
        rows.append(
            [
                100 + ed,
                f"Brown Poem {ed}",
                None,
                1000 + ed,
                ed,
                1990 + ed,
                1,
                "Sterling Brown",
                7,
                "poetry",
                10.0,
                11.0,
                10.0,
                11.0,
            ]
        )
    # Hughes -- concentrated: poem 200 in editions 1-3, poem 201 in edition 4.
    for ed in range(1, 4):
        rows.append(
            [
                200,
                "Mother to Son",
                None,
                2000 + ed,
                ed,
                1990 + ed,
                2,
                "Langston Hughes",
                7,
                "poetry",
                20.0,
                21.0,
                20.0,
                21.0,
            ]
        )
    rows.append(
        [
            201,
            "I, Too",
            None,
            2004,
            4,
            1994,
            2,
            "Langston Hughes",
            7,
            "poetry",
            20.0,
            21.0,
            20.0,
            21.0,
        ]
    )
    return rows


@pytest.fixture
def built():
    import author_form_concentration as afc

    raw = make_raw(poetry_rows())
    outputs = afc.compute(raw)
    table = build_poet_dispersal(outputs, raw, raw, min_editions=2)
    return table.set_index("author_name")


# ── top_poem_coverage math ──────────────────────────────────────────────────────


def test_brown_coverage_below_half(built):
    row = built.loc["Sterling Brown"]
    assert row["poetry_editions"] == 4
    assert row["top_poem_editions"] == 1
    assert row["top_poem_coverage"] == pytest.approx(0.25)


def test_hughes_coverage_above_half(built):
    row = built.loc["Langston Hughes"]
    assert row["poetry_editions"] == 4
    assert row["top_poem"] == "Mother to Son"
    assert row["top_poem_editions"] == 3
    assert row["top_poem_coverage"] == pytest.approx(0.75)


# ── effective poems: Brown is more dispersed than Hughes ─────────────────────────


def test_brown_more_effective_poems_than_hughes(built):
    brown = built.loc["Sterling Brown"]
    hughes = built.loc["Langston Hughes"]
    # Brown's 4 poems are evenly spread (effective ~= 4); Hughes is lopsided.
    assert brown["effective_poems_count"] > hughes["effective_poems_count"]
    assert brown["distinct_poems"] == 4
    assert hughes["distinct_poems"] == 2


# ── ranking + filtering ──────────────────────────────────────────────────────────


def test_ranked_by_editions_then_coverage(built):
    # Tie on poetry_editions (4 each) -> lower coverage first -> Brown leads.
    assert built.index[0] == "Sterling Brown"


def test_min_editions_filter():
    import author_form_concentration as afc

    raw = make_raw(poetry_rows())
    outputs = afc.compute(raw)
    table = build_poet_dispersal(outputs, raw, raw, min_editions=5)
    assert table.empty


def test_distinct_works_all_forms_counts_every_work(built):
    # raw_all == raw here (poetry-only fixture), so the all-forms distinct-works
    # count equals each poet's distinct poems. Brown spreads across more works.
    brown = built.loc["Sterling Brown"]
    hughes = built.loc["Langston Hughes"]
    assert brown["distinct_works_all_forms"] == 4
    assert hughes["distinct_works_all_forms"] == 2
    assert brown["distinct_works_all_forms"] > hughes["distinct_works_all_forms"]


# ── leaf-work filtering ──────────────────────────────────────────────────────────


def test_filter_leaf_works_drops_containers():
    raw = make_raw(poetry_rows())
    kept = filter_leaf_works(raw, {200})  # treat Hughes' poem 200 as a container
    assert 200 not in set(kept["work_id"])
    # leaf poems are untouched
    assert {101, 102, 103, 104}.issubset(set(kept["work_id"]))


def test_filter_leaf_works_empty_container_set_is_noop():
    raw = make_raw(poetry_rows())
    kept = filter_leaf_works(raw, set())
    assert set(kept["work_id"]) == set(raw["work_id"])
