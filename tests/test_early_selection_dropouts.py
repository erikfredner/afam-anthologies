from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "reselection"))

from early_selection_dropouts import (  # noqa: E402
    build_author_dropouts,
    build_work_dropouts,
    count_period_editions,
    detect_norton_first_year,
    filter_root_works,
)


def make_raw(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "work_id": "w",
        "work_title": "Work",
        "parent_id": None,
        "edition_id": 1,
        "anthology_publication_year": 1980,
        "series_id": None,
        "author_id": "a",
        "author_name": "Author",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_detects_early_cutoff_from_first_norton_series_edition():
    raw = make_raw(
        [
            {"edition_id": 1, "anthology_publication_year": 1980, "series_id": None},
            {"edition_id": 2, "anthology_publication_year": 1997, "series_id": 3},
            {"edition_id": 3, "anthology_publication_year": 2025, "series_id": "3"},
        ]
    )

    assert detect_norton_first_year(raw) == 1997


def test_author_dropouts_require_pre_norton_selection_and_zero_2010_plus():
    raw = make_raw(
        [
            {
                "author_id": "a1",
                "author_name": "Pre Norton",
                "edition_id": 1,
                "anthology_publication_year": 1980,
            },
            {
                "author_id": "a1",
                "author_name": "Pre Norton",
                "edition_id": 2,
                "anthology_publication_year": 1985,
            },
            {
                "author_id": "a1",
                "author_name": "Pre Norton",
                "edition_id": 3,
                "anthology_publication_year": 1997,
                "series_id": 3,
            },
            {
                "author_id": "a2",
                "author_name": "Contemporary Return",
                "edition_id": 1,
                "anthology_publication_year": 1980,
            },
            {
                "author_id": "a2",
                "author_name": "Contemporary Return",
                "edition_id": 4,
                "anthology_publication_year": 2010,
            },
        ]
    )
    counts = count_period_editions(raw, early_before=1997, contemporary_from=2010)

    dropouts = build_author_dropouts(
        raw,
        early_before=1997,
        contemporary_from=2010,
        early_edition_count=counts["early"],
        min_early_count=1,
    ).set_index("author_id")

    assert set(dropouts.index) == {"a1"}
    assert dropouts.loc["a1", "early_selection_count"] == 2
    assert dropouts.loc["a1", "post_early_pre_contemporary_count"] == 1
    assert dropouts.loc["a1", "contemporary_selection_count"] == 0


def test_first_norton_edition_itself_is_not_early():
    raw = make_raw(
        [
            {
                "author_id": "norton_only",
                "author_name": "Norton Only",
                "edition_id": 2,
                "anthology_publication_year": 1997,
                "series_id": 3,
            },
            {
                "author_id": "early",
                "author_name": "Early",
                "edition_id": 1,
                "anthology_publication_year": 1980,
            },
        ]
    )

    dropouts = build_author_dropouts(
        raw,
        early_before=1997,
        contemporary_from=2010,
        early_edition_count=1,
        min_early_count=1,
    )

    assert set(dropouts["author_id"]) == {"early"}


def test_work_counts_deduplicate_multi_author_rows_per_edition():
    raw = make_raw(
        [
            {
                "work_id": "w1",
                "work_title": "Shared Work",
                "author_id": "a1",
                "author_name": "Alice",
                "edition_id": 1,
                "anthology_publication_year": 1980,
            },
            {
                "work_id": "w1",
                "work_title": "Shared Work",
                "author_id": "a2",
                "author_name": "Bob",
                "edition_id": 1,
                "anthology_publication_year": 1980,
            },
            {
                "work_id": "w1",
                "work_title": "Shared Work",
                "author_id": "a1",
                "author_name": "Alice",
                "edition_id": 2,
                "anthology_publication_year": 1985,
            },
        ]
    )

    dropouts = build_work_dropouts(
        raw,
        early_before=1997,
        contemporary_from=2010,
        early_edition_count=2,
        min_early_count=1,
    ).set_index("work_id")

    assert dropouts.loc["w1", "early_selection_count"] == 2
    assert dropouts.loc["w1", "author_names"] == "Alice; Bob"


def test_root_only_filter_excludes_excerpts():
    raw = make_raw(
        [
            {"work_id": "root", "parent_id": None},
            {"work_id": "excerpt", "parent_id": "root"},
        ]
    )

    filtered = filter_root_works(raw)

    assert set(filtered["work_id"]) == {"root"}


def test_ranking_uses_early_count_before_name():
    raw = make_raw(
        [
            {
                "author_id": "a1",
                "author_name": "Zed",
                "edition_id": 1,
                "anthology_publication_year": 1980,
            },
            {
                "author_id": "a2",
                "author_name": "Alpha",
                "edition_id": 1,
                "anthology_publication_year": 1980,
            },
            {
                "author_id": "a2",
                "author_name": "Alpha",
                "edition_id": 2,
                "anthology_publication_year": 1985,
            },
        ]
    )

    dropouts = build_author_dropouts(
        raw,
        early_before=1997,
        contemporary_from=2010,
        early_edition_count=2,
        min_early_count=1,
    )

    assert list(dropouts["author_id"]) == ["a2", "a1"]
