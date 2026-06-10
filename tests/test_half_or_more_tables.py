from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "overlap"))

from authors_in_half_or_more_afam_eds import build_table as build_author_table  # noqa: E402
from works_in_half_or_more_afam_eds import build_table as build_work_table  # noqa: E402


def test_author_table_includes_reselection_rate() -> None:
    raw = pd.DataFrame(
        [
            {
                "author_id": "a1",
                "author_name": "Author One",
                "birth_year": 1900,
                "edition_count": 3,
                "reselection_count": 2,
                "opportunities": 4,
                "reselection_rate": 0.5,
            }
        ]
    )

    table = build_author_table(raw)

    assert table.columns.tolist() == ["Author", "Selections", "Reselections"]
    assert table.loc[0, "Author"] == "Author One"
    assert table.loc[0, "Selections"] == 3
    assert table.loc[0, "Reselections"] == "2/4"


def test_author_table_sorts_ties_by_author_last_name() -> None:
    raw = pd.DataFrame(
        [
            {
                "author_id": "a1",
                "author_name": "Paul Laurence Dunbar",
                "birth_year": 1872,
                "edition_count": 3,
                "reselection_count": 2,
                "opportunities": 4,
                "reselection_rate": 0.5,
            },
            {
                "author_id": "a2",
                "author_name": "W. E. B. Du Bois",
                "birth_year": 1868,
                "edition_count": 3,
                "reselection_count": 2,
                "opportunities": 4,
                "reselection_rate": 0.5,
            },
            {
                "author_id": "a3",
                "author_name": "Martin Luther King Jr.",
                "birth_year": 1929,
                "edition_count": 3,
                "reselection_count": 2,
                "opportunities": 4,
                "reselection_rate": 0.5,
            },
        ]
    )

    table = build_author_table(raw)

    assert table["Author"].tolist() == [
        "W. E. B. Du Bois",
        "Paul Laurence Dunbar",
        "Martin Luther King Jr.",
    ]


def test_work_table_includes_reselection_rate_after_grouping_authors() -> None:
    raw = pd.DataFrame(
        [
            {
                "work_id": "w1",
                "work_title": "A Work",
                "author_name": "Author One",
                "edition_count": 3,
                "reselection_count": 2,
                "opportunities": 4,
                "reselection_rate": 0.5,
            },
            {
                "work_id": "w1",
                "work_title": "A Work",
                "author_name": "Author Two",
                "edition_count": 3,
                "reselection_count": 2,
                "opportunities": 4,
                "reselection_rate": 0.5,
            },
        ]
    )

    table = build_work_table(raw)

    assert table.columns.tolist() == ["Work", "Selections", "Reselections"]
    assert table.loc[0, "Work"] == '"A Work" by Author One and Author Two'
    assert table.loc[0, "Selections"] == 3
    assert table.loc[0, "Reselections"] == "2/4"


def test_work_table_sorts_ties_by_author_last_name() -> None:
    raw = pd.DataFrame(
        [
            {
                "work_id": "w1",
                "work_title": "Work by Dunbar",
                "author_name": "Paul Laurence Dunbar",
                "edition_count": 3,
                "reselection_count": 2,
                "opportunities": 4,
                "reselection_rate": 0.5,
            },
            {
                "work_id": "w2",
                "work_title": "Work by Du Bois",
                "author_name": "W. E. B. Du Bois",
                "edition_count": 3,
                "reselection_count": 2,
                "opportunities": 4,
                "reselection_rate": 0.5,
            },
        ]
    )

    table = build_work_table(raw)

    assert table["Work"].tolist() == [
        '"Work by Du Bois" by W. E. B. Du Bois',
        '"Work by Dunbar" by Paul Laurence Dunbar',
    ]
