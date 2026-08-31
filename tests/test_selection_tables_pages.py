"""Unit tests for the docs/ page generator in scripts/build_selection_tables.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_selection_tables import (  # noqa: E402
    build_author_rows,
    build_work_rows,
    format_life_dates,
    format_percent,
    render_table,
    split_qualifier,
)

# ── formatting helpers ──────────────────────────────────────────────────


def test_life_dates_with_both_years():
    """A dead author gets an en-dashed range."""
    assert format_life_dates(1901, 1967) == "1901–1967"


def test_life_dates_for_living_author():
    """A null death year becomes a birth-only note."""
    assert format_life_dates(1944, None) == "b. 1944"


def test_life_dates_with_neither_year():
    """An author with no dates on record gets an em dash, not "b. nan"."""
    assert format_life_dates(None, None) == "—"


def test_split_qualifier_separates_angle_brackets():
    """The DB's disambiguating suffix comes off the title but is not discarded."""
    assert split_qualifier("The New Negro <essay>") == ("The New Negro", "essay")


def test_split_qualifier_leaves_plain_titles_alone():
    """A title with no suffix passes through with an empty qualifier."""
    assert split_qualifier("If We Must Die") == ("If We Must Die", "")


def test_split_qualifier_ignores_mid_string_angle_brackets():
    """Only a trailing bracket is a qualifier; one inside the title is content."""
    assert split_qualifier("A <b> in the middle") == ("A <b> in the middle", "")


def test_format_percent_at_full_coverage():
    """26 of 26 reads as 100%, not 100.0%."""
    assert format_percent(26, 26) == "100%"


def test_format_percent_rounds_to_a_whole_number():
    """A single selection out of 26 rounds rather than showing 3.846%."""
    assert format_percent(13, 26) == "50%"
    assert format_percent(1, 26) == "4%"


def test_format_percent_without_a_denominator():
    """An entity that debuted in the last anthology has had no opportunity."""
    assert format_percent(0, 0) == "—"


# ── row builders ────────────────────────────────────────────────────────


@pytest.fixture
def authors():
    """Two authors tied on selections, to exercise the surname tie-break."""
    return pd.DataFrame(
        [
            {
                "author_name": "Paul Laurence Dunbar",
                "birth_year": 1872,
                "death_year": 1906,
                "edition_count": 20,
                "reselection_count": 19,
                "opportunities": 20,
                "reselection_rate": 0.95,
            },
            {
                "author_name": "W. E. B. Du Bois",
                "birth_year": 1868,
                "death_year": 1963,
                "edition_count": 20,
                "reselection_count": 18,
                "opportunities": 20,
                "reselection_rate": 0.9,
            },
        ]
    )


def test_author_rows_break_ties_by_surname(authors):
    """Du Bois sorts before Dunbar, per afam.names.author_sort_key."""
    rows = build_author_rows(authors, 26)
    assert [row[0]["html"] for row in rows] == [
        "W. E. B. Du Bois",
        "Paul Laurence Dunbar",
    ]


def test_author_rows_carry_four_columns(authors):
    """Author, Dates, Selected, Reselected -- both records as percentages."""
    rows = build_author_rows(authors, 26)
    assert len(rows[0]) == 4
    assert rows[0][2]["html"] == "77%"
    assert rows[0][3]["html"] == "90%"


def test_author_rows_blank_reselection_without_opportunities(authors):
    """An author debuting in the most recent anthology has no rate, not 0%."""
    debut = authors.iloc[[0]].assign(
        reselection_count=0, opportunities=0, reselection_rate=None
    )
    row = build_author_rows(debut, 26)[0]
    assert row[3]["html"] == "—"
    # Ranked just below a genuine 0% rather than left as an unsortable blank.
    assert row[3]["sort"] == "-1"


def _work_row(**overrides):
    defaults = {
        "work_id": 1,
        "work_title": "A Work",
        "parent_id": None,
        "parent_work_title": None,
        "author_name": "Author One",
        "edition_count": 15,
        "reselection_count": 14,
        "opportunities": 20,
        "reselection_rate": 0.7,
    }
    return {**defaults, **overrides}


def test_work_rows_join_co_authors():
    """The SQL emits one row per author; the page shows one row per work."""
    df = pd.DataFrame(
        [_work_row(author_name="Author One"), _work_row(author_name="Author Two")]
    )
    rows = build_work_rows(df, 26)
    assert len(rows) == 1
    assert rows[0][1]["html"] == "Author One and Author Two"


def test_work_rows_leave_unattributed_works_blank():
    """
    Spirituals and folk songs have no author on record, and the cell stays empty.

    Not "Anonymous": the absence of attribution is the evidence, and naming an
    author would assert a claim the sources do not make.
    """
    df = pd.DataFrame([_work_row(work_title="Steal Away to Jesus", author_name=None)])
    assert build_work_rows(df, 26)[0][1]["html"] == ""


def test_work_rows_show_em_dash_for_a_root_work():
    """A work with no parent has nothing to put in the In column."""
    assert build_work_rows(pd.DataFrame([_work_row()]), 26)[0][2]["html"] == "—"


def test_work_rows_name_the_parent_work():
    """An excerpt names the book it came from, qualifier and all."""
    df = pd.DataFrame(
        [
            _work_row(
                work_title="The New Negro <essay>",
                parent_id=99,
                parent_work_title="The New Negro <edited collection>",
            )
        ]
    )
    row = build_work_rows(df, 26)[0]
    assert row[0]["html"] == 'The New Negro <span class="qual">(essay)</span>'
    assert (
        row[2]["html"] == 'The New Negro <span class="qual">(edited collection)</span>'
    )
    # Sorting keys drop the qualifier so the two columns alphabetize on the title.
    assert row[0]["sort"] == "The New Negro"


# ── rendering ───────────────────────────────────────────────────────────


def test_render_table_escapes_markup_in_names():
    """Ampersands and angle brackets must not reach the browser as markup."""
    df = pd.DataFrame([_work_row(author_name="Ida B. Wells & <Co>")])
    html = render_table(
        ["Work", "Author", "In", "Selections", "Reselections"], build_work_rows(df, 26)
    )
    assert "Ida B. Wells &amp; &lt;Co&gt;" in html
    assert "<Co>" not in html


def test_render_table_marks_numeric_columns_for_sorting():
    """Numeric cells carry data-sort and the num class the sort script reads."""
    html = render_table(
        ["Work", "Author", "In", "Selected", "Reselected"],
        build_work_rows(pd.DataFrame([_work_row()]), 26),
    )
    assert '<th class="num">Selected<span class="arrow"></span></th>' in html
    assert '<td class="num" data-sort="15">58%</td>' in html
