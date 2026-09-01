"""Unit tests for the docs/ page generator in scripts/build_selection_tables.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_selection_tables import (  # noqa: E402
    Column,
    add_peer_delta,
    build_author_rows,
    build_work_rows,
    format_life_dates,
    format_peer_delta,
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


# ── peer delta ──────────────────────────────────────────────────────────


def _cohort(**overrides):
    """Three entities debuting together, each with 10 post-debut opportunities."""
    rows = [
        {"debut_edition_id": 1, "reselection_count": 8, "opportunities": 10},
        {"debut_edition_id": 1, "reselection_count": 2, "opportunities": 10},
        {"debut_edition_id": 1, "reselection_count": 2, "opportunities": 10},
    ]
    return pd.DataFrame([{**row, **overrides} for row in rows])


def test_peer_delta_excludes_the_focal_entity():
    """The baseline is the *other* cohort members, not the cohort including self."""
    deltas = add_peer_delta(_cohort())["peer_delta_pp"].tolist()
    # 8/10 against peers' 4/20 = 20%: a 60-point gap, not the 40 points that
    # including the focal row in its own baseline (8/10 vs 12/30) would give.
    assert deltas[0] == pytest.approx(60.0)
    # And the two low performers each sit below a baseline that includes the 8.
    assert deltas[1] == pytest.approx(20.0 - 50.0)


def test_peer_delta_is_zero_when_a_cohort_is_uniform():
    """Everyone at the same rate is nobody above or below it."""
    uniform = _cohort(reselection_count=5)
    assert add_peer_delta(uniform)["peer_delta_pp"].tolist() == [0.0, 0.0, 0.0]


def test_peer_delta_is_undefined_without_opportunities():
    """An entity debuting in the most recent anthology has no peers to beat."""
    debut = _cohort(opportunities=0, reselection_count=0)
    assert add_peer_delta(debut)["peer_delta_pp"].isna().all()


def test_peer_delta_is_undefined_for_a_lone_debut():
    """A cohort of one has no baseline; the row must be blank, not 0."""
    lone = _cohort().iloc[[0]]
    assert add_peer_delta(lone)["peer_delta_pp"].isna().all()


def test_peer_delta_compares_within_cohorts_only():
    """Two cohorts in one frame are scored against themselves, not each other."""
    df = pd.concat(
        [_cohort(), _cohort().assign(debut_edition_id=2, reselection_count=1)],
        ignore_index=True,
    )
    deltas = add_peer_delta(df)["peer_delta_pp"].tolist()
    assert deltas[0] == pytest.approx(60.0)
    # The second cohort is uniform, so its members are all at their peers' rate
    # despite reselecting far less often than the first cohort.
    assert deltas[3:] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (51.4, "+51"),
        (-17.6, "−18"),
        (0.0, "0"),
        (0.4, "0"),
        (float("nan"), "—"),
        (None, "—"),
    ],
)
def test_format_peer_delta(value, expected):
    """Signed integers, a true minus sign, and an em dash where undefined."""
    assert format_peer_delta(value) == expected


@pytest.fixture
def authors():
    """Two authors tied on selections, to exercise the surname tie-break."""
    return pd.DataFrame(
        [
            {
                "author_name": "Paul Laurence Dunbar",
                "birth_year": 1872,
                "death_year": 1906,
                "debut_edition_id": 1,
                "edition_count": 20,
                "reselection_count": 19,
                "opportunities": 20,
                "reselection_rate": 0.95,
            },
            {
                "author_name": "W. E. B. Du Bois",
                "birth_year": 1868,
                "death_year": 1963,
                "debut_edition_id": 1,
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


def test_author_rows_carry_five_columns(authors):
    """Author, Dates, Selected, Reselected, vs. peers."""
    rows = build_author_rows(authors, 26)
    assert len(rows[0]) == 5
    assert rows[0][2]["html"] == "77%"
    assert rows[0][3]["html"] == "90%"
    # Du Bois reselected 18/20; his one cohort peer, Dunbar, 19/20.
    assert rows[0][4]["html"] == "−5"


def test_author_rows_blank_reselection_without_opportunities(authors):
    """An author debuting in the most recent anthology has no rate, not 0%."""
    debut = authors.iloc[[0]].assign(
        reselection_count=0, opportunities=0, reselection_rate=None
    )
    row = build_author_rows(debut, 26)[0]
    assert row[3]["html"] == "—"
    # No opportunities means no peer comparison either.
    assert row[4]["html"] == "—"
    # Marked missing rather than given a low sentinel rank: having had no chance
    # to be reselected is not the same as having been passed over.
    assert row[3]["missing"] and row[3]["sort"] is None
    assert row[4]["missing"] and row[4]["sort"] is None


def _work_row(**overrides):
    defaults = {
        "work_id": 1,
        "work_title": "A Work",
        "parent_id": None,
        "parent_work_title": None,
        "author_name": "Author One",
        "debut_edition_id": 1,
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
        ["Work", "Author", "In", "Selected", "Reselected", "vs. peers"],
        build_work_rows(df, 26),
    )
    assert "Ida B. Wells &amp; &lt;Co&gt;" in html
    assert "<Co>" not in html


def test_render_table_flags_missing_cells():
    """A cell with nothing to rank carries data-missing, so the sort sinks it."""
    df = pd.DataFrame([_work_row(reselection_count=0, opportunities=0)])
    html = render_table(
        ["Work", "Author", "In", "Selected", "Reselected", "vs. peers"],
        build_work_rows(df, 26),
    )
    # The root work's "In" cell, plus the two undefined rate cells.
    assert html.count("data-missing") == 3
    # A cell that does have a value never gets the marker.
    assert '<td class="num" data-sort="15">58%</td>' in html


def test_missing_cells_carry_no_sort_sentinel():
    """Absent values are flagged, not ranked -- no -1 or -1000 in the output."""
    df = pd.DataFrame([_work_row(reselection_count=0, opportunities=0)])
    html = render_table(
        ["Work", "Author", "In", "Selected", "Reselected", "vs. peers"],
        build_work_rows(df, 26),
    )
    assert 'data-sort="-1"' not in html
    assert 'data-sort="-1000"' not in html


def test_unattributed_work_sorts_as_missing():
    """The 298 authorless works sink rather than heading an A-Z author sort."""
    df = pd.DataFrame([_work_row(author_name=None)])
    author_cell = build_work_rows(df, 26)[0][1]
    assert author_cell["html"] == ""
    assert author_cell["missing"]


def test_render_table_marks_numeric_columns_for_sorting():
    """Numeric cells carry data-sort and the num class the sort script reads."""
    html = render_table(
        ["Work", "Author", "In", "Selected", "Reselected"],
        build_work_rows(pd.DataFrame([_work_row()]), 26),
    )
    assert '<th class="num">Selected<span class="arrow"></span></th>' in html
    assert '<td class="num" data-sort="15">58%</td>' in html


# ── filter row ──────────────────────────────────────────────────────────


def _work_table(headers=None):
    """One rendered works table, for asserting on its head markup."""
    return render_table(
        headers or ["Work", "Author", "In", "Selected", "Reselected", "vs. peers"],
        build_work_rows(pd.DataFrame([_work_row()]), 26),
        unit="works",
    )


def test_headings_and_filters_are_separate_rows():
    """
    The sort script collects its clickable headings from tr.cols alone.

    The filter boxes sit in a tr.filters of td cells, so a click in a filter box
    can never be read as a request to sort the column.
    """
    html = _work_table()
    assert '<tr class="cols"><th>Work' in html
    assert '<tr class="filters"><td>' in html
    assert '<tr class="filters"><th' not in html


def test_text_columns_get_a_substring_box():
    """A column of names filters by what is typed appearing anywhere in a cell."""
    html = _work_table()
    assert (
        '<input type="search" data-column="1" data-kind="text"'
        ' placeholder="filter" aria-label="Filter by Author">' in html
    )


def test_numeric_columns_get_a_lowest_and_highest_box():
    """A column of numbers filters by range, not by substring."""
    html = _work_table()
    assert html.count('data-kind="min"') == 3  # Selected, Reselected, vs. peers
    assert html.count('data-kind="max"') == 3
    assert 'data-column="3" data-kind="min" placeholder="min"' in html
    assert 'aria-label="Highest Selected"' in html


def test_an_explicit_range_overrides_the_numeric_default():
    """
    Dates is not a numeric column, but "born between" is the useful filter.

    Its placeholders say "from"/"to" rather than "min"/"max", since the reader
    is bounding a span of years rather than a quantity.
    """
    html = render_table(
        [Column("Dates", filter="range", low="from", high="to")],
        [[{"html": "1901–1967", "sort": 1901, "numeric": False, "missing": False}]],
        unit="authors",
    )
    assert 'data-kind="min" placeholder="from" aria-label="Lowest Dates"' in html
    assert 'data-kind="max" placeholder="to" aria-label="Highest Dates"' in html


def test_a_column_can_opt_out_of_filtering():
    """filter="none" leaves the cell empty rather than dropping it, so the
    filter row keeps one cell per column and the columns stay aligned."""
    html = render_table(
        [Column("Work", filter="none")],
        [[{"html": "A Work", "sort": None, "numeric": False, "missing": False}]],
    )
    assert '<tr class="filters"><td></td></tr>' in html


def test_the_table_names_what_it_counts():
    """The script says "12 of 3,236 works", so it needs the noun."""
    assert '<table data-unit="works">' in _work_table()
    assert 'data-unit="rows"' in render_table(
        ["Work"],
        [[{"html": "A Work", "sort": None, "numeric": False, "missing": False}]],
    )


def test_bare_string_headings_still_work():
    """Most columns want the default, and say so by staying plain strings."""
    html = _work_table()
    assert '<th>Work<span class="arrow"></span></th>' in html
    assert 'data-column="0" data-kind="text"' in html
