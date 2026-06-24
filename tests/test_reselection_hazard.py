from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "reselection"))

from reselection_hazard import (  # noqa: E402
    build_person_period,
    lifetable_survival,
)
from author_vs_work_debut_reselection import (  # noqa: E402
    add_entry_group,
    build_author_rows,
    build_edition_table,
    build_work_rows,
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


# Four editions, one author per row, used across several tests.
# edition_order: ed1->0 (2000), ed2->1 (2010), ed3->2 (2020), ed4->3 (2030).
def _four_edition_authors(appearances: list[tuple[str, int]]) -> tuple:
    rows = [
        {"author_id": a, "edition_id": e, "anthology_publication_year": yr}
        for a, e, yr in [
            (a, e, {1: 2000, 2: 2010, 3: 2020, 4: 2030}[e]) for a, e in appearances
        ]
    ]
    return make_raw(rows)


def _pp(raw, editions, *, event_model="first", scope="all"):
    return build_person_period(
        build_author_rows(raw),
        "author_id",
        "author",
        editions,
        event_model=event_model,
        scope=scope,
    )


def test_left_truncation_rows_start_after_debut():
    # a1 debuts in ed1 (order 0) and is reselected in ed3 (order 2). Filler
    # authors materialize ed2 and ed4 so all four edition orders exist.
    raw, editions = _four_edition_authors([("a1", 1), ("a1", 3), ("f2", 2), ("f4", 4)])
    pp = _pp(raw, editions, event_model="recurrent")
    a1 = pp[pp["entity_id"] == "a1"].sort_values("edition_order")

    # No row for the debut edition itself; rows begin at debut + 1.
    assert list(a1["t_since_debut"]) == [1, 2, 3]
    assert a1.iloc[0]["edition_order"] == 1
    assert list(a1["event"]) == [0, 1, 0]


def test_last_edition_debut_is_censored_with_zero_rows():
    # a_last debuts in the final edition (order 3): no successor -> no rows.
    raw, editions = _four_edition_authors([("a1", 1), ("a_last", 4)])
    pp = _pp(raw, editions, event_model="recurrent")
    assert (pp["entity_id"] == "a_last").sum() == 0
    assert (pp["entity_id"] == "a1").sum() > 0


def test_first_event_truncates_at_first_reselection():
    # a1 reselected in ed2 (order 1) and ed4 (order 3); first-event stops at ed2.
    raw, editions = _four_edition_authors([("a1", 1), ("a1", 2), ("a1", 4)])
    pp = _pp(raw, editions, event_model="first")
    a1 = pp[pp["entity_id"] == "a1"].sort_values("edition_order")
    assert list(a1["t_since_debut"]) == [1]
    assert list(a1["event"]) == [1]


def test_recurrent_keeps_all_rows_and_tracks_prior_selections():
    # a1 selected in ed1 (debut), ed2 (order1), ed4 (order3); filler f3 makes ed3.
    raw, editions = _four_edition_authors([("a1", 1), ("a1", 2), ("a1", 4), ("f3", 3)])
    pp = _pp(raw, editions, event_model="recurrent")
    a1 = pp[pp["entity_id"] == "a1"].sort_values("edition_order")
    assert list(a1["t_since_debut"]) == [1, 2, 3]
    assert list(a1["event"]) == [1, 0, 1]
    # prior_selections counts post-debut events strictly before the current row.
    assert list(a1["prior_selections"]) == [0, 1, 1]


def test_debut_decade_and_year_recorded():
    raw, editions = _four_edition_authors([("a1", 1), ("a1", 3)])
    pp = _pp(raw, editions, event_model="first")
    row = pp[pp["entity_id"] == "a1"].iloc[0]
    assert row["debut_year"] == 2000
    assert row["debut_decade"] == 2000


def test_cross_series_scope_excludes_same_series_editions():
    # Two editions in series 7, one standalone edition (its own group).
    raw, editions = make_raw(
        [
            {
                "author_id": "a1",
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "series_id": 7,
            },
            {
                "author_id": "a1",
                "edition_id": 2,
                "anthology_publication_year": 2010,
                "series_id": 7,
            },
            {"author_id": "a1", "edition_id": 3, "anthology_publication_year": 2020},
        ]
    )
    pp = build_person_period(
        build_author_rows(raw),
        "author_id",
        "author",
        editions,
        event_model="recurrent",
        scope="cross-series",
    )
    a1 = pp[pp["entity_id"] == "a1"]
    # The same-series ed2 is dropped from the risk set; only the standalone ed3
    # (a different entry_group) is at risk.
    assert list(a1["edition_id"]) == [3]


def test_works_carry_form_from_map():
    rows = [
        {"work_id": 10, "edition_id": 1, "anthology_publication_year": 2000},
        {"work_id": 10, "edition_id": 3, "anthology_publication_year": 2020},
    ]
    raw, editions = make_raw(rows)
    form_map = pd.Series({10: "poetry"}, name="form_name")
    pp = build_person_period(
        build_work_rows(raw),
        "work_id",
        "work",
        editions,
        event_model="first",
        scope="all",
        form_map=form_map,
    )
    assert (pp["form_name"] == "poetry").all()


def test_lifetable_survival_matches_hand_computation():
    # Three authors debut in ed1 (order 0). a1 reselected at t=1, a2 at t=2,
    # a3 never. First-event life table:
    #   t1: 3 at risk, 1 event -> h=1/3, S=2/3
    #   t2: 2 at risk, 1 event -> h=1/2, S=1/3
    #   t3: 1 at risk, 0 events -> h=0,   S=1/3
    # a4 debuts in the final edition (order 3) -> 0 rows, but materializes ed4 so
    # that a3 (never reselected) is at risk through t=3.
    raw, editions = _four_edition_authors(
        [
            ("a1", 1),
            ("a1", 2),
            ("a2", 1),
            ("a2", 3),
            ("a3", 1),
            ("a4", 4),
        ]
    )
    pp = _pp(raw, editions, event_model="first")
    lt = lifetable_survival(pp).sort_values("t_since_debut").reset_index(drop=True)

    assert list(lt["n_at_risk"]) == [3, 2, 1]
    assert list(lt["n_events"]) == [1, 1, 0]
    assert lt.loc[0, "hazard"] == pytest.approx(1 / 3)
    assert lt.loc[0, "survival"] == pytest.approx(2 / 3)
    assert lt.loc[1, "survival"] == pytest.approx(1 / 3)
    assert lt.loc[2, "survival"] == pytest.approx(1 / 3)
    # Survival is monotone non-increasing and bounded by 1.
    assert (lt["survival"].diff().dropna() <= 1e-12).all()
    assert (lt["survival"] <= 1.0).all()


def test_empty_person_period_returns_typed_frame():
    empty = pd.DataFrame(columns=["author_id", "edition_id", "entry_group"])
    editions = pd.DataFrame(
        columns=[
            "edition_id",
            "anthology_publication_year",
            "edition_order",
            "entry_group",
        ]
    )
    pp = build_person_period(empty, "author_id", "author", editions)
    assert pp.empty
    assert "t_since_debut" in pp.columns
