from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "reselection"))

from gender_reselection import (  # noqa: E402
    aggregate_by_year,
    build_gender_group,
    compute_cumulative,
    compute_opportunities,
)


# ── Helper ─────────────────────────────────────────────────────────────────────


def make_selections(rows):
    return pd.DataFrame(rows, columns=["author_id", "gender_group", "edition_key", "ek_year"])


# ── build_gender_group ─────────────────────────────────────────────────────────


def test_build_gender_group_male():
    assert build_gender_group("Male") == "Male"


def test_build_gender_group_female():
    assert build_gender_group("Female") == "Female"


def test_build_gender_group_empty():
    assert build_gender_group("") == "Unmarked"


def test_build_gender_group_other():
    assert build_gender_group("Other") == "Unmarked"


# ── compute_opportunities ──────────────────────────────────────────────────────


def test_basic_opportunity_generation():
    """Author in ed1 (1941) gets opportunity for ed2 (1968); not the reverse."""
    rows = [
        {"author_id": "a1", "gender_group": "Male", "edition_key": "ed1", "ek_year": 1941},
        {"author_id": "a1", "gender_group": "Male", "edition_key": "ed2", "ek_year": 1968},
    ]
    sel = make_selections(rows)
    opps = compute_opportunities(sel)
    assert len(opps) == 1
    assert opps.iloc[0]["edition_key"] == "ed2"
    assert opps.iloc[0]["ek_year"] == 1968


def test_debut_excluded():
    """Author's first edition is never itself an opportunity."""
    rows = [
        {"author_id": "a1", "gender_group": "Female", "edition_key": "ed1", "ek_year": 1929},
    ]
    sel = make_selections(rows)
    opps = compute_opportunities(sel)
    assert len(opps) == 0


def test_selected_flag_correct():
    """Opportunity marked selected=1 only if author appears in that edition."""
    rows = [
        {"author_id": "a1", "gender_group": "Male", "edition_key": "ed1", "ek_year": 1929},
        {"author_id": "a1", "gender_group": "Male", "edition_key": "ed2", "ek_year": 1941},
        # a2 only debuts in ed1 — does not appear in ed2
        {"author_id": "a2", "gender_group": "Female", "edition_key": "ed1", "ek_year": 1929},
    ]
    sel = make_selections(rows)
    opps = compute_opportunities(sel)

    a1_opp = opps[(opps["author_id"] == "a1") & (opps["edition_key"] == "ed2")]
    a2_opp = opps[(opps["author_id"] == "a2") & (opps["edition_key"] == "ed2")]

    assert len(a1_opp) == 1
    assert a1_opp.iloc[0]["selected"] == 1
    assert len(a2_opp) == 1
    assert a2_opp.iloc[0]["selected"] == 0


# ── aggregate_by_year ──────────────────────────────────────────────────────────


def test_aggregate_by_year_totals():
    """Sum of opportunities across genders matches expected cross-product count."""
    # 2 Male authors + 1 Female debut in ed1(1929); ed2(1941) exists (only m1 selected)
    rows = [
        {"author_id": "m1", "gender_group": "Male", "edition_key": "ed1", "ek_year": 1929},
        {"author_id": "m2", "gender_group": "Male", "edition_key": "ed1", "ek_year": 1929},
        {"author_id": "f1", "gender_group": "Female", "edition_key": "ed1", "ek_year": 1929},
        {"author_id": "m1", "gender_group": "Male", "edition_key": "ed2", "ek_year": 1941},
    ]
    sel = make_selections(rows)
    opps = compute_opportunities(sel)
    agg = aggregate_by_year(opps)
    # m1, m2, f1 each get exactly one opportunity at ed2
    assert agg["opportunities"].sum() == 3


# ── compute_cumulative ─────────────────────────────────────────────────────────


def test_compute_cumulative_monotone():
    """cum_opportunities and cum_reselections are non-decreasing per gender."""
    rows = [
        {"author_id": "m1", "gender_group": "Male", "edition_key": "ed1", "ek_year": 1929},
        {"author_id": "m1", "gender_group": "Male", "edition_key": "ed2", "ek_year": 1941},
        {"author_id": "m1", "gender_group": "Male", "edition_key": "ed3", "ek_year": 1968},
        {"author_id": "m2", "gender_group": "Male", "edition_key": "ed2", "ek_year": 1941},
        {"author_id": "m2", "gender_group": "Male", "edition_key": "ed3", "ek_year": 1968},
    ]
    sel = make_selections(rows)
    opps = compute_opportunities(sel)
    agg = aggregate_by_year(opps)
    cum = compute_cumulative(agg)

    male_cum = cum[cum["gender_group"] == "Male"].sort_values("ek_year")
    cum_opps = male_cum["cum_opportunities"].tolist()
    cum_sel = male_cum["cum_reselections"].tolist()

    for i in range(1, len(cum_opps)):
        assert cum_opps[i] >= cum_opps[i - 1]
        assert cum_sel[i] >= cum_sel[i - 1]
