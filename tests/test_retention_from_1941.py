from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "reselection"))

from retention_from_1941 import compute_retention  # noqa: E402


# ── Helper ─────────────────────────────────────────────────────────────────────


def make_long(rows):
    return pd.DataFrame(
        rows, columns=["edition_key", "ek_year", "author_id", "norm_title"]
    )


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_100_percent_retention():
    """Later edition contains all 1941 authors and works → 100% both."""
    rows = [
        {"edition_key": "62", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "62", "ek_year": 1941, "author_id": "a2", "norm_title": "w2"},
        {"edition_key": "B", "ek_year": 1968, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 1968, "author_id": "a2", "norm_title": "w2"},
    ]
    df = compute_retention(make_long(rows), "62", after_year=1941)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["author_pct"] == pytest.approx(100.0)
    assert row["work_pct"] == pytest.approx(100.0)


def test_0_percent_retention():
    """Later edition shares no authors or works with 1941 → 0% both."""
    rows = [
        {"edition_key": "62", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "62", "ek_year": 1941, "author_id": "a2", "norm_title": "w2"},
        {"edition_key": "B", "ek_year": 1968, "author_id": "a3", "norm_title": "w3"},
        {"edition_key": "B", "ek_year": 1968, "author_id": "a4", "norm_title": "w4"},
    ]
    df = compute_retention(make_long(rows), "62", after_year=1941)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["author_pct"] == pytest.approx(0.0)
    assert row["work_pct"] == pytest.approx(0.0)


def test_partial_retention():
    """1 of 2 authors shared → 50%; 1 of 4 works shared → 25%."""
    rows = [
        {"edition_key": "62", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "62", "ek_year": 1941, "author_id": "a1", "norm_title": "w2"},
        {"edition_key": "62", "ek_year": 1941, "author_id": "a2", "norm_title": "w3"},
        {"edition_key": "62", "ek_year": 1941, "author_id": "a2", "norm_title": "w4"},
        {"edition_key": "B", "ek_year": 1968, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 1968, "author_id": "a3", "norm_title": "w9"},
    ]
    df = compute_retention(make_long(rows), "62", after_year=1941)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["author_pct"] == pytest.approx(50.0)
    assert row["work_pct"] == pytest.approx(25.0)


def test_reference_edition_excluded():
    """The reference edition itself must not appear in the output."""
    rows = [
        {"edition_key": "62", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 1968, "author_id": "a1", "norm_title": "w1"},
    ]
    df = compute_retention(make_long(rows), "62", after_year=1941)
    assert "62" not in df["edition_key"].values


def test_output_sorted_by_year():
    """Output must be sorted by year ascending."""
    rows = [
        {"edition_key": "62", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "D", "ek_year": 2004, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 1968, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "C", "ek_year": 1996, "author_id": "a2", "norm_title": "w2"},
    ]
    df = compute_retention(make_long(rows), "62", after_year=1941)
    years = df["year"].tolist()
    assert years == sorted(years)


def test_author_pct_gte_work_pct():
    """Later edition takes both 1941 authors but only some works → author_pct ≥ work_pct."""
    rows = [
        {"edition_key": "62", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "62", "ek_year": 1941, "author_id": "a1", "norm_title": "w2"},
        {"edition_key": "62", "ek_year": 1941, "author_id": "a2", "norm_title": "w3"},
        {"edition_key": "62", "ek_year": 1941, "author_id": "a2", "norm_title": "w4"},
        {"edition_key": "B", "ek_year": 1968, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 1968, "author_id": "a2", "norm_title": "w9"},
    ]
    df = compute_retention(make_long(rows), "62", after_year=1941)
    row = df.iloc[0]
    assert row["author_pct"] == pytest.approx(100.0)
    assert row["work_pct"] == pytest.approx(25.0)
    assert row["author_pct"] >= row["work_pct"]


def test_after_year_excludes_earlier_editions():
    """Editions from 1941 and before must be excluded when after_year=1941."""
    rows = [
        {"edition_key": "62", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        # Same year as reference — should be excluded
        {"edition_key": "X", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        # Before reference year — should be excluded
        {"edition_key": "Y", "ek_year": 1929, "author_id": "a1", "norm_title": "w1"},
        # After reference year — should be included
        {"edition_key": "Z", "ek_year": 1950, "author_id": "a1", "norm_title": "w1"},
    ]
    df = compute_retention(make_long(rows), "62", after_year=1941)
    assert set(df["edition_key"].tolist()) == {"Z"}
