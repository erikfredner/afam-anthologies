from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "reselection"))

from retention_from_1929 import compute_retention  # noqa: E402


# ── Helper ─────────────────────────────────────────────────────────────────────


def make_long(rows):
    return pd.DataFrame(
        rows, columns=["edition_key", "ek_year", "author_id", "norm_title"]
    )


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_100_percent_retention():
    """Later edition contains all 1929 authors and works → 100% both."""
    rows = [
        # 1929 reference
        {"edition_key": "67", "ek_year": 1929, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "67", "ek_year": 1929, "author_id": "a2", "norm_title": "w2"},
        # Later edition with same authors and works
        {"edition_key": "B", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 1941, "author_id": "a2", "norm_title": "w2"},
    ]
    df = compute_retention(make_long(rows), "67")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["author_pct"] == pytest.approx(100.0)
    assert row["work_pct"] == pytest.approx(100.0)


def test_0_percent_retention():
    """Later edition shares no authors or works with 1929 → 0% both."""
    rows = [
        # 1929 reference
        {"edition_key": "67", "ek_year": 1929, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "67", "ek_year": 1929, "author_id": "a2", "norm_title": "w2"},
        # Entirely different later edition
        {"edition_key": "B", "ek_year": 1941, "author_id": "a3", "norm_title": "w3"},
        {"edition_key": "B", "ek_year": 1941, "author_id": "a4", "norm_title": "w4"},
    ]
    df = compute_retention(make_long(rows), "67")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["author_pct"] == pytest.approx(0.0)
    assert row["work_pct"] == pytest.approx(0.0)


def test_partial_retention():
    """1 of 2 authors shared → 50%; 1 of 4 works shared → 25%."""
    rows = [
        # 1929: authors a1, a2; works w1, w2, w3, w4
        {"edition_key": "67", "ek_year": 1929, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "67", "ek_year": 1929, "author_id": "a1", "norm_title": "w2"},
        {"edition_key": "67", "ek_year": 1929, "author_id": "a2", "norm_title": "w3"},
        {"edition_key": "67", "ek_year": 1929, "author_id": "a2", "norm_title": "w4"},
        # Later edition: only a1 (of 2 authors), only w1 (of 4 works)
        {"edition_key": "B", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 1941, "author_id": "a3", "norm_title": "w9"},
    ]
    df = compute_retention(make_long(rows), "67")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["author_pct"] == pytest.approx(50.0)
    assert row["work_pct"] == pytest.approx(25.0)


def test_reference_edition_excluded():
    """The reference edition itself must not appear in the output."""
    rows = [
        {"edition_key": "67", "ek_year": 1929, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
    ]
    df = compute_retention(make_long(rows), "67")
    assert "67" not in df["edition_key"].values


def test_output_sorted_by_year():
    """Output must be sorted by year ascending."""
    rows = [
        {"edition_key": "67", "ek_year": 1929, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "D", "ek_year": 2004, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "C", "ek_year": 1968, "author_id": "a2", "norm_title": "w2"},
    ]
    df = compute_retention(make_long(rows), "67")
    years = df["year"].tolist()
    assert years == sorted(years)


def test_author_pct_gte_work_pct():
    """Later edition takes both 1929 authors but only some works → author_pct ≥ work_pct."""
    rows = [
        # 1929: 2 authors, 4 distinct works
        {"edition_key": "67", "ek_year": 1929, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "67", "ek_year": 1929, "author_id": "a1", "norm_title": "w2"},
        {"edition_key": "67", "ek_year": 1929, "author_id": "a2", "norm_title": "w3"},
        {"edition_key": "67", "ek_year": 1929, "author_id": "a2", "norm_title": "w4"},
        # Later edition: both authors retained, but only 1 of 4 works
        {"edition_key": "B", "ek_year": 1941, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 1941, "author_id": "a2", "norm_title": "w9"},
    ]
    df = compute_retention(make_long(rows), "67")
    row = df.iloc[0]
    assert row["author_pct"] == pytest.approx(100.0)
    assert row["work_pct"] == pytest.approx(25.0)
    assert row["author_pct"] >= row["work_pct"]
