from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "reselection"))

from cumulative_pairwise_agreement import (  # noqa: E402
    compute_snapshots,
    jaccard,
)


# ── Helper ─────────────────────────────────────────────────────────────────────


def make_long_df(
    rows: list[dict],
) -> pd.DataFrame:
    """Build a long DataFrame with the schema compute_snapshots expects.

    Each dict must supply: edition_key, ek_year, author_id, norm_title.
    """
    return pd.DataFrame(rows, columns=["edition_key", "ek_year", "author_id", "norm_title"])


# ── jaccard ────────────────────────────────────────────────────────────────────


def test_jaccard_partial_overlap():
    assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3)


def test_jaccard_equal_sets():
    assert jaccard(frozenset({"x", "y"}), frozenset({"x", "y"})) == pytest.approx(1.0)


def test_jaccard_disjoint():
    assert jaccard(frozenset({"a"}), frozenset({"b"})) == pytest.approx(0.0)


def test_jaccard_both_empty_returns_nan():
    result = jaccard(frozenset(), frozenset())
    assert math.isnan(result)


# ── 2-edition snapshot ─────────────────────────────────────────────────────────


def test_two_edition_snapshot():
    """Two editions: A at year 1 (authors 1,2; works w1,w2) and B at year 2 (authors 2,3; works w2,w3)."""
    rows = [
        # Edition A, year 1
        {"edition_key": "A", "ek_year": 1, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "A", "ek_year": 1, "author_id": "a2", "norm_title": "w2"},
        # Edition B, year 2
        {"edition_key": "B", "ek_year": 2, "author_id": "a2", "norm_title": "w2"},
        {"edition_key": "B", "ek_year": 2, "author_id": "a3", "norm_title": "w3"},
    ]
    df = make_long_df(rows)
    snaps = compute_snapshots(df)

    # Only one snapshot (at year 2, when the first pair forms)
    assert len(snaps) == 1
    snap = snaps.iloc[0]
    assert snap["year"] == 2
    assert snap["n_pairs"] == 1

    # Author Jaccard: {a1,a2} vs {a2,a3} → 1/3
    assert snap["author_median"] == pytest.approx(1 / 3)

    # Work Jaccard: {w1,w2} vs {w2,w3} → 1/3
    assert snap["work_median"] == pytest.approx(1 / 3)


# ── Cumulative property ────────────────────────────────────────────────────────


def test_cumulative_pair_counts():
    """3 editions at years 1, 2, 3: pairs should be 0→1→3 cumulative."""
    rows = [
        {"edition_key": "A", "ek_year": 1, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 2, "author_id": "a2", "norm_title": "w2"},
        {"edition_key": "C", "ek_year": 3, "author_id": "a3", "norm_title": "w3"},
    ]
    df = make_long_df(rows)
    snaps = compute_snapshots(df)

    # Year 1: 0 pairs → no snapshot; year 2: 1 pair; year 3: 3 pairs
    assert len(snaps) == 2
    assert snaps.iloc[0]["n_pairs"] == 1
    assert snaps.iloc[1]["n_pairs"] == 3


# ── Same-year editions ─────────────────────────────────────────────────────────


def test_same_year_editions_included():
    """Two editions both at year 1 should produce a snapshot at year 1."""
    rows = [
        {"edition_key": "A", "ek_year": 1, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 1, "author_id": "a2", "norm_title": "w2"},
    ]
    df = make_long_df(rows)
    snaps = compute_snapshots(df)

    assert len(snaps) == 1
    assert snaps.iloc[0]["year"] == 1
    assert snaps.iloc[0]["n_pairs"] == 1


# ── n_pairs monotonically non-decreasing ──────────────────────────────────────


def test_n_pairs_monotone():
    rows = [
        {"edition_key": "A", "ek_year": 1, "author_id": "a1", "norm_title": "w1"},
        {"edition_key": "B", "ek_year": 2, "author_id": "a2", "norm_title": "w2"},
        {"edition_key": "C", "ek_year": 3, "author_id": "a3", "norm_title": "w3"},
        {"edition_key": "D", "ek_year": 4, "author_id": "a4", "norm_title": "w4"},
    ]
    df = make_long_df(rows)
    snaps = compute_snapshots(df)
    pairs = snaps["n_pairs"].tolist()
    assert all(pairs[i] <= pairs[i + 1] for i in range(len(pairs) - 1))


# ── Author Jaccard ≥ Work Jaccard (hypothesis validation) ─────────────────────


def test_author_jaccard_gte_work_jaccard():
    """Anthologies share all authors but differ in specific works chosen.

    This exercises the core hypothesis: anthologies agree more on *who* to
    include than on *which works* to represent them with.
    """
    # Three editions all including the same two authors (a1, a2),
    # but each picks different works for each author.
    rows = [
        # Edition A
        {"edition_key": "A", "ek_year": 1, "author_id": "a1", "norm_title": "work_a1_v1"},
        {"edition_key": "A", "ek_year": 1, "author_id": "a2", "norm_title": "work_a2_v1"},
        # Edition B
        {"edition_key": "B", "ek_year": 2, "author_id": "a1", "norm_title": "work_a1_v2"},
        {"edition_key": "B", "ek_year": 2, "author_id": "a2", "norm_title": "work_a2_v2"},
        # Edition C
        {"edition_key": "C", "ek_year": 3, "author_id": "a1", "norm_title": "work_a1_v3"},
        {"edition_key": "C", "ek_year": 3, "author_id": "a2", "norm_title": "work_a2_v3"},
    ]
    df = make_long_df(rows)
    snaps = compute_snapshots(df)

    last = snaps.iloc[-1]
    # Author Jaccard should be 1.0 (all editions share the same two authors)
    assert last["author_median"] == pytest.approx(1.0)
    # Work Jaccard should be 0.0 (each edition picks unique works)
    assert last["work_median"] == pytest.approx(0.0)
    # The main claim: author >= work
    assert last["author_median"] >= last["work_median"]


def test_compute_snapshots_accepts_db_work_ids():
    rows = [
        {"edition_key": "A", "ek_year": 1, "author_id": "a1", "work_id": "w1"},
        {"edition_key": "B", "ek_year": 2, "author_id": "a1", "work_id": "w2"},
    ]
    df = pd.DataFrame(rows)

    snaps = compute_snapshots(df)

    assert len(snaps) == 1
    assert snaps.iloc[0]["author_median"] == pytest.approx(1.0)
    assert snaps.iloc[0]["work_median"] == pytest.approx(0.0)
