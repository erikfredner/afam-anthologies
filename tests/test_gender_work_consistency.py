from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "gender"))

from gender_work_consistency import (  # noqa: E402
    assign_canonical_form,
    cliffs_delta,
    collapse_gender,
    compute_transitions,
    compute_unit_consistency,
    _jaccard,
    _unit_pairs,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


RAW_COLUMNS = [
    "work_id",
    "parent_id",
    "edition_id",
    "edition_year",
    "series_id",
    "author_id",
    "author_name",
    "gender",
    "form_id",
    "form_name",
]

SEL_COLUMNS = [
    "author_id",
    "author_name",
    "gender",
    "form",
    "edition_id",
    "edition_year",
    "series_identity",
    "work_id",
]


def make_raw(rows):
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def make_sel(rows):
    return pd.DataFrame(rows, columns=SEL_COLUMNS)


# ── assign_canonical_form ──────────────────────────────────────────────────────


def test_canonical_form_picks_lowest_form_id():
    raw = make_raw(
        [
            [1, None, 100, 1929, None, 10, "A", "Male", 3, "poetry"],
            [1, None, 100, 1929, None, 10, "A", "Male", 1, "nonfiction"],
        ]
    )
    forms = assign_canonical_form(raw)
    assert forms[1] == "nonfiction"  # lowest form_id wins


def test_canonical_form_excerpt_inherits_parent():
    raw = make_raw(
        [
            [1, None, 100, 1929, None, 10, "A", "Male", 1, "nonfiction"],
            [2, 1, 100, 1929, None, 10, "A", "Male", None, None],  # excerpt of 1
        ]
    )
    forms = assign_canonical_form(raw)
    assert forms[2] == "nonfiction"


def test_canonical_form_unknown_when_no_form_no_parent():
    raw = make_raw(
        [[3, None, 100, 1929, None, 10, "A", "Male", None, None]],
    )
    forms = assign_canonical_form(raw)
    assert forms[3] == "Unknown"


# ── collapse_gender ────────────────────────────────────────────────────────────


def test_collapse_gender_single_label_and_unknown():
    raw = make_raw(
        [
            [1, None, 100, 1929, None, 10, "A", "Female", 3, "poetry"],
            [2, None, 100, 1929, None, 10, "A", "Female", 3, "poetry"],
            [3, None, 100, 1929, None, 11, "B", None, 1, "nonfiction"],
        ]
    )
    g = collapse_gender(raw)
    assert g[10] == "Female"
    assert g[11] == "Unknown"


def test_collapse_gender_multi_gender_is_deterministic():
    raw = make_raw(
        [
            [1, None, 100, 1929, None, 10, "A", "Male", 3, "poetry"],
            [2, None, 101, 1941, None, 10, "A", "Female", 3, "poetry"],
        ]
    )
    g = collapse_gender(raw)
    assert g[10] == "Female"  # first in sorted order, stable


# ── Jaccard + pairing ──────────────────────────────────────────────────────────


def test_jaccard_basic():
    assert _jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert _jaccard(set(), set()) == 0.0
    assert _jaccard({"a"}, {"a"}) == 1.0


def test_unit_pairs_cross_series_excludes_same_series():
    order = {100: 0, 101: 1, 102: 2}
    series = {100: "series_1", 101: "series_1", 102: "series_2"}
    pairs = _unit_pairs([100, 101, 102], order, series, "cross_series")
    assert (100, 101) not in pairs  # same series excluded
    assert set(pairs) == {(100, 102), (101, 102)}


def test_unit_pairs_consecutive_uses_global_adjacency():
    order = {100: 0, 101: 1, 102: 2, 103: 3}
    series = {100: "s1", 101: "s2", 102: "s3", 103: "s4"}
    # unit skips edition 101 (order 1): its editions are 100, 102, 103
    pairs = _unit_pairs([100, 102, 103], order, series, "consecutive")
    assert set(pairs) == {(102, 103)}  # only globally-adjacent pair


# ── compute_unit_consistency ───────────────────────────────────────────────────


def _sample_sel():
    # author 1, poetry, 3 editions:
    #   ed100 (series_1): {w1, w2}
    #   ed101 (series_1): {w2, w3}
    #   ed102 (series_2): {w1}
    rows = []
    for w in (1, 2):
        rows.append([1, "A", "Female", "poetry", 100, 1929, "series_1", w])
    for w in (2, 3):
        rows.append([1, "A", "Female", "poetry", 101, 1941, "series_1", w])
    rows.append([1, "A", "Female", "poetry", 102, 1996, "series_2", 1])
    return make_sel(rows)


def test_unit_consistency_cross_series():
    units = compute_unit_consistency(_sample_sel(), "cross_series")
    assert len(units) == 1
    row = units.iloc[0]
    # pairs (100,102)->0.5, (101,102)->0.0  => mean 0.25
    assert row["n_pairs"] == 2
    assert np.isclose(row["mean_jaccard"], 0.25)
    assert row["debut_year"] == 1929
    assert row["span_years"] == 67


def test_unit_consistency_consecutive():
    units = compute_unit_consistency(_sample_sel(), "consecutive")
    row = units.iloc[0]
    # pairs (100,101)->1/3, (101,102)->0  => mean 1/6
    assert row["n_pairs"] == 2
    assert np.isclose(row["mean_jaccard"], 1 / 6)


def test_unit_with_single_edition_dropped():
    sel = make_sel([[1, "A", "Male", "poetry", 100, 1929, "series_1", 1]])
    assert compute_unit_consistency(sel, "cross_series").empty
    assert compute_unit_consistency(sel, "consecutive").empty


# ── compute_transitions ────────────────────────────────────────────────────────


def test_transitions_are_consecutive_only():
    trans = compute_transitions(_sample_sel())
    pairs = set(zip(trans["edition_earlier"], trans["edition_later"]))
    assert pairs == {(100, 101), (101, 102)}
    t = trans.set_index(["edition_earlier", "edition_later"])
    assert np.isclose(t.loc[(100, 101), "jaccard"], 1 / 3)
    assert t.loc[(100, 101), "midpoint_year"] == (1929 + 1941) / 2


# ── cliffs_delta ───────────────────────────────────────────────────────────────


def test_cliffs_delta_sign():
    assert cliffs_delta([0.5, 0.6], [0.1, 0.2]) == 1.0
    assert cliffs_delta([0.1, 0.2], [0.5, 0.6]) == -1.0
    assert np.isclose(cliffs_delta([0.3], [0.3]), 0.0)
