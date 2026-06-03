from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "predictability"))

from work_selection_probability_model import (  # noqa: E402
    author_prior_rate_table,
    build_form_map,
    compute_edition_total_pages,
    compute_work_volume_spans,
    heuristic_score,
    length_zscore_by_form,
    prior_editions_count,
    zscore_column,
)


# ── compute_work_volume_spans ─────────────────────────────────────────────────


def test_span_basic():
    raw = pd.DataFrame(
        {
            "work_id": [1, 2, 3],
            "volume_id": [10, 10, 10],
            "toc_page": [1.0, 20.0, 40.0],
            "toc_next": [20.0, 40.0, 50.0],
            "last_toc_page": [50.0, 50.0, 50.0],
        }
    )
    spans = compute_work_volume_spans(raw).set_index("work_id")["span"]
    assert spans[1] == 19
    assert spans[2] == 20
    assert spans[3] == 10


def test_span_fallback_when_toc_next_null():
    """When toc_next is NULL, span = last_toc_page - toc_page + 1."""
    raw = pd.DataFrame(
        {
            "work_id": [1],
            "volume_id": [10],
            "toc_page": [40.0],
            "toc_next": [np.nan],
            "last_toc_page": [50.0],
        }
    )
    spans = compute_work_volume_spans(raw).set_index("work_id")["span"]
    assert spans[1] == 11


def test_span_negative_clipped_to_zero():
    raw = pd.DataFrame(
        {
            "work_id": [1],
            "volume_id": [10],
            "toc_page": [50.0],
            "toc_next": [30.0],  # would yield -20
            "last_toc_page": [60.0],
        }
    )
    spans = compute_work_volume_spans(raw).set_index("work_id")["span"]
    assert spans[1] == 0


# ── build_form_map ────────────────────────────────────────────────────────────


def test_form_direct_assignment():
    form_rows = pd.DataFrame(
        {"work_id": [1, 2], "form_id": [3, 4], "form_name": ["poetry", "fiction"]}
    )
    parent_map = pd.DataFrame(columns=["work_id", "parent_id"])
    out = build_form_map(form_rows, parent_map)
    assert out[1] == "poetry"
    assert out[2] == "fiction"


def test_form_multi_form_collapses_to_lowest_form_id():
    """When a work has multiple form rows, lowest form_id wins (stable preference)."""
    form_rows = pd.DataFrame(
        {
            "work_id": [1, 1],
            "form_id": [4, 3],
            "form_name": ["fiction", "poetry"],
        }
    )
    parent_map = pd.DataFrame(columns=["work_id", "parent_id"])
    out = build_form_map(form_rows, parent_map)
    assert out[1] == "poetry"  # form_id=3 < form_id=4


def test_form_excerpt_inherits_from_parent():
    """An excerpt without its own form row inherits the parent's form."""
    form_rows = pd.DataFrame(
        {"work_id": [10], "form_id": [4], "form_name": ["fiction"]}
    )
    parent_map = pd.DataFrame({"work_id": [100], "parent_id": [10]})
    out = build_form_map(form_rows, parent_map)
    assert out[10] == "fiction"
    assert out[100] == "fiction"  # inherited


# ── length_zscore_by_form ─────────────────────────────────────────────────────


def test_length_z_zero_at_form_mean():
    frame = pd.DataFrame(
        {
            "form": ["poetry"] * 10,
            "pct_pages_of_edition": [0.002] * 5 + [0.004] * 5,
        }
    )
    z = length_zscore_by_form(frame, min_n=5)
    # mean = 0.003; rows at 0.002 should have negative z, rows at 0.004 positive z
    assert (z[frame["pct_pages_of_edition"] == 0.002] < 0).all()
    assert (z[frame["pct_pages_of_edition"] == 0.004] > 0).all()


def test_length_z_uses_proportion_not_raw_pages():
    """A 10-page poem in a 500pp anthology should look longer than 10 pages in a
    2500pp anthology. Using proportion rather than raw pages catches this.
    """
    # Two anthologies with the same form. Same raw pages (10) → identical raw-z
    # would be 0. But edition A is much smaller, so its pct_pages is much higher.
    frame = pd.DataFrame(
        {
            "form": ["poetry"] * 6,
            "pages": [10, 10, 10, 10, 10, 10],
            "pct_pages_of_edition": [0.02, 0.02, 0.02, 0.004, 0.004, 0.004],
        }
    )
    z = length_zscore_by_form(frame, value_col="pct_pages_of_edition", min_n=5)
    # The 0.02-share rows should be positive (above mean), 0.004 negative.
    assert (z.iloc[:3] > 0).all()
    assert (z.iloc[3:] < 0).all()


def test_length_z_nan_for_small_form():
    frame = pd.DataFrame(
        {
            "form": ["poetry"] * 6 + ["screenplay"] * 2,
            "pct_pages_of_edition": [0.001] * 6 + [0.05, 0.05],
        }
    )
    z = length_zscore_by_form(frame, min_n=5)
    # screenplay rows (only 2) should be NaN
    assert z[frame["form"] == "screenplay"].isna().all()


# ── compute_edition_total_pages ───────────────────────────────────────────────


def test_edition_total_pages_from_volume_metadata():
    """Edition total = sum of (last_toc_page - first_toc_page + 1) per volume."""
    raw = pd.DataFrame(
        {
            "edition_id": [1, 1, 1, 2, 2],
            "volume_id": [10, 10, 11, 20, 20],  # ed.1 has 2 volumes, ed.2 has 1
            "first_toc_page": [1, 1, 1, 1, 1],
            "last_toc_page": [500, 500, 600, 800, 800],
        }
    )
    totals = compute_edition_total_pages(raw)
    assert totals[1] == 500 + 600  # 1100
    assert totals[2] == 800


def test_edition_total_pages_drops_volumes_with_null_metadata():
    raw = pd.DataFrame(
        {
            "edition_id": [1, 1],
            "volume_id": [10, 11],
            "first_toc_page": [1, np.nan],
            "last_toc_page": [500, 600],
        }
    )
    totals = compute_edition_total_pages(raw)
    # Volume 11 dropped (NaN first_toc_page); edition total = 500 from volume 10.
    assert totals[1] == 500


# ── prior_editions_count ──────────────────────────────────────────────────────


def test_prior_count_increments_with_each_new_edition():
    frame = pd.DataFrame(
        {
            "work_id": [1, 1, 1, 2],
            "edition_id": [100, 101, 102, 100],
            "year": [1929, 1941, 1968, 1929],
        }
    )
    counts = prior_editions_count(frame)
    counts.index = frame.index
    # work 1: 1929 → 0 priors, 1941 → 1 prior, 1968 → 2 priors
    # work 2: 1929 → 0 priors
    assert counts.tolist() == [0, 1, 2, 0]


def test_prior_count_multi_volume_same_edition_counts_once():
    """Two volumes of the same edition should not double-count prior."""
    frame = pd.DataFrame(
        {
            "work_id": [1, 1, 1],
            "volume_id": [10, 11, 12],
            "edition_id": [100, 100, 101],  # 100 = 2 volumes
            "year": [1929, 1929, 1941],
        }
    )
    counts = prior_editions_count(frame)
    counts.index = frame.index
    # Both edition-100 rows have 0 priors; edition-101 row has 1 prior (edition 100).
    assert counts.tolist() == [0, 0, 1]


# ── author_prior_rate_table ───────────────────────────────────────────────────


def test_author_rate_is_debut_when_no_priors():
    author_rows = pd.DataFrame(
        {
            "author_id": [1, 1],
            "edition_id": [100, 101],
            "year": [1929, 1941],
        }
    )
    edition_years = pd.DataFrame({"edition_id": [100, 101], "year": [1929, 1941]})
    out = author_prior_rate_table(author_rows, edition_years).set_index(
        ["author_id", "edition_id"]
    )
    # At edition 100 (1929), author 1's first year = 1929. No prior editions.
    assert out.loc[(1, 100), "author_is_debut"]
    assert np.isnan(out.loc[(1, 100), "author_prior_selection_rate"])
    # At edition 101 (1941): 1 prior opportunity (edition 100), selected → rate=1.0
    assert not out.loc[(1, 101), "author_is_debut"]
    assert out.loc[(1, 101), "author_prior_selection_rate"] == 1.0


def test_author_rate_partial_selection():
    """Author selected in 1 of 2 prior opportunities → rate = 0.5."""
    author_rows = pd.DataFrame(
        {
            "author_id": [1, 1],
            "edition_id": [100, 102],
            "year": [1929, 1968],
        }
    )
    edition_years = pd.DataFrame(
        {
            "edition_id": [100, 101, 102],
            "year": [1929, 1941, 1968],
        }
    )
    out = author_prior_rate_table(author_rows, edition_years).set_index(
        ["author_id", "edition_id"]
    )
    # At edition 102 (1968): opportunities = {100, 101} (both year >= 1929 and < 1968).
    # Selections = {100} → rate = 1/2.
    assert out.loc[(1, 102), "author_prior_selection_rate"] == 0.5


# ── zscore_column ─────────────────────────────────────────────────────────────


def test_zscore_column_zero_variance_returns_zeros():
    s = pd.Series([5, 5, 5, 5])
    z = zscore_column(s)
    assert (z == 0).all()


def test_zscore_column_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore_column(s)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=1) - 1.0) < 1e-9


# ── heuristic_score sign convention ───────────────────────────────────────────


def test_heuristic_sign_convention():
    """Higher prior + higher author_rate − higher length_z − higher pct = more probable."""
    frame = pd.DataFrame(
        {
            "prior_editions_count": [0, 5, 0, 0, 0],
            "author_prior_selection_rate": [0.0, 0.0, 1.0, 0.0, 0.0],
            "length_z_by_form": [0.0, 0.0, 0.0, 2.0, 0.0],
            "pct_pages_of_edition": [0.0, 0.0, 0.0, 0.0, 0.5],
        }
    )
    s = heuristic_score(frame)
    baseline = s.iloc[0]
    # Higher prior_count → MORE probable
    assert s.iloc[1] > baseline
    # Higher author_rate → MORE probable
    assert s.iloc[2] > baseline
    # Higher length_z → LESS probable
    assert s.iloc[3] < baseline
    # Higher pct_pages → LESS probable
    assert s.iloc[4] < baseline
