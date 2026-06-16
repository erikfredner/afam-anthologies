"""Tests for analysis/concentration/author_form_concentration.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "concentration"))
sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "reselection"))

from author_form_concentration import compute, prepare_plot_data  # noqa: E402


def _raw_row(**kw) -> dict:
    defaults = {
        "work_id": 1,
        "work_title": "Work 1",
        "parent_id": None,
        "volume_id": 100,
        "edition_id": 1000,
        "edition_year": 1990,
        "author_id": 1,
        "author_name": "Author A",
        "form_id": 3,
        "form_name": "poetry",
        "toc_page": 10,
        "toc_next": 20,
        "first_toc_page": 1,
        "last_toc_page": 500,
    }
    defaults.update(kw)
    return defaults


def _unit(concentration: pd.DataFrame, author_id: int, form_name: str = "poetry"):
    return concentration[
        (concentration["author_id"] == author_id)
        & (concentration["form_name"] == form_name)
    ].iloc[0]


def test_two_equal_span_works_in_two_editions_are_evenly_spread():
    raw = pd.DataFrame(
        [
            _raw_row(work_id=1, work_title="A", volume_id=101, edition_id=1),
            _raw_row(
                work_id=2,
                work_title="B",
                volume_id=102,
                edition_id=1,
                toc_page=20,
                toc_next=30,
            ),
            _raw_row(work_id=1, work_title="A", volume_id=201, edition_id=2),
            _raw_row(
                work_id=2,
                work_title="B",
                volume_id=202,
                edition_id=2,
                toc_page=20,
                toc_next=30,
            ),
        ]
    )
    outputs = compute(raw)
    per_work = outputs["per_work"].sort_values("work_id")
    row = _unit(outputs["concentration"], 1)

    assert per_work["page_p"].tolist() == pytest.approx([0.5, 0.5])
    assert row["hhi_page"] == pytest.approx(0.5)
    assert row["effective_works_page"] == pytest.approx(2.0)
    assert row["normalized_entropy_page"] == pytest.approx(1.0)
    assert row["gini_page"] == pytest.approx(0.0)
    assert row["signature_work_share_page"] == pytest.approx(0.5)
    assert row["total_selections"] == 4


def test_single_work_unit_is_kept_with_nan_normalized_entropy():
    raw = pd.DataFrame([_raw_row(work_id=10, work_title="Only Work")])
    row = _unit(compute(raw)["concentration"], 1)

    assert row["distinct_works"] == 1
    assert row["hhi_page"] == pytest.approx(1.0)
    assert row["effective_works_page"] == pytest.approx(1.0)
    assert row["signature_work_share_page"] == pytest.approx(1.0)
    assert row["gini_page"] == pytest.approx(0.0)
    assert np.isnan(row["normalized_entropy_page"])


def test_skewed_unit_has_high_signature_share_and_low_effective_works():
    raw = pd.DataFrame(
        [
            _raw_row(work_id=1, work_title="Dominant", toc_page=1, toc_next=91),
            _raw_row(
                work_id=2,
                work_title="Minor",
                volume_id=102,
                toc_page=91,
                toc_next=101,
            ),
        ]
    )
    row = _unit(compute(raw)["concentration"], 1)

    assert row["signature_work_share_page"] == pytest.approx(0.9)
    assert row["effective_works_page"] < row["distinct_works"]


def test_all_zero_span_unit_marks_page_stats_degenerate_but_count_stats_finite():
    raw = pd.DataFrame(
        [
            _raw_row(work_id=1, work_title="Zero A", toc_page=10, toc_next=10),
            _raw_row(
                work_id=2,
                work_title="Zero B",
                volume_id=102,
                toc_page=20,
                toc_next=20,
            ),
        ]
    )
    row = _unit(compute(raw)["concentration"], 1)

    assert bool(row["page_weights_degenerate"]) is True
    assert row["n_zero_span_workedition"] == 2
    assert np.isnan(row["hhi_page"])
    assert np.isnan(row["effective_works_page"])
    assert row["hhi_count"] == pytest.approx(0.5)
    assert row["effective_works_count"] == pytest.approx(2.0)


def test_poetry_essay_work_splits_half_its_span_to_each_form_unit():
    raw = pd.DataFrame(
        [
            _raw_row(
                work_id=1,
                work_title="Hybrid",
                form_id=3,
                form_name="poetry",
                toc_page=10,
                toc_next=20,
            ),
            _raw_row(
                work_id=1,
                work_title="Hybrid",
                form_id=4,
                form_name="essay",
                toc_page=10,
                toc_next=20,
            ),
            _raw_row(
                work_id=2,
                work_title="Poem",
                volume_id=102,
                form_id=3,
                form_name="poetry",
                toc_page=20,
                toc_next=25,
            ),
        ]
    )
    per_work = compute(raw)["per_work"]
    poetry = per_work[per_work["form_name"] == "poetry"].set_index("work_title")
    essay = per_work[per_work["form_name"] == "essay"].set_index("work_title")

    assert poetry.loc["Hybrid", "page_p"] == pytest.approx(0.5)
    assert poetry.loc["Poem", "page_p"] == pytest.approx(0.5)
    assert essay.loc["Hybrid", "page_p"] == pytest.approx(1.0)


def test_count_and_page_weighting_can_diverge():
    raw = pd.DataFrame(
        [
            _raw_row(work_id=1, work_title="Long Once", volume_id=101, edition_id=1),
            _raw_row(
                work_id=2,
                work_title="Short Twice",
                volume_id=102,
                edition_id=1,
                toc_page=20,
                toc_next=30,
            ),
            _raw_row(
                work_id=2,
                work_title="Short Twice",
                volume_id=202,
                edition_id=2,
                toc_page=10,
                toc_next=20,
            ),
        ]
    )
    # Make edition 1 skewed: work 1 has 90 pages, work 2 has 10.
    raw.loc[raw["work_id"] == 1, ["toc_page", "toc_next"]] = [1, 91]
    row = _unit(compute(raw)["concentration"], 1)

    assert row["hhi_page"] != pytest.approx(row["hhi_count"])
    assert row["hhi_count"] > row["hhi_page"]


def test_prepare_plot_data_filters_single_work_units_and_ranks_within_form():
    concentration = pd.DataFrame(
        [
            {
                "author_id": 1,
                "author_name": "A One",
                "author_last_name": "One",
                "form_name": "poetry",
                "distinct_works": 1,
                "total_selections": 10,
                "page_weights_degenerate": False,
                "hhi_page": 1.0,
                "effective_works_page": 1.0,
                "signature_work_share_page": 1.0,
            },
            {
                "author_id": 2,
                "author_name": "A Two",
                "author_last_name": "Two",
                "form_name": "poetry",
                "distinct_works": 2,
                "total_selections": 10,
                "page_weights_degenerate": False,
                "hhi_page": 0.5,
                "effective_works_page": 2.0,
                "signature_work_share_page": 0.5,
            },
            {
                "author_id": 3,
                "author_name": "A Three",
                "author_last_name": "Three",
                "form_name": "poetry",
                "distinct_works": 3,
                "total_selections": 10,
                "page_weights_degenerate": False,
                "hhi_page": 0.8,
                "effective_works_page": 1.25,
                "signature_work_share_page": 0.8,
            },
        ]
    )

    plotted = prepare_plot_data(concentration, min_selections=5, max_forms=None)

    assert set(plotted["author_id"]) == {2, 3}
    percentiles = plotted.set_index("author_id")["concentration_percentile_page"]
    assert percentiles.loc[2] == pytest.approx(0.5)
    assert percentiles.loc[3] == pytest.approx(1.0)
