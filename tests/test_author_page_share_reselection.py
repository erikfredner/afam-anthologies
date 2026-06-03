"""Tests for viz/reselection/author_page_share_reselection.py

Covers helpers that don't need DB access: page-span computation, multi-author
and multi-form splitting, and the reselection-rate kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "reselection"))

from author_page_share_reselection import (  # noqa: E402
    build_author_form_counts,
    build_author_form_per_edition,
    build_author_per_edition,
    build_work_per_edition,
    compute_reselection_rates,
    compute_work_volume_spans,
    edition_form_total_pages,
    edition_total_pages,
)


def _raw_row(**kw) -> dict:
    defaults = {
        "work_id": 1,
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


# ── compute_work_volume_spans ────────────────────────────────────────────────


class TestWorkVolumeSpans:
    def test_span_is_toc_next_minus_toc_page(self):
        raw = pd.DataFrame([_raw_row(toc_page=10, toc_next=25)])
        spans = compute_work_volume_spans(raw)
        assert spans.iloc[0]["span"] == 15

    def test_fallback_when_toc_next_null(self):
        raw = pd.DataFrame([_raw_row(toc_page=180, toc_next=None, last_toc_page=200)])
        spans = compute_work_volume_spans(raw)
        # 200 - 180 + 1 = 21
        assert spans.iloc[0]["span"] == 21

    def test_negative_span_clipped_to_zero(self):
        raw = pd.DataFrame([_raw_row(toc_page=30, toc_next=20)])
        spans = compute_work_volume_spans(raw)
        assert spans.iloc[0]["span"] == 0


# ── Multi-author and multi-form splits ────────────────────────────────────────


class TestAuthorFormCounts:
    def test_two_author_two_form_work_has_correct_counts(self):
        raw = pd.DataFrame(
            [
                _raw_row(author_id=1, form_name="poetry"),
                _raw_row(author_id=1, form_name="song"),
                _raw_row(author_id=2, form_name="poetry"),
                _raw_row(author_id=2, form_name="song"),
            ]
        )
        counts = build_author_form_counts(raw)
        assert counts.iloc[0]["author_count"] == 2
        assert counts.iloc[0]["form_count"] == 2

    def test_work_without_form_has_form_count_one(self):
        raw = pd.DataFrame([_raw_row(form_id=None, form_name=None)])
        counts = build_author_form_counts(raw)
        assert counts.iloc[0]["form_count"] == 1


class TestAuthorPagesSplit:
    def test_two_authors_split_pages_equally(self):
        # 10-page work shared by 2 authors → each gets 5 pages in author_per_edition
        raw = pd.DataFrame(
            [
                _raw_row(author_id=1, author_name="A", toc_page=10, toc_next=20),
                _raw_row(author_id=2, author_name="B", toc_page=10, toc_next=20),
            ]
        )
        work_spans = compute_work_volume_spans(raw)
        wpe = build_work_per_edition(raw, work_spans)
        counts = build_author_form_counts(raw)
        ape = build_author_per_edition(raw, wpe, counts)
        ape = ape.sort_values("author_id").reset_index(drop=True)
        assert ape.loc[0, "author_pages"] == pytest.approx(5.0)
        assert ape.loc[1, "author_pages"] == pytest.approx(5.0)

    def test_single_author_gets_full_span(self):
        raw = pd.DataFrame([_raw_row(toc_page=10, toc_next=30)])
        work_spans = compute_work_volume_spans(raw)
        wpe = build_work_per_edition(raw, work_spans)
        counts = build_author_form_counts(raw)
        ape = build_author_per_edition(raw, wpe, counts)
        assert ape.iloc[0]["author_pages"] == pytest.approx(20.0)


class TestAuthorFormSplit:
    def test_one_author_two_form_work_splits_pages_per_form(self):
        # 10-page work, 1 author, 2 forms → each (author, form) gets 5 pages
        raw = pd.DataFrame(
            [
                _raw_row(toc_page=10, toc_next=20, form_name="poetry"),
                _raw_row(toc_page=10, toc_next=20, form_name="song"),
            ]
        )
        work_spans = compute_work_volume_spans(raw)
        wpe = build_work_per_edition(raw, work_spans)
        counts = build_author_form_counts(raw)
        afe = build_author_form_per_edition(raw, wpe, counts)
        by_form = afe.set_index("form_name")["author_form_pages"]
        assert by_form["poetry"] == pytest.approx(5.0)
        assert by_form["song"] == pytest.approx(5.0)

    def test_two_author_two_form_work_splits_pages_four_ways(self):
        # 20-page work, 2 authors, 2 forms → each (author, form) gets 5 pages
        raw = pd.DataFrame(
            [
                _raw_row(author_id=1, toc_page=10, toc_next=30, form_name="poetry"),
                _raw_row(author_id=1, toc_page=10, toc_next=30, form_name="song"),
                _raw_row(author_id=2, toc_page=10, toc_next=30, form_name="poetry"),
                _raw_row(author_id=2, toc_page=10, toc_next=30, form_name="song"),
            ]
        )
        work_spans = compute_work_volume_spans(raw)
        wpe = build_work_per_edition(raw, work_spans)
        counts = build_author_form_counts(raw)
        afe = build_author_form_per_edition(raw, wpe, counts)
        assert np.allclose(afe["author_form_pages"].to_numpy(dtype=float), 5.0)
        assert len(afe) == 4

    def test_work_without_form_assigned_to_unknown(self):
        raw = pd.DataFrame(
            [_raw_row(form_id=None, form_name=None, toc_page=10, toc_next=20)]
        )
        work_spans = compute_work_volume_spans(raw)
        wpe = build_work_per_edition(raw, work_spans)
        counts = build_author_form_counts(raw)
        afe = build_author_form_per_edition(raw, wpe, counts)
        assert afe.iloc[0]["form_name"] == "Unknown"
        assert afe.iloc[0]["author_form_pages"] == pytest.approx(10.0)


# ── Edition denominators ──────────────────────────────────────────────────────


class TestEditionTotals:
    def test_edition_total_pages_sums_unique_works(self):
        raw = pd.DataFrame(
            [
                # Two distinct works in the same edition
                _raw_row(work_id=1, toc_page=10, toc_next=20),
                _raw_row(work_id=2, toc_page=30, toc_next=50),
            ]
        )
        work_spans = compute_work_volume_spans(raw)
        wpe = build_work_per_edition(raw, work_spans)
        totals = edition_total_pages(wpe)
        assert totals.iloc[0]["edition_pages"] == 30  # 10 + 20

    def test_edition_form_totals_match_edition_total(self):
        # When each work has exactly one form, summing form pages should equal
        # the edition total.
        raw = pd.DataFrame(
            [
                _raw_row(work_id=1, toc_page=10, toc_next=20, form_name="poetry"),
                _raw_row(work_id=2, toc_page=30, toc_next=50, form_name="fiction"),
            ]
        )
        work_spans = compute_work_volume_spans(raw)
        wpe = build_work_per_edition(raw, work_spans)
        counts = build_author_form_counts(raw)
        form_totals = edition_form_total_pages(raw, wpe, counts)
        assert form_totals["form_pages"].sum() == pytest.approx(30.0)


# ── Reselection rate kernel ───────────────────────────────────────────────────


class TestReselectionRates:
    def _make_author_per_edition(self, rows: list[dict]) -> pd.DataFrame:
        """rows: list of {author_id, edition_id, edition_year} with optional author_name."""
        df = pd.DataFrame(rows)
        if "author_name" not in df.columns:
            df["author_name"] = "A" + df["author_id"].astype(str)
        df["author_pages"] = 10.0
        return df

    def test_debut_in_oldest_with_two_of_three_subsequent_yields_two_thirds(self):
        # Editions: 1980, 1990, 2000, 2010
        # Author 1 debuts in 1980 and reappears in 1990, 2010 (not 2000)
        ape = self._make_author_per_edition(
            [
                {"author_id": 1, "edition_id": 1, "edition_year": 1980},
                {"author_id": 1, "edition_id": 2, "edition_year": 1990},
                {"author_id": 1, "edition_id": 4, "edition_year": 2010},
                # Filler author so editions 2-4 exist
                {"author_id": 99, "edition_id": 3, "edition_year": 2000},
            ]
        )
        all_eds = pd.DataFrame(
            [
                {"edition_id": 1, "edition_year": 1980},
                {"edition_id": 2, "edition_year": 1990},
                {"edition_id": 3, "edition_year": 2000},
                {"edition_id": 4, "edition_year": 2010},
            ]
        )
        rates = compute_reselection_rates(ape, all_eds)
        a1 = rates[rates["author_id"] == 1].iloc[0]
        assert a1["subsequent_count"] == 3
        assert a1["reselection_count"] == 2
        assert a1["reselection_prob"] == pytest.approx(2 / 3)

    def test_debut_in_newest_edition_excluded(self):
        ape = self._make_author_per_edition(
            [
                {"author_id": 1, "edition_id": 1, "edition_year": 1980},
                {"author_id": 2, "edition_id": 2, "edition_year": 2010},
            ]
        )
        all_eds = pd.DataFrame(
            [
                {"edition_id": 1, "edition_year": 1980},
                {"edition_id": 2, "edition_year": 2010},
            ]
        )
        rates = compute_reselection_rates(ape, all_eds)
        assert (rates["author_id"] == 2).sum() == 0
        assert (rates["author_id"] == 1).sum() == 1

    def test_always_reselected_author_yields_one(self):
        ape = self._make_author_per_edition(
            [
                {"author_id": 1, "edition_id": 1, "edition_year": 1980},
                {"author_id": 1, "edition_id": 2, "edition_year": 1990},
                {"author_id": 1, "edition_id": 3, "edition_year": 2000},
            ]
        )
        all_eds = pd.DataFrame(
            [
                {"edition_id": 1, "edition_year": 1980},
                {"edition_id": 2, "edition_year": 1990},
                {"edition_id": 3, "edition_year": 2000},
            ]
        )
        rates = compute_reselection_rates(ape, all_eds)
        assert rates.iloc[0]["reselection_prob"] == pytest.approx(1.0)
        assert rates.iloc[0]["n_appearances"] == 3
