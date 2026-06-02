"""
Tests for analysis/predictability_over_time.py

Covers helpers that don't need DB access: page-span computation,
subgroup binning, gini, Mann-Kendall, and the per-edition metric kernel
exercised through compute_all_metrics on synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "predictability"))

from predictability_over_time import (  # noqa: E402
    birth_cohort,
    compute_all_metrics,
    compute_edition_metrics,
    compute_work_volume_spans,
    debut_era,
    gini,
    mann_kendall,
)


# ── compute_work_volume_spans ─────────────────────────────────────────────────


class TestWorkVolumeSpans:
    def _row(self, **kw):
        defaults = {
            "work_id": 1,
            "volume_id": 100,
            "toc_page": 10,
            "toc_next": 20,
            "last_toc_page": 200,
        }
        defaults.update(kw)
        return defaults

    def test_span_is_toc_next_minus_toc_page(self):
        raw = pd.DataFrame([self._row(toc_page=10, toc_next=25)])
        spans = compute_work_volume_spans(raw)
        assert spans.iloc[0]["span"] == 15

    def test_fallback_when_toc_next_null(self):
        raw = pd.DataFrame([self._row(toc_page=180, toc_next=None, last_toc_page=200)])
        spans = compute_work_volume_spans(raw)
        # 200 - 180 + 1 = 21
        assert spans.iloc[0]["span"] == 21

    def test_negative_span_clipped_to_zero(self):
        raw = pd.DataFrame([self._row(toc_page=30, toc_next=20)])
        spans = compute_work_volume_spans(raw)
        assert spans.iloc[0]["span"] == 0

    def test_deduplicates_multi_author_rows(self):
        # Same (work, volume) appears twice (two authors) — span computed once.
        rows = [self._row(work_id=1, volume_id=100, toc_page=10, toc_next=20)] * 2
        spans = compute_work_volume_spans(pd.DataFrame(rows))
        assert len(spans) == 1
        assert spans.iloc[0]["span"] == 10


# ── Subgroup binning ──────────────────────────────────────────────────────────


class TestBinning:
    def test_birth_cohort_boundaries(self):
        assert birth_cohort(1850) == "≤1900"
        assert birth_cohort(1900) == "≤1900"
        assert birth_cohort(1901) == "1901-1950"
        assert birth_cohort(1950) == "1901-1950"
        assert birth_cohort(1951) == ">1950"
        assert birth_cohort(None) == "unknown"
        assert birth_cohort(float("nan")) == "unknown"

    def test_debut_era_boundaries(self):
        assert debut_era(1929) == "pre-1970"
        assert debut_era(1969) == "pre-1970"
        assert debut_era(1970) == "1970-1995"
        assert debut_era(1995) == "1970-1995"
        assert debut_era(1996) == "post-1995"
        assert debut_era(2025) == "post-1995"


# ── Gini ──────────────────────────────────────────────────────────────────────


class TestGini:
    def test_uniform_distribution_is_zero(self):
        assert gini(np.array([5.0, 5.0, 5.0, 5.0])) == 0.0

    def test_empty_or_all_zero_is_zero(self):
        assert gini(np.array([])) == 0.0
        assert gini(np.array([0.0, 0.0])) == 0.0

    def test_highly_concentrated_is_close_to_one(self):
        v = np.zeros(100)
        v[0] = 100.0
        assert gini(v) > 0.8


# ── Mann-Kendall ──────────────────────────────────────────────────────────────


class TestMannKendall:
    def test_strictly_increasing_series_is_significant(self):
        s, p = mann_kendall(np.arange(20, dtype=float))
        assert s > 0
        assert p < 0.001

    def test_strictly_decreasing_series_is_significant(self):
        s, p = mann_kendall(np.arange(20, 0, -1, dtype=float))
        assert s < 0
        assert p < 0.001

    def test_flat_series_is_not_significant(self):
        rng = np.random.default_rng(0)
        # Random noise around a constant mean — should be non-significant
        s, p = mann_kendall(rng.normal(0, 1, size=100))
        assert p > 0.01


# ── compute_edition_metrics (the kernel) ──────────────────────────────────────


def _make_author_long(rows: list[dict]) -> pd.DataFrame:
    """rows: list of {author_id, edition_id, year, page_w (default 10)}."""
    df = pd.DataFrame(rows)
    if "page_w" not in df.columns:
        df["page_w"] = 10
    if "slot_w" not in df.columns:
        df["slot_w"] = 1
    return df


class TestEditionMetrics:
    def test_all_debut_edition_has_zero_repeat(self):
        """E with entities none of whom appeared before → fraction_repeat = 0."""
        long = _make_author_long(
            [
                # Prior edition: authors 1..10
                *[
                    {"author_id": i, "edition_id": 1, "year": 1990}
                    for i in range(1, 11)
                ],
                # Target edition: authors 100..109 (disjoint)
                *[
                    {"author_id": i, "edition_id": 2, "year": 2000}
                    for i in range(100, 110)
                ],
            ]
        )
        m = compute_edition_metrics(long, "author_id", 2000, 2)
        assert m is not None
        assert m["fraction_repeat_slot"] == 0.0
        assert m["fraction_frequent_canon_slot"] == 0.0
        assert m["recall_at_k"] == 0.0

    def test_full_recall_when_e_equals_prior_canon(self):
        """E is exactly the top-K of prior canon → recall_at_k = 1.0."""
        long = _make_author_long(
            [
                # Author 1 in 3 prior editions, author 2 in 2, author 3 in 1, 4..6 in 1 each
                *[
                    {"author_id": 1, "edition_id": e, "year": 1980 + e}
                    for e in range(1, 4)
                ],
                *[
                    {"author_id": 2, "edition_id": e, "year": 1980 + e}
                    for e in range(1, 3)
                ],
                *[{"author_id": i, "edition_id": 1, "year": 1981} for i in range(3, 7)],
                # Target edition (year 1990): authors 1, 2 — the top-2 by prior_count
                {"author_id": 1, "edition_id": 99, "year": 1990},
                {"author_id": 2, "edition_id": 99, "year": 1990},
                # Need ≥ MIN_SUBGROUP_SIZE (5) in target — add more from prior canon
                {"author_id": 3, "edition_id": 99, "year": 1990},
                {"author_id": 4, "edition_id": 99, "year": 1990},
                {"author_id": 5, "edition_id": 99, "year": 1990},
            ]
        )
        m = compute_edition_metrics(long, "author_id", 1990, 99)
        assert m is not None
        # Top-5 by prior_count includes 1, 2, then ties among 3..6; E = {1..5} ⊂ top-K
        assert m["fraction_repeat_slot"] == 1.0
        assert m["recall_at_k"] >= 0.8  # 4 or 5 of top-K hit

    def test_disjoint_edition_yields_zero_metrics(self):
        long = _make_author_long(
            [
                *[
                    {"author_id": i, "edition_id": 1, "year": 1980}
                    for i in range(1, 11)
                ],
                *[
                    {"author_id": 100 + i, "edition_id": 2, "year": 1990}
                    for i in range(5)
                ],
            ]
        )
        m = compute_edition_metrics(long, "author_id", 1990, 2)
        assert m is not None
        assert m["recall_at_k"] == 0.0
        assert m["fraction_repeat_slot"] == 0.0

    def test_returns_none_when_e_smaller_than_min_subgroup(self):
        long = _make_author_long(
            [
                *[
                    {"author_id": i, "edition_id": 1, "year": 1980}
                    for i in range(1, 11)
                ],
                *[{"author_id": i, "edition_id": 2, "year": 1990} for i in range(1, 4)],
            ]
        )
        m = compute_edition_metrics(long, "author_id", 1990, 2)
        assert m is None

    def test_perfectly_predictable_edition_has_high_auc(self):
        """Construct a prior pool where in_E is a step function of prior_count."""
        rows = []
        # 30 prior editions; we'll seed authors with varying counts then
        # let everyone with prior_count >= 5 appear in E.
        for prior_count in range(1, 11):
            n_authors = 10
            for k in range(n_authors):
                aid = prior_count * 1000 + k
                for ed in range(1, prior_count + 1):
                    rows.append(
                        {
                            "author_id": aid,
                            "edition_id": ed,
                            "year": 1900 + ed,
                        }
                    )
                # Author appears in E iff prior_count >= 5
                if prior_count >= 5:
                    rows.append(
                        {
                            "author_id": aid,
                            "edition_id": 999,
                            "year": 2000,
                        }
                    )
        long = _make_author_long(rows)
        m = compute_edition_metrics(long, "author_id", 2000, 999)
        assert m is not None
        # Step function is highly separable → AUC should be near 1.0
        assert m["logit_auc"] > 0.9

    def test_random_edition_has_auc_near_half(self):
        rng = np.random.default_rng(42)
        rows = []
        n_authors = 200
        for aid in range(n_authors):
            prior_count = rng.integers(1, 10)
            for ed in range(prior_count):
                rows.append({"author_id": aid, "edition_id": ed, "year": 1900 + ed})
        # Randomly include half in E, regardless of prior_count
        in_e = rng.choice(n_authors, size=n_authors // 2, replace=False)
        for aid in in_e:
            rows.append({"author_id": int(aid), "edition_id": 999, "year": 2000})
        long = _make_author_long(rows)
        m = compute_edition_metrics(long, "author_id", 2000, 999)
        assert m is not None
        assert 0.35 < m["logit_auc"] < 0.65

    def test_page_weighted_equals_slot_weighted_with_equal_spans(self):
        """All entities have the same page_w → fraction_repeat_page == fraction_repeat_slot."""
        long = _make_author_long(
            [
                *[
                    {"author_id": i, "edition_id": 1, "year": 1980, "page_w": 10}
                    for i in range(1, 11)
                ],
                *[
                    {"author_id": i, "edition_id": 2, "year": 1990, "page_w": 10}
                    for i in [1, 2, 3, 100, 101, 102]
                ],
            ]
        )
        m = compute_edition_metrics(long, "author_id", 1990, 2)
        assert m is not None
        assert abs(m["fraction_repeat_page"] - m["fraction_repeat_slot"]) < 1e-9


# ── compute_all_metrics (excludes earliest year) ──────────────────────────────


class TestAllMetrics:
    def test_excludes_earliest_year(self):
        long = _make_author_long(
            [
                *[
                    {"author_id": i, "edition_id": 1, "year": 1929}
                    for i in range(1, 21)
                ],
                *[
                    {"author_id": i, "edition_id": 2, "year": 1990}
                    for i in range(1, 21)
                ],
            ]
        )
        # All-subgroup tagging
        long["sg_gender"] = "unknown"
        long["sg_birth_cohort"] = "unknown"
        long["sg_debut_era"] = "pre-1970"
        out = compute_all_metrics(long, "authors")
        years = set(out["year"].unique())
        assert 1929 not in years
        assert 1990 in years
