"""
Tests for analysis/predictability/simulate_naaal1996_selection.py

Covers (against the DB-shaped wide frame the migrated script consumes):
  - build_work_frame: prior_count, in_naaal, only_root filter, zero-prior exclusion
  - build_author_frame: prior_count, in_naaal, multi-author rows, zero-prior exclusion
  - fit_logit_safe: slope direction recovery with synthetic data
  - TestTautologicalZeros: regression guards showing the pre-fix negative-slope bug
    and confirming the fix restores a positive slope
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "predictability"))

from simulate_naaal1996_selection import (  # noqa: E402
    build_author_frame,
    build_work_frame,
    fit_logit_safe,
)

# A synthetic target edition id (stands in for NAAAL ed.1).
TARGET_ID = 999


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Minimal wide (work × author × edition) frame.

    Columns mirror queries/works-authors-per-afam-edition.sql; missing columns
    are filled so callers only specify what each test needs.
    """
    defaults = {
        "work_id": 0,
        "work_title": "",
        "parent_id": np.nan,
        "parent_work_title": np.nan,
        "edition_id": 0,
        "anthology_publication_year": 1990,
        "series_id": np.nan,
        "edition_number": np.nan,
        "author_id": np.nan,
        "author_name": np.nan,
    }
    df = pd.DataFrame(rows)
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


# ── build_work_frame ──────────────────────────────────────────────────────────


class TestBuildWorkFrame:
    def _simple_df(self) -> pd.DataFrame:
        """
        edition 1: w1, w2
        edition 2: w1
        target (999): w1, w2  (w3 is target-only → excluded by prior_count >= 1)
        """
        return make_df(
            [
                dict(work_id=1, edition_id=1),
                dict(work_id=2, edition_id=1),
                dict(work_id=1, edition_id=2),
                dict(work_id=1, edition_id=TARGET_ID),
                dict(work_id=2, edition_id=TARGET_ID),
                dict(work_id=3, edition_id=TARGET_ID),
            ]
        )

    def test_prior_count_correct(self):
        frame = build_work_frame(
            self._simple_df(), only_root=False, target_id=TARGET_ID
        )
        counts = frame.set_index("work_id")["prior_count"]
        assert counts[1] == 2  # editions 1 + 2
        assert counts[2] == 1  # edition 1 only

    def test_in_naaal_correct(self):
        frame = build_work_frame(
            self._simple_df(), only_root=False, target_id=TARGET_ID
        )
        naaal = frame.set_index("work_id")["in_naaal"]
        assert naaal[1] == 1
        assert naaal[2] == 1

    def test_naaal_not_counted_in_prior(self):
        """Target-edition appearances must not inflate prior_count."""
        rows = [
            dict(work_id=1, edition_id=1),
            dict(work_id=1, edition_id=2),
            dict(work_id=1, edition_id=TARGET_ID),
        ]
        frame = build_work_frame(make_df(rows), only_root=False, target_id=TARGET_ID)
        assert frame.iloc[0]["prior_count"] == 2

    def test_only_root_filter_excludes_excerpts(self):
        rows = [
            dict(work_id=10, parent_id=np.nan, edition_id=1),
            dict(work_id=11, parent_id=10, edition_id=1),
            dict(work_id=10, parent_id=np.nan, edition_id=TARGET_ID),
        ]
        frame = build_work_frame(make_df(rows), only_root=True, target_id=TARGET_ID)
        assert 11 not in frame["work_id"].values
        assert 10 in frame["work_id"].values


# ── build_author_frame ────────────────────────────────────────────────────────


class TestBuildAuthorFrame:
    def test_prior_count_and_in_naaal(self):
        """Author in 2 prior editions + target → prior_count=2, in_naaal=1.
        Author in target only → excluded (prior_count=0)."""
        rows = [
            dict(work_id=1, author_id=1, author_name="Alice", edition_id=1),
            dict(work_id=2, author_id=1, author_name="Alice", edition_id=2),
            dict(work_id=3, author_id=1, author_name="Alice", edition_id=TARGET_ID),
            dict(work_id=4, author_id=2, author_name="Bob", edition_id=TARGET_ID),
        ]
        frame = build_author_frame(make_df(rows), target_id=TARGET_ID)
        assert set(frame["author_id"].tolist()) == {1}
        alice = frame[frame["author_id"] == 1].iloc[0]
        assert alice["prior_count"] == 2
        assert alice["in_naaal"] == 1

    def test_multi_author_rows_each_counted(self):
        """A work with two authors yields one row per author in the wide frame."""
        rows = [
            dict(work_id=1, author_id=1, author_name="Alice", edition_id=1),
            dict(work_id=1, author_id=2, author_name="Bob", edition_id=1),
            dict(work_id=1, author_id=1, author_name="Alice", edition_id=TARGET_ID),
            dict(work_id=1, author_id=2, author_name="Bob", edition_id=TARGET_ID),
        ]
        frame = build_author_frame(make_df(rows), target_id=TARGET_ID)
        assert set(frame["author_id"].tolist()) == {1, 2}
        assert all(frame["prior_count"] == 1)
        assert all(frame["in_naaal"] == 1)


# ── TestTautologicalZeros ─────────────────────────────────────────────────────


class TestTautologicalZeros:
    """Regression guards for the pre-fix bug.

    A work with prior_count == 0 is necessarily in_naaal == 1 because it only
    exists in the dataset via the target edition. Including such items inflates
    the intercept and can invert the regression slope. The fix is prior_count >= 1.
    """

    def test_zero_prior_works_excluded(self):
        """Works appearing only in the target edition must not appear."""
        rows = [
            dict(work_id=1, edition_id=1),
            dict(work_id=1, edition_id=TARGET_ID),
            dict(work_id=2, edition_id=TARGET_ID),
        ]
        frame = build_work_frame(make_df(rows), only_root=False, target_id=TARGET_ID)
        assert 2 not in frame["work_id"].values
        assert 1 in frame["work_id"].values

    def test_negative_slope_without_filter_positive_with_filter(self):
        """
        Demonstrate the tautological-zeros bug with deterministic data.

        Without the prior_count >= 1 filter, the 1000 items at prior_count=1
        (mostly zeros) overwhelm the 100 tautological positives at prior_count=0,
        dragging the overall slope negative — exactly the bug in the real data.
        """
        rows = []
        rows.extend([{"prior_count": 0, "in_naaal": 1}] * 100)
        rows.extend([{"prior_count": 1, "in_naaal": 1}] * 50)
        rows.extend([{"prior_count": 1, "in_naaal": 0}] * 950)
        rows.extend([{"prior_count": 2, "in_naaal": 1}] * 50)
        rows.extend([{"prior_count": 2, "in_naaal": 0}] * 50)

        df = pd.DataFrame(rows)
        y = df["in_naaal"]
        x = df["prior_count"]

        result_with_zeros = fit_logit_safe(y, x)
        assert result_with_zeros.params["prior_count"] < 0, (
            "Expected negative slope when tautological zeros are included"
        )

        mask = df["prior_count"] >= 1
        result_filtered = fit_logit_safe(y[mask], x[mask])
        assert result_filtered.params["prior_count"] > 0, (
            "Expected positive slope after filtering out zero-prior items"
        )


# ── fit_logit_safe ────────────────────────────────────────────────────────────


class TestFitLogitSafe:
    def test_recovers_positive_slope(self):
        """Synthetic data with true logit = −2 + 0.8·x → fitted slope > 0."""
        rng = np.random.default_rng(0)
        x_arr = rng.integers(1, 10, size=500)
        log_odds = -2 + 0.8 * x_arr
        p = 1 / (1 + np.exp(-log_odds))
        y = pd.Series(rng.binomial(1, p).astype(float))
        x = pd.Series(x_arr.astype(float))
        result = fit_logit_safe(y, x)
        assert result.params["prior_count"] > 0

    def test_recovers_negative_slope(self):
        """Synthetic data with true logit = 1 − 0.8·x → fitted slope < 0."""
        rng = np.random.default_rng(1)
        x_arr = rng.integers(1, 10, size=500)
        log_odds = 1 - 0.8 * x_arr
        p = 1 / (1 + np.exp(-log_odds))
        y = pd.Series(rng.binomial(1, p).astype(float))
        x = pd.Series(x_arr.astype(float))
        result = fit_logit_safe(y, x)
        assert result.params["prior_count"] < 0

    def test_perfect_separation_does_not_crash(self):
        """Complete separation triggers the regularised fallback — must not raise."""
        x = pd.Series([1] * 100 + [5] * 100, dtype=float)
        y = pd.Series([0] * 100 + [1] * 100, dtype=float)
        result = fit_logit_safe(y, x)  # should not raise
        assert result.params["prior_count"] > 0
