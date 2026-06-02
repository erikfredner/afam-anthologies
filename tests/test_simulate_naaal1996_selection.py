"""
Tests for analysis/simulate_naaal1996_selection.py

Covers:
  - assign_edition_key: series-based, volume-stripped, and multi-volume deduplication
  - build_work_frame: prior_count, in_naaal, only_root filter, zero-prior exclusion
  - build_author_frame: prior_count, in_naaal, multi-author expansion, zero-prior exclusion
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
    NAAAL_KEY,
    assign_edition_key,
    build_author_frame,
    build_work_frame,
    fit_logit_safe,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Minimal anthology DataFrame with edition_key pre-set.

    Fills missing required columns with empty strings so callers only need to
    specify the columns relevant to each test.
    """
    defaults = {
        "work_id": "",
        "work_title": "",
        "parent_work_id": "",
        "parent_work_title": "",
        "author_ids": "",
        "author_names": "",
        "anthology_id": "",
        "anthology_title": "",
        "series_id": "",
        "anthology_edition": "",
        "anthology_volume": "",
        "anthology_publication_year": 1990,
        "edition_key": "",
    }
    df = pd.DataFrame(rows)
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    str_cols = [c for c, v in defaults.items() if isinstance(v, str)]
    df[str_cols] = df[str_cols].astype(str)
    return df


# ── assign_edition_key ────────────────────────────────────────────────────────


class TestAssignEditionKey:
    def _base(self, **kwargs) -> pd.DataFrame:
        row = {
            "work_id": "w1",
            "anthology_id": "a1",
            "anthology_title": "Some Title",
            "series_id": "",
            "anthology_edition": "1",
            "anthology_volume": "",
            "anthology_publication_year": "1990",
            **{k: str(v) for k, v in kwargs.items()},
        }
        return pd.DataFrame([row])

    def test_series_key(self):
        df = self._base(series_id="3", anthology_edition="1")
        result = assign_edition_key(df)
        assert result.iloc[0]["edition_key"] == "3|1"

    def test_volume_stripped_key(self):
        df = self._base(
            anthology_title="Black Voices, Vol. 2",
            anthology_edition="1",
            anthology_volume="2",
        )
        result = assign_edition_key(df)
        assert result.iloc[0]["edition_key"] == "Black Voices|1"

    def test_multi_volume_same_edition_key(self):
        """Both volumes of the same edition must collapse to one key."""
        rows = [
            {
                "work_id": "w1",
                "anthology_id": "v1",
                "anthology_title": "Big Anthology, Vol. 1",
                "series_id": "",
                "anthology_edition": "2",
                "anthology_volume": "1",
                "anthology_publication_year": "1990",
            },
            {
                "work_id": "w2",
                "anthology_id": "v2",
                "anthology_title": "Big Anthology, Vol. 2",
                "series_id": "",
                "anthology_edition": "2",
                "anthology_volume": "2",
                "anthology_publication_year": "1990",
            },
        ]
        result = assign_edition_key(pd.DataFrame(rows))
        keys = result["edition_key"].unique()
        assert len(keys) == 1, f"Expected 1 unique key, got {list(keys)}"


# ── build_work_frame ──────────────────────────────────────────────────────────


class TestBuildWorkFrame:
    def _simple_df(self) -> pd.DataFrame:
        """
        anthA: w1, w2
        anthB: w1
        NAAAL: w1, w2  (w3 is NAAAL-only → excluded by prior_count >= 1 filter)
        """
        return make_df(
            [
                dict(work_id="w1", anthology_id="a1", edition_key="anthA"),
                dict(work_id="w2", anthology_id="a1", edition_key="anthA"),
                dict(work_id="w1", anthology_id="b1", edition_key="anthB"),
                dict(work_id="w1", anthology_id="naaal", edition_key=NAAAL_KEY),
                dict(work_id="w2", anthology_id="naaal", edition_key=NAAAL_KEY),
                dict(work_id="w3", anthology_id="naaal", edition_key=NAAAL_KEY),
            ]
        )

    def test_prior_count_correct(self):
        frame = build_work_frame(self._simple_df(), only_root=False)
        counts = frame.set_index("work_id")["prior_count"]
        assert counts["w1"] == 2  # in anthA + anthB
        assert counts["w2"] == 1  # in anthA only

    def test_in_naaal_correct(self):
        frame = build_work_frame(self._simple_df(), only_root=False)
        naaal = frame.set_index("work_id")["in_naaal"]
        assert naaal["w1"] == 1
        assert naaal["w2"] == 1

    def test_naaal_not_counted_in_prior(self):
        """NAAAL appearances must not inflate prior_count."""
        rows = [
            dict(work_id="w1", anthology_id="a1", edition_key="anthA"),
            dict(work_id="w1", anthology_id="a2", edition_key="anthB"),
            dict(work_id="w1", anthology_id="naaal", edition_key=NAAAL_KEY),
        ]
        frame = build_work_frame(make_df(rows), only_root=False)
        assert frame.iloc[0]["prior_count"] == 2

    def test_multi_volume_counted_once(self):
        """Two rows sharing the same edition_key count as 1 prior appearance."""
        rows = [
            dict(work_id="w1", anthology_id="v1", edition_key="anthA|1"),
            dict(work_id="w1", anthology_id="v2", edition_key="anthA|1"),
            dict(work_id="w1", anthology_id="naaal", edition_key=NAAAL_KEY),
        ]
        frame = build_work_frame(make_df(rows), only_root=False)
        assert frame.iloc[0]["prior_count"] == 1

    def test_only_root_filter_excludes_excerpts(self):
        rows = [
            dict(
                work_id="root",
                parent_work_id="",
                parent_work_title="",
                anthology_id="a1",
                edition_key="anthA",
            ),
            dict(
                work_id="excerpt",
                parent_work_id="root",
                parent_work_title="Root",
                anthology_id="a1",
                edition_key="anthA",
            ),
            dict(
                work_id="root",
                parent_work_id="",
                parent_work_title="",
                anthology_id="naaal",
                edition_key=NAAAL_KEY,
            ),
        ]
        frame = build_work_frame(make_df(rows), only_root=True)
        assert "excerpt" not in frame["work_id"].values
        assert "root" in frame["work_id"].values


# ── build_author_frame ────────────────────────────────────────────────────────


class TestBuildAuthorFrame:
    def test_prior_count_and_in_naaal(self):
        """Author in 2 non-NAAAL editions + NAAAL → prior_count=2, in_naaal=1.
        Author in NAAAL only → excluded (prior_count=0)."""
        rows = [
            dict(
                work_id="w1",
                author_ids="a1",
                author_names="Alice",
                anthology_id="x1",
                edition_key="ed1",
            ),
            dict(
                work_id="w2",
                author_ids="a1",
                author_names="Alice",
                anthology_id="x2",
                edition_key="ed2",
            ),
            dict(
                work_id="w3",
                author_ids="a1",
                author_names="Alice",
                anthology_id="naaal",
                edition_key=NAAAL_KEY,
            ),
            dict(
                work_id="w4",
                author_ids="a2",
                author_names="Bob",
                anthology_id="naaal",
                edition_key=NAAAL_KEY,
            ),  # NAAAL-only → excluded
        ]
        frame = build_author_frame(make_df(rows))
        assert set(frame["author_id"].tolist()) == {"a1"}
        alice = frame[frame["author_id"] == "a1"].iloc[0]
        assert alice["prior_count"] == 2
        assert alice["in_naaal"] == 1

    def test_comma_separated_author_ids_expanded(self):
        """A work with two comma-separated author_ids yields one row per author."""
        rows = [
            dict(
                work_id="w1",
                author_ids="a1, a2",
                author_names="Alice; Bob",
                anthology_id="x1",
                edition_key="ed1",
            ),
            dict(
                work_id="w1",
                author_ids="a1, a2",
                author_names="Alice; Bob",
                anthology_id="naaal",
                edition_key=NAAAL_KEY,
            ),
        ]
        frame = build_author_frame(make_df(rows))
        assert set(frame["author_id"].tolist()) == {"a1", "a2"}
        assert all(frame["prior_count"] == 1)
        assert all(frame["in_naaal"] == 1)


# ── TestTautologicalZeros ─────────────────────────────────────────────────────


class TestTautologicalZeros:
    """Regression guards for the pre-fix bug.

    A work (or author) with prior_count == 0 is necessarily in_naaal == 1
    because it only exists in the dataset via NAAAL. Including such items inflates
    the intercept and can invert the regression slope. The fix is prior_count >= 1.
    """

    def test_zero_prior_works_excluded(self):
        """Works appearing only in NAAAL must not appear in the frame."""
        rows = [
            dict(work_id="classic", anthology_id="a1", edition_key="ed1"),
            dict(work_id="classic", anthology_id="naaal", edition_key=NAAAL_KEY),
            dict(work_id="debut", anthology_id="naaal", edition_key=NAAAL_KEY),
        ]
        frame = build_work_frame(make_df(rows), only_root=False)
        assert "debut" not in frame["work_id"].values
        assert "classic" in frame["work_id"].values

    def test_negative_slope_without_filter_positive_with_filter(self):
        """
        Demonstrate the tautological-zeros bug with deterministic data.

        Structure mirrors the real dataset:
          - 100 items at prior_count=0, ALL in NAAAL (tautological positives)
          - 1000 items at prior_count=1, only 5% in NAAAL
          - 100 items at prior_count=2, 50% in NAAAL

        Mean-x analysis proves slope direction:
          Without filter: mean_x(y=1)=0.75 < mean_x(y=0)=1.05  → negative slope
          With filter:    mean_x(y=1)=1.50 > mean_x(y=0)=1.05  → positive slope

        The 1000 items at prior_count=1 (mostly zeros) overwhelm the 100
        tautological positives at prior_count=0, dragging the overall slope
        negative — exactly the bug that appeared in the real data.
        """
        rows = []
        # Zero-prior items: in NAAAL by construction
        rows.extend([{"prior_count": 0, "in_naaal": 1}] * 100)
        # Many items with one prior appearance, low inclusion rate
        rows.extend([{"prior_count": 1, "in_naaal": 1}] * 50)
        rows.extend([{"prior_count": 1, "in_naaal": 0}] * 950)
        # Fewer items with two prior appearances, higher inclusion rate
        rows.extend([{"prior_count": 2, "in_naaal": 1}] * 50)
        rows.extend([{"prior_count": 2, "in_naaal": 0}] * 50)

        df = pd.DataFrame(rows)
        y = df["in_naaal"]
        x = df["prior_count"]

        # Without filter: negative slope (the bug)
        result_with_zeros = fit_logit_safe(y, x)
        assert result_with_zeros.params["prior_count"] < 0, (
            "Expected negative slope when tautological zeros are included"
        )

        # With filter: positive slope (the fix)
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
