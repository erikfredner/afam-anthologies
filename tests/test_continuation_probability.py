from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "reselection"))

from continuation_probability import continuation_stats  # noqa: E402


def test_continuation_stats_point_estimates():
    stats = continuation_stats(np.array([1, 1, 2, 3]), min_n=1)

    assert stats["k"].tolist() == [1, 2]
    assert stats["n_at_least_k"].tolist() == [4, 2]
    assert stats["n_at_least_k_plus_1"].tolist() == [2, 1]
    assert stats["p_continue"].tolist() == pytest.approx([0.5, 0.5])


def test_continuation_stats_ci_brackets_point_estimate():
    stats = continuation_stats(np.array([1, 1, 1, 2, 2, 3, 4, 4]), min_n=1)

    assert (stats["ci_lo"] <= stats["p_continue"]).all()
    assert (stats["p_continue"] <= stats["ci_hi"]).all()
    assert (stats["ci_lo"] >= 0).all()
    assert (stats["ci_hi"] <= 1).all()


def test_continuation_stats_respects_denominator_floor():
    # N(>=1)=10, N(>=2)=3: with min_n=5 only k=1 survives.
    counts = np.array([1] * 7 + [2, 3, 5])

    stats = continuation_stats(counts, min_n=5)

    assert stats["k"].tolist() == [1]
    assert stats.loc[0, "p_continue"] == pytest.approx(0.3)
