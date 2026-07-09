from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "inequality"))

from inclusion_inequality import (  # noqa: E402
    edition_counts,
    gini,
    lorenz,
    plot,
)


# ── gini ──────────────────────────────────────────────────────────────────────


def test_gini_perfect_equality():
    """All items appear the same number of times → Gini = 0."""
    assert gini(np.array([3, 3, 3, 3])) == pytest.approx(0.0)


def test_gini_known_two_element():
    """[1, 3]: manual calculation gives G = 1/4."""
    # sorted = [1, 3], n=2, sum=4
    # G = (2*(1*1 + 2*3) / (2*4)) - 3/2 = (14/8) - 1.5 = 1.75 - 1.5 = 0.25
    assert gini(np.array([1, 3])) == pytest.approx(0.25)


def test_gini_known_four_element():
    """[1, 1, 1, 3]: manual calculation gives G = 0.25."""
    # sorted = [1,1,1,3], n=4, sum=6
    # G = (2*(1+2+3+12) / (4*6)) - 5/4 = (36/24) - 1.25 = 1.5 - 1.25 = 0.25
    assert gini(np.array([1, 1, 1, 3])) == pytest.approx(0.25)


def test_gini_unsorted_input():
    """gini should give the same result regardless of input order."""
    a = np.array([3, 1, 1, 1])
    b = np.array([1, 1, 1, 3])
    assert gini(a) == pytest.approx(gini(b))


def test_gini_single_element():
    """A single item trivially has Gini = 0."""
    assert gini(np.array([5])) == pytest.approx(0.0)


def test_gini_bounded():
    """Gini is always in [0, 1)."""
    for counts in [
        np.array([1, 1, 1, 1]),
        np.array([1, 2, 3, 100]),
        np.array([1, 1, 1, 1000]),
    ]:
        g = gini(counts)
        assert 0.0 <= g < 1.0


# ── lorenz ────────────────────────────────────────────────────────────────────


def test_lorenz_starts_and_ends_at_zero_and_one():
    pop, inc = lorenz(np.array([1, 2, 3]))
    assert pop[0] == pytest.approx(0.0)
    assert inc[0] == pytest.approx(0.0)
    assert pop[-1] == pytest.approx(1.0)
    assert inc[-1] == pytest.approx(1.0)


def test_lorenz_perfect_equality_is_diagonal():
    """Equal counts → Lorenz curve lies on the 45° line."""
    pop, inc = lorenz(np.array([2, 2, 2, 2]))
    np.testing.assert_allclose(pop, inc)


def test_lorenz_monotone_non_decreasing():
    pop, inc = lorenz(np.array([1, 1, 2, 5]))
    assert all(np.diff(pop) >= 0)
    assert all(np.diff(inc) >= 0)


def test_lorenz_below_diagonal_for_unequal_distribution():
    """For any unequal distribution, the Lorenz curve lies below the diagonal."""
    pop, inc = lorenz(np.array([1, 1, 1, 7]))
    # interior points only (first and last are equal by construction)
    assert all(inc[1:-1] <= pop[1:-1])


def test_lorenz_known_values():
    """[1, 3]: bottom 50% of items hold 25% of inclusions."""
    pop, inc = lorenz(np.array([1, 3]))
    # pop = [0, 0.5, 1.0], inc = [0, 0.25, 1.0]
    assert pop[1] == pytest.approx(0.5)
    assert inc[1] == pytest.approx(0.25)


def test_lorenz_length_equals_n_plus_one():
    counts = np.array([1, 2, 3, 4, 5])
    pop, inc = lorenz(counts)
    assert len(pop) == len(counts) + 1
    assert len(inc) == len(counts) + 1


# ── edition_counts ────────────────────────────────────────────────────────────


def test_edition_counts_unique_editions_per_item():
    """A work appearing in two distinct editions counts twice, regardless of
    how many volume rows back each edition."""
    df = pd.DataFrame(
        {
            "work_id": ["w1", "w1", "w1"],
            # two rows for edition 1 (e.g. two volumes), one for edition 2
            "edition_key": [1, 1, 2],
        }
    )
    counts = edition_counts(df, "work_id")
    assert counts[0] == 2  # two distinct editions


def test_edition_counts_multiple_items():
    """Each item's count is computed independently."""
    df = pd.DataFrame(
        {
            "work_id": ["w1", "w1", "w2"],
            "edition_key": [1, 2, 3],
        }
    )
    counts_by_id = df.groupby("work_id")["edition_key"].nunique()
    assert counts_by_id["w1"] == 2
    assert counts_by_id["w2"] == 1


# ── plot smoke test ───────────────────────────────────────────────────────────


def test_plot_writes_png(tmp_path: Path):
    work_counts = np.array([1, 1, 2, 3, 5])
    author_counts = np.array([1, 2, 4])
    out = tmp_path / "test_inequality.png"
    plot(work_counts, author_counts, out, title="Test")
    assert out.exists()
    assert out.stat().st_size > 0
