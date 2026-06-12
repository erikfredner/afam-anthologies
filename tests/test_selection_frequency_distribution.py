from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "inequality"))

from selection_frequency_distribution import edition_counts, exact_k_stats  # noqa: E402


def test_exact_k_stats_counts_each_entity_once():
    ks, ns, pcts = exact_k_stats(np.array([1, 1, 2, 3]))

    assert ks.tolist() == [1, 2, 3]
    assert ns.tolist() == [2, 1, 1]
    assert pcts.tolist() == [50.0, 25.0, 25.0]


def test_exact_k_stats_pcts_sum_to_100():
    counts = np.array([1, 1, 1, 2, 4, 4, 7])
    _, ns, pcts = exact_k_stats(counts)

    assert ns.sum() == len(counts)
    assert pcts.sum() == pytest.approx(100.0)


def test_exact_k_stats_includes_empty_intermediate_bins():
    ks, ns, _ = exact_k_stats(np.array([1, 3]))

    assert ks.tolist() == [1, 2, 3]
    assert ns.tolist() == [1, 0, 1]


def test_edition_counts_deduplicates_and_drops_missing_ids():
    df = pd.DataFrame(
        {
            "work_id": ["w1", "w1", "w1", "w2", None],
            "edition_id": [1, 1, 2, 1, 3],
        }
    )

    counts = edition_counts(df, "work_id")

    assert sorted(counts.tolist()) == [1, 2]
