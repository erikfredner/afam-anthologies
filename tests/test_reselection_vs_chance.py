from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "reselection"))

from reselection_vs_chance import (  # noqa: E402
    compute_per_edition_stats,
    expand_authors,
    precompute_birth_years,
    run_stats,
    sorted_edition_ids,
)


def make_raw(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "edition_id": None,
        "anthology_publication_year": None,
        "work_id": None,
        "author_id": None,
        "author_birth_year": 1900,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_compute_per_edition_stats_uses_seen_sets_and_birth_year_pool():
    raw = make_raw(
        [
            {
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "work_id": "w1",
                "author_id": "a1",
                "author_birth_year": 1900,
            },
            {
                "edition_id": 1,
                "anthology_publication_year": 2000,
                "work_id": "w2",
                "author_id": "a2",
                "author_birth_year": 1910,
            },
            {
                "edition_id": 2,
                "anthology_publication_year": 2010,
                "work_id": "w1",
                "author_id": "a1",
                "author_birth_year": 1900,
            },
            {
                "edition_id": 2,
                "anthology_publication_year": 2010,
                "work_id": "w3",
                "author_id": "a3",
                "author_birth_year": 1920,
            },
            {
                "edition_id": 3,
                "anthology_publication_year": 2020,
                "work_id": "w2",
                "author_id": "a2",
                "author_birth_year": 1910,
            },
            {
                "edition_id": 3,
                "anthology_publication_year": 2020,
                "work_id": "w4",
                "author_id": "a4",
                "author_birth_year": 1980,
            },
        ]
    )

    work_max_by, author_by = precompute_birth_years(raw)
    stats = compute_per_edition_stats(
        raw,
        expand_authors(raw),
        sorted_edition_ids(raw),
        work_max_by,
        author_by,
        window=5,
    ).set_index("edition_id")

    assert stats.loc[2, "w_resel"] == 1
    assert stats.loc[2, "a_resel"] == 1
    assert stats.loc[2, "pool_works"] == 3
    assert stats.loc[3, "pool_authors"] == 4


def test_run_stats_returns_standard_test_p_values_and_handles_zero_slots():
    stats = pd.DataFrame(
        [
            {
                "w_slots": 2,
                "w_resel": 0,
                "p_w_chance": 0.5,
                "a_slots": 2,
                "a_resel": 1,
                "p_a_chance": 0.25,
            },
            {
                "w_slots": 2,
                "w_resel": 1,
                "p_w_chance": 0.75,
                "a_slots": 2,
                "a_resel": 2,
                "p_a_chance": 0.5,
            },
        ]
    )

    out = run_stats(stats)

    assert out["total_w_slots"] == 4
    assert out["total_a_resel"] == 3
    assert out["w_obs_rate"] == pytest.approx(0.25)
    assert 0.0 <= out["binom_w_p"] <= 1.0
    assert 0.0 <= out["binom_a_p"] <= 1.0
    assert 0.0 <= out["chi2_p"] <= 1.0
    assert 0.0 <= out["sign_w_p"] <= 1.0

    empty = pd.DataFrame(
        [
            {
                "w_slots": 0,
                "w_resel": 0,
                "p_w_chance": 0.0,
                "a_slots": 0,
                "a_resel": 0,
                "p_a_chance": 0.0,
            }
        ]
    )
    empty_out = run_stats(empty)
    assert math.isnan(empty_out["w_obs_rate"])
    assert math.isnan(empty_out["a_obs_rate"])
    assert math.isnan(empty_out["binom_w_p"])
    assert math.isnan(empty_out["binom_a_p"])
