from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "growth"))

from pool_growth_rates import (
    _author_step_matrices,
    apply_scope,
    bootstrap_delta_beta,
    build_steps,
    compute_curves,
    fit_exponential,
    fit_heaps,
    loyalty_matched_null,
    paired_novelty_test,
)

COLUMNS = [
    "work_id",
    "edition_id",
    "anthology_publication_year",
    "series_id",
    "edition_number",
    "parent_id",
    "author_id",
    "author_birth_year",
]


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a works-per-afam-edition-shaped frame from partial rows."""
    defaults = dict.fromkeys(COLUMNS)
    frame = pd.DataFrame([{**defaults, **row} for row in rows], columns=COLUMNS)
    frame["anthology_publication_year"] = frame["anthology_publication_year"].astype(
        int
    )
    return frame


def simple_df() -> pd.DataFrame:
    """Two editions: edition 1 debuts w1/w2, edition 2 repeats w1 and adds w3."""
    return make_df(
        [
            {
                "work_id": 1,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": 10,
            },
            {
                "work_id": 2,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": 20,
            },
            {
                "work_id": 1,
                "edition_id": 2,
                "anthology_publication_year": 1941,
                "author_id": 10,
            },
            {
                "work_id": 3,
                "edition_id": 2,
                "anthology_publication_year": 1941,
                "author_id": 30,
            },
        ]
    )


def curves_of(df: pd.DataFrame, time_axis: str = "edition") -> pd.DataFrame:
    return compute_curves(df, build_steps(df, time_axis))


# ── build_steps ───────────────────────────────────────────────────────────────


def test_build_steps_edition_axis_orders_by_year_then_edition_id():
    df = make_df(
        [
            {"work_id": 1, "edition_id": 9, "anthology_publication_year": 1971},
            {"work_id": 2, "edition_id": 4, "anthology_publication_year": 1971},
            {"work_id": 3, "edition_id": 7, "anthology_publication_year": 1929},
        ]
    )
    steps = build_steps(df, "edition")
    assert [s.edition_ids for s in steps] == [(7,), (4,), (9,)]
    assert [s.index for s in steps] == [1, 2, 3]


def test_build_steps_year_axis_merges_same_year_editions():
    df = make_df(
        [
            {"work_id": 1, "edition_id": 9, "anthology_publication_year": 1971},
            {"work_id": 2, "edition_id": 4, "anthology_publication_year": 1971},
            {"work_id": 3, "edition_id": 7, "anthology_publication_year": 1929},
        ]
    )
    steps = build_steps(df, "year")
    assert [s.edition_ids for s in steps] == [(7,), (4, 9)]
    assert [s.label for s in steps] == ["1929", "1971"]


def test_build_steps_rejects_unknown_axis():
    with pytest.raises(ValueError, match="unknown time_axis"):
        build_steps(simple_df(), "decade")


# ── compute_curves ────────────────────────────────────────────────────────────


def test_compute_curves_basic_accumulation():
    curves = curves_of(simple_df())
    assert curves["pool_authors"].tolist() == [2, 3]
    assert curves["pool_works"].tolist() == [2, 3]
    assert curves["new_works"].tolist() == [2, 1]
    assert curves["slots_works"].tolist() == [2, 4]
    assert curves["novelty_works"].tolist() == [1.0, 0.5]
    assert curves["retention_works"].tolist() == [0.0, 0.5]


def test_compute_curves_pools_are_monotone():
    curves = curves_of(simple_df())
    assert curves["pool_authors"].is_monotonic_increasing
    assert curves["pool_works"].is_monotonic_increasing
    assert curves["slots_authors"].is_monotonic_increasing
    assert curves["slots_works"].is_monotonic_increasing


def test_multi_author_work_counts_once_as_work_and_once_per_author():
    df = make_df(
        [
            {
                "work_id": 1,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": 10,
            },
            {
                "work_id": 1,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": 20,
            },
        ]
    )
    curves = curves_of(df)
    assert curves["pool_works"].tolist() == [1]
    assert curves["pool_authors"].tolist() == [2]


def test_unauthored_work_grows_only_the_work_pool():
    df = make_df(
        [
            {
                "work_id": 1,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": 10,
            },
            {
                "work_id": 2,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": None,
            },
        ]
    )
    curves = curves_of(df)
    assert curves["pool_works"].tolist() == [2]
    assert curves["pool_authors"].tolist() == [1]
    assert curves["works_per_author"].tolist() == [2.0]


def test_repeated_work_in_later_edition_is_not_new():
    curves = curves_of(simple_df())
    assert curves["new_authors"].tolist() == [2, 1]
    assert curves["novelty_authors"].tolist() == [1.0, 0.5]


def test_steps_with_no_selections_yield_nan_novelty():
    df = make_df(
        [
            {
                "work_id": 1,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": 10,
            },
            {
                "work_id": 2,
                "edition_id": 2,
                "anthology_publication_year": 1941,
                "author_id": None,
            },
        ]
    )
    curves = curves_of(df)
    # Edition 2 contributes a work but no author, so author novelty is undefined.
    assert np.isnan(curves["novelty_authors"].iloc[1])
    assert curves["novelty_works"].iloc[1] == 1.0


# ── apply_scope ───────────────────────────────────────────────────────────────


def scope_df() -> pd.DataFrame:
    return make_df(
        [
            {
                "work_id": 1,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": 10,
            },
            # An excerpt of work 1, by the same author.
            {
                "work_id": 2,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": 10,
                "parent_id": 1,
            },
            # An excerpt by an author with no root work of their own.
            {
                "work_id": 3,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": 20,
                "parent_id": 1,
            },
            {
                "work_id": 4,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": None,
            },
        ]
    )


def test_apply_scope_all_keeps_everything():
    curves = curves_of(apply_scope(scope_df(), root_only=False, authored_only=False))
    assert curves["pool_works"].tolist() == [4]
    assert curves["pool_authors"].tolist() == [2]


def test_apply_scope_root_only_drops_excerpts_and_excerpt_only_authors():
    curves = curves_of(apply_scope(scope_df(), root_only=True, authored_only=False))
    assert curves["pool_works"].tolist() == [2]  # work 1 and unauthored work 4
    assert curves["pool_authors"].tolist() == [1]  # author 20 had only an excerpt


def test_apply_scope_authored_only_drops_unauthored_works():
    curves = curves_of(apply_scope(scope_df(), root_only=False, authored_only=True))
    assert curves["pool_works"].tolist() == [3]
    assert curves["pool_authors"].tolist() == [2]


def test_apply_scope_does_not_mutate_input():
    df = scope_df()
    before = len(df)
    apply_scope(df, root_only=True, authored_only=True)
    assert len(df) == before


# ── fits ──────────────────────────────────────────────────────────────────────


def test_fit_heaps_recovers_a_known_exponent():
    slots = np.array([10, 20, 40, 80, 160, 320], dtype=float)
    pool = 3.0 * slots**0.7
    beta, se, r2, log_k = fit_heaps(slots, pool)
    assert beta == pytest.approx(0.7, abs=1e-9)
    assert np.exp(log_k) == pytest.approx(3.0, rel=1e-9)
    assert r2 == pytest.approx(1.0)
    assert se == pytest.approx(0.0, abs=1e-9)


def test_fit_heaps_ignores_nonpositive_points():
    slots = np.array([0, 10, 20, 40, 80, 160], dtype=float)
    pool = np.array([0, 3, 6, 12, 24, 48], dtype=float)
    beta, *_ = fit_heaps(slots, pool)
    assert beta == pytest.approx(1.0, abs=1e-9)


def test_fit_heaps_returns_nan_with_too_few_points():
    beta, se, r2, log_k = fit_heaps(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert all(np.isnan(v) for v in (beta, se, r2, log_k))


def test_fit_exponential_recovers_a_known_rate():
    pool = 5.0 * 1.25 ** np.arange(6)
    rate, se, r2 = fit_exponential(pool)
    assert rate == pytest.approx(0.25, abs=1e-9)
    assert r2 == pytest.approx(1.0)
    assert se == pytest.approx(0.0, abs=1e-8)


# ── paired novelty test ───────────────────────────────────────────────────────


def test_paired_novelty_drops_the_first_step_and_counts_correctly():
    curves = pd.DataFrame(
        {
            "novelty_authors": [1.0, 0.2, 0.3, 0.1, 0.4, 0.5],
            "novelty_works": [1.0, 0.6, 0.7, 0.5, 0.8, 0.4],
        }
    )
    result = paired_novelty_test(curves)
    assert result["n_steps"] == 5
    assert result["n_works_higher"] == 4
    assert result["median_diff"] == pytest.approx(0.4)


# ── bootstrap machinery ───────────────────────────────────────────────────────


def test_author_step_matrices_reproduce_the_observed_curves():
    """Summing every author exactly once must rebuild the real curves."""
    df = simple_df()
    steps = build_steps(df, "edition")
    curves = compute_curves(df, steps)
    a_new, a_slots, w_new, w_slots, un_new, un_slots = _author_step_matrices(df, steps)
    ones = np.ones(a_new.shape[0])

    assert np.cumsum(ones @ a_new).tolist() == curves["pool_authors"].tolist()
    assert np.cumsum(ones @ a_slots).tolist() == curves["slots_authors"].tolist()
    assert np.cumsum(ones @ w_new + un_new).tolist() == curves["pool_works"].tolist()
    assert (
        np.cumsum(ones @ w_slots + un_slots).tolist() == curves["slots_works"].tolist()
    )


def test_author_step_matrices_route_unauthored_works_to_the_baseline():
    df = make_df(
        [
            {
                "work_id": 1,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": 10,
            },
            {
                "work_id": 2,
                "edition_id": 1,
                "anthology_publication_year": 1929,
                "author_id": None,
            },
        ]
    )
    steps = build_steps(df, "edition")
    _, _, w_new, _, un_new, un_slots = _author_step_matrices(df, steps)
    assert w_new.sum() == 1  # only the authored work
    assert un_new.tolist() == [1.0]
    assert un_slots.tolist() == [1.0]


def fittable_df() -> pd.DataFrame:
    """Six editions with enough authors and works for the Heaps fits to run.

    Each edition repeats every earlier author but brings two fresh works per
    author, so the work pool outruns the author pool by construction.
    """
    rows = []
    for edition in range(1, 7):
        for author in range(1, edition + 1):
            for work in range(2):
                rows.append(
                    {
                        "work_id": author * 100 + edition * 10 + work,
                        "edition_id": edition,
                        "anthology_publication_year": 1900 + edition * 10,
                        "author_id": author,
                    }
                )
    return make_df(rows)


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    df = fittable_df()
    steps = build_steps(df, "edition")
    a = bootstrap_delta_beta(df, steps, n_boot=25, seed=7)
    b = bootstrap_delta_beta(df, steps, n_boot=25, seed=7)
    assert np.array_equal(a["_draws"], b["_draws"], equal_nan=True)
    assert np.isfinite(a["delta_beta_mean"])


def test_bootstrap_finds_a_positive_gap_when_works_outrun_authors():
    df = fittable_df()
    steps = build_steps(df, "edition")
    result = bootstrap_delta_beta(df, steps, n_boot=50, seed=7)
    assert result["delta_beta_lo"] > 0


# ── loyalty-matched null ──────────────────────────────────────────────────────


def null_curves(retention: float) -> pd.DataFrame:
    """Six steps with a constant author retention rate and 10 works each."""
    n_works = np.full(6, 10)
    pool_authors = np.arange(1, 7) * 5
    return pd.DataFrame(
        {
            "n_works": n_works,
            "retention_authors": np.full(6, retention),
            "slots_authors": np.cumsum(np.full(6, 5)),
            "pool_authors": pool_authors,
            "slots_works": np.cumsum(n_works),
            "pool_works": np.cumsum(n_works),
        }
    )


def test_loyalty_null_with_zero_retention_fills_the_pool_with_every_slot():
    result = loyalty_matched_null(null_curves(0.0), n_sims=20, seed=1)
    # Every work is new, so the null work pool equals cumulative slots (60).
    assert result["null_pool_last_mean"] == pytest.approx(60.0)


def test_loyalty_null_with_full_retention_stalls_the_pool():
    result = loyalty_matched_null(null_curves(1.0), n_sims=20, seed=1)
    # Step 1 seeds 10 works; afterwards every draw is capped at the pool, so the
    # pool never grows past that first anthology.
    assert result["null_pool_last_mean"] == pytest.approx(10.0)


def test_loyalty_null_reports_the_observed_gap_and_is_deterministic():
    curves = null_curves(0.5)
    a = loyalty_matched_null(curves, n_sims=30, seed=3)
    b = loyalty_matched_null(curves, n_sims=30, seed=3)
    assert np.array_equal(a["_draws"], b["_draws"], equal_nan=True)
    beta_a, *_ = fit_heaps(curves["slots_authors"], curves["pool_authors"])
    beta_w, *_ = fit_heaps(curves["slots_works"], curves["pool_works"])
    assert a["observed_delta_beta"] == pytest.approx(beta_w - beta_a)


def test_loyalty_null_p_value_is_never_zero():
    """The add-one estimator keeps an all-exceed result reported as a bound."""
    result = loyalty_matched_null(null_curves(0.9), n_sims=20, seed=5)
    assert result["p_null_ge_observed"] >= 1 / 21
