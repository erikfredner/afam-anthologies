"""Unit tests for analysis/vernacular/vernacular_reselection_test.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "vernacular"))
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "reselection"))

from author_vs_work_debut_reselection import (  # noqa: E402
    add_entry_group,
    build_edition_table,
    compute_work_records,
)
from vernacular_reselection_test import (  # noqa: E402
    SPECS,
    build_comparison,
    build_groups,
    build_monte_carlo,
    comparison_row,
    empirical_two_sided_p,
    metric_counts,
    monte_carlo,
    run_rng,
    sample_indices,
)

SPEC_BY_NAME = {spec.name: spec for spec in SPECS}


# ── Fixtures ─────────────────────────────────────────────────────────────────


def make_page_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "volume_id": None,
        "edition_id": None,
        "edition_year": 2000,
        "work_id": None,
        "work_title": "Untitled",
        "toc_page": 1,
        "toc_next": 2,
        "volume_number": 1,
        "author_name": None,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def make_work_records(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "work_id": None,
        "edition_id": None,
        "anthology_publication_year": None,
        "series_id": None,
        "parent_id": None,
        "author_id": None,
    }
    raw = pd.DataFrame([{**defaults, **row} for row in rows])
    raw = add_entry_group(raw)
    return compute_work_records(raw, build_edition_table(raw))


def make_classified(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "group": "non_vernacular",
        "subsequent_count": 1,
        "reselection_count": 0,
        "is_reselected": False,
        "cross_series_subsequent_count": 1,
        "cross_series_reselection_count": 0,
        "is_reselected_cross_series": False,
        "debut_edition_id": 1,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


# ── build_groups ─────────────────────────────────────────────────────────────


def test_build_groups_classification():
    """Vernacular-range hit → V; other work in E_v edition → NV; work only in
    a non-vernacular edition → excluded."""
    page_df = make_page_df(
        [
            {"volume_id": 10, "edition_id": 1, "work_id": 1, "toc_page": 5},
            {"volume_id": 10, "edition_id": 1, "work_id": 2, "toc_page": 50},
            {"volume_id": 20, "edition_id": 2, "work_id": 1, "toc_page": 30},
            {"volume_id": 20, "edition_id": 2, "work_id": 3, "toc_page": 40},
        ]
    )
    ranges = pd.DataFrame(
        [
            {
                "edition_title": "E1",
                "volume_id": 10,
                "slot": None,
                "page_start": 1,
                "page_end": 10,
            }
        ]
    )
    work_records = make_work_records(
        [
            {"work_id": 1, "edition_id": 1, "anthology_publication_year": 2000},
            {"work_id": 2, "edition_id": 1, "anthology_publication_year": 2000},
            {"work_id": 1, "edition_id": 2, "anthology_publication_year": 2010},
            {"work_id": 3, "edition_id": 2, "anthology_publication_year": 2010},
        ]
    )

    classified, info = build_groups(page_df, ranges, work_records)
    groups = classified.set_index("work_key")["group"]

    assert groups.to_dict() == {"1": "vernacular", "2": "non_vernacular"}
    assert info["n_editions_total"] == 2
    assert info["n_editions_vernacular"] == 1
    assert info["n_vernacular_works"] == 1
    assert info["n_non_vernacular_works"] == 1


def test_build_groups_id_dtype_mismatch():
    """Float work ids on the page side match string ids on the records side."""
    page_df = make_page_df(
        [{"volume_id": 10, "edition_id": 1, "work_id": 1.0, "toc_page": 5}]
    )
    ranges = pd.DataFrame(
        [
            {
                "edition_title": "E1",
                "volume_id": 10,
                "slot": None,
                "page_start": 1,
                "page_end": 10,
            }
        ]
    )
    work_records = make_work_records(
        [{"work_id": "1", "edition_id": 1, "anthology_publication_year": 2000}]
    )

    classified, _ = build_groups(page_df, ranges, work_records)
    assert classified["group"].tolist() == ["vernacular"]


# ── metric_counts / comparison_row ───────────────────────────────────────────


def test_metric_counts_ever_and_opportunity():
    frame = make_classified(
        [
            {"subsequent_count": 3, "reselection_count": 2, "is_reselected": True},
            {"subsequent_count": 5, "reselection_count": 0, "is_reselected": False},
        ]
    )
    assert metric_counts(frame, SPEC_BY_NAME["ever_all"]) == (1, 2)
    assert metric_counts(frame, SPEC_BY_NAME["opportunity_all"]) == (2, 8)


def test_comparison_row_hand_computed():
    row = comparison_row("ever_all", "desc", 8, 10, 5, 10)
    assert row["vernacular_rate"] == pytest.approx(0.8)
    assert row["non_vernacular_rate"] == pytest.approx(0.5)
    assert row["rate_difference"] == pytest.approx(0.3)
    assert row["risk_ratio"] == pytest.approx(1.6)
    assert row["odds_ratio"] == pytest.approx(4.0)
    for key in ("chi2_p", "fisher_p", "two_proportion_z_p"):
        assert 0.0 < row[key] <= 1.0
    assert row["vernacular_ci_lo"] < 0.8 < row["vernacular_ci_hi"]


def test_comparison_row_empty_group():
    row = comparison_row("ever_all", "desc", 0, 0, 5, 10)
    assert pd.isna(row["vernacular_rate"])
    assert row["non_vernacular_rate"] == pytest.approx(0.5)


def test_build_comparison_eligibility_filter():
    """Debuts with no subsequent opportunity stay out of the denominator."""
    classified = make_classified(
        [
            {"group": "vernacular", "is_reselected": True},
            {"group": "vernacular", "subsequent_count": 0, "is_reselected": False},
            {"group": "non_vernacular", "is_reselected": False},
        ]
    )
    comparison = build_comparison(classified).set_index("metric")
    ever = comparison.loc["ever_all"]
    assert ever["vernacular_n"] == 1
    assert ever["vernacular_k"] == 1
    assert ever["non_vernacular_n"] == 1


# ── Monte Carlo ──────────────────────────────────────────────────────────────


def test_sample_indices_stratified_counts_and_cap():
    rng = np.random.default_rng(0)
    pools = {"a": np.arange(0, 5), "b": np.arange(5, 8)}
    idx = sample_indices(rng, pools, {"a": 2, "b": 5})
    assert len(idx) == 2 + 3  # b capped at pool size
    assert sum(i < 5 for i in idx) == 2
    assert len(set(idx)) == len(idx)  # without replacement


def test_empirical_two_sided_p_edges():
    rates = np.zeros(99)
    assert empirical_two_sided_p(rates, 1.0) == pytest.approx(2 / 100)
    assert empirical_two_sided_p(rates, 0.0) == pytest.approx(1.0)


def test_empirical_two_sided_p_ignores_nan_draws():
    """NaN draws are excluded from N; an all-NaN null yields NaN, never a
    spuriously significant p."""
    assert np.isnan(empirical_two_sided_p(np.full(99, np.nan), 1.0))
    assert np.isnan(empirical_two_sided_p(np.zeros(99), float("nan")))
    half_nan = np.concatenate([np.zeros(49), np.full(50, np.nan)])
    assert empirical_two_sided_p(half_nan, 1.0) == pytest.approx(2 / 50)


def test_monte_carlo_disjoint_strata_yields_nan():
    """Stratified draws with no NV works in any vernacular stratum produce a
    NaN null and NaN p instead of a fake significant result."""
    vern = make_classified(
        [{"group": "vernacular", "debut_edition_id": 1, "is_reselected": True}] * 4
    )
    nonvern = make_classified([{"group": "non_vernacular", "debut_edition_id": 2}] * 10)
    res = monte_carlo(
        vern, nonvern, SPEC_BY_NAME["ever_all"], 50, np.random.default_rng(0), True
    )
    assert res["sample_size"] == 0
    assert res["sample_shortfall"] == 4
    for key in ("null_mean", "null_sd", "null_lo", "null_hi", "empirical_p"):
        assert np.isnan(res[key])


def test_monte_carlo_deterministic():
    vern = make_classified([{"group": "vernacular", "is_reselected": True}] * 5)
    nonvern = make_classified(
        [
            {"group": "non_vernacular", "is_reselected": flag}
            for flag in [True, False] * 25
        ]
    )
    spec = SPEC_BY_NAME["ever_all"]

    res1 = monte_carlo(vern, nonvern, spec, 200, np.random.default_rng(7), False)
    res2 = monte_carlo(vern, nonvern, spec, 200, np.random.default_rng(7), False)
    assert res1 == res2
    assert res1["null_sd"] > 0


def test_monte_carlo_extreme_observed():
    """All-True vernacular vs all-False pool: observed beats every draw."""
    vern = make_classified([{"group": "vernacular", "is_reselected": True}] * 5)
    nonvern = make_classified([{"group": "non_vernacular"}] * 50)
    res = monte_carlo(
        vern, nonvern, SPEC_BY_NAME["ever_all"], 200, np.random.default_rng(7), False
    )
    assert res["vernacular_rate"] == pytest.approx(1.0)
    assert res["null_mean"] == pytest.approx(0.0)
    assert res["null_lo"] == pytest.approx(0.0)
    assert res["null_hi"] == pytest.approx(0.0)
    assert res["empirical_p"] == pytest.approx(2 / 201)
    assert res["sample_size"] == 5
    assert res["sample_shortfall"] == 0


def test_monte_carlo_stratified_shortfall():
    """A stratum with fewer NV than V works caps the draw and reports it."""
    vern = make_classified(
        [{"group": "vernacular", "debut_edition_id": 1, "is_reselected": True}] * 4
    )
    nonvern = make_classified(
        [{"group": "non_vernacular", "debut_edition_id": 1}] * 3
        + [{"group": "non_vernacular", "debut_edition_id": 2}] * 10
    )
    res = monte_carlo(
        vern, nonvern, SPEC_BY_NAME["ever_all"], 50, np.random.default_rng(0), True
    )
    assert res["sample_size"] == 3
    assert res["sample_shortfall"] == 1


def test_monte_carlo_opportunity_metric():
    """Opportunity draws pool numerators over denominators, not per-work means."""
    vern = make_classified(
        [{"group": "vernacular", "subsequent_count": 4, "reselection_count": 2}]
    )
    nonvern = make_classified(
        [{"group": "non_vernacular", "subsequent_count": 2, "reselection_count": 1}]
        * 20
    )
    res = monte_carlo(
        vern,
        nonvern,
        SPEC_BY_NAME["opportunity_all"],
        50,
        np.random.default_rng(0),
        False,
    )
    assert res["vernacular_rate"] == pytest.approx(0.5)
    assert res["null_mean"] == pytest.approx(0.5)
    assert res["empirical_p"] == pytest.approx(1.0)


def test_build_monte_carlo_shapes():
    classified = make_classified(
        [
            {"group": "vernacular", "is_reselected": True, "reselection_count": 1},
            {"group": "vernacular"},
            {"group": "non_vernacular", "is_reselected": True, "reselection_count": 1},
            {"group": "non_vernacular"},
            {"group": "non_vernacular"},
        ]
    )
    mc = build_monte_carlo(classified, 20, seed=0)
    assert len(mc) == len(SPECS) * 2  # unmatched + stratified per metric
    assert set(mc["mode"]) == {"unmatched", "stratified"}
    assert (mc["n_draws"] == 20).all()


def test_build_monte_carlo_runs_are_order_independent():
    """Each metric×mode run draws from its own name-keyed stream, so a run's
    result matches a standalone run with the same seed regardless of SPECS
    ordering or other runs."""
    classified = make_classified(
        [{"group": "vernacular", "is_reselected": True}] * 3
        + [
            {"group": "non_vernacular", "is_reselected": flag}
            for flag in [True, False] * 10
        ]
    )
    mc = build_monte_carlo(classified, 50, seed=11).set_index(["metric", "mode"])

    spec = SPEC_BY_NAME["opportunity_cross_series"]
    eligible = classified[classified[spec.eligible_col] > 0]
    standalone = monte_carlo(
        eligible[eligible["group"] == "vernacular"],
        eligible[eligible["group"] == "non_vernacular"],
        spec,
        50,
        run_rng(11, spec.name, "stratified"),
        True,
    )
    batched = mc.loc[(spec.name, "stratified")]
    assert batched["null_mean"] == pytest.approx(standalone["null_mean"])
    assert batched["empirical_p"] == pytest.approx(standalone["empirical_p"])


def test_run_rng_streams_are_stable_and_distinct():
    a1 = run_rng(5, "ever_all", "unmatched").random(4)
    a2 = run_rng(5, "ever_all", "unmatched").random(4)
    b = run_rng(5, "ever_all", "stratified").random(4)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)


def test_build_groups_warns_on_unknown_volume(capsys):
    """A range volume_id absent from the corpus triggers a WARNING instead of
    silently shrinking E_v (mirrors vernacular_works.compute_edition_shares)."""
    page_df = make_page_df(
        [{"volume_id": 10, "edition_id": 1, "work_id": 1, "toc_page": 5}]
    )
    ranges = pd.DataFrame(
        [
            {
                "edition_title": "E1",
                "volume_id": 10,
                "slot": None,
                "page_start": 1,
                "page_end": 10,
            },
            {
                "edition_title": "Ghost",
                "volume_id": 999,
                "slot": None,
                "page_start": 1,
                "page_end": 10,
            },
        ]
    )
    work_records = make_work_records(
        [{"work_id": 1, "edition_id": 1, "anthology_publication_year": 2000}]
    )

    _, info = build_groups(page_df, ranges, work_records)
    assert "WARNING: vernacular volumes not found in AFAM corpus: [999]" in (
        capsys.readouterr().out
    )
    assert info["n_editions_vernacular"] == 1
