"""Regression tests for the author/work agreement model stress test.

These tests cover the methodological invariants added after peer review.  In
particular, they prevent excerpt handling, latent-work identity, and the fit
criterion from silently reverting to the confirmatory original specification.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "overlap"))

import author_work_agreement_models as awm  # noqa: E402
from simulate_author_work_overlap import EditionTarget  # noqa: E402


# -- Work-unit sensitivity ----------------------------------------------------


@pytest.fixture
def excerpt_frame() -> pd.DataFrame:
    """A root and its excerpt overlap in e1; e2 selects only the excerpt."""
    return pd.DataFrame(
        [
            {
                "edition_key": "e1",
                "work_id": "root",
                "parent_id": None,
                "author_id": "a1",
                "title": "Whole work",
            },
            {
                "edition_key": "e1",
                "work_id": "excerpt",
                "parent_id": "root",
                "author_id": "a1",
                "title": "Excerpt",
            },
            {
                "edition_key": "e2",
                "work_id": "excerpt",
                "parent_id": "root",
                "author_id": "a1",
                "title": "Excerpt",
            },
            {
                "edition_key": "e2",
                "work_id": "other",
                "parent_id": None,
                "author_id": "a2",
                "title": "Other work",
            },
        ]
    )


def _selection_units(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(
        df[["edition_key", "work_id", "author_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )


def test_preprocess_work_units_drop_include_and_coalesce(
    excerpt_frame: pd.DataFrame,
) -> None:
    original = excerpt_frame.copy(deep=True)

    dropped = awm.preprocess_work_units(excerpt_frame, "drop")
    included = awm.preprocess_work_units(excerpt_frame, "include")
    coalesced = awm.preprocess_work_units(excerpt_frame, "coalesce")

    assert _selection_units(dropped) == {
        ("e1", "root", "a1"),
        ("e2", "other", "a2"),
    }
    assert _selection_units(included) == {
        ("e1", "root", "a1"),
        ("e1", "excerpt", "a1"),
        ("e2", "excerpt", "a1"),
        ("e2", "other", "a2"),
    }
    assert _selection_units(coalesced) == {
        ("e1", "root", "a1"),
        ("e2", "root", "a1"),
        ("e2", "other", "a2"),
    }
    pd.testing.assert_frame_equal(excerpt_frame, original)


def test_preprocess_work_units_rejects_unknown_mode(
    excerpt_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="work unit|mode"):
        awm.preprocess_work_units(excerpt_frame, "parents-maybe")


def test_coalescing_normalizes_float_parent_ids_from_nullable_database_column() -> None:
    df = pd.DataFrame(
        [
            {"edition_key": "e1", "work_id": 1, "parent_id": np.nan},
            {"edition_key": "e2", "work_id": 2, "parent_id": 1.0},
        ]
    )

    coalesced = awm.preprocess_work_units(df, "coalesce")

    assert set(coalesced["work_id"]) == {"1"}


# -- Shared latent oeuvre -----------------------------------------------------


def test_shared_phantom_identity_can_recur_across_editions() -> None:
    eligible = [("a1", ["w1"])]
    target = EditionTarget(
        per_author_work_counts=[2], authored_work_count=2, authorless_work_count=0
    )

    sims = [
        awm.sample_edition_author_first(
            target,
            eligible,
            None,
            0.0,
            [],
            np.random.default_rng(seed),
            lam=1.0,
            phantom_prefix=edition,
            phantom_scope="shared",
        )
        for seed, edition in enumerate(("e1", "e2"))
    ]

    # Each edition must take the sole observed and sole latent work.  A shared
    # oeuvre gives that latent work one stable identity despite edition labels.
    phantoms = [{w for w in sim.work_ids if w.startswith("~")} for sim in sims]
    assert len(phantoms[0]) == 1
    assert phantoms[0] == phantoms[1]
    assert sims[0].work_ids & sims[1].work_ids == sims[0].work_ids


def test_shared_phantoms_are_sampled_from_the_whole_latent_oeuvre() -> None:
    target = EditionTarget(
        per_author_work_counts=[1], authored_work_count=1, authorless_work_count=0
    )
    selected_phantoms: set[str] = set()
    for seed in range(50):
        sim = awm.sample_edition_author_first(
            target,
            [("a1", ["w1"])],
            None,
            0.0,
            [],
            np.random.default_rng(seed),
            lam=4.0,
            phantom_scope="shared",
        )
        selected_phantoms.update(w for w in sim.work_ids if w.startswith("~"))

    # Merely removing the edition namespace from the legacy implementation
    # would always emit latent index zero; a genuine shared pool samples IDs.
    assert len(selected_phantoms) > 1


def test_invalid_phantom_scope_is_rejected() -> None:
    target = EditionTarget(
        per_author_work_counts=[1], authored_work_count=1, authorless_work_count=0
    )
    with pytest.raises(ValueError, match=r"phantom[_ ]scope"):
        awm.sample_edition_author_first(
            target,
            [("a1", ["w1"])],
            None,
            0.0,
            [],
            np.random.default_rng(0),
            lam=1.0,
            phantom_scope="edition-ish",
        )


# -- Mixed author and work concentration -------------------------------------


def test_mixed_sampler_combines_gamma_alpha_and_inflation() -> None:
    eligible = [
        ("a0", ["a0_w1", "a0_w2"]),
        ("a1", ["a1_w1", "a1_w2"]),
        ("a2", ["a2_w1", "a2_w2"]),
        ("a3", ["a3_w1", "a3_w2"]),
    ]
    author_ranks = {f"a{i}": i + 1 for i in range(4)}
    author_to_works = {aid: set(wids) for aid, wids in eligible}
    work_freq = {wids[0]: 2 for _, wids in eligible} | {
        wids[1]: 1 for _, wids in eligible
    }
    corpus_ranks = awm.draw_author_corpus_ranks(
        author_to_works,
        np.random.default_rng(0),
        "frequency",
        work_freq,
        lam=1.0,
    )
    assert all(len(rank_map) == 4 for rank_map in corpus_ranks.values())
    target = EditionTarget(
        per_author_work_counts=[2, 1], authored_work_count=3, authorless_work_count=0
    )

    sim = awm.sample_edition_author_first(
        target,
        eligible,
        corpus_ranks,
        30.0,
        [],
        np.random.default_rng(4),
        author_ranks=author_ranks,
        gamma=30.0,
        lam=1.0,
        phantom_scope="shared",
    )

    assert sim.author_ids == {"a0", "a1"}
    assert sorted(map(len, sim.author_work_ids.values()), reverse=True) == [2, 1]
    # Observed ranks must remain meaningful after latent works extend the pool.
    assert "a0_w1" in sim.author_work_ids["a0"]
    assert "a1_w1" in sim.author_work_ids["a1"]


def test_model_config_exposes_mixed_parameters_without_breaking_legacy_positionals(
) -> None:
    legacy = awm.ModelConfig("legacy canon", "canon", 1.5, 2.0)
    mixed = awm.ModelConfig("mixed", "mixed", 1.25, 0.5, gamma=2.5)

    assert legacy.alpha == pytest.approx(1.5)
    assert legacy.lam == pytest.approx(2.0)
    assert mixed.family == "mixed"
    assert mixed.alpha == pytest.approx(1.25)
    assert mixed.gamma == pytest.approx(2.5)
    assert mixed.lam == pytest.approx(0.5)


# -- Nonredundant, covariance-aware fit --------------------------------------


def test_fit_vector_excludes_algebraically_redundant_difference() -> None:
    summary = awm.Summary(
        rate_a_gt_w=0.6,
        rate_tie=0.3,
        rate_w_gt_a=0.1,
        mean_jaccard_diff=999.0,
        mean_jaccard_authors=0.25,
        mean_jaccard_works=0.10,
    )

    assert awm.FIT_STATS == (
        "jaccard_authors",
        "jaccard_works",
        "rate_a_gt_w",
        "zero_share",
    )
    assert np.array_equal(
        awm.fit_stat_vector(summary, 0.75),
        np.array([0.25, 0.10, 0.60, 0.75]),
    )


def test_joint_fit_score_is_normalized_mahalanobis_distance() -> None:
    rng = np.random.default_rng(12)
    sims = rng.multivariate_normal(
        mean=[0.2, 0.1, 0.7, 0.5],
        cov=[
            [0.040, 0.025, 0.000, -0.010],
            [0.025, 0.030, 0.000, -0.008],
            [0.000, 0.000, 0.020, 0.006],
            [-0.010, -0.008, 0.006, 0.030],
        ],
        size=500,
    )
    obs = np.array([0.30, 0.08, 0.62, 0.58])
    delta = obs - sims.mean(axis=0)
    covariance = np.cov(sims, rowvar=False)
    expected = np.sqrt(delta @ np.linalg.pinv(covariance) @ delta / obs.size)

    # The implementation adds a tiny covariance ridge for degenerate grids.
    assert awm.joint_fit_score(obs, sims) == pytest.approx(expected, rel=5e-6)


def test_synthetic_likelihood_is_finite_for_singular_simulations() -> None:
    x = np.linspace(-1.0, 1.0, 20)
    sims = np.column_stack((x, 2 * x, -x, np.ones_like(x)))
    score = awm.synthetic_likelihood_score(np.array([0.1, 0.2, -0.1, 1.0]), sims)
    assert np.isfinite(score)


def test_synthetic_likelihood_penalizes_excess_dispersion() -> None:
    rng = np.random.default_rng(91)
    base = rng.normal(size=(1_000, 4))
    obs = np.zeros(4)

    compact = awm.synthetic_likelihood_score(obs, base * 0.5)
    diffuse = awm.synthetic_likelihood_score(obs, base * 2.0)

    assert compact < diffuse


def test_run_fit_mode_reports_joint_covariance_scores(monkeypatch) -> None:
    rng = np.random.default_rng(7)
    sims = rng.multivariate_normal(
        mean=[0.22, 0.08, 0.65, 0.52],
        cov=np.diag([0.002, 0.001, 0.004, 0.003]),
        size=80,
    )
    arrays = {
        "ja": sims[:, 0],
        "jw": sims[:, 1],
        "rate": sims[:, 2],
        "zero": sims[:, 3],
        "diff": sims[:, 0] - sims[:, 1],
    }
    observed = awm.Summary(
        rate_a_gt_w=0.62,
        rate_tie=0.30,
        rate_w_gt_a=0.08,
        mean_jaccard_diff=0.16,
        mean_jaccard_authors=0.24,
        mean_jaccard_works=0.08,
    )
    data = SimpleNamespace(observed=observed, obs_zero_share=0.50)
    args = SimpleNamespace(
        lambdas=[1.0],
        gammas=[1.5],
        fit_trials=len(sims),
        rank_by="random",
        decouple_allocation=False,
        phantom_scope="shared",
    )
    monkeypatch.setattr(awm, "simulate_model", lambda *args, **kwargs: arrays)

    fit = awm.run_fit_mode(data, args, np.random.default_rng(0))
    row = fit.iloc[0]
    obs_vector = awm.fit_stat_vector(observed, 0.50)

    assert row["joint_rms"] == pytest.approx(awm.joint_fit_score(obs_vector, sims))
    assert row["synthetic_nll"] == pytest.approx(
        awm.synthetic_likelihood_score(obs_vector, sims)
    )
    assert row["joint_p"] == pytest.approx(awm.joint_predictive_p(obs_vector, sims))
    assert np.isfinite(
        row[["joint_rms", "synthetic_nll", "joint_p"]].astype(float)
    ).all()


# -- Outcome-derived eligibility diagnostics ---------------------------------


def test_eligibility_leakage_uses_other_editions_at_or_before_focal_year() -> None:
    df = pd.DataFrame(
        [
            # w1/a1 are independently observed in a same-year edition; w2/a2
            # are only observed again later and therefore leak into e1.
            ("e1", 1900, "w1", "a1"),
            ("e1", 1900, "w2", "a2"),
            ("e1", 1900, "w2", "a2"),  # duplicated row must not inflate counts
            ("e2", 1900, "w1", "a1"),
            ("e2", 1900, "w3", "a3"),
            ("e3", 1910, "w2", "a2"),
            ("e3", 1910, "w4", "a4"),
        ],
        columns=[
            "edition_key",
            "anthology_publication_year",
            "work_id",
            "author_id",
        ],
    )

    diagnostics = awm.eligibility_leakage_diagnostics(df).set_index("edition_key")

    assert set(diagnostics.index) == {"e1", "e2", "e3"}
    assert (diagnostics[["n_authors", "n_works"]] == 2).all().all()
    assert (diagnostics[["leaked_authors", "leaked_works"]] == 1).all().all()
    assert np.allclose(diagnostics["author_leakage_rate"], 0.5)
    assert np.allclose(diagnostics["work_leakage_rate"], 0.5)


def test_earliest_unreplicated_edition_is_fully_self_seeded() -> None:
    df = pd.DataFrame(
        [
            ("first", 1890, "w0", "a0"),
            ("later", 1900, "w1", "a1"),
        ],
        columns=[
            "edition_key",
            "anthology_publication_year",
            "work_id",
            "author_id",
        ],
    )

    first = (
        awm.eligibility_leakage_diagnostics(df)
        .set_index("edition_key")
        .loc["first"]
    )
    assert first["author_leakage_rate"] == pytest.approx(1.0)
    assert first["work_leakage_rate"] == pytest.approx(1.0)
