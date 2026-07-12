"""Focused invariants for the chronological author/work bundle stress test."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "overlap"))

import author_work_agreement_stress_test as stress  # noqa: E402


def _row(
    work: str,
    author: str | None,
    edition: int,
    year: int,
    *,
    afam: int = 1,
) -> dict[str, object]:
    return {
        "work_id": work,
        "parent_id": None,
        "author_id": author,
        "edition_id": edition,
        "edition_key": str(edition),
        "anthology_publication_year": year,
        "series_id": None,
        "edition_number": 1,
        "is_afam": afam,
    }


def test_snapshot_is_strict_and_shared_latent_ids_are_stable() -> None:
    frame = pd.DataFrame(
        [
            _row("w1", "a1", 1, 1980),
            _row("w2", "a1", 2, 1990, afam=0),
            _row("future", "a1", 3, 1998),
        ]
    )

    early = stress.make_snapshot(frame, 1990, strict=True, lam=1.0)
    late = stress.make_snapshot(frame, 1998, strict=True, lam=1.0)

    assert "w2" not in early.author_works["a1"]
    assert "future" not in late.author_works["a1"]
    assert stress.latent_work_id("a1", 0) in early.author_works["a1"]
    assert stress.latent_work_id("a1", 0) in late.author_works["a1"]


def test_smoothed_conditional_q_uses_tied_midranks_for_unseen_works() -> None:
    frame = pd.DataFrame(
        [
            _row("w_seen", "a1", 1, 1980),
            _row("w_zero", "a1", 2, 1981, afam=0),
        ]
    )

    snapshot = stress.make_snapshot(frame, 1990, strict=True, lam=1.0)
    latent = [w for w in snapshot.author_works["a1"] if w.startswith("~shared|")]

    assert len(latent) == 2
    tied = snapshot.work_logrank["w_zero"]
    assert all(snapshot.work_logrank[wid] == pytest.approx(tied) for wid in latent)
    assert snapshot.work_logrank["w_seen"] < tied


def test_exact_product_subset_sampler_matches_closed_form_probabilities() -> None:
    works = ("w1", "w2", "w3")
    logrank = {"w1": 0.0, "w2": math.log(2.0), "w3": math.log(4.0)}
    rng = np.random.default_rng(73)
    draws = Counter(
        stress.sample_product_subset(works, logrank, 1.0, 2, rng) for _ in range(20_000)
    )

    expected = {
        ("w1", "w2"): 4 / 7,
        ("w1", "w3"): 2 / 7,
        ("w2", "w3"): 1 / 7,
    }
    for subset, probability in expected.items():
        assert draws[subset] / 20_000 == pytest.approx(probability, abs=0.015)


def test_simulation_preserves_every_bundle_and_authorless_margin() -> None:
    snapshot = stress.Snapshot(
        author_works={
            "a1": ("a1w1", "a1w2", "a1w3"),
            "a2": ("a2w1", "a2w2"),
            "a3": ("a3w1",),
        },
        author_logrank={"a1": 0.0, "a2": math.log(2), "a3": math.log(3)},
        work_logrank={
            "a1w1": 0.0,
            "a1w2": math.log(2),
            "a1w3": math.log(3),
            "a2w1": 0.0,
            "a2w2": math.log(2),
            "a3w1": 0.0,
        },
        authorless_works=("x1", "x2", "x3"),
        authorless_logrank={"x1": 0.0, "x2": math.log(2), "x3": math.log(3)},
    )

    simulated = stress.simulate_edition(
        (2, 1), 2, snapshot, gamma=1.0, alpha=1.0, rng=np.random.default_rng(9)
    )

    assert sorted(map(len, simulated.author_works.values()), reverse=True) == [2, 1]
    assert len(simulated.authors) == 2
    assert len(simulated.authorless) == 2
    assert len(simulated.works) == 5


def test_coauthored_work_is_excluded_once_and_reported_once() -> None:
    full = pd.DataFrame(
        [
            _row("solo", "a1", 1, 1980),
            _row("joint", "a1", 2, 1981),
            _row("joint", "a2", 2, 1981),
        ]
    )
    pool, coauthored = stress.split_coauthored(full)
    snapshot = stress.make_snapshot(pool, 1990, strict=True, lam=0.0)
    observation = stress.observe_edition(
        full[full["edition_id"].eq(2)], snapshot, coauthored
    )

    assert coauthored == {"joint"}
    assert "joint" not in set(pool["work_id"])
    assert observation.coauthored_selections == 1
    assert observation.known_authored_works == 0


def test_build_series_history_links_immediate_strictly_earlier_predecessor() -> None:
    frame = pd.DataFrame(
        [
            _row("w1", "a1", 1, 1970) | {"series_id": 3.0},
            _row("w2", "a2", 2, 1980) | {"series_id": 3.0},
            _row("w3", "a3", 3, 1990) | {"series_id": 3.0},
            _row("w4", "a4", 4, 1985),  # series-less: never linked
            _row("w5", "a5", 5, 1985) | {"series_id": 9.0},
            _row("w6", "a6", 6, 1985) | {"series_id": 9.0},  # same-year tie
        ]
    )

    history = stress.build_series_history(frame)

    # Each successor links to the most recent strictly earlier same-series
    # edition; a same-year sibling is not a predecessor.
    assert history.pred_key == {"2": "1", "3": "2"}
    assert history.prev_sets("3") == (frozenset({"a2"}), frozenset({"w2"}))
    assert history.prev_sets("1") == (frozenset(), frozenset())
    assert history.prev_sets("6") == (frozenset(), frozenset())


def _two_author_snapshot() -> stress.Snapshot:
    return stress.Snapshot(
        author_works={"a1": ("w1", "w2"), "a2": ("w3", "w4")},
        author_logrank={"a1": 0.0, "a2": 0.0},
        work_logrank={"w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 0.0},
        authorless_works=("x1", "x2"),
        authorless_logrank={"x1": 0.0, "x2": 0.0},
    )


def _one_slot_observation() -> stress.EditionObservation:
    return stress.EditionObservation(
        slots=(("a1", ("w1",)),),
        authorless=("x1",),
        total_authors=1,
        total_authored_works=1,
        total_authorless=1,
        known_authors=1,
        known_authored_works=1,
        known_authorless=1,
        coauthored_selections=0,
    )


def test_score_observation_with_zero_deltas_matches_legacy_path() -> None:
    snapshot = _two_author_snapshot()
    observation = _one_slot_observation()

    legacy = stress.score_observation(observation, snapshot, gamma=1.0, alpha=0.5)
    with_context = stress.score_observation(
        observation,
        snapshot,
        gamma=1.0,
        alpha=0.5,
        delta_author=0.0,
        delta_work=0.0,
        prev_authors=frozenset({"a1"}),
        prev_works=frozenset({"w1", "x1"}),
    )

    assert with_context == legacy


def test_carryover_boost_raises_probability_of_predecessor_choices() -> None:
    snapshot = _two_author_snapshot()
    observation = _one_slot_observation()

    base = stress.score_observation(observation, snapshot, gamma=0.0, alpha=0.0)
    boosted = stress.score_observation(
        observation,
        snapshot,
        gamma=0.0,
        alpha=0.0,
        delta_author=3.0,
        delta_work=3.0,
        prev_authors=frozenset({"a1"}),
        prev_works=frozenset({"w1", "x1"}),
    )

    assert boosted[0] > base[0]  # author component
    assert boosted[1] > base[1]  # work bundle component
    assert boosted[2] > base[2]  # authorless component


def test_carryover_bundle_tables_recompute_only_affected_authors() -> None:
    snapshot = _two_author_snapshot()
    base = stress.bundle_tables(snapshot, alpha=0.7, max_k=2)

    adjusted = stress.carryover_bundle_tables(
        snapshot, 0.7, 2, base, frozenset({"w1"}), delta_work=2.0
    )

    assert adjusted["a2"] is base["a2"]
    expected = stress.log_elementary(np.array([-0.7 * 0.0 + 2.0, -0.7 * 0.0]), 2)
    np.testing.assert_allclose(adjusted["a1"], expected)
    # Zero boost (or no overlap) must return the base tables object untouched.
    assert (
        stress.carryover_bundle_tables(
            snapshot, 0.7, 2, base, frozenset({"w1"}), delta_work=0.0
        )
        is base
    )
    assert (
        stress.carryover_bundle_tables(
            snapshot, 0.7, 2, base, frozenset(), delta_work=2.0
        )
        is base
    )


def test_series_rank_basis_counts_distinct_series_not_editions() -> None:
    rows = [
        # a1 appears in two editions of one series; a2 in two distinct series.
        _row("w1", "a1", 1, 1970) | {"series_id": 3.0},
        _row("w1", "a1", 2, 1980) | {"series_id": 3.0},
        _row("w2", "a2", 3, 1975) | {"series_id": 8.0},
        _row("w3", "a2", 4, 1981) | {"series_id": 9.0},
    ]
    frame = pd.DataFrame(rows)

    by_editions = stress.make_snapshot(
        frame, 1990, strict=True, lam=0.0, rank_basis="editions"
    )
    by_series = stress.make_snapshot(
        frame, 1990, strict=True, lam=0.0, rank_basis="series"
    )

    # Two editions each: tied under the legacy basis.
    assert by_editions.author_logrank["a1"] == by_editions.author_logrank["a2"]
    # One series versus two: the cross-series author outranks the repeater.
    assert by_series.author_logrank["a2"] < by_series.author_logrank["a1"]

    with pytest.raises(ValueError, match="rank basis"):
        stress.make_snapshot(frame, 1990, strict=True, lam=0.0, rank_basis="pages")


def test_simulated_carryover_reproduces_predecessor_under_large_boost() -> None:
    snapshot = _two_author_snapshot()

    simulated = [
        stress.simulate_edition(
            (1,),
            1,
            snapshot,
            gamma=0.0,
            alpha=0.0,
            rng=np.random.default_rng(seed),
            delta_author=20.0,
            delta_work=20.0,
            prev_authors=frozenset({"a2"}),
            prev_works=frozenset({"w4", "x2"}),
        )
        for seed in range(30)
    ]

    assert all(sim.author_works == {"a2": {"w4"}} for sim in simulated)
    assert all(sim.authorless == {"x2"} for sim in simulated)


def test_corpus_metrics_reports_within_series_jaccards_separately() -> None:
    editions = {
        "e1": stress.SimulatedCorpusEdition({"a1": {"w1"}, "a2": {"w2"}}, set()),
        "e2": stress.SimulatedCorpusEdition({"a1": {"w1"}, "a3": {"w3"}}, set()),
        "e3": stress.SimulatedCorpusEdition({"a1": {"w9"}, "a4": {"w4"}}, set()),
    }
    metadata = {
        "e1": (1990, "3"),
        "e2": (2000, "3"),  # same series as e1
        "e3": (2010, None),
    }

    metrics = dict(
        zip(
            stress.METRIC_NAMES,
            stress.corpus_metrics(editions, metadata),
            strict=True,
        )
    )

    # Within pair (e1, e2): authors {a1,a2} vs {a1,a3}, works {w1,w2} vs {w1,w3}.
    assert metrics["jaccard_authors_within"] == pytest.approx(1 / 3)
    assert metrics["jaccard_works_within"] == pytest.approx(1 / 3)
    # Cross-series means exclude the within-series pair.
    assert metrics["jaccard_authors"] == pytest.approx((1 / 3 + 1 / 3) / 2)
    assert metrics["jaccard_works"] == pytest.approx(0.0)


def test_combine_scores_sums_raw_fields_and_rederives_ratios() -> None:
    fresh = dict.fromkeys(stress.RAW_SCORE_FIELDS, 0.0) | {
        "total_log_score": -10.0,
        "fresh_log_score": -10.0,
        "choices": 4,
        "fresh_choices": 4,
        "n_total_authors": 8,
        "n_known_authors": 6,
    }
    chained = dict.fromkeys(stress.RAW_SCORE_FIELDS, 0.0) | {
        "total_log_score": -6.0,
        "chained_log_score": -6.0,
        "choices": 2,
        "chained_choices": 2,
        "n_total_authors": 4,
        "n_known_authors": 4,
    }

    combined = stress.combine_scores(fresh, chained)

    assert combined["total_log_score"] == pytest.approx(-16.0)
    assert combined["log_score_per_choice"] == pytest.approx(-16.0 / 6)
    assert combined["chained_log_per_choice"] == pytest.approx(-3.0)
    assert combined["fresh_log_per_choice"] == pytest.approx(-2.5)
    assert combined["author_coverage"] == pytest.approx(10 / 12)


def test_authorless_identity_receives_likelihood_and_is_simulated() -> None:
    frame = pd.DataFrame(
        [
            _row("w1", "a1", 1, 1980),
            _row("x1", None, 1, 1980),
            _row("x2", None, 2, 1981, afam=0),
        ]
    )
    snapshot = stress.make_snapshot(frame, 1990, strict=True, lam=0.0)
    observation = stress.EditionObservation(
        slots=(("a1", ("w1",)),),
        authorless=("x1",),
        total_authors=1,
        total_authored_works=1,
        total_authorless=1,
        known_authors=1,
        known_authored_works=1,
        known_authorless=1,
        coauthored_selections=0,
    )

    author_lp, work_lp, authorless_lp, choices = stress.score_observation(
        observation, snapshot, gamma=0.0, alpha=1.0
    )
    simulated = stress.simulate_edition(
        (1,), 1, snapshot, gamma=0.0, alpha=1.0, rng=np.random.default_rng(4)
    )

    assert np.isfinite([author_lp, work_lp, authorless_lp]).all()
    assert choices == 2
    assert len(simulated.authorless) == 1
