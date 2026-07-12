"""Unit tests for analysis/overlap/author_work_agreement_models.py helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "overlap"))

from author_work_agreement_models import (  # noqa: E402
    FIT_STATS,
    build_synthetic_pool,
    collect_pair_stats,
    draw_author_corpus_ranks,
    draw_author_pool_ranks,
    draw_global_work_ranks,
    fit_objective,
    random_feasible_matching,
    run_synthetic_trial,
    sample_edition_author_first,
    sample_edition_work_first,
    summarize_pairs,
    weights_for_subset,
    zero_share_rate,
    zipf_weights,
)
from simulate_author_work_overlap import EditionTarget  # noqa: E402


# ── Zipf weights and rankings ─────────────────────────────────────────────────


def test_zipf_weights_uniform_at_alpha_zero():
    w = zipf_weights(5, 0.0)
    assert np.allclose(w, 0.2)


def test_zipf_weights_monotone_and_normalized():
    w = zipf_weights(6, 1.5)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(np.diff(w) < 0)


def test_weights_for_subset_renormalizes_over_eligible():
    rank_map = {"w1": 1, "w2": 2, "w4": 4}
    w = weights_for_subset(["w2", "w4"], rank_map, alpha=1.0)
    # ∝ [1/2, 1/4] → [2/3, 1/3]
    assert np.allclose(w, [2 / 3, 1 / 3])


def test_draw_author_corpus_ranks_frequency_orders_by_freq_then_id():
    author_to_works = {"a": {"w1", "w2", "w3"}}
    freq = {"w1": 1, "w2": 5, "w3": 1}
    ranks = draw_author_corpus_ranks(
        author_to_works, np.random.default_rng(0), "frequency", freq
    )
    assert ranks["a"] == {"w2": 1, "w1": 2, "w3": 3}


def test_draw_global_work_ranks_random_is_permutation():
    ranks = draw_global_work_ranks(
        ["w1", "w2", "w3"], np.random.default_rng(0), "random"
    )
    assert sorted(ranks.tolist()) == [1.0, 2.0, 3.0]


def test_ranks_fixed_within_trial_make_editions_converge():
    corpus = [f"w{i:02d}" for i in range(10)]
    rng = np.random.default_rng(0)
    ranks = draw_author_corpus_ranks({"a": set(corpus)}, rng, "random")
    target = EditionTarget(
        per_author_work_counts=[3], authored_work_count=3, authorless_work_count=0
    )
    eligible = [("a", sorted(corpus))]
    sim1 = sample_edition_author_first(target, eligible, ranks, 8.0, [], rng)
    sim2 = sample_edition_author_first(target, eligible, ranks, 8.0, [], rng)
    top_work = next(w for w, r in ranks["a"].items() if r == 1)
    assert top_work in sim1.work_ids
    assert top_work in sim2.work_ids
    assert len(sim1.work_ids & sim2.work_ids) >= 2


# ── Real-data samplers ────────────────────────────────────────────────────────


def test_author_first_sampler_preserves_shape():
    eligible = [
        ("a1", ["w1", "w2", "w3"]),
        ("a2", ["w4", "w5"]),
        ("a3", ["w6"]),
    ]
    corpora = {aid: set(wids) for aid, wids in eligible}
    target = EditionTarget(
        per_author_work_counts=[2, 1, 1],
        authored_work_count=4,
        authorless_work_count=0,
    )
    rng = np.random.default_rng(1)
    sim = sample_edition_author_first(target, eligible, None, 0.0, [], rng)
    assert sorted((len(v) for v in sim.author_work_ids.values()), reverse=True) == [
        2,
        1,
        1,
    ]
    assert len(sim.author_ids) == 3
    for aid, wids in sim.author_work_ids.items():
        assert wids <= corpora[aid]
    assert sim.authorless_work_ids == set()


def test_author_first_sampler_raises_when_no_author_can_satisfy_count():
    eligible = [("a1", ["w1"])]
    target = EditionTarget(
        per_author_work_counts=[2], authored_work_count=2, authorless_work_count=0
    )
    with pytest.raises(RuntimeError):
        sample_edition_author_first(
            target, eligible, None, 0.0, [], np.random.default_rng(0)
        )


def test_work_first_sampler_draws_distinct_works_and_induces_coauthors():
    work_to_authors = {
        "w1": frozenset({"a1", "a2"}),
        "w2": frozenset({"a3"}),
    }
    target = EditionTarget(
        per_author_work_counts=[], authored_work_count=2, authorless_work_count=0
    )
    sim = sample_edition_work_first(
        target,
        ["w1", "w2"],
        None,
        0.0,
        work_to_authors,
        [],
        np.random.default_rng(0),
    )
    assert sim.work_ids == {"w1", "w2"}
    assert sim.author_ids == {"a1", "a2", "a3"}


def test_work_first_sampler_raises_when_pool_too_small():
    target = EditionTarget(
        per_author_work_counts=[], authored_work_count=3, authorless_work_count=0
    )
    with pytest.raises(RuntimeError):
        sample_edition_work_first(
            target, ["w1", "w2"], None, 0.0, {}, [], np.random.default_rng(0)
        )


def test_samplers_draw_authorless_works():
    target = EditionTarget(
        per_author_work_counts=[1], authored_work_count=1, authorless_work_count=2
    )
    rng = np.random.default_rng(0)
    sim = sample_edition_author_first(
        target, [("a1", ["w1"])], None, 0.0, ["x1", "x2", "x3"], rng
    )
    assert len(sim.authorless_work_ids) == 2
    assert sim.authorless_work_ids <= {"x1", "x2", "x3"}


# ── Author-canon sampler ──────────────────────────────────────────────────────


def _canon_pool():
    eligible = [(f"a{i}", [f"a{i}_w1", f"a{i}_w2"]) for i in range(8)]
    return eligible, [aid for aid, _ in eligible]


def test_author_canon_preserves_shape_and_concentrates_on_top_authors():
    eligible, aids = _canon_pool()
    ranks = {aid: r for r, aid in enumerate(aids, start=1)}  # a0 is rank 1
    target = EditionTarget(
        per_author_work_counts=[2, 1, 1],
        authored_work_count=4,
        authorless_work_count=0,
    )
    rng = np.random.default_rng(0)
    sims = [
        sample_edition_author_first(
            target, eligible, None, 0.0, [], rng, author_ranks=ranks, gamma=12.0
        )
        for _ in range(5)
    ]
    for sim in sims:
        counts = sorted((len(v) for v in sim.author_work_ids.values()), reverse=True)
        assert counts == [2, 1, 1]
        # with near-deterministic gamma, the top-ranked authors are always chosen
        assert {"a0", "a1", "a2"} == sim.author_ids


def test_author_canon_requires_ranks():
    eligible, _ = _canon_pool()
    target = EditionTarget(
        per_author_work_counts=[1], authored_work_count=1, authorless_work_count=0
    )
    with pytest.raises(ValueError):
        sample_edition_author_first(
            target, eligible, None, 0.0, [], np.random.default_rng(0), gamma=2.0
        )


def test_draw_author_pool_ranks_frequency_orders_by_freq_then_id():
    ranks = draw_author_pool_ranks(
        ["a1", "a2", "a3"],
        np.random.default_rng(0),
        "frequency",
        {"a1": 2, "a2": 9, "a3": 2},
    )
    assert ranks == {"a2": 1, "a1": 2, "a3": 3}


def test_zero_share_rate_counts_disjoint_shared_authors():
    aw = {
        "1|1": {"a1": {"w1"}, "a2": {"w2"}},
        "2|1": {"a1": {"w1"}, "a2": {"w9"}, "a3": {"w3"}},
    }
    # shared authors a1 (shares w1) and a2 (disjoint) → 1 of 2
    assert zero_share_rate(aw, ["1|1", "2|1"]) == pytest.approx(0.5)


# ── Corpus inflation (λ) ──────────────────────────────────────────────────────


def test_lambda_inflation_lets_small_corpora_absorb_large_counts():
    # m=1, λ=1 → effective corpus 2, so a count of 2 is feasible: the one real
    # work plus exactly one phantom pick
    eligible = [("a1", ["w1"])]
    target = EditionTarget(
        per_author_work_counts=[2], authored_work_count=2, authorless_work_count=0
    )
    sim = sample_edition_author_first(
        target,
        eligible,
        None,
        0.0,
        [],
        np.random.default_rng(0),
        lam=1.0,
        phantom_prefix="e1",
    )
    works = sim.author_work_ids["a1"]
    assert len(works) == 2
    assert "w1" in works
    phantoms = {w for w in works if w.startswith("~")}
    assert len(phantoms) == 1
    assert all(w.startswith("~e1|a1|") for w in phantoms)


def test_lambda_zero_never_produces_phantoms():
    eligible = [("a1", ["w1", "w2", "w3"])]
    target = EditionTarget(
        per_author_work_counts=[2], authored_work_count=2, authorless_work_count=0
    )
    sim = sample_edition_author_first(
        target, eligible, None, 0.0, [], np.random.default_rng(0), lam=0.0
    )
    assert sim.author_work_ids["a1"] <= {"w1", "w2", "w3"}


def test_phantom_picks_never_shared_across_editions():
    eligible = [("a1", ["w1", "w2"])]
    target = EditionTarget(
        per_author_work_counts=[2], authored_work_count=2, authorless_work_count=0
    )
    rng = np.random.default_rng(0)
    sims = [
        sample_edition_author_first(
            target, eligible, None, 0.0, [], rng, lam=4.0, phantom_prefix=ek
        )
        for ek in ("e1", "e2")
    ]
    ph1 = {w for w in sims[0].work_ids if w.startswith("~")}
    ph2 = {w for w in sims[1].work_ids if w.startswith("~")}
    assert not ph1 & ph2
    assert sims[0].work_ids & sims[1].work_ids <= {"w1", "w2"}


def test_lambda_with_alpha_uses_an_inflated_ranked_pool():
    eligible = [("a1", ["w1", "w2"])]
    ranks = draw_author_corpus_ranks(
        {"a1": {"w1", "w2"}},
        np.random.default_rng(0),
        "frequency",
        {"w1": 2, "w2": 1},
        lam=1.0,
    )
    target = EditionTarget(
        per_author_work_counts=[1], authored_work_count=1, authorless_work_count=0
    )
    sim = sample_edition_author_first(
        target,
        eligible,
        ranks,
        1.0,
        [],
        np.random.default_rng(0),
        lam=1.0,
        phantom_scope="shared",
    )
    assert len(sim.author_work_ids["a1"]) == 1


# ── Decoupled allocation ──────────────────────────────────────────────────────


def test_random_feasible_matching_respects_thresholds():
    # the count of 3 can only go to the size-3 entry
    for seed in range(5):
        assign = random_feasible_matching([1, 3], [3, 1], np.random.default_rng(seed))
        assert assign == [1, 0]


def test_random_feasible_matching_is_a_feasible_permutation():
    sizes = [5, 2, 4, 1]
    counts = [4, 2, 2, 1]
    assign = random_feasible_matching(sizes, counts, np.random.default_rng(0))
    assert sorted(assign) == [0, 1, 2, 3]
    assert all(sizes[j] >= c for j, c in zip(assign, counts))


def test_decouple_preserves_shape_and_author_set_under_gamma():
    eligible, aids = _canon_pool()
    ranks = {aid: r for r, aid in enumerate(aids, start=1)}
    target = EditionTarget(
        per_author_work_counts=[2, 1, 1],
        authored_work_count=4,
        authorless_work_count=0,
    )
    sim = sample_edition_author_first(
        target,
        eligible,
        None,
        0.0,
        [],
        np.random.default_rng(0),
        author_ranks=ranks,
        gamma=12.0,
        decouple=True,
    )
    counts = sorted((len(v) for v in sim.author_work_ids.values()), reverse=True)
    assert counts == [2, 1, 1]
    assert sim.author_ids == {"a0", "a1", "a2"}


def test_decouple_breaks_rank_count_coupling():
    eligible = [("a0", ["x1", "x2", "x3"]), ("a1", ["y1", "y2", "y3"])]
    ranks = {"a0": 1, "a1": 2}
    target = EditionTarget(
        per_author_work_counts=[2, 1], authored_work_count=3, authorless_work_count=0
    )
    # coupled: with near-deterministic gamma the top-ranked author always
    # takes the larger allocation
    for seed in range(20):
        sim = sample_edition_author_first(
            target,
            eligible,
            None,
            0.0,
            [],
            np.random.default_rng(seed),
            author_ranks=ranks,
            gamma=50.0,
        )
        assert len(sim.author_work_ids["a0"]) == 2
    # decoupled: the larger allocation lands on the lower-ranked author too
    got_a1_two = any(
        len(
            sample_edition_author_first(
                target,
                eligible,
                None,
                0.0,
                [],
                np.random.default_rng(seed),
                author_ranks=ranks,
                gamma=50.0,
                decouple=True,
            ).author_work_ids["a1"]
        )
        == 2
        for seed in range(20)
    )
    assert got_a1_two


# ── Fit objective ─────────────────────────────────────────────────────────────


def test_fit_objective_is_rms_of_stat_zs():
    row = {f"{s}_z": 2.0 for s in FIT_STATS}
    assert fit_objective(row) == pytest.approx(2.0)
    row = {f"{s}_z": 0.0 for s in FIT_STATS}
    row[f"{FIT_STATS[0]}_z"] = -5.0
    assert fit_objective(row) == pytest.approx(5.0 / np.sqrt(len(FIT_STATS)))


# ── Pair metrics ──────────────────────────────────────────────────────────────


def test_collect_pair_stats_jaccard_values():
    author_sets = {"1|1": {"a", "b"}, "2|1": {"b", "c"}}
    work_sets = {"1|1": {"w1", "w2"}, "2|1": {"w2", "w3"}}
    stats = collect_pair_stats(work_sets, author_sets, ["1|1", "2|1"])
    assert len(stats) == 1
    s = stats[0]
    assert (s.shared_authors, s.shared_works) == (1, 1)
    assert s.jaccard_authors == pytest.approx(1 / 3)
    assert s.jaccard_works == pytest.approx(1 / 3)


def test_collect_pair_stats_empty_sets_jaccard_zero():
    stats = collect_pair_stats(
        {"1|1": set(), "2|1": set()}, {"1|1": set(), "2|1": set()}, ["1|1", "2|1"]
    )
    assert stats[0].jaccard_authors == 0.0
    assert stats[0].jaccard_works == 0.0


def test_collect_pair_stats_excludes_within_series_pairs():
    sets = {"1|1": {"x"}, "1|2": {"x"}, "60": {"x"}}
    stats = collect_pair_stats(sets, sets, ["1|1", "1|2", "60"])
    # 1|1-vs-1|2 excluded; both standalone pairs kept
    assert len(stats) == 2


def test_summarize_pairs_rates_sum_to_one():
    author_sets = {"a": {"x", "y"}, "b": {"x"}, "c": {"z"}}
    work_sets = {"a": {"w"}, "b": {"w"}, "c": {"v"}}
    summ = summarize_pairs(collect_pair_stats(work_sets, author_sets, ["a", "b", "c"]))
    assert summ.rate_a_gt_w + summ.rate_tie + summ.rate_w_gt_a == pytest.approx(1.0)


# ── Synthetic mode ────────────────────────────────────────────────────────────


def test_build_synthetic_pool_deterministic_under_seed():
    a = build_synthetic_pool(200, 5.0, np.random.default_rng(7))
    b = build_synthetic_pool(200, 5.0, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert a.min() >= 1
    assert abs(a.mean() - 5.0) < 1.5


def test_synthetic_trial_infeasible_returns_none():
    sizes = np.full(5, 1)  # only 5 works in the whole pool
    assert (
        run_synthetic_trial("work", 0.0, 3, sizes, 4, 4, np.random.default_rng(0))
        is None
    )
    assert (
        run_synthetic_trial("author", 0.0, 3, sizes, 4, 4, np.random.default_rng(0))
        is None
    )


def test_author_first_k1_never_yields_more_shared_works_than_authors():
    rng = np.random.default_rng(3)
    for _ in range(10):
        sizes = build_synthetic_pool(30, 3.0, rng)
        summ = run_synthetic_trial("author", 0.0, 1, sizes, 5, 10, rng)
        assert summ is not None
        assert summ.rate_w_gt_a == 0.0


def test_high_alpha_k3_inverts_count_metric_but_not_jaccard():
    """Fractal canonicity with k >= 2: shared works outnumber shared authors
    (count inversion), while J(works) <= J(authors) stays structural under
    equal works-per-author."""
    rng = np.random.default_rng(4)
    summaries = []
    for _ in range(10):
        sizes = np.full(20, 6)
        summ = run_synthetic_trial("author", 12.0, 3, sizes, 6, 10, rng)
        assert summ is not None
        summaries.append(summ)
    assert np.mean([s.rate_w_gt_a for s in summaries]) > np.mean(
        [s.rate_a_gt_w for s in summaries]
    )
    for s in summaries:
        assert s.mean_jaccard_diff >= -1e-9


def test_work_first_trial_runs_and_induces_author_agreement():
    rng = np.random.default_rng(5)
    sizes = build_synthetic_pool(50, 4.0, rng)
    summ = run_synthetic_trial("work", 2.0, 2, sizes, 6, 10, rng)
    assert summ is not None
    assert summ.rate_a_gt_w + summ.rate_tie + summ.rate_w_gt_a == pytest.approx(1.0)
