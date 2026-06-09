from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "overlap"))

from simulate_author_work_overlap import (  # noqa: E402
    DEFAULT_N,
    EditionTarget,
    build_author_to_all_works,
    build_elig_by_author,
    build_real_targets,
    precompute_first_publication_years,
    sample_simulated_edition,
)


def make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "edition_key": "",
        "anthology_publication_year": None,
        "work_id": "",
        "author_id": None,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_default_simulation_count_is_1k():
    assert DEFAULT_N == 1_000


def test_publication_year_pools_exclude_future_authors_and_works():
    df = make_df(
        [
            {
                "edition_key": "early",
                "anthology_publication_year": 2000,
                "work_id": "w_early",
                "author_id": "a_early",
            },
            {
                "edition_key": "early",
                "anthology_publication_year": 2000,
                "work_id": "w_same_year",
                "author_id": "a_same_year",
            },
            {
                "edition_key": "future",
                "anthology_publication_year": 2010,
                "work_id": "w_future",
                "author_id": "a_future",
            },
        ]
    )

    work_first, author_first = precompute_first_publication_years(df)
    eligible = build_elig_by_author(
        ["early", "future"],
        {"early": 2000, "future": 2010},
        build_author_to_all_works(df),
        author_first,
        work_first,
    )

    early = dict(eligible["early"])
    assert set(early) == {"a_early", "a_same_year"}
    assert early["a_early"] == ["w_early"]
    assert early["a_same_year"] == ["w_same_year"]
    assert "a_future" not in early

    future = dict(eligible["future"])
    assert set(future) == {"a_early", "a_same_year", "a_future"}


def test_build_real_targets_preserves_author_and_work_distribution():
    df = make_df(
        [
            {
                "edition_key": "ed",
                "anthology_publication_year": 2000,
                "work_id": "w1",
                "author_id": "a1",
            },
            {
                "edition_key": "ed",
                "anthology_publication_year": 2000,
                "work_id": "w2",
                "author_id": "a1",
            },
            {
                "edition_key": "ed",
                "anthology_publication_year": 2000,
                "work_id": "w2",
                "author_id": "a2",
            },
            {
                "edition_key": "ed",
                "anthology_publication_year": 2000,
                "work_id": "anon",
                "author_id": None,
            },
        ]
    )

    target = build_real_targets(df)["ed"]

    assert target.per_author_work_counts == [2, 1]
    assert target.authored_work_count == 2
    assert target.authorless_work_count == 1


def test_sample_simulated_edition_preserves_counts_and_conditional_works():
    target = EditionTarget(
        per_author_work_counts=[2, 1, 1],
        authored_work_count=4,
        authorless_work_count=1,
    )
    eligible_by_author = [
        ("a1", ["w1", "w2", "w3"]),
        ("a2", ["w4"]),
        ("a3", ["w5"]),
    ]

    sim = sample_simulated_edition(
        target,
        eligible_by_author,
        ["anon1", "anon2"],
        random.Random(2),
    )

    assert sorted(len(wids) for wids in sim.author_work_ids.values()) == [1, 1, 2]
    assert len(sim.author_ids) == 3
    assert len(set().union(*sim.author_work_ids.values())) == 4
    assert len(sim.authorless_work_ids) == 1
    assert len(sim.work_ids) == 5

    eligible_lookup = {aid: set(wids) for aid, wids in eligible_by_author}
    for aid, selected_work_ids in sim.author_work_ids.items():
        assert selected_work_ids <= eligible_lookup[aid]


def test_sample_simulated_edition_preserves_coauthored_duplicate_slots():
    target = EditionTarget(
        per_author_work_counts=[1, 1],
        authored_work_count=1,
        authorless_work_count=0,
    )
    eligible_by_author = [
        ("a1", ["shared", "solo1"]),
        ("a2", ["shared", "solo2"]),
    ]

    sim = sample_simulated_edition(
        target,
        eligible_by_author,
        [],
        random.Random(1),
    )

    assert sim.author_ids == {"a1", "a2"}
    assert sim.work_ids == {"shared"}
    assert all(wids == {"shared"} for wids in sim.author_work_ids.values())
