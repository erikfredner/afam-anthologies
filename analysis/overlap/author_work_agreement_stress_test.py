"""Chronological stress test for author- versus work-centered selection.

This runner implements the adversarial design requested after review of
``author_work_agreement_models.py``.  It does not fit the same corpus summaries
it later evaluates.  Instead it:

* uses candidates seen in *any* earlier anthology record (never the focal
  edition) and reports the resulting identity coverage;
* compares drop/include/coalesce definitions of a work;
* gives latent unseen works stable, shared identities;
* gives uniform, author-only, work-only, and mixed models identical edition
  author-count and per-author allocation margins;
* tunes parameters by prequential conditional log score before a cutoff;
* freezes ranks and parameters, then scores later anthologies by identity;
* reserves pairwise agreement and temporal slopes for a split-simulation joint
  posterior-predictive check.

Within-series carryover: a new edition of an existing series (NAAAL ed.2
after ed.1) is largely a revision of its predecessor — observed work
retention into a same-series successor averages ~74% versus ~18% into other
editions.  A single work-rank effect cannot represent both regimes, and the
post-1998 held-out period is dominated by NAAAL follow-ups, so any "work
effect" estimated without series structure conflates cross-editor consensus
with series inertia.  The model therefore adds two carryover log-boosts,
``delta_author`` and ``delta_work``, applied to candidates that appeared in
the focal edition's most recent same-series predecessor (a realized-history
covariate, never a peek at the focal outcome).  Rank scores are additionally
decontaminated by counting distinct *series* rather than editions
(``--rank-basis``), so a work carried through four NAAAL editions counts once
toward cross-editor canonicity.  Held-out scores are reported separately for
editions with and without a same-series predecessor, and the simulation
chains carryover through *simulated* predecessors so the posterior-predictive
corpus reproduces series autocorrelation generatively.

The database has anthology dates rather than original publication dates or
complete oeuvres.  Accordingly, the candidate pool is explicitly a broader
anthology-record proxy, not bibliographic eligibility.  Choices outside the
frozen pool are conditioned on as observed novelty and reported as coverage;
they are never smuggled into the risk set from the focal outcome.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata

from afam import DATA_DIR
from afam.db import query as query_db

sys.path.insert(0, str(Path(__file__).parent))

from author_work_agreement_models import (  # noqa: E402
    WORK_UNIT_MODES,
    preprocess_work_units,
)


DEFAULT_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT_LAMBDAS = (0.0, 1.0, 3.0, 7.0)
DEFAULT_DELTAS = (0.0, 2.0, 4.0, 6.0)
DEFAULT_CUTOFFS = (1998, 1991)
DEFAULT_SEED = 20260711
DEFAULT_TRIALS = 400

RANK_BASES = ("series", "editions")

OUT_GRID = DATA_DIR / "author_work_agreement_stress_train_grid.csv"
OUT_HOLDOUT = DATA_DIR / "author_work_agreement_stress_holdout.csv"
OUT_PPC = DATA_DIR / "author_work_agreement_stress_ppc.csv"
OUT_COVERAGE = DATA_DIR / "author_work_agreement_stress_coverage.csv"
OUT_SERIES_GAP = DATA_DIR / "author_work_agreement_stress_series_gap.csv"


SQL = r"""
SELECT DISTINCT
    w.id AS work_id,
    w.parent_id,
    e.id AS edition_id,
    e.id::text AS edition_key,
    e.year AS anthology_publication_year,
    e.series_id,
    e.edition_number,
    a.id AS author_id,
    CASE WHEN EXISTS (
        SELECT 1
        FROM data_edition_literary_traditions elt
        JOIN data_literarytradition lt ON lt.id = elt.literarytradition_id
        WHERE elt.edition_id = e.id
          AND lt.name = 'African-American Literature'
    ) THEN 1 ELSE 0 END AS is_afam
FROM data_work w
JOIN data_workinanthology wia ON wia.work_id = w.id
JOIN data_volume v ON v.id = wia.volume_id
JOIN data_edition e ON e.id = v.edition_id
LEFT JOIN data_work_authors wa ON wa.work_id = w.id
LEFT JOIN data_author a ON a.id = wa.author_id
WHERE e.year IS NOT NULL
"""


@dataclass(frozen=True)
class Snapshot:
    """Candidate identities and ranks frozen strictly before an outcome."""

    author_works: dict[str, tuple[str, ...]]
    author_logrank: dict[str, float]
    work_logrank: dict[str, float]
    authorless_works: tuple[str, ...]
    authorless_logrank: dict[str, float]


@dataclass(frozen=True)
class EditionObservation:
    """Known-identity portion of one anthology, conditional on its novelty."""

    slots: tuple[tuple[str, tuple[str, ...]], ...]
    authorless: tuple[str, ...]
    total_authors: int
    total_authored_works: int
    total_authorless: int
    known_authors: int
    known_authored_works: int
    known_authorless: int
    coauthored_selections: int


@dataclass(frozen=True)
class SimulatedCorpusEdition:
    author_works: dict[str, set[str]]
    authorless: set[str]

    @property
    def authors(self) -> set[str]:
        return set(self.author_works)

    @property
    def works(self) -> set[str]:
        authored = (
            set().union(*self.author_works.values()) if self.author_works else set()
        )
        return authored | self.authorless


@dataclass(frozen=True)
class SimulationKernel:
    author_ids: np.ndarray
    feasible_indices: dict[int, np.ndarray]
    base_logweights: dict[int, np.ndarray]
    positions: dict[int, dict[str, int]]


@dataclass(frozen=True)
class SeriesHistory:
    """Same-series predecessor links and realized edition contents.

    ``pred_key`` maps an edition_key to the edition_key of the most recent
    strictly earlier edition in the same series (across the full record, not
    just AFAM-tagged editions, because series inertia does not respect the
    corpus tag).  ``contents`` holds each edition's realized author and work
    sets in the active work-unit space.
    """

    pred_key: dict[str, str]
    contents: dict[str, tuple[frozenset[str], frozenset[str]]]

    def prev_sets(self, key: str) -> tuple[frozenset[str], frozenset[str]]:
        predecessor = self.pred_key.get(key)
        if predecessor is None:
            return frozenset(), frozenset()
        return self.contents[predecessor]


EMPTY_HISTORY = SeriesHistory(pred_key={}, contents={})


def build_series_history(df: pd.DataFrame) -> SeriesHistory:
    """Link every edition to its most recent same-series predecessor."""
    editions = df[
        ["edition_key", "anthology_publication_year", "series_id"]
    ].drop_duplicates("edition_key")
    pred_key: dict[str, str] = {}
    for _, group in editions.dropna(subset=["series_id"]).groupby("series_id"):
        ordered = group.sort_values(["anthology_publication_year", "edition_key"])
        keys = list(ordered["edition_key"])
        years = list(ordered["anthology_publication_year"])
        for i in range(1, len(keys)):
            earlier = [j for j in range(i) if years[j] < years[i]]
            if earlier:
                pred_key[keys[i]] = keys[earlier[-1]]
    contents = {
        str(key): (
            frozenset(group["author_id"].dropna()),
            frozenset(group["work_id"]),
        )
        for key, group in df.groupby("edition_key")
    }
    return SeriesHistory(pred_key=pred_key, contents=contents)


def _csv_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def load_all_anthology_frame(work_unit: str) -> pd.DataFrame:
    """Load all dated anthology selections and apply one work estimand."""
    raw = query_db(SQL)
    raw["anthology_publication_year"] = raw["anthology_publication_year"].astype(int)
    df = preprocess_work_units(raw, work_unit)
    df["work_id"] = df["work_id"].astype(str)
    df["author_id"] = df["author_id"].map(lambda x: None if pd.isna(x) else str(int(x)))
    df["edition_key"] = df["edition_key"].astype(str)
    return df


def split_coauthored(df: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Exclude multi-author work bundles rather than silently dropping coauthors."""
    authored = df.dropna(subset=["author_id"])
    counts = authored.groupby("work_id")["author_id"].nunique()
    coauthored = set(counts[counts > 1].index.astype(str))
    return df[~df["work_id"].isin(coauthored)].copy(), coauthored


def _midrank_log(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    keys = sorted(values)
    ranks = rankdata(-np.array([values[k] for k in keys]), method="average")
    return {k: float(math.log(r)) for k, r in zip(keys, ranks, strict=True)}


def latent_work_id(author_id: str, index: int) -> str:
    return f"~shared|{author_id}|{index}"


def make_snapshot(
    df: pd.DataFrame,
    boundary_year: int,
    *,
    strict: bool,
    lam: float,
    rank_basis: str = "series",
) -> Snapshot:
    """Build ranks from records before (or through) ``boundary_year``.

    Candidate identities come from every anthology in the database; selection
    scores come only from AFAM editions in the same historical window.  With
    ``rank_basis="series"`` (the default), selection counts are the number of
    distinct *series* (series-less editions each count as their own), so a
    work carried through several editions of one series counts once toward
    cross-editor canonicity instead of once per revision.
    ``rank_basis="editions"`` restores the legacy edition-count behavior.
    """
    if rank_basis not in RANK_BASES:
        raise ValueError(f"Unknown rank basis {rank_basis!r}; choose {RANK_BASES}.")
    in_window = (
        df["anthology_publication_year"] < boundary_year
        if strict
        else df["anthology_publication_year"] <= boundary_year
    )
    pool = df[in_window]
    authored_pool = pool.dropna(subset=["author_id"])
    author_works: dict[str, tuple[str, ...]] = {
        str(aid): tuple(sorted(g["work_id"].unique()))
        for aid, g in authored_pool.groupby("author_id")
    }

    afam = pool[pool["is_afam"].eq(1)]
    if rank_basis == "series":
        afam = afam.assign(
            _count_unit=afam["series_id"]
            .map(lambda s: f"s{s}" if pd.notna(s) else None)
            .fillna("e" + afam["edition_key"])
        )
    else:
        afam = afam.assign(_count_unit=afam["edition_id"])
    author_counts = (
        afam.dropna(subset=["author_id"]).groupby("author_id")["_count_unit"].nunique()
    )
    author_values = {a: float(author_counts.get(a, 0)) for a in author_works}
    author_logrank = _midrank_log(author_values)

    work_counts = afam.groupby("work_id")["_count_unit"].nunique()
    work_values: dict[str, float] = {}
    inflated_author_works: dict[str, tuple[str, ...]] = {}
    for aid, works in author_works.items():
        denominator = float(author_counts.get(aid, 0))
        smoothed_unseen_q = 0.5 / (denominator + 1.0)
        for wid in works:
            # Conditional work recurrence given that this author is selected.
            work_values[wid] = (float(work_counts.get(wid, 0)) + 0.5) / (
                denominator + 1.0
            )
        phantoms = tuple(
            latent_work_id(aid, i) for i in range(int(round(lam * len(works))))
        )
        # Latent works participate in the same smoothed-q ranking as observed
        # zero-count works.  Assigning them an arbitrary rank after ranking
        # would break tied midranks and make lambda change the score scale.
        work_values.update({wid: smoothed_unseen_q for wid in phantoms})
        inflated_author_works[aid] = works + phantoms

    work_logrank = _midrank_log(work_values)

    authored_ids = set(authored_pool["work_id"])
    authorless_pool = pool[
        pool["author_id"].isna() & ~pool["work_id"].isin(authored_ids)
    ]
    authorless_works = tuple(sorted(authorless_pool["work_id"].unique()))
    authorless_counts = (
        afam[afam["author_id"].isna() & ~afam["work_id"].isin(authored_ids)]
        .groupby("work_id")["_count_unit"]
        .nunique()
    )
    authorless_logrank = _midrank_log(
        {w: float(authorless_counts.get(w, 0)) for w in authorless_works}
    )
    return Snapshot(
        author_works=inflated_author_works,
        author_logrank=author_logrank,
        work_logrank=work_logrank,
        authorless_works=authorless_works,
        authorless_logrank=authorless_logrank,
    )


def observe_edition(
    group: pd.DataFrame,
    snapshot: Snapshot,
    coauthored: set[str],
) -> EditionObservation:
    """Condition on novelty counts and retain only identities in the snapshot."""
    authored = group.dropna(subset=["author_id"])
    authored_ids = set(authored["work_id"])
    total_authorless_set = (
        set(group.loc[group["author_id"].isna(), "work_id"]) - authored_ids
    )
    slots: list[tuple[str, tuple[str, ...]]] = []
    for aid, ag in authored.groupby("author_id"):
        aid = str(aid)
        if aid not in snapshot.author_works:
            continue
        known = tuple(sorted(set(ag["work_id"]) & set(snapshot.author_works[aid])))
        # Keep a known author even when every selected work is novel.  That
        # scores the author identity with a zero-known-work bundle while
        # conditioning the unmodeled work identities on their observed novelty.
        slots.append((aid, known))
    slots.sort(key=lambda item: (-len(item[1]), item[0]))
    known_authorless = tuple(
        sorted(total_authorless_set & set(snapshot.authorless_works))
    )
    return EditionObservation(
        slots=tuple(slots),
        authorless=known_authorless,
        total_authors=int(authored["author_id"].nunique()),
        total_authored_works=int(authored["work_id"].nunique()),
        total_authorless=len(total_authorless_set),
        known_authors=len(slots),
        known_authored_works=len(set().union(*(set(w) for _, w in slots)))
        if slots
        else 0,
        known_authorless=len(known_authorless),
        coauthored_selections=int(
            group.loc[group["work_id"].isin(coauthored), "work_id"].nunique()
        ),
    )


def logcomb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def log_elementary(logweights: np.ndarray, max_k: int) -> np.ndarray:
    """Log elementary-symmetric-polynomial table through order ``max_k``."""
    upto = min(max_k, len(logweights))
    dp = np.full(upto + 1, -np.inf)
    dp[0] = 0.0
    seen = 0
    for lw in logweights:
        seen += 1
        for j in range(min(upto, seen), 0, -1):
            dp[j] = np.logaddexp(dp[j], dp[j - 1] + lw)
    return dp


def bundle_tables(
    snapshot: Snapshot, alpha: float, max_k: int
) -> dict[str, np.ndarray]:
    return {
        aid: log_elementary(
            np.array([-alpha * snapshot.work_logrank[w] for w in works]), max_k
        )
        for aid, works in snapshot.author_works.items()
    }


def _boosted_work_logweights(
    works: tuple[str, ...],
    logrank: dict[str, float],
    alpha: float,
    prev_works: frozenset[str],
    delta_work: float,
) -> np.ndarray:
    """Per-work log weight with the same-series carryover boost applied."""
    return np.array(
        [-alpha * logrank[w] + (delta_work if w in prev_works else 0.0) for w in works]
    )


def carryover_bundle_tables(
    snapshot: Snapshot,
    alpha: float,
    max_k: int,
    base_tables: dict[str, np.ndarray],
    prev_works: frozenset[str],
    delta_work: float,
) -> dict[str, np.ndarray]:
    """Bundle tables with carryover boosts; only affected authors recomputed."""
    if delta_work == 0.0 or not prev_works:
        return base_tables
    tables = dict(base_tables)
    for aid, works in snapshot.author_works.items():
        if any(w in prev_works for w in works):
            tables[aid] = log_elementary(
                _boosted_work_logweights(
                    works, snapshot.work_logrank, alpha, prev_works, delta_work
                ),
                max_k,
            )
    return tables


def score_observation(
    observation: EditionObservation,
    snapshot: Snapshot,
    gamma: float,
    alpha: float,
    tables: dict[str, np.ndarray] | None = None,
    *,
    delta_author: float = 0.0,
    delta_work: float = 0.0,
    prev_authors: frozenset[str] = frozenset(),
    prev_works: frozenset[str] = frozenset(),
) -> tuple[float, float, float, int]:
    """Conditional log score for author bundles, works, and authorless works.

    ``tables`` must already include the carryover adjustment when
    ``delta_work`` is nonzero (see :func:`carryover_bundle_tables`).
    """
    if not observation.slots and not observation.authorless:
        return 0.0, 0.0, 0.0, 0
    max_k = max((len(works) for _, works in observation.slots), default=0)
    if tables is None:
        tables = carryover_bundle_tables(
            snapshot,
            alpha,
            max_k,
            bundle_tables(snapshot, alpha, max_k),
            prev_works,
            delta_work,
        )
    unused = set(snapshot.author_works)
    author_lp = work_lp = authorless_lp = 0.0
    choices = 0
    for actual_author, picked in observation.slots:
        k = len(picked)
        candidates = [a for a in unused if len(snapshot.author_works[a]) >= k]
        logs = np.array(
            [
                -gamma * snapshot.author_logrank[a]
                + (delta_author if a in prev_authors else 0.0)
                + tables[a][k]
                - logcomb(len(snapshot.author_works[a]), k)
                for a in candidates
            ]
        )
        author_lp += float(logs[candidates.index(actual_author)] - logsumexp(logs))
        work_lp += float(
            sum(
                -alpha * snapshot.work_logrank[w]
                + (delta_work if w in prev_works else 0.0)
                for w in picked
            )
            - tables[actual_author][k]
        )
        unused.remove(actual_author)
        choices += 1

    if observation.authorless:
        k = len(observation.authorless)
        logweights = _boosted_work_logweights(
            snapshot.authorless_works,
            snapshot.authorless_logrank,
            alpha,
            prev_works,
            delta_work,
        )
        table = log_elementary(logweights, k)
        authorless_lp = float(
            sum(
                -alpha * snapshot.authorless_logrank[w]
                + (delta_work if w in prev_works else 0.0)
                for w in observation.authorless
            )
            - table[k]
        )
        choices += 1
    return author_lp, work_lp, authorless_lp, choices


RAW_SCORE_FIELDS = (
    "author_log_score",
    "work_log_score",
    "authorless_log_score",
    "total_log_score",
    "choices",
    "chained_log_score",
    "chained_choices",
    "fresh_log_score",
    "fresh_choices",
    "n_total_authors",
    "n_known_authors",
    "n_total_authored_works",
    "n_known_authored_works",
    "n_total_authorless",
    "n_known_authorless",
    "coauthored_selection_rows",
)


def finalize_score(raw: dict[str, float]) -> dict[str, float]:
    """Add the derived per-choice and coverage fields to a raw score dict."""
    out = dict(raw)
    out["log_score_per_choice"] = (
        raw["total_log_score"] / raw["choices"] if raw["choices"] else float("nan")
    )
    out["chained_log_per_choice"] = (
        raw["chained_log_score"] / raw["chained_choices"]
        if raw["chained_choices"]
        else float("nan")
    )
    out["fresh_log_per_choice"] = (
        raw["fresh_log_score"] / raw["fresh_choices"]
        if raw["fresh_choices"]
        else float("nan")
    )
    for name, known, total in (
        ("author_coverage", "n_known_authors", "n_total_authors"),
        ("authored_work_coverage", "n_known_authored_works", "n_total_authored_works"),
        ("authorless_coverage", "n_known_authorless", "n_total_authorless"),
    ):
        out[name] = raw[known] / raw[total] if raw[total] else float("nan")
    return out


def combine_scores(*parts: dict[str, float]) -> dict[str, float]:
    """Sum the raw fields of several period scores and re-derive the ratios."""
    raw = {field: sum(p[field] for p in parts) for field in RAW_SCORE_FIELDS}
    return finalize_score(raw)


def score_period(
    pool_df: pd.DataFrame,
    outcomes: pd.DataFrame,
    coauthored: set[str],
    gamma: float,
    alpha: float,
    lam: float,
    *,
    prequential: bool,
    cutoff: int,
    delta_author: float = 0.0,
    delta_work: float = 0.0,
    history: SeriesHistory = EMPTY_HISTORY,
    rank_basis: str = "series",
    snapshot_cache: dict[tuple[int, bool, float], Snapshot] | None = None,
    bundle_cache: dict[tuple[int, bool, float, float, int], dict[str, np.ndarray]]
    | None = None,
) -> dict[str, float]:
    """Score a period either prequentially or against one frozen snapshot.

    Ranks (and, when frozen, the candidate pool) never look past the cutoff,
    but the carryover covariate always uses the focal edition's realized
    same-series predecessor: that is history available at prediction time,
    not outcome leakage.
    """
    author_lp = work_lp = authorless_lp = 0.0
    choices = 0
    chained_lp = fresh_lp = 0.0
    chained_choices = fresh_choices = 0
    totals = Counter()
    cache = snapshot_cache if snapshot_cache is not None else {}
    bundles = bundle_cache if bundle_cache is not None else {}

    def cached_snapshot(year: int, strict: bool) -> Snapshot:
        key = (year, strict, lam)
        if key not in cache:
            cache[key] = make_snapshot(
                pool_df, year, strict=strict, lam=lam, rank_basis=rank_basis
            )
        return cache[key]

    frozen = None if prequential else cached_snapshot(cutoff, False)
    for _, group in outcomes.groupby("edition_id", sort=False):
        year = int(group["anthology_publication_year"].iloc[0])
        edition_key = str(group["edition_key"].iloc[0])
        snapshot = cached_snapshot(year, True) if prequential else frozen
        assert snapshot is not None
        obs = observe_edition(group, snapshot, coauthored)
        max_k = max((len(works) for _, works in obs.slots), default=0)
        bundle_key = (year if prequential else cutoff, prequential, lam, alpha, max_k)
        if bundle_key not in bundles:
            bundles[bundle_key] = bundle_tables(snapshot, alpha, max_k)
        prev_authors, prev_works = history.prev_sets(edition_key)
        tables = carryover_bundle_tables(
            snapshot, alpha, max_k, bundles[bundle_key], prev_works, delta_work
        )
        a_lp, w_lp, x_lp, n_choices = score_observation(
            obs,
            snapshot,
            gamma,
            alpha,
            tables,
            delta_author=delta_author,
            delta_work=delta_work,
            prev_authors=prev_authors,
            prev_works=prev_works,
        )
        author_lp += a_lp
        work_lp += w_lp
        authorless_lp += x_lp
        choices += n_choices
        if edition_key in history.pred_key:
            chained_lp += a_lp + w_lp + x_lp
            chained_choices += n_choices
        else:
            fresh_lp += a_lp + w_lp + x_lp
            fresh_choices += n_choices
        totals.update(
            total_authors=obs.total_authors,
            total_authored_works=obs.total_authored_works,
            total_authorless=obs.total_authorless,
            known_authors=obs.known_authors,
            known_authored_works=obs.known_authored_works,
            known_authorless=obs.known_authorless,
            coauthored_selections=obs.coauthored_selections,
        )
    return finalize_score(
        {
            "author_log_score": author_lp,
            "work_log_score": work_lp,
            "authorless_log_score": authorless_lp,
            "total_log_score": author_lp + work_lp + authorless_lp,
            "choices": choices,
            "chained_log_score": chained_lp,
            "chained_choices": chained_choices,
            "fresh_log_score": fresh_lp,
            "fresh_choices": fresh_choices,
            "n_total_authors": totals["total_authors"],
            "n_known_authors": totals["known_authors"],
            "n_total_authored_works": totals["total_authored_works"],
            "n_known_authored_works": totals["known_authored_works"],
            "n_total_authorless": totals["total_authorless"],
            "n_known_authorless": totals["known_authorless"],
            "coauthored_selection_rows": float(totals["coauthored_selections"]),
        }
    )


def fit_grid(
    pool_df: pd.DataFrame,
    outcomes: pd.DataFrame,
    coauthored: set[str],
    grid: list[float],
    lambdas: list[float],
    deltas: list[float],
    cutoff: int,
    work_unit: str,
    history: SeriesHistory,
    rank_basis: str,
) -> pd.DataFrame:
    """Prequential grid over (γ, α, λ, δ_A, δ_W).

    The carryover deltas only touch editions with a same-series predecessor,
    so the fresh editions are scored once per (γ, α, λ) and only the chained
    editions are re-scored across the delta grid.
    """
    chained_mask = outcomes["edition_key"].isin(history.pred_key)
    fresh_outcomes = outcomes[~chained_mask]
    chained_outcomes = outcomes[chained_mask]
    rows: list[dict[str, float | str | int]] = []
    snapshot_cache: dict[tuple[int, bool, float], Snapshot] = {}
    bundle_cache: dict[tuple[int, bool, float, float, int], dict[str, np.ndarray]] = {}
    empty_raw = finalize_score(dict.fromkeys(RAW_SCORE_FIELDS, 0.0))
    for lam in lambdas:
        for alpha in grid:
            for gamma in grid:
                shared = dict(
                    prequential=True,
                    cutoff=cutoff,
                    history=history,
                    rank_basis=rank_basis,
                    snapshot_cache=snapshot_cache,
                    bundle_cache=bundle_cache,
                )
                fresh = (
                    score_period(
                        pool_df,
                        fresh_outcomes,
                        coauthored,
                        gamma,
                        alpha,
                        lam,
                        **shared,
                    )
                    if not fresh_outcomes.empty
                    else empty_raw
                )
                for delta_author in deltas:
                    for delta_work in deltas:
                        chained = (
                            score_period(
                                pool_df,
                                chained_outcomes,
                                coauthored,
                                gamma,
                                alpha,
                                lam,
                                delta_author=delta_author,
                                delta_work=delta_work,
                                **shared,
                            )
                            if not chained_outcomes.empty
                            else empty_raw
                        )
                        rows.append(
                            {
                                "work_unit": work_unit,
                                "cutoff": cutoff,
                                "gamma": gamma,
                                "alpha": alpha,
                                "lambda": lam,
                                "delta_author": delta_author,
                                "delta_work": delta_work,
                                **combine_scores(fresh, chained),
                            }
                        )
    return pd.DataFrame(rows)


def family_best(grid: pd.DataFrame) -> dict[str, pd.Series]:
    """Best prequential row in eight genuinely nested model families."""
    no_series = grid["delta_author"].eq(0) & grid["delta_work"].eq(0)
    masks = {
        "uniform": grid["gamma"].eq(0) & grid["alpha"].eq(0) & no_series,
        "author-only": grid["gamma"].gt(0) & grid["alpha"].eq(0) & no_series,
        "work-only": grid["gamma"].eq(0) & grid["alpha"].gt(0) & no_series,
        "mixed": grid["gamma"].gt(0) & grid["alpha"].gt(0) & no_series,
        "series-only": grid["gamma"].eq(0) & grid["alpha"].eq(0) & ~no_series,
        "author+series": grid["gamma"].gt(0) & grid["alpha"].eq(0) & ~no_series,
        "work+series": grid["gamma"].eq(0) & grid["alpha"].gt(0) & ~no_series,
        "mixed+series": grid["gamma"].gt(0) & grid["alpha"].gt(0) & ~no_series,
    }
    result: dict[str, pd.Series] = {}
    for family, mask in masks.items():
        subset = grid[mask]
        if not subset.empty:
            result[family] = subset.loc[subset["total_log_score"].idxmax()]
    return result


def sample_product_subset_from_logweights(
    works: tuple[str, ...],
    logweights: np.ndarray,
    k: int,
    rng: np.random.Generator,
    suffix: np.ndarray | None = None,
) -> tuple[str, ...]:
    """Exact subset draw with probability ∝ the product of exp(logweights)."""
    if k == 0:
        return ()
    if k >= len(works):
        return tuple(works)
    if np.all(logweights == logweights[0]):
        picked = rng.choice(len(works), size=k, replace=False)
        return tuple(sorted(works[int(i)] for i in picked))
    n = len(works)
    if suffix is None:
        suffix = product_suffix_from_logweights(logweights, k)
    chosen: list[str] = []
    need = k
    for i, wid in enumerate(works):
        if need == 0:
            break
        if n - i == need:
            chosen.extend(works[i:])
            break
        log_include = logweights[i] + suffix[i + 1, need - 1]
        probability = float(np.exp(log_include - suffix[i, need]))
        if rng.random() < probability:
            chosen.append(wid)
            need -= 1
    return tuple(chosen)


def product_suffix_from_logweights(logweights: np.ndarray, k: int) -> np.ndarray:
    """Suffix DP over explicit per-work log weights."""
    n = len(logweights)
    suffix = np.full((n + 1, k + 1), -np.inf)
    suffix[:, 0] = 0.0
    for i in range(n - 1, -1, -1):
        lw = logweights[i]
        for j in range(1, min(k, n - i) + 1):
            suffix[i, j] = np.logaddexp(suffix[i + 1, j], lw + suffix[i + 1, j - 1])
    return suffix


def sample_product_subset(
    works: tuple[str, ...],
    logrank: dict[str, float],
    alpha: float,
    k: int,
    rng: np.random.Generator,
    suffix: np.ndarray | None = None,
) -> tuple[str, ...]:
    """Exact subset draw with probability proportional to product weights."""
    if alpha == 0 and 0 < k < len(works):
        picked = rng.choice(len(works), size=k, replace=False)
        return tuple(sorted(works[int(i)] for i in picked))
    logweights = np.array([-alpha * logrank[w] for w in works])
    return sample_product_subset_from_logweights(works, logweights, k, rng, suffix)


def product_suffix(
    works: tuple[str, ...], logrank: dict[str, float], alpha: float, k: int
) -> np.ndarray:
    """Suffix DP reused across posterior-predictive trials."""
    return product_suffix_from_logweights(
        np.array([-alpha * logrank[w] for w in works]), k
    )


def build_simulation_kernel(
    snapshot: Snapshot,
    gamma: float,
    tables: dict[str, np.ndarray],
    counts: set[int],
) -> SimulationKernel:
    authors = np.array(list(snapshot.author_works), dtype=object)
    sizes = np.array([len(snapshot.author_works[str(a)]) for a in authors])
    feasible: dict[int, np.ndarray] = {}
    logweights: dict[int, np.ndarray] = {}
    positions: dict[int, dict[str, int]] = {}
    for k in counts:
        indices = np.flatnonzero(sizes >= k)
        feasible[k] = indices
        logweights[k] = np.array(
            [
                -gamma * snapshot.author_logrank[str(authors[i])]
                + tables[str(authors[i])][k]
                - logcomb(int(sizes[i]), k)
                for i in indices
            ]
        )
        positions[k] = {str(authors[int(i)]): pos for pos, i in enumerate(indices)}
    return SimulationKernel(authors, feasible, logweights, positions)


def _carryover_author_logweights(
    snapshot: Snapshot,
    kernel: SimulationKernel,
    base_tables: dict[str, np.ndarray],
    adjusted_tables: dict[str, np.ndarray],
    counts: set[int],
    delta_author: float,
    prev_authors: frozenset[str],
) -> dict[int, np.ndarray]:
    """Per-k author log weights patched for one edition's carryover boosts."""
    adjusted: dict[int, np.ndarray] = {}
    for k in counts:
        logs = kernel.base_logweights[k].copy()
        pos = kernel.positions[k]
        if delta_author != 0.0:
            for author in prev_authors:
                position = pos.get(author)
                if position is not None:
                    logs[position] += delta_author
        for author, table in adjusted_tables.items():
            if table is base_tables.get(author):
                continue
            position = pos.get(author)
            if position is not None:
                logs[position] += table[k] - base_tables[author][k]
        adjusted[k] = logs
    return adjusted


def simulate_edition(
    counts: tuple[int, ...],
    authorless_count: int,
    snapshot: Snapshot,
    gamma: float,
    alpha: float,
    rng: np.random.Generator,
    tables: dict[str, np.ndarray] | None = None,
    kernel: SimulationKernel | None = None,
    subset_cache: dict[tuple[str, int], np.ndarray] | None = None,
    *,
    delta_author: float = 0.0,
    delta_work: float = 0.0,
    prev_authors: frozenset[str] = frozenset(),
    prev_works: frozenset[str] = frozenset(),
) -> SimulatedCorpusEdition:
    max_k = max(counts, default=0)
    tables = tables if tables is not None else bundle_tables(snapshot, alpha, max_k)
    if kernel is None:
        kernel = build_simulation_kernel(snapshot, gamma, tables, set(counts))
    carryover = bool(prev_works or prev_authors) and (
        delta_work != 0.0 or delta_author != 0.0
    )
    adjusted_tables = tables
    author_logweights: dict[int, np.ndarray] = kernel.base_logweights
    if carryover:
        adjusted_tables = carryover_bundle_tables(
            snapshot, alpha, max_k, tables, prev_works, delta_work
        )
        author_logweights = _carryover_author_logweights(
            snapshot,
            kernel,
            tables,
            adjusted_tables,
            set(counts),
            delta_author,
            prev_authors,
        )
    used = np.zeros(len(kernel.author_ids), dtype=bool)
    cache = subset_cache if subset_cache is not None else {}
    author_works: dict[str, set[str]] = {}
    for k in counts:
        feasible = kernel.feasible_indices[k]
        available = ~used[feasible]
        candidate_indices = feasible[available]
        logs = author_logweights[k][available]
        probabilities = np.exp(logs - logsumexp(logs))
        selected_position = int(rng.choice(len(candidate_indices), p=probabilities))
        author_index = int(candidate_indices[selected_position])
        author = str(kernel.author_ids[author_index])
        oeuvre = snapshot.author_works[author]
        boosted = (
            delta_work != 0.0 and carryover and any(w in prev_works for w in oeuvre)
        )
        if boosted:
            author_works[author] = set(
                sample_product_subset_from_logweights(
                    oeuvre,
                    _boosted_work_logweights(
                        oeuvre, snapshot.work_logrank, alpha, prev_works, delta_work
                    ),
                    k,
                    rng,
                )
            )
        else:
            suffix = None
            if alpha > 0 and 0 < k < len(oeuvre):
                suffix_key = (author, k)
                if suffix_key not in cache:
                    cache[suffix_key] = product_suffix(
                        oeuvre,
                        snapshot.work_logrank,
                        alpha,
                        k,
                    )
                suffix = cache[suffix_key]
            author_works[author] = set(
                sample_product_subset(
                    oeuvre,
                    snapshot.work_logrank,
                    alpha,
                    k,
                    rng,
                    suffix,
                )
            )
        used[author_index] = True
    authorless_boosted = (
        delta_work != 0.0
        and carryover
        and any(w in prev_works for w in snapshot.authorless_works)
    )
    if authorless_boosted:
        authorless = set(
            sample_product_subset_from_logweights(
                snapshot.authorless_works,
                _boosted_work_logweights(
                    snapshot.authorless_works,
                    snapshot.authorless_logrank,
                    alpha,
                    prev_works,
                    delta_work,
                ),
                authorless_count,
                rng,
            )
        )
    else:
        authorless_suffix = None
        if alpha > 0 and 0 < authorless_count < len(snapshot.authorless_works):
            key = ("~authorless", authorless_count)
            if key not in cache:
                cache[key] = product_suffix(
                    snapshot.authorless_works,
                    snapshot.authorless_logrank,
                    alpha,
                    authorless_count,
                )
            authorless_suffix = cache[key]
        authorless = set(
            sample_product_subset(
                snapshot.authorless_works,
                snapshot.authorless_logrank,
                alpha,
                authorless_count,
                rng,
                authorless_suffix,
            )
        )
    return SimulatedCorpusEdition(author_works, authorless)


def heldout_shapes(
    outcomes: pd.DataFrame,
    snapshot: Snapshot,
    coauthored: set[str],
) -> tuple[
    dict[str, tuple[tuple[int, ...], int]],
    dict[str, SimulatedCorpusEdition],
    dict[str, tuple[int, str | None]],
    pd.DataFrame,
]:
    shapes: dict[str, tuple[tuple[int, ...], int]] = {}
    observed: dict[str, SimulatedCorpusEdition] = {}
    metadata: dict[str, tuple[int, str | None]] = {}
    coverage: list[dict[str, float | str | int]] = []
    for _, group in outcomes.groupby("edition_id", sort=False):
        key = str(group["edition_key"].iloc[0])
        obs = observe_edition(group, snapshot, coauthored)
        counts = tuple(len(works) for _, works in obs.slots)
        shapes[key] = (counts, len(obs.authorless))
        observed[key] = SimulatedCorpusEdition(
            {author: set(works) for author, works in obs.slots}, set(obs.authorless)
        )
        series = group["series_id"].iloc[0]
        metadata[key] = (
            int(group["anthology_publication_year"].iloc[0]),
            None if pd.isna(series) else str(int(series)),
        )
        coverage.append(
            {
                "edition_key": key,
                "year": metadata[key][0],
                "author_coverage": obs.known_authors / obs.total_authors
                if obs.total_authors
                else np.nan,
                "authored_work_coverage": obs.known_authored_works
                / obs.total_authored_works
                if obs.total_authored_works
                else np.nan,
                "authorless_coverage": obs.known_authorless / obs.total_authorless
                if obs.total_authorless
                else np.nan,
                "coauthored_selection_rows": obs.coauthored_selections,
            }
        )
    return shapes, observed, metadata, pd.DataFrame(coverage)


METRIC_NAMES = (
    "jaccard_authors",
    "jaccard_works",
    "rate_a_gt_w",
    "zero_share",
    "author_year_slope",
    "work_year_slope",
    "jaccard_authors_within",
    "jaccard_works_within",
)


def corpus_metrics(
    editions: dict[str, SimulatedCorpusEdition],
    metadata: dict[str, tuple[int, str | None]],
) -> np.ndarray:
    """Cross-series pair summaries plus mean within-series Jaccards."""
    rows: list[tuple[float, float, float, float]] = []
    within: list[tuple[float, float]] = []
    zero = shared_instances = 0
    for left, right in combinations(sorted(editions), 2):
        a_left, a_right = editions[left].authors, editions[right].authors
        w_left, w_right = editions[left].works, editions[right].works
        shared_a, shared_w = len(a_left & a_right), len(w_left & w_right)
        union_a, union_w = len(a_left | a_right), len(w_left | w_right)
        ja = shared_a / union_a if union_a else 0.0
        jw = shared_w / union_w if union_w else 0.0
        series_left, series_right = metadata[left][1], metadata[right][1]
        if series_left is not None and series_left == series_right:
            within.append((ja, jw))
            continue
        gap = math.log1p(abs(metadata[left][0] - metadata[right][0]))
        rows.append((ja, jw, float(shared_a > shared_w), gap))
        for author in a_left & a_right:
            shared_instances += 1
            zero += not (
                editions[left].author_works[author]
                & editions[right].author_works[author]
            )
    if not rows:
        return np.zeros(len(METRIC_NAMES))
    matrix = np.asarray(rows, dtype=float)
    within_matrix = np.asarray(within, dtype=float) if within else np.zeros((0, 2))

    def slope(values: np.ndarray) -> float:
        x = matrix[:, 3]
        return float(np.polyfit(x, values, 1)[0]) if len(np.unique(x)) > 1 else 0.0

    return np.array(
        [
            matrix[:, 0].mean(),
            matrix[:, 1].mean(),
            matrix[:, 2].mean(),
            zero / shared_instances if shared_instances else 0.0,
            slope(matrix[:, 0]),
            slope(matrix[:, 1]),
            within_matrix[:, 0].mean() if len(within_matrix) else 0.0,
            within_matrix[:, 1].mean() if len(within_matrix) else 0.0,
        ]
    )


def split_joint_check(
    observed: np.ndarray, simulations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit covariance on half the simulations and calibrate on the other half."""
    split = len(simulations) // 2
    reference, calibration = simulations[:split], simulations[split:]
    mean = reference.mean(axis=0)
    covariance = np.cov(reference, rowvar=False)
    ridge = max(0.05 * float(np.trace(covariance)) / len(observed), 1e-12)
    inverse = np.linalg.pinv(covariance + ridge * np.eye(len(observed)))
    delta = observed - mean
    q_observed = float(delta @ inverse @ delta)
    centered = calibration - mean
    q_calibration = np.einsum("ij,jk,ik->i", centered, inverse, centered)
    p_value = float(
        (np.count_nonzero(q_calibration >= q_observed) + 1) / (len(q_calibration) + 1)
    )
    return mean, simulations.std(axis=0), q_observed, p_value


def posterior_predictive_check(
    shapes: dict[str, tuple[tuple[int, ...], int]],
    observed: dict[str, SimulatedCorpusEdition],
    metadata: dict[str, tuple[int, str | None]],
    snapshot: Snapshot,
    gamma: float,
    alpha: float,
    trials: int,
    rng: np.random.Generator,
    *,
    delta_author: float = 0.0,
    delta_work: float = 0.0,
    history: SeriesHistory = EMPTY_HISTORY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Simulate the held-out corpus, chaining carryover through the trajectory.

    Editions are simulated in chronological order.  When a simulated edition's
    same-series predecessor is itself in the simulated period, the carryover
    covariate uses the *simulated* predecessor's contents, so within-series
    autocorrelation is generated, not conditioned on.  Predecessors outside the
    simulated period contribute their realized contents.
    """
    observed_metrics = corpus_metrics(observed, metadata)
    simulations = np.empty((trials, len(METRIC_NAMES)))
    max_k = max((max(counts, default=0) for counts, _ in shapes.values()), default=0)
    tables = bundle_tables(snapshot, alpha, max_k)
    all_counts = {k for counts, _ in shapes.values() for k in counts}
    kernel = build_simulation_kernel(snapshot, gamma, tables, all_counts)
    subset_cache: dict[tuple[str, int], np.ndarray] = {}
    order = sorted(shapes, key=lambda key: (metadata[key][0], key))
    for trial in range(trials):
        corpus: dict[str, SimulatedCorpusEdition] = {}
        for key in order:
            counts, authorless = shapes[key]
            predecessor = history.pred_key.get(key)
            if predecessor is not None and predecessor in corpus:
                simulated_prev = corpus[predecessor]
                prev_authors = frozenset(simulated_prev.authors)
                prev_works = frozenset(simulated_prev.works)
            else:
                prev_authors, prev_works = history.prev_sets(key)
            corpus[key] = simulate_edition(
                counts,
                authorless,
                snapshot,
                gamma,
                alpha,
                rng,
                tables,
                kernel,
                subset_cache,
                delta_author=delta_author,
                delta_work=delta_work,
                prev_authors=prev_authors,
                prev_works=prev_works,
            )
        simulations[trial] = corpus_metrics(corpus, metadata)
    mean, sd, q_value, p_value = split_joint_check(observed_metrics, simulations)
    return observed_metrics, mean, sd, q_value, p_value


def series_gap_summary(afam: pd.DataFrame, work_unit: str) -> pd.DataFrame:
    """Descriptive motivation: within-series versus cross-series reselection."""
    editions = {
        str(key): {
            "year": int(group["anthology_publication_year"].iloc[0]),
            "series": group["series_id"].iloc[0],
            "works": set(group["work_id"]),
            "authors": set(group["author_id"].dropna()),
        }
        for key, group in afam.groupby("edition_key")
    }
    ordered = sorted(editions, key=lambda key: (editions[key]["year"], key))
    buckets: dict[bool, list[tuple[float, float, float, float]]] = {
        True: [],
        False: [],
    }
    for left, right in combinations(ordered, 2):
        e_left, e_right = editions[left], editions[right]
        same = pd.notna(e_left["series"]) and e_left["series"] == e_right["series"]
        shared_w = len(e_left["works"] & e_right["works"])
        shared_a = len(e_left["authors"] & e_right["authors"])
        buckets[same].append(
            (
                shared_w / len(e_left["works"]) if e_left["works"] else 0.0,
                shared_a / len(e_left["authors"]) if e_left["authors"] else 0.0,
                shared_w / len(e_left["works"] | e_right["works"]),
                shared_a / len(e_left["authors"] | e_right["authors"]),
            )
        )
    rows = []
    for same, values in buckets.items():
        matrix = np.asarray(values, dtype=float)
        rows.append(
            {
                "work_unit": work_unit,
                "pair_type": "within-series" if same else "cross-series",
                "n_pairs": len(values),
                "work_retention": matrix[:, 0].mean() if len(values) else np.nan,
                "author_retention": matrix[:, 1].mean() if len(values) else np.nan,
                "jaccard_works": matrix[:, 2].mean() if len(values) else np.nan,
                "jaccard_authors": matrix[:, 3].mean() if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_analysis(args: argparse.Namespace) -> tuple[pd.DataFrame, ...]:
    all_grid: list[pd.DataFrame] = []
    holdout_rows: list[dict[str, float | str | int]] = []
    ppc_rows: list[dict[str, float | str | int]] = []
    coverage_rows: list[pd.DataFrame] = []
    series_gap_rows: list[pd.DataFrame] = []
    rng = np.random.default_rng(args.seed)

    for work_unit in args.work_units:
        print(f"\n===== WORK UNIT: {work_unit} =====", flush=True)
        full = load_all_anthology_frame(work_unit)
        pool, coauthored = split_coauthored(full)
        afam = full[full["is_afam"].eq(1)]
        history = build_series_history(full)
        gap = series_gap_summary(afam, work_unit)
        series_gap_rows.append(gap)
        print("Within- vs cross-series reselection (motivating the carryover term):")
        print(
            gap.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
            flush=True,
        )
        for cutoff in args.cutoffs:
            train = afam[
                afam["anthology_publication_year"].between(
                    args.min_train_year, cutoff, inclusive="both"
                )
            ]
            test = afam[afam["anthology_publication_year"] > cutoff]
            if train.empty or test.empty:
                print(f"Skipping cutoff {cutoff}: empty train or test period.")
                continue
            print(
                f"Cutoff {cutoff}: {train['edition_id'].nunique()} train / "
                f"{test['edition_id'].nunique()} held-out editions; fitting grid...",
                flush=True,
            )
            fitted = fit_grid(
                pool,
                train,
                coauthored,
                args.grid,
                args.lambdas,
                args.deltas,
                cutoff,
                work_unit,
                history,
                args.rank_basis,
            )
            all_grid.append(fitted)
            best = family_best(fitted)
            for family, params in best.items():
                gamma = float(params["gamma"])
                alpha = float(params["alpha"])
                lam = float(params["lambda"])
                delta_author = float(params["delta_author"])
                delta_work = float(params["delta_work"])
                heldout = score_period(
                    pool,
                    test,
                    coauthored,
                    gamma,
                    alpha,
                    lam,
                    prequential=False,
                    cutoff=cutoff,
                    delta_author=delta_author,
                    delta_work=delta_work,
                    history=history,
                    rank_basis=args.rank_basis,
                )
                row: dict[str, float | str | int] = {
                    "work_unit": work_unit,
                    "cutoff": cutoff,
                    "family": family,
                    "gamma": gamma,
                    "alpha": alpha,
                    "lambda": lam,
                    "delta_author": delta_author,
                    "delta_work": delta_work,
                    "train_log_score": float(params["total_log_score"]),
                    "train_log_score_per_choice": float(params["log_score_per_choice"]),
                    **heldout,
                }
                holdout_rows.append(row)

                snapshot = make_snapshot(
                    pool, cutoff, strict=False, lam=lam, rank_basis=args.rank_basis
                )
                shapes, observed, metadata, coverage = heldout_shapes(
                    test, snapshot, coauthored
                )
                coverage.insert(0, "family", family)
                coverage.insert(0, "cutoff", cutoff)
                coverage.insert(0, "work_unit", work_unit)
                coverage_rows.append(coverage)
                observed_metrics, mean, sd, q_value, p_value = (
                    posterior_predictive_check(
                        shapes,
                        observed,
                        metadata,
                        snapshot,
                        gamma,
                        alpha,
                        args.trials,
                        rng,
                        delta_author=delta_author,
                        delta_work=delta_work,
                        history=history,
                    )
                )
                for name, obs_value, sim_mean, sim_sd in zip(
                    METRIC_NAMES, observed_metrics, mean, sd, strict=True
                ):
                    ppc_rows.append(
                        {
                            "work_unit": work_unit,
                            "cutoff": cutoff,
                            "family": family,
                            "gamma": gamma,
                            "alpha": alpha,
                            "lambda": lam,
                            "delta_author": delta_author,
                            "delta_work": delta_work,
                            "metric": name,
                            "observed": obs_value,
                            "sim_mean": sim_mean,
                            "sim_sd": sim_sd,
                            "z": (obs_value - sim_mean) / sim_sd
                            if sim_sd > 0
                            else np.nan,
                            "joint_q": q_value,
                            "joint_p": p_value,
                            "trials": args.trials,
                        }
                    )
                print(
                    f"  {family:<13} γ={gamma:g} α={alpha:g} λ={lam:g} "
                    f"δA={delta_author:g} δW={delta_work:g}  "
                    f"held-out log/choice={heldout['log_score_per_choice']:.3f} "
                    f"(chained={heldout['chained_log_per_choice']:.3f}, "
                    f"fresh={heldout['fresh_log_per_choice']:.3f})  "
                    f"coverage A/W={heldout['author_coverage']:.1%}/"
                    f"{heldout['authored_work_coverage']:.1%}  joint p={p_value:.3f}",
                    flush=True,
                )

    grid_df = pd.concat(all_grid, ignore_index=True) if all_grid else pd.DataFrame()
    holdout_df = pd.DataFrame(holdout_rows)
    ppc_df = pd.DataFrame(ppc_rows)
    coverage_df = (
        pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame()
    )
    series_gap_df = (
        pd.concat(series_gap_rows, ignore_index=True)
        if series_gap_rows
        else pd.DataFrame()
    )
    return grid_df, holdout_df, ppc_df, coverage_df, series_gap_df


def print_summary(holdout: pd.DataFrame, ppc: pd.DataFrame) -> None:
    if holdout.empty:
        return
    print("\n===== HELD-OUT MODEL COMPARISON (higher log score is better) =====")
    columns = [
        "work_unit",
        "cutoff",
        "family",
        "gamma",
        "alpha",
        "lambda",
        "delta_author",
        "delta_work",
        "log_score_per_choice",
        "chained_log_per_choice",
        "fresh_log_per_choice",
        "author_coverage",
        "authored_work_coverage",
    ]
    print(
        holdout[columns]
        .sort_values(
            ["work_unit", "cutoff", "log_score_per_choice"],
            ascending=[True, True, False],
        )
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    if not ppc.empty:
        joint = ppc.drop_duplicates(["work_unit", "cutoff", "family"])[
            ["work_unit", "cutoff", "family", "joint_q", "joint_p"]
        ]
        print("\n===== RESERVED JOINT POSTERIOR-PREDICTIVE CHECK =====")
        print(joint.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-units",
        nargs="+",
        choices=WORK_UNIT_MODES,
        default=list(WORK_UNIT_MODES),
    )
    parser.add_argument(
        "--grid",
        type=_csv_floats,
        default=list(DEFAULT_GRID),
        help="comma-separated gamma/alpha grid",
    )
    parser.add_argument(
        "--lambdas",
        type=_csv_floats,
        default=list(DEFAULT_LAMBDAS),
        help="comma-separated shared latent-work multipliers",
    )
    parser.add_argument(
        "--deltas",
        type=_csv_floats,
        default=list(DEFAULT_DELTAS),
        help="comma-separated same-series carryover log-boosts for both the "
        "author and work grids (0 must be included to fit the no-series "
        "families)",
    )
    parser.add_argument(
        "--rank-basis",
        choices=RANK_BASES,
        default="series",
        help="count prior selections by distinct series (decontaminated "
        "default) or by distinct editions (legacy, inflates within-series "
        "repeats)",
    )
    parser.add_argument(
        "--cutoffs",
        type=_csv_ints,
        default=list(DEFAULT_CUTOFFS),
        help="comma-separated fixed temporal cutoffs",
    )
    parser.add_argument("--min-train-year", type=int, default=1968)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()
    if args.trials < 20:
        parser.error("--trials must be at least 20 for split calibration")
    if 0.0 not in args.deltas:
        parser.error("--deltas must include 0 so the no-series families exist")

    grid, holdout, ppc, coverage, series_gap = run_analysis(args)
    print_summary(holdout, ppc)
    if args.save_csv:
        outputs = (
            (grid, OUT_GRID),
            (holdout, OUT_HOLDOUT),
            (ppc, OUT_PPC),
            (coverage, OUT_COVERAGE),
            (series_gap, OUT_SERIES_GAP),
        )
        for frame, path in outputs:
            frame.to_csv(path, index=False)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
