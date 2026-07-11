"""
author_work_agreement_models.py
--------------------------------
Is "editors agree more about authors than works" (A > W) a substantive
literary finding, or the expected outcome of pool sizes (~600 authors vs.
~3000 works) under almost any selection process?

Compares two generative model families, each in uniform and
concentration-weighted variants, against the observed data and on fully
synthetic pools:

  Model              # authors  per-author counts  total works  choice rule
  -----------------  ---------  -----------------  -----------  --------------------------
  author-uniform     exact      exact              emergent     uniform authors + works
  author-weighted    exact      exact              emergent     Zipf(alpha) within author
  author-canon       exact      exact              emergent     Zipf(gamma) over AUTHORS,
                                                                uniform works within author
  work-uniform       induced    induced            exact        uniform over work pool
  work-weighted      induced    induced            exact        Zipf(beta) over work pool

The author-canon model is the essay's thesis made generative: a fixed canon of
must-have authors (per-trial fixed author ranking; editors converge on the
same names) with free work choice within each selected author.

Corpus inflation (--lam): for author-family models with uniform work choice
(author-uniform, author-canon), each author's effective corpus is their
observed anthologized works plus round(lam * observed) unobserved works; a
pick that lands on an unobserved work becomes a unique, never-shared
selection. This models the fact that editors choose from full oeuvres, not
from the corpus the anthologies happen to have printed. --mode fit
grid-searches (gamma, lam) for the author-canon model jointly against all
five observed statistics (J(authors), J(works), D, the count rate, and the
zero-shared-works share), reporting RMS z per cell and the best-fitting cell.

--decouple-allocation (robustness): re-match the per-author work-count
multiset to the selected authors uniformly at random over feasible
assignments, separating "canonical authors are chosen" from "canonical
authors get the larger allocations."

Author-first models (the author is canonical): draw authors from each
edition's publication-year-eligible pool, then draw works within each chosen
author's corpus. "Emergent" total works: unlike simulate_author_work_overlap's
exact-shape sampler, each author's works are drawn independently, so
coauthored collisions are not forced to match the observed count (rare in
practice; alpha=0 should match the exact null within Monte Carlo noise).

Work-first models (the work is canonical): draw works directly from the
eligible work pool; the author set is induced by the chosen works. The
per-author work-count multiset is *not* preserved — that is the model's point.

Weighted variants implement "fractal canonicity": work ranks are a fixed
property drawn ONCE PER TRIAL and shared by every simulated edition in that
trial, so independent editions converge on the same core works. --rank-by
random (default) draws a fresh random ranking each trial (the clean null:
does *any* consistent core structure produce W > A?). --rank-by frequency
ranks by observed selection frequency — confirmatory/upper-bound only, since
it injects the observed canon into the null.

Structural note (single-author works, equal works-per-author k): work-level
Jaccard can never exceed author-level Jaccard under author-first selection
(|W∩| <= k|A∩| and |W∪| >= k|A∪|), however concentrated the within-author
choice. The *count* metric (shared_works > shared_authors) can invert with
k >= 2 plus concentration. Jaccard inversion requires skewed per-author
allocations concentrated on commonly selected authors — which the real
editions' shapes provide.

Metrics per edition pair (cross-series only by default, as elsewhere):
primary D = mean Jaccard(authors) − Jaccard(works); secondary = share of
pairs with shared_authors > shared_works (continuity with
simulate_author_work_overlap.py).

Note: unlike simulate_author_work_overlap.py (where --only-root-works is
opt-in), this script uses the shared afam.cli flag, so root-only is the
DEFAULT; pass --include-excerpts to keep excerpt works.

Usage:
    uv run python analysis/overlap/author_work_agreement_models.py
    uv run python analysis/overlap/author_work_agreement_models.py --mode all --save-csv
    uv run python analysis/overlap/author_work_agreement_models.py --mode synthetic \\
        --sweep-trials 200 --alphas 0,0.5,1,2,4 --k-values 1,2,3,5
    uv run python analysis/overlap/author_work_agreement_models.py --mode real \\
        --n 1000 --rank-by frequency
    uv run python analysis/overlap/author_work_agreement_models.py --mode fit \\
        --fit-trials 200 --gammas 0,0.5,1,1.5,2,3 --lambdas 0,0.5,1,2,4
    uv run python analysis/overlap/author_work_agreement_models.py --mode real \\
        --models author-canon --alphas 1.5 --lam 2 --decouple-allocation
"""

from __future__ import annotations

import argparse
import bisect
import sys
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle

from afam import DATA_DIR
from afam.cli import add_root_works_flag, add_save_csv_flag
from afam.db import query as query_db
from afam.sql import query_path
from afam.viz_style import AUTHOR_COLOR, DPI, OUTPUT_DIR, WORK_COLOR

sys.path.insert(0, str(Path(__file__).parent))

from simulate_author_work_overlap import (  # noqa: E402
    EditionTarget,
    SimulatedEdition,
    build_author_to_all_works,
    build_elig_by_author,
    build_real_targets,
    fmt_p,
    fmt_pct,
    precompute_first_publication_years,
)

OUT_REAL = OUTPUT_DIR / "author_work_agreement_models_real.png"
OUT_SWEEP = OUTPUT_DIR / "author_work_agreement_models_sweep.png"
OUT_FIT = OUTPUT_DIR / "author_work_agreement_models_fit.png"

DEFAULT_N = 1_000
DEFAULT_SWEEP_TRIALS = 200
DEFAULT_FIT_TRIALS = 200
DEFAULT_SEED = 42
DEFAULT_ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)
DEFAULT_GAMMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT_LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0)
DEFAULT_K_VALUES = (1, 2, 3, 5)
DEFAULT_MEAN_CORPUS = (2.0, 5.0, 10.0)

MODEL_NAMES = (
    "author-uniform",
    "author-weighted",
    "author-canon",
    "work-uniform",
    "work-weighted",
)


# ── Zipf weights and fixed-per-trial work rankings ────────────────────────────


@lru_cache(maxsize=None)
def zipf_weights(n: int, alpha: float) -> np.ndarray:
    """Normalized weights w_r ∝ r^(-alpha) over ranks 1..n; alpha=0 → uniform."""
    ranks = np.arange(1, n + 1, dtype=float)
    w = ranks**-alpha
    return w / w.sum()


def weights_for_subset(
    wids: list[str], rank_map: dict[str, int], alpha: float
) -> np.ndarray:
    """Zipf weights over an eligible subset, using ranks from the full corpus."""
    ranks = np.array([rank_map[w] for w in wids], dtype=float)
    w = ranks**-alpha
    return w / w.sum()


def compute_work_frequency(df: pd.DataFrame) -> dict[str, int]:
    """Map work_id → number of distinct editions that selected it."""
    counts = df.groupby("work_id")["edition_key"].nunique()
    return {str(k): int(v) for k, v in counts.items()}


def compute_author_frequency(df: pd.DataFrame) -> dict[str, int]:
    """Map author_id → number of distinct editions that selected them."""
    sub = df[df["author_id"].notna()]
    counts = sub.groupby("author_id")["edition_key"].nunique()
    return {str(k): int(v) for k, v in counts.items()}


def draw_author_pool_ranks(
    author_ids: list[str],
    rng: np.random.Generator,
    rank_by: str,
    author_freq: dict[str, int] | None = None,
) -> dict[str, int]:
    """One fixed 1-based rank per author over the whole pool — the author canon
    for the author-canon model. Drawn once per trial and shared by every
    simulated edition in that trial."""
    if rank_by == "frequency":
        freq = author_freq or {}
        order = sorted(author_ids, key=lambda a: (-freq.get(a, 0), a))
        return {a: r for r, a in enumerate(order, start=1)}
    perm = rng.permutation(len(author_ids))
    return {a: int(r) + 1 for a, r in zip(sorted(author_ids), perm)}


def draw_author_corpus_ranks(
    author_to_works: dict[str, set[str]],
    rng: np.random.Generator,
    rank_by: str,
    work_freq: dict[str, int] | None = None,
) -> dict[str, dict[str, int]]:
    """One fixed 1-based rank per work within each author's full corpus.

    Drawn once per trial and shared by every simulated edition in that trial —
    the cross-editor consistency the weighted models exist to create.
    """
    ranks: dict[str, dict[str, int]] = {}
    for aid, wids in author_to_works.items():
        corpus = sorted(wids)
        if rank_by == "frequency":
            order = sorted(corpus, key=lambda w: (-(work_freq or {}).get(w, 0), w))
            ranks[aid] = {w: r for r, w in enumerate(order, start=1)}
        else:
            perm = rng.permutation(len(corpus))
            ranks[aid] = {w: int(r) + 1 for w, r in zip(corpus, perm)}
    return ranks


def draw_global_work_ranks(
    work_ids: list[str],
    rng: np.random.Generator,
    rank_by: str,
    work_freq: dict[str, int] | None = None,
) -> np.ndarray:
    """One fixed 1-based rank per work over the global pool, aligned to work_ids."""
    n = len(work_ids)
    if rank_by == "frequency":
        freq = work_freq or {}
        order = sorted(range(n), key=lambda i: (-freq.get(work_ids[i], 0), work_ids[i]))
        ranks = np.empty(n, dtype=float)
        for r, i in enumerate(order, start=1):
            ranks[i] = r
        return ranks
    return rng.permutation(n).astype(float) + 1.0


# ── Data loading and pools ────────────────────────────────────────────────────


def load_divergence_frame(only_root_works: bool) -> pd.DataFrame:
    """Load the shared divergence query with edition keys (as the sibling does)."""
    df = query_db(query_path("work-selection-divergence"))
    df["edition_key"] = df.apply(
        lambda r: (
            f"{int(r['series_id'])}|{r['edition_number']}"
            if pd.notna(r["series_id"])
            else str(r["edition_id"])
        ),
        axis=1,
    )
    df["anthology_publication_year"] = df["anthology_publication_year"].astype(int)
    if only_root_works:
        df = df[df["parent_id"].isna()].copy()
    return df


def build_work_to_authors(df: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Map each authored work_id → the full set of its author_ids."""
    result: dict[str, frozenset[str]] = {}
    for wid, grp in df[df["author_id"].notna()].groupby("work_id"):
        result[str(wid)] = frozenset(grp["author_id"].astype(str))
    return result


def build_eligible_authored_works(
    sorted_keys: list[str],
    edition_year: dict[str, int],
    work_first_year: dict[str, int],
    work_to_authors: dict[str, frozenset[str]],
) -> dict[str, list[str]]:
    """Per edition: authored works first observed no later than its year."""
    all_authored = sorted(work_to_authors)
    return {
        ek: [
            w
            for w in all_authored
            if work_first_year.get(w, edition_year[ek] + 1) <= edition_year[ek]
        ]
        for ek in sorted_keys
    }


# ── Samplers ──────────────────────────────────────────────────────────────────


def _weighted_pick(n: int, weights: np.ndarray, k: int, rng: np.random.Generator):
    """k distinct indices from range(n), p ∝ weights (Gumbel top-k, i.e.
    successive weighted draws without replacement)."""
    if k >= n:
        return np.arange(n)
    keys = np.log(weights) + rng.gumbel(size=n)
    return np.argpartition(keys, n - k)[n - k :]


def _sample_authorless(
    count: int, pool: list[str], rng: np.random.Generator
) -> set[str]:
    if count == 0:
        return set()
    if len(pool) < count:
        raise RuntimeError(
            f"Eligible authorless pool ({len(pool)}) smaller than target ({count})."
        )
    picked = rng.choice(len(pool), size=count, replace=False)
    return {pool[int(i)] for i in picked}


def _effective_size(m: int, lam: float) -> int:
    """Observed corpus size m plus round(lam * m) unobserved (phantom) works."""
    return m + int(round(lam * m))


def random_feasible_matching(
    sizes: list[int], counts: list[int], rng: np.random.Generator
) -> list[int]:
    """Random feasible assignment of a work-count multiset to author slots.

    Returns assign with assign[ci] = entry index j such that
    sizes[j] >= counts[ci], each entry used exactly once. Counts are processed
    largest-first with a uniform pick among the feasible remaining entries;
    because feasibility is a threshold (an entry feasible for a count is
    feasible for every smaller one), largest-first greedy succeeds whenever
    any perfect matching exists, so no backtracking is needed."""
    order = sorted(range(len(counts)), key=lambda ci: -counts[ci])
    unused = list(range(len(sizes)))
    assign = [0] * len(counts)
    for ci in order:
        feasible = [j for j in unused if sizes[j] >= counts[ci]]
        if not feasible:
            raise RuntimeError(
                "No feasible matching of work counts to selected authors."
            )
        j = feasible[int(rng.integers(len(feasible)))]
        unused.remove(j)
        assign[ci] = j
    return assign


def sample_edition_author_first(
    target: EditionTarget,
    eligible_by_author: list[tuple[str, list[str]]],
    corpus_ranks: dict[str, dict[str, int]] | None,
    alpha: float,
    eligible_authorless: list[str],
    rng: np.random.Generator,
    max_rejects: int = 64,
    author_ranks: dict[str, int] | None = None,
    gamma: float = 0.0,
    lam: float = 0.0,
    phantom_prefix: str = "",
    decouple: bool = False,
) -> SimulatedEdition:
    """Author-first: draw distinct authors satisfying the observed per-author
    work-count multiset, then draw each author's works (uniform or Zipf(alpha)
    over their full-corpus ranks, renormalized to the eligible subset).

    gamma > 0 turns on the author-canon variant: authors are drawn with
    probability ∝ pool_rank^(−gamma) (Plackett–Luce via sequential Gumbel
    argmax over the qualifying prefix) instead of uniformly, so independent
    editions converge on the same top-ranked authors.

    lam > 0 inflates each author's corpus with round(lam * m) unobserved
    works (uniform work choice only, alpha == 0). A pick that lands on an
    unobserved work becomes a unique, never-shared work id namespaced by
    phantom_prefix (pass the edition key so phantoms can never collide
    across editions).

    decouple=True re-matches the work-count multiset to the selected authors
    uniformly at random over feasible assignments, severing the link between
    an author's canon rank and the size of their allocation."""
    if lam > 0 and alpha > 0:
        raise ValueError("lam (corpus inflation) requires alpha == 0")
    entries = sorted(
        eligible_by_author, key=lambda t: (-_effective_size(len(t[1]), lam), t[0])
    )
    aids = [aid for aid, _ in entries]
    neg_sizes = [-_effective_size(len(wids), lam) for _, wids in entries]

    author_keys: np.ndarray | None = None
    if gamma > 0:
        if author_ranks is None:
            raise ValueError("author_ranks is required when gamma > 0")
        rank_arr = np.array([author_ranks[a] for a in aids], dtype=float)
        author_keys = -gamma * np.log(rank_arr) + rng.gumbel(size=len(aids))

    used: set[str] = set()
    assigned: list[tuple[int, int]] = []  # (entry index, work count)
    for count in target.per_author_work_counts:  # sorted descending
        n_elig = bisect.bisect_right(neg_sizes, -count)
        idx: int | None = None
        if author_keys is not None:
            j = int(np.argmax(author_keys[:n_elig])) if n_elig else 0
            if n_elig == 0 or author_keys[j] == -np.inf:
                raise RuntimeError(
                    f"No unused eligible author with >= {count} eligible works."
                )
            author_keys[j] = -np.inf
            idx = j
        else:
            if n_elig > len(used):
                for _ in range(max_rejects):
                    j = int(rng.integers(n_elig))
                    if aids[j] not in used:
                        idx = j
                        break
            if idx is None:
                candidates = [j for j in range(n_elig) if aids[j] not in used]
                if not candidates:
                    raise RuntimeError(
                        f"No unused eligible author with >= {count} eligible works."
                    )
                idx = candidates[int(rng.integers(len(candidates)))]
        used.add(aids[idx])
        assigned.append((idx, count))

    if decouple:
        idxs = [i for i, _ in assigned]
        counts = [c for _, c in assigned]
        sizes = [-neg_sizes[i] for i in idxs]
        matching = random_feasible_matching(sizes, counts, rng)
        assigned = [(idxs[j], c) for j, c in zip(matching, counts)]

    author_work_ids: dict[str, set[str]] = {}
    for idx, count in assigned:
        aid, wids = entries[idx]
        m = len(wids)
        n_ph = 0
        if lam > 0:
            n_phantom_pool = _effective_size(m, lam) - m
            if n_phantom_pool > 0:
                n_ph = int(rng.hypergeometric(n_phantom_pool, m, count))
        n_real = count - n_ph
        if alpha > 0:
            if corpus_ranks is None:
                raise ValueError("corpus_ranks is required when alpha > 0")
            w = weights_for_subset(wids, corpus_ranks[aid], alpha)
            picked = _weighted_pick(m, w, n_real, rng)
        elif n_real < m:
            picked = rng.choice(m, size=n_real, replace=False)
        else:
            picked = np.arange(m)
        works = {wids[int(i)] for i in picked}
        works.update(f"~{phantom_prefix}|{aid}|{j}" for j in range(n_ph))
        author_work_ids[aid] = works

    authorless = _sample_authorless(
        target.authorless_work_count, eligible_authorless, rng
    )
    return SimulatedEdition(author_work_ids, authorless)


def sample_edition_work_first(
    target: EditionTarget,
    eligible_works: list[str],
    rank_arr: np.ndarray | None,
    beta: float,
    work_to_authors: dict[str, frozenset[str]],
    eligible_authorless: list[str],
    rng: np.random.Generator,
) -> SimulatedEdition:
    """Work-first: draw the observed number of authored works directly from the
    eligible pool (uniform or Zipf(beta) over global ranks); authors are
    induced by the chosen works (coauthors all included)."""
    n = target.authored_work_count
    m = len(eligible_works)
    if m < n:
        raise RuntimeError(
            f"Eligible authored-work pool ({m}) smaller than target ({n})."
        )
    if beta > 0:
        if rank_arr is None:
            raise ValueError("rank_arr is required when beta > 0")
        w = rank_arr**-beta
        picked = _weighted_pick(m, w / w.sum(), n, rng)
    else:
        picked = rng.choice(m, size=n, replace=False)

    author_work_ids: dict[str, set[str]] = {}
    for i in picked:
        wid = eligible_works[int(i)]
        for aid in work_to_authors[wid]:
            author_work_ids.setdefault(aid, set()).add(wid)

    authorless = _sample_authorless(
        target.authorless_work_count, eligible_authorless, rng
    )
    return SimulatedEdition(author_work_ids, authorless)


# ── Pair metrics ──────────────────────────────────────────────────────────────


class PairStat(NamedTuple):
    shared_authors: int
    shared_works: int
    jaccard_authors: float
    jaccard_works: float


def same_series(ki: str, kj: str) -> bool:
    if "|" not in ki or "|" not in kj:
        return False
    return ki.rsplit("|", 1)[0] == kj.rsplit("|", 1)[0]


def _jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def collect_pair_stats(
    work_sets: dict[str, set],
    author_sets: dict[str, set],
    sorted_keys: list[str],
    cross_series_only: bool = True,
) -> list[PairStat]:
    """Shared counts and Jaccards for every requested edition pair."""
    stats = []
    for ki, kj in combinations(sorted_keys, 2):
        if cross_series_only and same_series(ki, kj):
            continue
        a_i, a_j = author_sets[ki], author_sets[kj]
        w_i, w_j = work_sets[ki], work_sets[kj]
        stats.append(
            PairStat(
                len(a_i & a_j), len(w_i & w_j), _jaccard(a_i, a_j), _jaccard(w_i, w_j)
            )
        )
    return stats


@dataclass(frozen=True)
class Summary:
    """Pair-level agreement summary for one (real or simulated) corpus."""

    rate_a_gt_w: float
    rate_tie: float
    rate_w_gt_a: float
    mean_jaccard_diff: float  # mean J(authors) − J(works); > 0 → A > W
    mean_jaccard_authors: float
    mean_jaccard_works: float


def summarize_pairs(stats: list[PairStat]) -> Summary:
    n = len(stats)
    if n == 0:
        raise ValueError("No edition pairs to summarize.")
    gt = sum(1 for s in stats if s.shared_authors > s.shared_works)
    tie = sum(1 for s in stats if s.shared_authors == s.shared_works)
    ja = sum(s.jaccard_authors for s in stats) / n
    jw = sum(s.jaccard_works for s in stats) / n
    return Summary(gt / n, tie / n, (n - gt - tie) / n, ja - jw, ja, jw)


def zero_share_rate(
    author_work_ids: dict[str, dict[str, set[str]]],
    sorted_keys: list[str],
    cross_series_only: bool = True,
) -> float:
    """Over all (pair, shared author) instances: share with no common work by
    that author — the essay's "shared author, zero shared works" statistic."""
    shared = zero = 0
    for ki, kj in combinations(sorted_keys, 2):
        if cross_series_only and same_series(ki, kj):
            continue
        di, dj = author_work_ids[ki], author_work_ids[kj]
        for aid in di.keys() & dj.keys():
            shared += 1
            if not (di[aid] & dj[aid]):
                zero += 1
    return zero / shared if shared else float("nan")


# ── Model configurations ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelConfig:
    label: str
    family: str  # "author" | "canon" | "work"
    alpha: float  # α (within-author), γ (author canon), or β (work canon)
    lam: float = 0.0  # corpus inflation (author family, uniform work choice)


def build_model_configs(
    models: list[str], alphas: list[float], lam: float = 0.0
) -> list[ModelConfig]:
    weighted = [a for a in alphas if a > 0]
    suffix = f", λ={lam:g}" if lam > 0 else ""
    configs: list[ModelConfig] = []
    for name in models:
        if name == "author-uniform":
            configs.append(
                ModelConfig(f"author-first uniform (α=0{suffix})", "author", 0.0, lam)
            )
        elif name == "author-weighted":
            # within-author Zipf needs a real corpus ranking, so λ is not applied
            configs.extend(
                ModelConfig(f"author-first Zipf α={a:g}", "author", a) for a in weighted
            )
        elif name == "author-canon":
            configs.extend(
                ModelConfig(
                    f"author-canon Zipf γ={a:g}{suffix} (works uniform)",
                    "canon",
                    a,
                    lam,
                )
                for a in weighted
            )
        elif name == "work-uniform":
            configs.append(ModelConfig("work-first uniform (β=0)", "work", 0.0))
        elif name == "work-weighted":
            configs.extend(
                ModelConfig(f"work-first Zipf β={a:g}", "work", a) for a in weighted
            )
    return configs


# ── Real-data mode ────────────────────────────────────────────────────────────


def _metric_stats(obs: float, arr: np.ndarray) -> dict[str, float]:
    mean = float(arr.mean())
    sd = float(arr.std())
    z = (obs - mean) / sd if sd > 0 else float("inf")
    return {
        "sim_mean": mean,
        "sim_sd": sd,
        "z": z,
        "p_ge_obs": float((arr >= obs).mean()),
        "p_le_obs": float((arr <= obs).mean()),
    }


@dataclass
class RealData:
    """Pools, targets, and observed statistics computed once per session."""

    sorted_keys: list[str]
    real_targets: dict[str, EditionTarget]
    elig_by_author: dict[str, list[tuple[str, list[str]]]]
    eligible_authored: dict[str, list[str]]
    eligible_authorless: dict[str, list[str]]
    work_to_authors: dict[str, frozenset[str]]
    author_to_all_works: dict[str, set[str]]
    all_author_ids: list[str]
    global_works: list[str]
    ed_work_idx: dict[str, np.ndarray]
    cross_series_only: bool
    observed: Summary
    obs_zero_share: float
    n_pairs: int
    work_freq: dict[str, int]
    author_freq: dict[str, int]


def prepare_real_data(df: pd.DataFrame, cross_series_only: bool) -> RealData:
    """Build every pool and observed statistic real/fit mode needs."""
    edition_works = {
        ek: set(grp["work_id"].astype(str)) for ek, grp in df.groupby("edition_key")
    }
    edition_authors = {
        ek: set(grp["author_id"].dropna().astype(str))
        for ek, grp in df.groupby("edition_key")
    }
    edition_year = {
        ek: int(grp["anthology_publication_year"].min())
        for ek, grp in df.groupby("edition_key")
    }
    sorted_keys = sorted(edition_works, key=lambda k: (edition_year[k], k))

    work_first_year, author_first_year = precompute_first_publication_years(df)
    author_to_all_works = build_author_to_all_works(df)
    elig_by_author = build_elig_by_author(
        sorted_keys,
        edition_year,
        author_to_all_works,
        author_first_year,
        work_first_year,
    )
    real_targets = build_real_targets(df)
    work_to_authors = build_work_to_authors(df)
    eligible_authored = build_eligible_authored_works(
        sorted_keys, edition_year, work_first_year, work_to_authors
    )

    authored_wids = set(work_to_authors)
    authorless_pool = sorted(
        set(df.loc[df["author_id"].isna(), "work_id"].astype(str)) - authored_wids
    )
    eligible_authorless = {
        ek: [
            w
            for w in authorless_pool
            if work_first_year.get(w, edition_year[ek] + 1) <= edition_year[ek]
        ]
        for ek in sorted_keys
    }

    for ek in sorted_keys:
        t = real_targets[ek]
        if len(eligible_authored[ek]) < t.authored_work_count:
            raise RuntimeError(
                f"Edition {ek} ({edition_year[ek]}): eligible authored-work pool "
                f"({len(eligible_authored[ek])}) < target ({t.authored_work_count})."
            )
        if len(eligible_authorless[ek]) < t.authorless_work_count:
            raise RuntimeError(
                f"Edition {ek} ({edition_year[ek]}): eligible authorless pool "
                f"({len(eligible_authorless[ek])}) < target "
                f"({t.authorless_work_count})."
            )

    obs_stats = collect_pair_stats(
        edition_works, edition_authors, sorted_keys, cross_series_only
    )
    observed = summarize_pairs(obs_stats)

    real_author_works: dict[str, dict[str, set[str]]] = {ek: {} for ek in sorted_keys}
    for ek, grp in df[df["author_id"].notna()].groupby("edition_key"):
        real_author_works[ek] = {
            str(aid): set(g["work_id"].astype(str))
            for aid, g in grp.groupby("author_id")
        }
    obs_zero_share = zero_share_rate(real_author_works, sorted_keys, cross_series_only)

    global_works = sorted(authored_wids)
    gidx = {w: i for i, w in enumerate(global_works)}
    ed_work_idx = {
        ek: np.array([gidx[w] for w in eligible_authored[ek]], dtype=int)
        for ek in sorted_keys
    }

    return RealData(
        sorted_keys=sorted_keys,
        real_targets=real_targets,
        elig_by_author=elig_by_author,
        eligible_authored=eligible_authored,
        eligible_authorless=eligible_authorless,
        work_to_authors=work_to_authors,
        author_to_all_works=author_to_all_works,
        all_author_ids=sorted(author_to_all_works),
        global_works=global_works,
        ed_work_idx=ed_work_idx,
        cross_series_only=cross_series_only,
        observed=observed,
        obs_zero_share=obs_zero_share,
        n_pairs=len(obs_stats),
        work_freq=compute_work_frequency(df),
        author_freq=compute_author_frequency(df),
    )


def simulate_model(
    cfg: ModelConfig,
    data: RealData,
    n_trials: int,
    rng: np.random.Generator,
    rank_by: str,
    decouple: bool = False,
) -> dict[str, np.ndarray]:
    """Run one model config for n_trials against the real edition shapes."""
    fixed_corpus_ranks = fixed_global_ranks = fixed_author_ranks = None
    if rank_by == "frequency":
        fixed_corpus_ranks = draw_author_corpus_ranks(
            data.author_to_all_works, rng, "frequency", data.work_freq
        )
        fixed_global_ranks = draw_global_work_ranks(
            data.global_works, rng, "frequency", data.work_freq
        )
        fixed_author_ranks = draw_author_pool_ranks(
            data.all_author_ids, rng, "frequency", data.author_freq
        )

    arrays = {k: np.empty(n_trials) for k in ("rate", "diff", "ja", "jw", "zero")}
    print(f"Running {cfg.label}: {n_trials:,} trials", end="", flush=True)
    for trial in range(n_trials):
        if (trial + 1) % 200 == 0:
            print(".", end="", flush=True)

        sims: dict[str, SimulatedEdition] = {}
        if cfg.family in ("author", "canon"):
            alpha = cfg.alpha if cfg.family == "author" else 0.0
            gamma = cfg.alpha if cfg.family == "canon" else 0.0
            corpus_ranks = None
            if alpha > 0:
                corpus_ranks = fixed_corpus_ranks or draw_author_corpus_ranks(
                    data.author_to_all_works, rng, "random"
                )
            author_ranks = None
            if gamma > 0:
                author_ranks = fixed_author_ranks or draw_author_pool_ranks(
                    data.all_author_ids, rng, "random"
                )
            for ek in data.sorted_keys:
                sims[ek] = sample_edition_author_first(
                    data.real_targets[ek],
                    data.elig_by_author[ek],
                    corpus_ranks,
                    alpha,
                    data.eligible_authorless[ek],
                    rng,
                    author_ranks=author_ranks,
                    gamma=gamma,
                    lam=cfg.lam,
                    phantom_prefix=ek,
                    decouple=decouple,
                )
        else:
            global_ranks = None
            if cfg.alpha > 0:
                global_ranks = (
                    fixed_global_ranks
                    if fixed_global_ranks is not None
                    else draw_global_work_ranks(data.global_works, rng, "random")
                )
            for ek in data.sorted_keys:
                rank_arr = (
                    global_ranks[data.ed_work_idx[ek]]
                    if global_ranks is not None
                    else None
                )
                sims[ek] = sample_edition_work_first(
                    data.real_targets[ek],
                    data.eligible_authored[ek],
                    rank_arr,
                    cfg.alpha,
                    data.work_to_authors,
                    data.eligible_authorless[ek],
                    rng,
                )

        sim_w = {ek: s.work_ids for ek, s in sims.items()}
        sim_a = {ek: s.author_ids for ek, s in sims.items()}
        summ = summarize_pairs(
            collect_pair_stats(sim_w, sim_a, data.sorted_keys, data.cross_series_only)
        )
        arrays["rate"][trial] = summ.rate_a_gt_w
        arrays["diff"][trial] = summ.mean_jaccard_diff
        arrays["ja"][trial] = summ.mean_jaccard_authors
        arrays["jw"][trial] = summ.mean_jaccard_works
        arrays["zero"][trial] = zero_share_rate(
            {ek: s.author_work_ids for ek, s in sims.items()},
            data.sorted_keys,
            data.cross_series_only,
        )
    print(" done.")
    return arrays


STAT_ARRAY_KEYS = (
    ("jaccard_diff", "diff"),
    ("jaccard_authors", "ja"),
    ("jaccard_works", "jw"),
    ("rate_a_gt_w", "rate"),
    ("zero_share", "zero"),
)


def model_row(
    cfg: ModelConfig, data: RealData, arrays: dict[str, np.ndarray], n_trials: int
) -> dict:
    """Per-model result row: sim mean/sd/z/p against each observed statistic."""
    observed_vals = {
        "jaccard_diff": data.observed.mean_jaccard_diff,
        "jaccard_authors": data.observed.mean_jaccard_authors,
        "jaccard_works": data.observed.mean_jaccard_works,
        "rate_a_gt_w": data.observed.rate_a_gt_w,
        "zero_share": data.obs_zero_share,
    }
    row = {
        "model": cfg.label,
        "family": cfg.family,
        "alpha": cfg.alpha,
        "lam": cfg.lam,
        "n_trials": n_trials,
    }
    for name, key in STAT_ARRAY_KEYS:
        obs_value = observed_vals[name]
        row |= {
            f"{name}_{k}": v for k, v in _metric_stats(obs_value, arrays[key]).items()
        }
        row[f"{name}_obs"] = obs_value
    return row


def run_real_mode(
    data: RealData, args: argparse.Namespace, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Simulate every requested model against the real edition shapes."""
    configs = build_model_configs(args.models, args.alphas, args.lam)
    rows: list[dict] = []
    dists: dict[str, dict[str, np.ndarray]] = {}
    for cfg in configs:
        arrays = simulate_model(
            cfg, data, args.n, rng, args.rank_by, args.decouple_allocation
        )
        rows.append(model_row(cfg, data, arrays, args.n))
        dists[cfg.label] = arrays
    return pd.DataFrame(rows), dists


def print_observed_block(
    observed: Summary, obs_zero_share: float, n_pairs: int
) -> None:
    print()
    print("===== OBSERVED (real editions) =====")
    print(f"  Edition pairs:                  {n_pairs}")
    print(f"  mean J(authors):                {observed.mean_jaccard_authors:.4f}")
    print(f"  mean J(works):                  {observed.mean_jaccard_works:.4f}")
    print(f"  mean J(authors) − J(works):     {observed.mean_jaccard_diff:+.4f}")
    print(f"  shared_authors > shared_works:  {fmt_pct(observed.rate_a_gt_w)}")
    print(f"  ties:                           {fmt_pct(observed.rate_tie)}")
    print(f"  shared_works > shared_authors:  {fmt_pct(observed.rate_w_gt_a)}")
    print(f"  P(zero shared works | shared author): {fmt_pct(obs_zero_share)}")
    print()


def print_real_report(
    observed: Summary,
    results: pd.DataFrame,
    obs_zero_share: float,
    n_pairs: int,
    args: argparse.Namespace,
) -> None:
    print_observed_block(observed, obs_zero_share, n_pairs)
    if args.rank_by == "frequency":
        print(
            "NOTE: --rank-by frequency injects the observed selection frequencies\n"
            "into the weighted nulls; treat these as confirmatory upper bounds,\n"
            "not clean nulls.\n"
        )
    if args.decouple_allocation:
        print(
            "NOTE: --decouple-allocation randomly re-matches work counts to the\n"
            "selected authors (rank no longer buys the larger allocations).\n"
        )
    for _, r in results.iterrows():
        print(f"===== {r['model']} (N={int(r['n_trials']):,}) =====")
        _print_stat_lines(r)
        print()


def _print_stat_lines(r) -> None:
    stat_lines = (
        ("jaccard_authors", "J(authors)", False),
        ("jaccard_works", "J(works)", False),
        ("jaccard_diff", "D = J(authors) − J(works)", False),
        ("rate_a_gt_w", "rate shared_A > shared_W", True),
        ("zero_share", "P(zero works | shared author)", True),
    )
    for key, label, as_pct in stat_lines:
        fmt = fmt_pct if as_pct else lambda v: f"{v:+.4f}"
        print(
            f"  {label:<33} sim {fmt(r[f'{key}_sim_mean'])}"
            f" ± {fmt(r[f'{key}_sim_sd'])}"
            f"   obs {fmt(r[f'{key}_obs'])}"
            f"   z = {r[f'{key}_z']:+.2f}"
        )


# ── Fit mode: joint (γ, λ) grid for the author-canon model ────────────────────


FIT_STATS = (
    "jaccard_diff",
    "jaccard_authors",
    "jaccard_works",
    "rate_a_gt_w",
    "zero_share",
)


def fit_objective(row: dict) -> float:
    """RMS z across the five fitted statistics (lower = closer joint fit)."""
    zs = np.clip([row[f"{s}_z"] for s in FIT_STATS], -1e6, 1e6)
    return float(np.sqrt(np.mean(np.square(zs))))


def run_fit_mode(
    data: RealData, args: argparse.Namespace, rng: np.random.Generator
) -> pd.DataFrame:
    """Grid-search (γ, λ) for author-canon against all five observed stats.

    γ=0 cells are the uniform-author (author-first) model with inflation λ,
    so the grid nests the plain author-first null as its origin cell."""
    rows: list[dict] = []
    for lam in args.lambdas:
        for gamma in args.gammas:
            cfg = ModelConfig(
                f"author-canon γ={gamma:g}, λ={lam:g}", "canon", gamma, lam
            )
            arrays = simulate_model(
                cfg, data, args.fit_trials, rng, args.rank_by, args.decouple_allocation
            )
            row = model_row(cfg, data, arrays, args.fit_trials)
            row["gamma"] = gamma
            row["rms_z"] = fit_objective(row)
            rows.append(row)
    return pd.DataFrame(rows)


def print_fit_report(fit: pd.DataFrame, args: argparse.Namespace) -> None:
    print()
    print("===== JOINT (γ, λ) FIT: author-canon with corpus inflation =====")
    print("RMS z across five statistics (lower = closer joint fit);")
    print("rows γ (author-canon concentration), cols λ (corpus inflation):\n")
    pivot = fit.pivot(index="gamma", columns="lam", values="rms_z")
    print(pivot.to_string(float_format=lambda v: f"{v:8.2f}"))
    print()
    best = fit.loc[fit["rms_z"].idxmin()]
    print(
        f"Best cell: γ={best['gamma']:g}, λ={best['lam']:g}   "
        f"RMS z = {best['rms_z']:.2f}   "
        f"({int(best['n_trials'])} trials/cell)"
    )
    _print_stat_lines(best)
    print()


# ── Synthetic mode ────────────────────────────────────────────────────────────


def build_synthetic_pool(
    n_authors: int, mean_corpus: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-author corpus sizes: geometric with mean `mean_corpus`, support >= 1.

    Synthetic pools use single-author works and no authorless works — a
    deliberate simplification relative to the real data.
    """
    if mean_corpus < 1:
        raise ValueError("mean_corpus must be >= 1")
    return rng.geometric(1.0 / mean_corpus, size=n_authors)


def run_synthetic_trial(
    family: str,
    alpha: float,
    k: int,
    sizes: np.ndarray,
    n_editions: int,
    authors_per_edition: int,
    rng: np.random.Generator,
) -> Summary | None:
    """One synthetic trial: sample n_editions editions and summarize all pairs.

    Author-first: `authors_per_edition` authors (uniform among those with
    corpus >= k), k works each, Zipf(alpha) over per-trial random corpus ranks.
    Work-first: authors_per_edition * k works, Zipf(alpha) over a per-trial
    random global ranking; authors induced. Returns None if the pool cannot
    support the edition size (infeasible cell).
    """
    sizes = np.asarray(sizes)
    n_authors = len(sizes)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    total = int(offsets[-1])
    work_author = np.repeat(np.arange(n_authors), sizes)

    author_sets: dict[str, set] = {}
    work_sets: dict[str, set] = {}
    keys = [str(e) for e in range(n_editions)]

    if family == "author":
        eligible = np.flatnonzero(sizes >= k)
        if len(eligible) < authors_per_edition:
            return None
        rank_within = None
        if alpha > 0:
            # Per-trial random rank of each work within its author's corpus,
            # shared across editions (corpora are contiguous slices, so one
            # author-keyed argsort ranks every corpus at once).
            u = rng.random(total)
            order = np.argsort(work_author + u)
            pos = np.empty(total, dtype=int)
            pos[order] = np.arange(total)
            rank_within = pos - offsets[work_author] + 1
        for key in keys:
            chosen = rng.choice(eligible, size=authors_per_edition, replace=False)
            picked: list[np.ndarray] = []
            for a in chosen:
                start, m = int(offsets[a]), int(sizes[a])
                if k >= m:
                    sel = np.arange(m)
                elif alpha > 0:
                    r = rank_within[start : start + m].astype(float)
                    gumbel_keys = -alpha * np.log(r) + rng.gumbel(size=m)
                    sel = np.argpartition(gumbel_keys, m - k)[m - k :]
                else:
                    sel = rng.choice(m, size=k, replace=False)
                picked.append(start + sel)
            author_sets[key] = set(chosen.tolist())
            work_sets[key] = set(np.concatenate(picked).tolist())
    else:
        n_works = authors_per_edition * k
        if total < n_works:
            return None
        logw = None
        if alpha > 0:
            rank_global = rng.permutation(total).astype(float) + 1.0
            logw = -alpha * np.log(rank_global)
        for key in keys:
            if logw is not None:
                gumbel_keys = logw + rng.gumbel(size=total)
                sel = np.argpartition(gumbel_keys, total - n_works)[total - n_works :]
            else:
                sel = rng.choice(total, size=n_works, replace=False)
            work_sets[key] = set(sel.tolist())
            author_sets[key] = set(np.unique(work_author[sel]).tolist())

    return summarize_pairs(
        collect_pair_stats(work_sets, author_sets, keys, cross_series_only=False)
    )


def run_sweep(args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    """Parameter sweep over (family, mean_corpus, alpha, k) synthetic cells."""
    rows: list[dict] = []
    for family in ("author", "work"):
        for mean_corpus in args.mean_corpus_values:
            print(
                f"Sweep: {family}-first, mean corpus {mean_corpus:g} "
                f"({len(args.alphas) * len(args.k_values)} cells × "
                f"{args.sweep_trials} trials)...",
                flush=True,
            )
            for alpha in args.alphas:
                for k in args.k_values:
                    diffs: list[float] = []
                    rates_a: list[float] = []
                    rates_w: list[float] = []
                    for _ in range(args.sweep_trials):
                        sizes = build_synthetic_pool(args.n_authors, mean_corpus, rng)
                        summ = run_synthetic_trial(
                            family,
                            alpha,
                            k,
                            sizes,
                            args.n_editions,
                            args.authors_per_edition,
                            rng,
                        )
                        if summ is None:
                            continue
                        diffs.append(summ.mean_jaccard_diff)
                        rates_a.append(summ.rate_a_gt_w)
                        rates_w.append(summ.rate_w_gt_a)
                    rows.append(
                        {
                            "family": family,
                            "mean_corpus": mean_corpus,
                            "alpha": alpha,
                            "k": k,
                            "n_feasible_trials": len(diffs),
                            "mean_jaccard_diff": (
                                float(np.mean(diffs)) if diffs else float("nan")
                            ),
                            "mean_rate_a_gt_w": (
                                float(np.mean(rates_a)) if rates_a else float("nan")
                            ),
                            "mean_rate_w_gt_a": (
                                float(np.mean(rates_w)) if rates_w else float("nan")
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def print_sweep_report(sweep: pd.DataFrame) -> None:
    print()
    print("===== SYNTHETIC SWEEP =====")
    print("Cells with mean D < 0 (works agree more than authors) are the W > A")
    print("region; blank cells were infeasible for the pool.\n")
    for family in ("author", "work"):
        for mean_corpus in sorted(sweep["mean_corpus"].unique()):
            sub = sweep[
                (sweep["family"] == family) & (sweep["mean_corpus"] == mean_corpus)
            ]
            print(
                f"--- {family}-first, mean corpus size {mean_corpus:g} "
                "(rows k, cols α) ---"
            )
            for name, col in (
                ("mean D = J(authors) − J(works)", "mean_jaccard_diff"),
                (
                    "mean share of pairs with shared_works > shared_authors",
                    "mean_rate_w_gt_a",
                ),
            ):
                pivot = sub.pivot(index="k", columns="alpha", values=col)
                print(f"{name}:")
                print(pivot.to_string(float_format=lambda v: f"{v:+.3f}"))
                print()


# ── Figures ───────────────────────────────────────────────────────────────────


def plot_real_panels(
    dists: dict[str, dict[str, np.ndarray]],
    observed: Summary,
    configs: list[ModelConfig],
    out: Path,
) -> None:
    """Small multiples: per-model distribution of simulated D vs. observed."""
    families = list(dict.fromkeys(c.family for c in configs))
    by_family = {f: [c for c in configs if c.family == f] for f in families}
    ncols = max(len(v) for v in by_family.values())
    nrows = len(families)

    all_vals = np.concatenate(
        [dists[c.label]["diff"] for c in configs] + [[observed.mean_jaccard_diff]]
    )
    lo, hi = float(all_vals.min()), float(all_vals.max())
    pad = 0.05 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.8 * nrows),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    for r, family in enumerate(families):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(by_family[family]):
                ax.axis("off")
                continue
            cfg = by_family[family][c]
            diffs = dists[cfg.label]["diff"]
            ax.hist(diffs, bins=40, color="0.7", edgecolor="none")
            ax.axvline(0, color="0.5", linestyle=":", linewidth=1)
            ax.axvline(
                observed.mean_jaccard_diff,
                color="black",
                linewidth=1.5,
                label="observed",
            )
            p_ge = float((diffs >= observed.mean_jaccard_diff).mean())
            ax.set_title(cfg.label, fontsize=9)
            ax.text(
                0.03,
                0.92,
                f"p(sim ≥ obs) = {fmt_p(p_ge)}",
                transform=ax.transAxes,
                fontsize=7.5,
                va="top",
            )
            ax.set_xlim(lo, hi)
            ax.set_yticks([])
            if r == nrows - 1:
                ax.set_xlabel("mean J(authors) − J(works) per trial", fontsize=8)
    axes[0][0].legend(loc="upper right", fontsize=7.5)
    fig.suptitle(
        "Simulated author-vs-work agreement gap by generative model\n"
        f"(black line = observed D = {observed.mean_jaccard_diff:+.4f}; "
        "D > 0 means authors agree more)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out}")


def plot_sweep_heatmaps(sweep: pd.DataFrame, out: Path) -> None:
    """Heatmap grid of mean D per (α, k) cell, faceted by family × mean corpus."""
    mean_corpus_values = sorted(sweep["mean_corpus"].unique())
    alphas = sorted(sweep["alpha"].unique())
    k_values = sorted(sweep["k"].unique())

    vmax = float(np.nanmax(np.abs(sweep["mean_jaccard_diff"])))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1e-3
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(
        "work_author", [WORK_COLOR, "#f7f7f7", AUTHOR_COLOR]
    )

    fig, axes = plt.subplots(
        2,
        len(mean_corpus_values),
        figsize=(3.4 * len(mean_corpus_values) + 1.5, 6.5),
        squeeze=False,
    )
    im = None
    for r, family in enumerate(("author", "work")):
        for c, mean_corpus in enumerate(mean_corpus_values):
            ax = axes[r][c]
            sub = sweep[
                (sweep["family"] == family) & (sweep["mean_corpus"] == mean_corpus)
            ]
            pivot = sub.pivot(index="k", columns="alpha", values="mean_jaccard_diff")
            pivot = pivot.reindex(index=k_values, columns=alphas)
            mat = pivot.to_numpy(dtype=float)
            im = ax.imshow(mat, cmap=cmap, norm=norm, origin="lower", aspect="auto")
            ax.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas], fontsize=8)
            ax.set_yticks(range(len(k_values)), [str(k) for k in k_values], fontsize=8)
            if r == 1:
                ax.set_xlabel("concentration α / β", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{family}-first\nworks per author k", fontsize=9)
            ax.set_title(f"mean corpus size {mean_corpus:g}", fontsize=9)
            for i in range(len(k_values)):
                for j in range(len(alphas)):
                    v = mat[i, j]
                    if np.isnan(v):
                        ax.text(
                            j,
                            i,
                            "n/a",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="0.45",
                        )
                        continue
                    color = "white" if abs(v) > 0.6 * vmax else "black"
                    weight = "bold" if v < 0 else "normal"
                    ax.text(
                        j,
                        i,
                        f"{v:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                        fontweight=weight,
                    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label(
        "mean J(authors) − J(works)\n(red: authors agree more; blue: works agree more)",
        fontsize=8,
    )
    fig.suptitle(
        "Where can works agree more than authors? Synthetic sweep\n"
        "(bold cells: W > A region; n/a: pool too small for the edition size)",
        fontsize=11,
    )
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out}")


def plot_fit_heatmap(fit: pd.DataFrame, out: Path) -> None:
    """Sequential heatmap (one hue, light→dark = worse fit) of RMS z over the
    (γ, λ) grid, best cell outlined."""
    gammas = sorted(fit["gamma"].unique())
    lams = sorted(fit["lam"].unique())
    pivot = fit.pivot(index="gamma", columns="lam", values="rms_z")
    mat = pivot.reindex(index=gammas, columns=lams).to_numpy(dtype=float)

    cmap = LinearSegmentedColormap.from_list("misfit", ["#f7f7f7", AUTHOR_COLOR])
    vmin = max(float(np.nanmin(mat)), 0.1)
    vmax = max(float(np.nanmax(mat)), vmin * 1.01)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(1.15 * len(lams) + 2.6, 0.72 * len(gammas) + 2.2))
    im = ax.imshow(mat, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    ax.set_xticks(range(len(lams)), [f"{v:g}" for v in lams], fontsize=8)
    ax.set_yticks(range(len(gammas)), [f"{v:g}" for v in gammas], fontsize=8)
    ax.set_xlabel("corpus inflation λ (unobserved works per observed work)", fontsize=9)
    ax.set_ylabel("author-canon concentration γ", fontsize=9)

    bi, bj = np.unravel_index(int(np.nanargmin(mat)), mat.shape)
    ax.add_patch(
        Rectangle(
            (bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2
        )
    )
    thresh = np.exp(0.6 * (np.log(vmax) - np.log(vmin)) + np.log(vmin))
    for i in range(len(gammas)):
        for j in range(len(lams)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if v > thresh else "black",
                fontweight="bold" if (i, j) == (bi, bj) else "normal",
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(
        "RMS z across five statistics (log scale; lower = better)", fontsize=8
    )
    ax.set_title(
        "Joint fit of the author-canon model with corpus inflation\n"
        f"(black box = best cell: γ={gammas[bi]:g}, λ={lams[bj]:g}, "
        f"RMS z = {mat[bi, bj]:.2f})",
        fontsize=10,
    )
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────


def _csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["real", "synthetic", "fit", "all"],
        default="real",
        help="real-data model comparison, synthetic parameter sweep, joint "
        "(γ, λ) fit of the author-canon model, or real+synthetic (all)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        metavar="INT",
        help=f"real-mode trials per model (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--sweep-trials",
        type=int,
        default=DEFAULT_SWEEP_TRIALS,
        metavar="INT",
        help=f"synthetic trials per sweep cell (default: {DEFAULT_SWEEP_TRIALS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="INT",
        help=f"random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="model variants to run in real mode (default: all four)",
    )
    parser.add_argument(
        "--alphas",
        type=_csv_floats,
        default=list(DEFAULT_ALPHAS),
        metavar="F,F,...",
        help="concentration values; >0 entries drive the weighted models and the "
        f"sweep axis (default: {','.join(str(a) for a in DEFAULT_ALPHAS)})",
    )
    parser.add_argument(
        "--rank-by",
        choices=["random", "frequency"],
        default="random",
        help="anchor for fixed work rankings in real mode: fresh random ranking "
        "per trial (clean null) or observed selection frequency (confirmatory)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.0,
        metavar="F",
        help="corpus inflation λ for author-family models with uniform work "
        "choice (author-uniform, author-canon): effective corpus = observed "
        "works + round(λ×observed) unobserved works; unobserved picks are "
        "unique never-shared selections (default: 0)",
    )
    parser.add_argument(
        "--decouple-allocation",
        action="store_true",
        help="re-match the per-author work-count multiset to the selected "
        "authors uniformly at random over feasible assignments (robustness "
        "check: canon rank no longer buys the larger allocations)",
    )
    parser.add_argument(
        "--gammas",
        type=_csv_floats,
        default=list(DEFAULT_GAMMAS),
        metavar="F,F,...",
        help="fit-mode author-canon concentration grid "
        f"(default: {','.join(str(g) for g in DEFAULT_GAMMAS)})",
    )
    parser.add_argument(
        "--lambdas",
        type=_csv_floats,
        default=list(DEFAULT_LAMBDAS),
        metavar="F,F,...",
        help="fit-mode corpus-inflation grid "
        f"(default: {','.join(str(v) for v in DEFAULT_LAMBDAS)})",
    )
    parser.add_argument(
        "--fit-trials",
        type=int,
        default=DEFAULT_FIT_TRIALS,
        metavar="INT",
        help=f"fit-mode trials per (γ, λ) cell (default: {DEFAULT_FIT_TRIALS})",
    )
    parser.add_argument(
        "--include-within-series",
        action="store_true",
        help="include within-series pairs (e.g. NAAAL ed.1 vs. ed.2) in real mode; "
        "excluded by default",
    )
    add_root_works_flag(parser)
    add_save_csv_flag(parser)
    parser.add_argument(
        "--n-authors",
        type=int,
        default=600,
        metavar="INT",
        help="synthetic pool size (default: 600)",
    )
    parser.add_argument(
        "--n-editions",
        type=int,
        default=12,
        metavar="INT",
        help="synthetic editions per trial (default: 12)",
    )
    parser.add_argument(
        "--authors-per-edition",
        type=int,
        default=60,
        metavar="INT",
        help="synthetic authors per edition (default: 60)",
    )
    parser.add_argument(
        "--k-values",
        type=_csv_ints,
        default=list(DEFAULT_K_VALUES),
        metavar="I,I,...",
        help="synthetic works-per-author values "
        f"(default: {','.join(str(k) for k in DEFAULT_K_VALUES)})",
    )
    parser.add_argument(
        "--mean-corpus-values",
        type=_csv_floats,
        default=list(DEFAULT_MEAN_CORPUS),
        metavar="F,F,...",
        help="synthetic mean corpus sizes "
        f"(default: {','.join(str(m) for m in DEFAULT_MEAN_CORPUS)})",
    )
    parser.add_argument(
        "--out-real",
        type=Path,
        default=OUT_REAL,
        metavar="PATH",
        help=f"real-mode figure path (default: {OUT_REAL})",
    )
    parser.add_argument(
        "--out-sweep",
        type=Path,
        default=OUT_SWEEP,
        metavar="PATH",
        help=f"sweep figure path (default: {OUT_SWEEP})",
    )
    parser.add_argument(
        "--out-fit",
        type=Path,
        default=OUT_FIT,
        metavar="PATH",
        help=f"fit figure path (default: {OUT_FIT})",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    data: RealData | None = None
    if args.mode in ("real", "fit", "all"):
        df = load_divergence_frame(args.only_root_works)
        data = prepare_real_data(df, not args.include_within_series)

    if args.mode in ("real", "all"):
        results, dists = run_real_mode(data, args, rng)
        print_real_report(
            data.observed, results, data.obs_zero_share, data.n_pairs, args
        )
        configs = build_model_configs(args.models, args.alphas, args.lam)
        plot_real_panels(dists, data.observed, configs, args.out_real)
        if args.save_csv:
            path = DATA_DIR / "author_work_agreement_models_real.csv"
            results.to_csv(path, index=False)
            print(f"CSV saved → {path}")

    if args.mode == "fit":
        fit = run_fit_mode(data, args, rng)
        print_observed_block(data.observed, data.obs_zero_share, data.n_pairs)
        print_fit_report(fit, args)
        plot_fit_heatmap(fit, args.out_fit)
        if args.save_csv:
            path = DATA_DIR / "author_work_agreement_models_fit.csv"
            fit.to_csv(path, index=False)
            print(f"CSV saved → {path}")

    if args.mode in ("synthetic", "all"):
        sweep = run_sweep(args, rng)
        print_sweep_report(sweep)
        plot_sweep_heatmaps(sweep, args.out_sweep)
        if args.save_csv:
            path = DATA_DIR / "author_work_agreement_models_sweep.csv"
            sweep.to_csv(path, index=False)
            print(f"CSV saved → {path}")


if __name__ == "__main__":
    main()
