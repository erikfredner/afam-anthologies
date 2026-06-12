"""
simulate_author_work_overlap.py
--------------------------------
Tests how surprising it is that anthology pairs share more authors in common
than works in common (~75% of real pairs).

For each of N simulation trials, each anthology edition's real selections are
replaced with random draws from its publication-year-eligible pool, preserving:
  - the real per-author work-count distribution (e.g. if the real edition has
    3 authors with 2 works and 10 authors with 1 work, the simulation does too)
  - the real authored-work and authorless-work counts
  - selected works are conditional on selected authors
  - authorless works (anonymous, traditional) are sampled independently

Per-edition eligible pools use real anthology publication years. Authors and
works first observed after the simulated anthology's publication year are not
eligible for that simulation.

Produces a figure (output/author_work_overlap_simulation.png) overlaying real
pairs on the simulated (shared_authors, shared_works) distribution.

By default only cross-series pairs are included (matching anthology_overlap_heatmap.py
and author_work_shared_scatter.py). Use --include-within-series to include
within-series pairs (e.g. NAAAL ed.1 vs. NAAAL ed.2).

Usage:
    uv run python analysis/overlap/simulate_author_work_overlap.py
    uv run python analysis/overlap/simulate_author_work_overlap.py --only-root-works
    uv run python analysis/overlap/simulate_author_work_overlap.py --n 1000 --seed 0
    uv run python analysis/overlap/simulate_author_work_overlap.py --include-within-series
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afam.db import query as query_db
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

OUT_FILE = OUTPUT_DIR / "author_work_overlap_simulation.png"

DEFAULT_N = 1_000
DEFAULT_SEED = 42


# ── Publication-year precomputation ───────────────────────────────────────────


def precompute_first_publication_years(
    df: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return first observed AFAM anthology year for works and authors."""
    work_first_year: dict[str, int] = {
        str(k): v
        for k, v in df.groupby("work_id")["anthology_publication_year"]
        .min()
        .dropna()
        .astype(int)
        .items()
    }
    author_first_year: dict[str, int] = {
        str(k): v
        for k, v in (
            df[["author_id", "anthology_publication_year"]]
            .dropna(subset=["author_id", "anthology_publication_year"])
            .groupby("author_id")["anthology_publication_year"]
            .min()
            .astype(int)
            .items()
        )
    }
    return work_first_year, author_first_year


# ── Simulation pool construction ──────────────────────────────────────────────


def build_author_to_all_works(df: pd.DataFrame) -> dict[str, set[str]]:
    """Map each author_id → all work_ids they authored anywhere in the dataset."""
    result: dict[str, set[str]] = {}
    for aid, grp in df[df["author_id"].notna()].groupby("author_id"):
        result[str(aid)] = set(grp["work_id"].astype(str))
    return result


def build_elig_by_author(
    sorted_keys: list[str],
    edition_year: dict[str, int],
    author_to_all_works: dict[str, set[str]],
    author_first_year: dict[str, int],
    work_first_year: dict[str, int],
) -> dict[str, list[tuple[str, list[str]]]]:
    """Return {ek: list of (author_id, eligible_work_ids)} — only authors with
    at least one eligible work in this edition's pool."""
    result: dict[str, list[tuple[str, list[str]]]] = {}
    for ek in sorted_keys:
        year = edition_year[ek]
        by_author: list[tuple[str, list[str]]] = []
        for aid, wids in author_to_all_works.items():
            if author_first_year.get(aid, year + 1) > year:
                continue
            eligible = sorted(
                w for w in wids if work_first_year.get(w, year + 1) <= year
            )
            if eligible:
                by_author.append((aid, eligible))
        result[ek] = by_author
    return result


@dataclass(frozen=True)
class EditionTarget:
    """Observed selection shape to preserve in a simulated edition."""

    per_author_work_counts: list[int]
    authored_work_count: int
    authorless_work_count: int


@dataclass(frozen=True)
class SimulatedEdition:
    """One simulated edition's selected authors and works."""

    author_work_ids: dict[str, set[str]]
    authorless_work_ids: set[str]

    @property
    def author_ids(self) -> set[str]:
        return set(self.author_work_ids)

    @property
    def work_ids(self) -> set[str]:
        authored = (
            set().union(*self.author_work_ids.values())
            if self.author_work_ids
            else set()
        )
        return authored | self.authorless_work_ids


def build_real_targets(df: pd.DataFrame) -> dict[str, EditionTarget]:
    """Return observed author/work selection targets for each real edition.

    Multi-author works contribute one work count to each co-author. The
    authored_work_count is unique works, so coauthored selections can make it
    smaller than the sum of per-author counts.
    """
    result: dict[str, EditionTarget] = {}
    for ek in df["edition_key"].drop_duplicates():
        ed_rows = df[(df["edition_key"] == ek) & df["author_id"].notna()]
        author_works: dict[str, set[str]] = {}
        for _, row in ed_rows.iterrows():
            wid = str(row["work_id"])
            aid = str(row["author_id"])
            author_works.setdefault(aid, set()).add(wid)
        authored_work_ids = set(ed_rows["work_id"].astype(str))
        authorless_work_ids = set(
            df.loc[
                (df["edition_key"] == ek) & df["author_id"].isna(),
                "work_id",
            ].astype(str)
        ) - authored_work_ids
        result[ek] = EditionTarget(
            per_author_work_counts=sorted(
                (len(v) for v in author_works.values()), reverse=True
            ),
            authored_work_count=len(authored_work_ids),
            authorless_work_count=len(authorless_work_ids),
        )
    return result


# ── Exact per-edition sampling ────────────────────────────────────────────────


def _choose_author_assignments(
    counts: list[int],
    eligible_by_author: list[tuple[str, list[str]]],
    rng: random.Random,
    duplicate_slots_needed: int,
) -> list[tuple[str, int, set[str]]] | None:
    """Choose distinct authors that can satisfy the requested work counts."""
    eligible_map = {aid: set(wids) for aid, wids in eligible_by_author}
    required_authors: list[str] = []

    if duplicate_slots_needed:
        work_to_authors: dict[str, list[str]] = {}
        for aid, wids in eligible_map.items():
            for wid in wids:
                work_to_authors.setdefault(wid, []).append(aid)

        shared_work_ids = [
            wid for wid, aids in work_to_authors.items() if len(set(aids)) >= 2
        ]
        rng.shuffle(shared_work_ids)
        for wid in shared_work_ids:
            available = [
                aid for aid in set(work_to_authors[wid]) if aid not in required_authors
            ]
            if len(available) < 2:
                continue
            required_authors.extend(rng.sample(available, 2))
            if len(required_authors) >= duplicate_slots_needed * 2:
                break
        if len(required_authors) < duplicate_slots_needed * 2:
            return None

    selected: list[tuple[str, int, set[str]]] = []
    used_authors: set[str] = set()
    remaining_counts = list(counts)

    for aid in required_authors:
        possible_counts = [c for c in remaining_counts if len(eligible_map[aid]) >= c]
        if not possible_counts:
            return None
        count = min(possible_counts)
        remaining_counts.remove(count)
        selected.append((aid, count, eligible_map[aid]))
        used_authors.add(aid)

    for count in sorted(remaining_counts, reverse=True):
        candidates = [
            (aid, wids)
            for aid, wids in eligible_map.items()
            if aid not in used_authors and len(wids) >= count
        ]
        if not candidates:
            return None
        aid, wids = rng.choice(candidates)
        selected.append((aid, count, wids))
        used_authors.add(aid)
    return selected


def _assign_author_works_exact(
    assignments: list[tuple[str, int, set[str]]],
    target_unique_works: int,
    rng: random.Random,
) -> dict[str, set[str]] | None:
    """Assign works so per-author counts and unique authored-work count match."""
    target_slots = sum(count for _, count, _ in assignments)
    duplicate_slots_needed = target_slots - target_unique_works
    if duplicate_slots_needed < 0:
        return None

    work_to_selected_authors: dict[str, set[str]] = {}
    author_eligible_works: dict[str, set[str]] = {}
    remaining_counts: dict[str, int] = {}
    author_work_ids: dict[str, set[str]] = {}

    for aid, _, wids in assignments:
        author_eligible_works[aid] = wids
    for aid, count, _ in assignments:
        remaining_counts[aid] = count
        author_work_ids[aid] = set()
    for aid, _, wids in assignments:
        for wid in wids:
            work_to_selected_authors.setdefault(wid, set()).add(aid)

    selected_works: set[str] = set()

    for _ in range(duplicate_slots_needed):
        candidates: list[tuple[str, list[str], int]] = []
        for wid, aids in work_to_selected_authors.items():
            available = [
                aid
                for aid in aids
                if remaining_counts[aid] > 0 and wid not in author_work_ids[aid]
            ]
            if wid in selected_works:
                if available:
                    candidates.append((wid, available, 1))
            elif len(available) >= 2:
                candidates.append((wid, available, 2))
        if not candidates:
            return None

        wid, available, n_authors = rng.choice(candidates)
        for aid in rng.sample(available, n_authors):
            author_work_ids[aid].add(wid)
            remaining_counts[aid] -= 1
        selected_works.add(wid)

    for aid, needed in remaining_counts.items():
        if needed == 0:
            continue
        candidates = sorted(
            author_eligible_works[aid] - selected_works - author_work_ids[aid]
        )
        if len(candidates) < needed:
            return None
        picked = set(rng.sample(candidates, needed))
        author_work_ids[aid].update(picked)
        selected_works.update(picked)

    unique_author_works = (
        set().union(*author_work_ids.values()) if author_work_ids else set()
    )
    if len(unique_author_works) != target_unique_works:
        return None
    return author_work_ids


def sample_simulated_edition(
    target: EditionTarget,
    eligible_by_author: list[tuple[str, list[str]]],
    eligible_authorless: list[str],
    rng: random.Random,
    max_attempts: int = 500,
) -> SimulatedEdition:
    """Sample one edition while exactly preserving observed selection counts."""
    counts = target.per_author_work_counts
    duplicate_slots_needed = sum(counts) - target.authored_work_count
    for _ in range(max_attempts):
        assignments = _choose_author_assignments(
            counts,
            eligible_by_author,
            rng,
            duplicate_slots_needed,
        )
        if assignments is None:
            continue
        author_work_ids = _assign_author_works_exact(
            assignments, target.authored_work_count, rng
        )
        if author_work_ids is None:
            continue
        if len(eligible_authorless) < target.authorless_work_count:
            break
        authorless = (
            set(rng.sample(eligible_authorless, target.authorless_work_count))
            if target.authorless_work_count
            else set()
        )
        return SimulatedEdition(author_work_ids, authorless)

    raise RuntimeError(
        "Unable to sample an exact simulated edition from the eligible pool "
        f"after {max_attempts} attempts."
    )


# ── Pairwise data collection ──────────────────────────────────────────────────


def _same_series(ki: str, kj: str) -> bool:
    if "|" not in ki or "|" not in kj:
        return False
    return ki.rsplit("|", 1)[0] == kj.rsplit("|", 1)[0]


def collect_pairs(
    work_sets: dict[str, set[str]],
    author_sets: dict[str, set[str]],
    sorted_keys: list[str],
    cross_series_only: bool = True,
) -> list[tuple[int, int]]:
    """Return (shared_authors, shared_works) for every requested edition pair."""
    results = []
    for ki, kj in combinations(sorted_keys, 2):
        if cross_series_only and _same_series(ki, kj):
            continue
        results.append(
            (len(author_sets[ki] & author_sets[kj]), len(work_sets[ki] & work_sets[kj]))
        )
    return results


# ── Formatting helpers ────────────────────────────────────────────────────────


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_p(p: float) -> str:
    if p == 0.0:
        return "0 (< 1/N)"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ── Figure ────────────────────────────────────────────────────────────────────


def plot_figure(
    real_pairs: list[tuple[int, int]],
    sim_counter: dict[tuple[int, int], int],
    out: Path,
) -> None:
    sim_coords = list(sim_counter.keys())
    sim_counts = [sim_counter[k] for k in sim_coords]
    max_count = max(sim_counts)
    sim_x = [k[0] for k in sim_coords]
    sim_y = [k[1] for k in sim_coords]
    norm_size = [200 * c / max_count for c in sim_counts]

    real_x = [p[0] for p in real_pairs]
    real_y = [p[1] for p in real_pairs]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        sim_x, sim_y, s=norm_size,
        color="0.65", alpha=0.5, linewidths=0,
        label="Simulated pairs (area ∝ frequency)",
    )
    ax.scatter(
        real_x, real_y, s=50,
        color="black", zorder=4, linewidths=0,
        label="Real pairs",
    )

    lim = max(max(sim_x + real_x, default=1), max(sim_y + real_y, default=1)) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=1, zorder=0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax.set_xlabel("Shared authors", fontsize=13)
    ax.set_ylabel("Shared works", fontsize=13)
    ax.set_title(
        "Shared authors vs. shared works: real pairs vs. simulation\n"
        "(dashed line = equal shared authors and works)",
        fontsize=11,
    )
    ax.text(
        0.72, 0.22, "More shared\nauthors",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=10, color="grey", style="italic",
    )
    ax.text(
        0.22, 0.72, "More shared\nworks",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=10, color="grey", style="italic",
    )
    ax.legend(loc="upper left", fontsize=10)

    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-root-works", action="store_true")
    parser.add_argument(
        "--include-within-series",
        action="store_true",
        help="Include within-series pairs (e.g. NAAAL ed.1 vs. ed.2). Excluded by default.",
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N, metavar="INT",
        help=f"Simulation trials (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, metavar="INT",
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_FILE, metavar="PATH",
        help=f"Output figure path (default: {OUT_FILE})",
    )
    args = parser.parse_args()

    # ── Load from DB ──────────────────────────────────────────────────────────
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

    if args.only_root_works:
        df = df[df["parent_id"].isna()].copy()

    # ── First observed publication years ─────────────────────────────────────
    work_first_year, author_first_year = precompute_first_publication_years(df)

    # ── Per-edition sets ──────────────────────────────────────────────────────
    edition_works: dict[str, set[str]] = {
        ek: set(grp["work_id"].astype(str))
        for ek, grp in df.groupby("edition_key")
    }
    edition_authors: dict[str, set[str]] = {
        ek: set(grp["author_id"].dropna().astype(str))
        for ek, grp in df.groupby("edition_key")
    }
    edition_year: dict[str, int] = {
        ek: int(grp["anthology_publication_year"].min())
        for ek, grp in df.groupby("edition_key")
    }
    sorted_keys = sorted(edition_works.keys(), key=lambda k: (edition_year[k], k))

    # ── Simulation pools ──────────────────────────────────────────────────────
    author_to_all_works = build_author_to_all_works(df)
    elig_by_author = build_elig_by_author(
        sorted_keys,
        edition_year,
        author_to_all_works,
        author_first_year,
        work_first_year,
    )
    real_targets = build_real_targets(df)

    # Authorless works sampled independently (no author to entail)
    authored_wids: set[str] = {w for wids in author_to_all_works.values() for w in wids}
    authorless_wids = (
        df[df["author_id"].isna()]
        .drop_duplicates("work_id")["work_id"]
        .astype(str)
    )
    all_authorless_wids = sorted(w for w in authorless_wids if w not in authored_wids)

    eligible_authorless: dict[str, list[str]] = {
        ek: [
            w for w in all_authorless_wids
            if work_first_year.get(w, edition_year[ek] + 1) <= edition_year[ek]
        ]
        for ek in sorted_keys
    }

    cross_series_only = not args.include_within_series

    # ── Real observation ──────────────────────────────────────────────────────
    real_pairs = collect_pairs(
        edition_works,
        edition_authors,
        sorted_keys,
        cross_series_only=cross_series_only,
    )
    obs_gt   = sum(1 for sa, sw in real_pairs if sa > sw)
    obs_tie  = sum(1 for sa, sw in real_pairs if sa == sw)
    total_pairs = len(real_pairs)
    obs_rate = obs_gt / total_pairs

    print("===== REAL DATA (publication-year pools) =====")
    print(f"  Edition pairs:              {total_pairs}")
    print(
        f"  shared_authors > shared_works:  {obs_gt:3d} / {total_pairs}  ({fmt_pct(obs_rate)})"
    )
    print(
        f"  Ties (equal):               {obs_tie:3d} / {total_pairs}  ({fmt_pct(obs_tie / total_pairs)})"
    )
    print(
        f"  shared_works > shared_authors:  {total_pairs - obs_gt - obs_tie:3d} / {total_pairs}  ({fmt_pct((total_pairs - obs_gt - obs_tie) / total_pairs)})"
    )
    print()

    # ── Simulation ────────────────────────────────────────────────────────────
    print(f"Running {args.n:,} simulations (seed={args.seed})...", end="", flush=True)
    rng = random.Random(args.seed)
    sim_rates: list[float] = []
    sim_counter: dict[tuple[int, int], int] = {}

    for trial in range(args.n):
        if (trial + 1) % 1000 == 0:
            print(".", end="", flush=True)

        sim_w: dict[str, set[str]] = {}
        sim_a: dict[str, set[str]] = {}

        for ek in sorted_keys:
            sim = sample_simulated_edition(
                real_targets[ek],
                elig_by_author[ek],
                eligible_authorless[ek],
                rng,
            )
            sim_w[ek] = sim.work_ids
            sim_a[ek] = sim.author_ids

        trial_pairs = collect_pairs(
            sim_w,
            sim_a,
            sorted_keys,
            cross_series_only=cross_series_only,
        )
        n_gt = sum(1 for sa, sw in trial_pairs if sa > sw)
        sim_rates.append(n_gt / total_pairs)
        for pair in trial_pairs:
            sim_counter[pair] = sim_counter.get(pair, 0) + 1

    print(" done.")
    print()

    # ── Report ────────────────────────────────────────────────────────────────
    arr    = np.array(sim_rates)
    mean   = float(arr.mean())
    std    = float(arr.std())
    median = float(np.median(arr))
    p95    = float(np.percentile(arr, 95))
    p99    = float(np.percentile(arr, 99))
    p999   = float(np.percentile(arr, 99.9))
    sim_max = float(arr.max())

    z     = (obs_rate - mean) / std if std > 0 else float("inf")
    emp_p = float((arr >= obs_rate).mean())

    print(f"===== SIMULATION (N={args.n:,}, publication-year pools, seed={args.seed}) =====")
    print(f"  Mean rate:         {fmt_pct(mean)} ± {fmt_pct(std)}")
    print(f"  Median:            {fmt_pct(median)}")
    print(f"  95th percentile:   {fmt_pct(p95)}")
    print(f"  99th percentile:   {fmt_pct(p99)}")
    print(f"  99.9th percentile: {fmt_pct(p999)}")
    print(f"  Max observed:      {fmt_pct(sim_max)}")
    print()
    print("===== REAL vs SIMULATION =====")
    print(f"  Observed rate:     {fmt_pct(obs_rate)}")
    print(f"  z-score:           {z:+.2f}")
    n_exceeded = int((arr >= obs_rate).sum())
    if n_exceeded == 0:
        print(
            f"  Empirical p:       < {1 / args.n:.4f}  "
            f"(0 / {args.n:,} simulations reached {fmt_pct(obs_rate)})"
        )
    else:
        print(
            f"  Empirical p:       {fmt_p(emp_p)}  "
            f"({n_exceeded} / {args.n:,} simulations reached {fmt_pct(obs_rate)})"
        )
    print()

    # ── Figure ────────────────────────────────────────────────────────────────
    plot_figure(real_pairs, sim_counter, args.out)


if __name__ == "__main__":
    main()
