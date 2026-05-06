"""
simulate_author_work_overlap.py
--------------------------------
Tests how surprising it is that anthology pairs share more authors in common
than works in common (~75% of real pairs).

For each of N simulation trials, each anthology edition's real selections are
replaced with random draws from its birth-year-eligible pool, preserving:
  - the real per-author work-count distribution (e.g. if the real edition has
    3 authors with 2 works and 10 authors with 1 work, the simulation does too)
  - selecting a work entails selecting its author (not the converse: an author
    with k eligible works gets k randomly chosen ones)
  - authorless works (anonymous, traditional) are sampled independently to
    preserve the total work count

Per-edition eligible pools use each edition's own birth-year cutoff
(max_author_birth_year + window).  Pairwise comparisons apply the stricter
min(cutoff_i, cutoff_j) filter — matching anthology_overlap_heatmap.py.

Produces a figure (viz/author_work_overlap_simulation.png) overlaying real
pairs on the simulated (shared_authors, shared_works) distribution.

By default only cross-series pairs are included (matching anthology_overlap_heatmap.py
and author_work_shared_scatter.py). Use --include-within-series to include
within-series pairs (e.g. NAAAL ed.1 vs. NAAAL ed.2).

Usage:
    uv run python analysis/simulate_author_work_overlap.py
    uv run python analysis/simulate_author_work_overlap.py --only-root-works
    uv run python analysis/simulate_author_work_overlap.py --n 1000 --seed 0
    uv run python analysis/simulate_author_work_overlap.py --include-within-series
"""

from __future__ import annotations

import argparse
import random
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
from dotenv import dotenv_values


ENV_FILE = Path(__file__).parent.parent / ".env"
QUERIES  = Path(__file__).parent.parent / "queries"
OUT_DIR  = Path(__file__).parent.parent / "viz"
OUT_FILE = OUT_DIR / "author_work_overlap_simulation.png"

BIRTH_YEAR_WINDOW = 5
DEFAULT_N = 10_000
DEFAULT_SEED = 42


# ── DB helpers ────────────────────────────────────────────────────────────────


def parse_db_params(env_file: Path) -> dict[str, str]:
    env = dotenv_values(env_file)
    raw = env["DATABASE_URL"]
    return {
        "host":     re.search(r"-h\s+(\S+)", raw).group(1),
        "user":     re.search(r"-U\s+(\S+)", raw).group(1),
        "password": re.search(r"PGPASSWORD=(\S+)", raw).group(1),
        "dbname":   raw.split()[-1],
    }


def query_db(params: dict[str, str], sql_file: Path) -> pd.DataFrame:
    sql = sql_file.read_text()
    with psycopg.connect(**params) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc.name for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


# ── Birth-year precomputation ─────────────────────────────────────────────────


def precompute_birth_years(
    df: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (work_max_birth_year, author_birth_year) dicts with str keys."""
    work_max_by: dict[str, int] = {
        str(k): v
        for k, v in df.groupby("work_id")["author_birth_year"]
        .max()
        .dropna()
        .astype(int)
        .items()
    }
    author_by: dict[str, int] = {
        str(k): v
        for k, v in (
            df[["author_id", "author_birth_year"]]
            .dropna(subset=["author_id", "author_birth_year"])
            .drop_duplicates("author_id")
            .set_index("author_id")["author_birth_year"]
            .astype(int)
            .items()
        )
    }
    return work_max_by, author_by


# ── Simulation pool construction ──────────────────────────────────────────────


def build_author_to_all_works(df: pd.DataFrame) -> dict[str, set[str]]:
    """Map each author_id → all work_ids they authored anywhere in the dataset."""
    result: dict[str, set[str]] = {}
    for aid, grp in df[df["author_id"].notna()].groupby("author_id"):
        result[str(aid)] = set(grp["work_id"].astype(str))
    return result


def build_elig_by_author(
    sorted_keys: list[str],
    edition_cutoff: dict[str, int],
    author_to_all_works: dict[str, set[str]],
    author_by: dict[str, int],
    work_max_by: dict[str, int],
) -> dict[str, list[tuple[str, list[str]]]]:
    """Return {ek: list of (author_id, eligible_work_ids)} — only authors with
    at least one eligible work in this edition's pool."""
    result: dict[str, list[tuple[str, list[str]]]] = {}
    for ek in sorted_keys:
        cutoff = edition_cutoff[ek]
        by_author: list[tuple[str, list[str]]] = []
        for aid, wids in author_to_all_works.items():
            if author_by.get(aid, cutoff) > cutoff:
                continue
            eligible = sorted(w for w in wids if work_max_by.get(w, cutoff) <= cutoff)
            if eligible:
                by_author.append((aid, eligible))
        result[ek] = by_author
    return result


def build_real_dist(
    df: pd.DataFrame,
    edition_works: dict[str, set[str]],
) -> dict[str, list[int]]:
    """Return {ek: sorted-descending per-author work counts} for the real edition.

    Only counts authored works; multi-author works contribute +1 to each co-author.
    """
    result: dict[str, list[int]] = {}
    for ek, works_in_ed in edition_works.items():
        ed_rows = df[(df["edition_key"] == ek) & df["author_id"].notna()]
        author_works: dict[str, set[str]] = {}
        for _, row in ed_rows.iterrows():
            wid = str(row["work_id"])
            if wid not in works_in_ed:
                continue
            aid = str(row["author_id"])
            author_works.setdefault(aid, set()).add(wid)
        result[ek] = sorted((len(v) for v in author_works.values()), reverse=True)
    return result


# ── Per-edition stats ─────────────────────────────────────────────────────────


def build_edition_max_by(
    edition_authors: dict[str, set[str]],
    author_by: dict[str, int],
    global_max_by: int,
) -> dict[str, int]:
    """Return {edition_key: max known author birth year}, fallback = global_max_by."""
    result: dict[str, int] = {}
    for ek, auths in edition_authors.items():
        years = [author_by[a] for a in auths if a in author_by]
        result[ek] = max(years) if years else global_max_by
    return result


# ── Pairwise data collection ──────────────────────────────────────────────────


def _same_series(ki: str, kj: str) -> bool:
    if "|" not in ki or "|" not in kj:
        return False
    return ki.rsplit("|", 1)[0] == kj.rsplit("|", 1)[0]


def collect_pairs(
    work_sets: dict[str, set[str]],
    author_sets: dict[str, set[str]],
    sorted_keys: list[str],
    pair_cutoffs: dict[tuple[str, str], int],
    work_max_by: dict[str, int],
    author_by: dict[str, int],
    cross_series_only: bool = True,
) -> list[tuple[int, int]]:
    """Return (shared_authors, shared_works) for every pair with birth-year filtering."""
    results = []
    for ki, kj in combinations(sorted_keys, 2):
        if cross_series_only and _same_series(ki, kj):
            continue
        pc = pair_cutoffs[(ki, kj)]
        wi = {w for w in work_sets[ki] if work_max_by.get(w, pc) <= pc}
        wj = {w for w in work_sets[kj] if work_max_by.get(w, pc) <= pc}
        ai = {a for a in author_sets[ki] if author_by.get(a, pc) <= pc}
        aj = {a for a in author_sets[kj] if author_by.get(a, pc) <= pc}
        results.append((len(ai & aj), len(wi & wj)))
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
        color="steelblue", alpha=0.5, linewidths=0,
        label="Simulated (bubble area ∝ frequency)",
    )
    ax.scatter(
        real_x, real_y, s=50,
        color="firebrick", zorder=4, linewidths=0,
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
        "--window", type=int, default=BIRTH_YEAR_WINDOW, metavar="INT",
        help=f"Birth-year window (default: {BIRTH_YEAR_WINDOW})",
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
    params = parse_db_params(ENV_FILE)
    df = query_db(params, QUERIES / "work-selection-divergence.sql")

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

    # ── Birth years ───────────────────────────────────────────────────────────
    work_max_by, author_by = precompute_birth_years(df)
    global_max_by = max(work_max_by.values()) if work_max_by else 1970

    # ── Per-edition sets ──────────────────────────────────────────────────────
    edition_works: dict[str, set[str]] = {
        ek: set(grp["work_id"].astype(str))
        for ek, grp in df.groupby("edition_key")
    }
    edition_authors: dict[str, set[str]] = {
        ek: set(grp["author_id"].dropna().astype(str))
        for ek, grp in df.groupby("edition_key")
    }

    sorted_keys = sorted(
        edition_works.keys(),
        key=lambda k: df.loc[df["edition_key"] == k, "anthology_publication_year"].min(),
    )

    # ── Per-edition cutoffs ───────────────────────────────────────────────────
    edition_max_by = build_edition_max_by(edition_authors, author_by, global_max_by)
    edition_cutoff = {ek: edition_max_by[ek] + args.window for ek in sorted_keys}

    # ── Simulation pools ──────────────────────────────────────────────────────
    author_to_all_works = build_author_to_all_works(df)
    elig_by_author = build_elig_by_author(
        sorted_keys, edition_cutoff, author_to_all_works, author_by, work_max_by
    )
    real_dist = build_real_dist(df, edition_works)

    # Authorless works sampled independently (no author to entail)
    authored_wids: set[str] = {w for wids in author_to_all_works.values() for w in wids}
    authorless_wids = (
        df[df["author_id"].isna()]
        .drop_duplicates("work_id")["work_id"]
        .astype(str)
    )
    all_authorless_wids = sorted(w for w in authorless_wids if w not in authored_wids)
    authorless_set = set(all_authorless_wids)

    eligible_authorless: dict[str, list[str]] = {
        ek: [
            w for w in all_authorless_wids
            if work_max_by.get(w, edition_cutoff[ek]) <= edition_cutoff[ek]
        ]
        for ek in sorted_keys
    }
    n_authorless: dict[str, int] = {}
    for ek in sorted_keys:
        ed_rows = df[(df["edition_key"] == ek) & df["author_id"].isna()]
        n_authorless[ek] = sum(
            1 for wid in ed_rows["work_id"].astype(str) if wid in authorless_set
        )

    # ── Precompute pairwise cutoffs ───────────────────────────────────────────
    pair_cutoffs: dict[tuple[str, str], int] = {}
    for ki, kj in combinations(sorted_keys, 2):
        pair_cutoffs[(ki, kj)] = min(edition_max_by[ki], edition_max_by[kj]) + args.window

    cross_series_only = not args.include_within_series

    # ── Real observation ──────────────────────────────────────────────────────
    real_pairs = collect_pairs(
        edition_works, edition_authors, sorted_keys, pair_cutoffs, work_max_by, author_by,
        cross_series_only=cross_series_only,
    )
    obs_gt   = sum(1 for sa, sw in real_pairs if sa > sw)
    obs_tie  = sum(1 for sa, sw in real_pairs if sa == sw)
    total_pairs = len(real_pairs)
    obs_rate = obs_gt / total_pairs

    print(f"===== REAL DATA (window={args.window}) =====")
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
            dist = real_dist[ek]  # [3, 2, 2, 1, 1, ...] descending
            available = list(elig_by_author[ek])
            rng.shuffle(available)

            selected_works: set[str] = set()
            selected_authors: set[str] = set()

            for i, c in enumerate(dist):
                if i >= len(available):
                    break
                aid, wids = available[i]
                selected_works.update(rng.sample(wids, min(c, len(wids))))
                selected_authors.add(aid)

            n_al = n_authorless[ek]
            if n_al > 0 and eligible_authorless[ek]:
                selected_works.update(
                    rng.sample(eligible_authorless[ek], min(n_al, len(eligible_authorless[ek])))
                )

            sim_w[ek] = selected_works
            sim_a[ek] = selected_authors

        trial_pairs = collect_pairs(
            sim_w, sim_a, sorted_keys, pair_cutoffs, work_max_by, author_by,
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

    print(
        f"===== SIMULATION (N={args.n:,}, window={args.window}, seed={args.seed}) ====="
    )
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
