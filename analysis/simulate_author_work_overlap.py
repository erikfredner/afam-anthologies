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

Usage:
    uv run python analysis/simulate_author_work_overlap.py
    uv run python analysis/simulate_author_work_overlap.py --only-root-works
    uv run python analysis/simulate_author_work_overlap.py --n 1000 --seed 0
"""

from __future__ import annotations

import argparse
import random
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILE = (
    Path(__file__).parent.parent / "data" / "2026-03-13 works per afam anthology.csv"
)
BIRTH_YEAR_WINDOW = 5
DEFAULT_N = 10_000
DEFAULT_SEED = 42


# ── Edition-key logic ─────────────────────────────────────────────────────────


def _strip_volume(title: str) -> str:
    return re.sub(r",?\s+[Vv]ol\.?\s+\d+\s*$", "", title).strip()


def assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    meta_rows: list[dict] = []
    for _, r in (
        df[
            [
                "anthology_id",
                "anthology_title",
                "series_id",
                "anthology_edition",
                "anthology_volume",
            ]
        ]
        .drop_duplicates()
        .iterrows()
    ):
        series_id = r["series_id"].strip()
        edition = r["anthology_edition"].strip()
        volume = r["anthology_volume"].strip()
        title = r["anthology_title"].strip()
        aid = r["anthology_id"]
        if series_id:
            key = f"{series_id}|{edition}"
        elif volume:
            key = f"{_strip_volume(title)}|{edition}"
        else:
            key = aid
        meta_rows.append({"anthology_id": aid, "edition_key": key})
    return df.merge(pd.DataFrame(meta_rows), on="anthology_id")


# ── Birth-year precomputation ─────────────────────────────────────────────────


def precompute_birth_years(
    df: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (work_max_birth_year, author_birth_year).

    Mirrors reselection_vs_chance.py logic exactly.
    """
    work_max_by: dict[str, int] = {}
    author_by: dict[str, int] = {}

    for _, row in df.iterrows():
        wid = row["work_id"].strip()
        aids_raw = row["author_ids"].strip()
        bys_raw = row["author_birth_years"].strip()
        if not aids_raw or not bys_raw:
            continue
        aids = [a.strip() for a in aids_raw.split(",") if a.strip()]
        bys_strs = [b.strip() for b in bys_raw.split(",") if b.strip()]
        years: list[int] = []
        for by_str in bys_strs:
            try:
                years.append(int(by_str))
            except ValueError:
                pass
        for aid, yr in zip(aids, years):
            if aid not in author_by:
                author_by[aid] = yr
        if wid and years:
            existing = work_max_by.get(wid)
            new_max = max(years)
            if existing is None or new_max > existing:
                work_max_by[wid] = new_max

    return work_max_by, author_by


# ── Author expansion ──────────────────────────────────────────────────────────


def expand_author_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Expand multi-author rows to one row per (edition_key, author_id)."""
    records: list[dict] = []
    for _, row in df[df["author_ids"] != ""].iterrows():
        ids = [i.strip() for i in row["author_ids"].split(",") if i.strip()]
        for aid in ids:
            records.append(
                {
                    "author_id": aid,
                    "edition_key": row["edition_key"],
                }
            )
    return pd.DataFrame(records)


# ── Simulation pool construction ──────────────────────────────────────────────


def build_author_to_all_works(df: pd.DataFrame) -> dict[str, set[str]]:
    """Map each author_id → all work_ids they authored anywhere in the dataset."""
    result: dict[str, set[str]] = {}
    for _, row in df[df["author_ids"] != ""].iterrows():
        wid = row["work_id"].strip()
        for aid in row["author_ids"].split(","):
            aid = aid.strip()
            if aid:
                result.setdefault(aid, set()).add(wid)
    return result


def build_elig_by_author(
    sorted_keys: list[str],
    edition_cutoff: dict[str, int],
    author_to_all_works: dict[str, set[str]],
    author_by: dict[str, int],
    work_max_by: dict[str, int],
) -> dict[str, list[tuple[str, list[str]]]]:
    """Return {ek: list of (author_id, eligible_work_ids)} — only authors with
    at least one eligible work in this edition's pool.

    'Eligible' means both the author and the work fall within the edition's
    birth-year cutoff.
    """
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

    Only counts authored works (authorless works tracked separately).
    Multi-author works contribute +1 to each co-author's count.
    """
    result: dict[str, list[int]] = {}
    for ek, works_in_ed in edition_works.items():
        ed_rows = df[df["edition_key"] == ek]
        author_works: dict[str, set[str]] = {}
        for _, row in ed_rows.iterrows():
            wid = row["work_id"].strip()
            if wid not in works_in_ed:
                continue
            aids_raw = row["author_ids"].strip()
            if not aids_raw:
                continue
            for aid in aids_raw.split(","):
                aid = aid.strip()
                if aid:
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


# ── Pairwise classification ───────────────────────────────────────────────────


def classify_pairs(
    work_sets: dict[str, set[str]],
    author_sets: dict[str, set[str]],
    sorted_keys: list[str],
    pair_cutoffs: dict[tuple[str, str], int],
    work_max_by: dict[str, int],
    author_by: dict[str, int],
) -> tuple[int, int, int]:
    """Return (n_authors_gt, n_ties, total_pairs).

    For each pair (ki, kj), applies pairwise birth-year filtering before
    comparing shared counts.
    """
    n_gt = n_tie = 0
    for ki, kj in combinations(sorted_keys, 2):
        pc = pair_cutoffs[(ki, kj)]
        wi = {w for w in work_sets[ki] if work_max_by.get(w, pc) <= pc}
        wj = {w for w in work_sets[kj] if work_max_by.get(w, pc) <= pc}
        ai = {a for a in author_sets[ki] if author_by.get(a, pc) <= pc}
        aj = {a for a in author_sets[kj] if author_by.get(a, pc) <= pc}
        sw = len(wi & wj)
        sa = len(ai & aj)
        if sa > sw:
            n_gt += 1
        elif sa == sw:
            n_tie += 1
    total = len(sorted_keys) * (len(sorted_keys) - 1) // 2
    return n_gt, n_tie, total


# ── Formatting helpers ────────────────────────────────────────────────────────


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_p(p: float) -> str:
    if p == 0.0:
        return "0 (< 1/N)"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-root-works", action="store_true")
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        metavar="INT",
        help=f"Simulation trials (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=BIRTH_YEAR_WINDOW,
        metavar="INT",
        help=f"Birth-year window (default: {BIRTH_YEAR_WINDOW})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="INT",
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_FILE, dtype=str, na_filter=False)
    df["anthology_publication_year"] = df["anthology_publication_year"].astype(int)
    df = assign_edition_key(df)
    if args.only_root_works:
        df = df[
            (df["parent_work_id"].str.strip() == "")
            & (df["parent_work_title"].str.strip() == "")
        ].copy()

    # ── Birth years ───────────────────────────────────────────────────────────
    work_max_by, author_by = precompute_birth_years(df)
    global_max_by = max(work_max_by.values()) if work_max_by else 1970

    # ── Per-edition sets ──────────────────────────────────────────────────────
    edition_works: dict[str, set[str]] = (
        df.groupby("edition_key")["work_id"].apply(set).to_dict()
    )
    expanded = expand_author_rows(df)
    edition_authors: dict[str, set[str]] = (
        expanded.groupby("edition_key")["author_id"].apply(set).to_dict()
    )

    sorted_keys = sorted(
        edition_works.keys(),
        key=lambda k: df.loc[
            df["edition_key"] == k, "anthology_publication_year"
        ].min(),
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

    # Authorless works: sampled independently (no author to entail)
    authored_wids: set[str] = set(
        w for wids in author_to_all_works.values() for w in wids
    )
    all_authorless_wids = sorted(
        {
            row["work_id"].strip()
            for _, row in df[df["author_ids"] == ""].iterrows()
            if row["work_id"].strip() not in authored_wids
        }
    )
    authorless_set = set(all_authorless_wids)
    eligible_authorless: dict[str, list[str]] = {
        ek: [
            w
            for w in all_authorless_wids
            if work_max_by.get(w, edition_cutoff[ek]) <= edition_cutoff[ek]
        ]
        for ek in sorted_keys
    }
    n_authorless: dict[str, int] = {}
    for ek in sorted_keys:
        ed_rows = df[df["edition_key"] == ek]
        n_authorless[ek] = sum(
            1
            for _, row in ed_rows.iterrows()
            if not row["author_ids"].strip()
            and row["work_id"].strip() in authorless_set
        )

    # ── Precompute pairwise cutoffs ───────────────────────────────────────────
    pair_cutoffs: dict[tuple[str, str], int] = {}
    for ki, kj in combinations(sorted_keys, 2):
        pair_cutoffs[(ki, kj)] = (
            min(edition_max_by[ki], edition_max_by[kj]) + args.window
        )

    # ── Real observation ──────────────────────────────────────────────────────
    obs_gt, obs_tie, total_pairs = classify_pairs(
        edition_works,
        edition_authors,
        sorted_keys,
        pair_cutoffs,
        work_max_by,
        author_by,
    )
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

            # Authorless works sampled independently
            n_al = n_authorless[ek]
            if n_al > 0 and eligible_authorless[ek]:
                selected_works.update(
                    rng.sample(
                        eligible_authorless[ek], min(n_al, len(eligible_authorless[ek]))
                    )
                )

            sim_w[ek] = selected_works
            sim_a[ek] = selected_authors

        n_gt, _, _ = classify_pairs(
            sim_w, sim_a, sorted_keys, pair_cutoffs, work_max_by, author_by
        )
        sim_rates.append(n_gt / total_pairs)

    print(" done.")
    print()

    # ── Report ────────────────────────────────────────────────────────────────
    arr = np.array(sim_rates)
    mean = float(arr.mean())
    std = float(arr.std())
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    p999 = float(np.percentile(arr, 99.9))
    sim_max = float(arr.max())

    z = (obs_rate - mean) / std if std > 0 else float("inf")
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
            f"  Empirical p:       < {1 / args.n:.4f}  (0 / {args.n:,} simulations reached {fmt_pct(obs_rate)})"
        )
    else:
        print(
            f"  Empirical p:       {fmt_p(emp_p)}  ({n_exceeded} / {args.n:,} simulations reached {fmt_pct(obs_rate)})"
        )


if __name__ == "__main__":
    main()
