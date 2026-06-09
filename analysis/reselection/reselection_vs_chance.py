"""
reselection_vs_chance.py
------------------------
Tests whether reselection of works and authors across African American
literary anthologies exceeds what pure chance would predict.

For each anthology edition (after the first, chronologically), computes
how many works/authors were already seen in earlier editions, then
compares that count against a null model: random draw from the
contemporaneous pool — works/authors by authors born within BIRTH_YEAR_WINDOW
years of the youngest author in that edition. This prevents works by
authors not yet born from inflating the denominator of the chance rate.

Statistical tests:
  - scipy.stats.binomtest for works vs weighted-average chance rate
  - scipy.stats.binomtest for authors vs weighted-average chance rate
  - scipy.stats.chi2_contingency (2×2) comparing work vs author reselection rates

Usage:
    uv run python analysis/reselection_vs_chance.py
    uv run python analysis/reselection_vs_chance.py --only-root-works
    uv run python analysis/reselection_vs_chance.py --window 10
    uv run python analysis/reselection_vs_chance.py --out delta.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, wilcoxon

from afam.db import query
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

BIRTH_YEAR_WINDOW = 5


def load_data(only_root_works: bool = False) -> pd.DataFrame:
    df = query(query_path("works-per-afam-edition"))
    if only_root_works:
        df = df[df["parent_id"].isna()]
    return df


# ── Birth-year precomputation ─────────────────────────────────────────────────


def precompute_birth_years(
    df: pd.DataFrame,
) -> tuple[dict, dict]:
    """Return (work_max_birth_year, author_birth_year).

    work_max_birth_year: work_id → max author birth year for that work.
    author_birth_year:   author_id → birth year.
    """
    work_max_by: dict = {}
    author_by: dict = {}

    for _, row in df.dropna(subset=["author_birth_year"]).iterrows():
        by = int(row["author_birth_year"])
        aid = row["author_id"]
        wid = row["work_id"]
        if aid not in author_by:
            author_by[aid] = by
        if wid not in work_max_by or by > work_max_by[wid]:
            work_max_by[wid] = by

    return work_max_by, author_by


# ── Author expansion ──────────────────────────────────────────────────────────


def expand_authors(df: pd.DataFrame) -> dict[object, set]:
    """Return {edition_id: set of author_ids}."""
    return {
        eid: set(g["author_id"].dropna())
        for eid, g in df.groupby("edition_id")
    }


# ── Edition ordering ──────────────────────────────────────────────────────────


def sorted_edition_ids(df: pd.DataFrame) -> list:
    ed_year = df.groupby("edition_id")["anthology_publication_year"].min()
    return [eid for eid, _ in sorted(ed_year.items(), key=lambda x: (x[1], x[0]))]


# ── Per-edition statistics ────────────────────────────────────────────────────


def compute_per_edition_stats(
    df: pd.DataFrame,
    ed_authors: dict[object, set],
    sorted_editions: list,
    work_max_by: dict,
    author_by: dict,
    window: int,
) -> pd.DataFrame:
    """Return DataFrame with one row per non-first edition."""
    ed_works: dict[object, set] = {
        eid: set(g["work_id"]) for eid, g in df.groupby("edition_id")
    }

    all_work_ids: set = set(df["work_id"])
    all_author_ids: set = set()
    for aids in ed_authors.values():
        all_author_ids.update(aids)

    global_max_by = max(work_max_by.values()) if work_max_by else 1950
    ed_year = df.groupby("edition_id")["anthology_publication_year"].min()

    rows: list[dict] = []
    seen_works: set = set()
    seen_authors: set = set()

    for i, eid in enumerate(sorted_editions):
        ed_w = ed_works.get(eid, set())
        ed_a = ed_authors.get(eid, set())

        if i == 0:
            seen_works.update(ed_w)
            seen_authors.update(ed_a)
            continue

        # Cutoff: max birth year of authors in this edition + window
        by_in_edition = [author_by[a] for a in ed_a if a in author_by]
        cutoff = (max(by_in_edition) if by_in_edition else global_max_by) + window

        # Contemporaneous pools
        pool_works = {w for w in all_work_ids if work_max_by.get(w, cutoff) <= cutoff}
        pool_authors = {a for a in all_author_ids if author_by.get(a, cutoff) <= cutoff}

        # Slots: edition's works/authors that fall within the contemporaneous pool
        w_slots = len(ed_w & pool_works)
        a_slots = len(ed_a & pool_authors)

        # Seen-before sets restricted to the contemporaneous pool
        seen_works_filt = seen_works & pool_works
        seen_authors_filt = seen_authors & pool_authors

        # Reselections
        w_resel = len(ed_w & seen_works_filt)
        a_resel = len(ed_a & seen_authors_filt)

        # Chance rates
        p_w_chance = len(seen_works_filt) / len(pool_works) if pool_works else 0.0
        p_a_chance = len(seen_authors_filt) / len(pool_authors) if pool_authors else 0.0

        rows.append(
            {
                "edition_id": eid,
                "year": int(ed_year[eid]),
                "cutoff": cutoff,
                "pool_works": len(pool_works),
                "pool_authors": len(pool_authors),
                "w_slots": w_slots,
                "w_resel": w_resel,
                "p_w_chance": p_w_chance,
                "a_slots": a_slots,
                "a_resel": a_resel,
                "p_a_chance": p_a_chance,
            }
        )

        seen_works.update(ed_w)
        seen_authors.update(ed_a)

    return pd.DataFrame(rows)


# ── Statistical tests ─────────────────────────────────────────────────────────


def run_stats(stats: pd.DataFrame) -> dict:
    total_w_slots = int(stats["w_slots"].sum())
    total_w_resel = int(stats["w_resel"].sum())
    total_a_slots = int(stats["a_slots"].sum())
    total_a_resel = int(stats["a_resel"].sum())

    # Weighted-average chance rate (weights = slot count per edition)
    w_chance = (
        float((stats["p_w_chance"] * stats["w_slots"]).sum() / total_w_slots)
        if total_w_slots
        else float("nan")
    )
    a_chance = (
        float((stats["p_a_chance"] * stats["a_slots"]).sum() / total_a_slots)
        if total_a_slots
        else float("nan")
    )

    # Binomial tests (direction-appropriate one-sided)
    # Works: observed < expected → test whether below chance
    binom_w_p = (
        binomtest(total_w_resel, total_w_slots, p=w_chance, alternative="less").pvalue
        if total_w_slots and not pd.isna(w_chance)
        else float("nan")
    )
    # Authors: observed > expected → test whether above chance
    binom_a_p = (
        binomtest(total_a_resel, total_a_slots, p=a_chance, alternative="greater").pvalue
        if total_a_slots and not pd.isna(a_chance)
        else float("nan")
    )

    # Chi-square 2×2: works vs authors reselection rates
    table = [
        [total_w_resel, total_w_slots - total_w_resel],
        [total_a_resel, total_a_slots - total_a_resel],
    ]
    if total_w_slots == 0 or total_a_slots == 0:
        chi2_stat, chi2_p, chi2_dof = float("nan"), float("nan"), 1
    else:
        try:
            chi2_stat, chi2_p, chi2_dof, _ = chi2_contingency(table, correction=False)
        except ValueError:
            chi2_stat, chi2_p, chi2_dof = float("nan"), float("nan"), 1

    # Sign tests: per-edition consistency (how many editions go in the expected direction?)
    n_ed = len(stats)
    w_obs_per_ed = stats["w_resel"] / stats["w_slots"].replace(0, float("nan"))
    a_obs_per_ed = stats["a_resel"] / stats["a_slots"].replace(0, float("nan"))
    w_neg_ed = int((w_obs_per_ed < stats["p_w_chance"]).sum())
    a_pos_ed = int((a_obs_per_ed > stats["p_a_chance"]).sum())
    sign_w = binomtest(w_neg_ed, n_ed, p=0.5, alternative="greater")
    sign_a = binomtest(a_pos_ed, n_ed, p=0.5, alternative="greater")

    # Paired Wilcoxon: is author delta systematically greater than work delta per edition?
    w_delta_per_ed = (w_obs_per_ed - stats["p_w_chance"]).dropna()
    a_delta_per_ed = (a_obs_per_ed - stats["p_a_chance"]).dropna()
    paired_diff = (a_delta_per_ed - w_delta_per_ed).dropna()
    if paired_diff.empty or (paired_diff == 0).all():
        wilcox_stat, wilcox_p = float("nan"), float("nan")
    else:
        wilcox_stat, wilcox_p = wilcoxon(paired_diff, alternative="greater")

    return {
        "total_w_slots": total_w_slots,
        "total_w_resel": total_w_resel,
        "w_obs_rate": total_w_resel / total_w_slots if total_w_slots else float("nan"),
        "w_chance_rate": w_chance,
        "w_ratio": (total_w_resel / total_w_slots) / w_chance
        if total_w_slots and w_chance
        else float("inf"),
        "binom_w_p": binom_w_p,
        "total_a_slots": total_a_slots,
        "total_a_resel": total_a_resel,
        "a_obs_rate": total_a_resel / total_a_slots if total_a_slots else float("nan"),
        "a_chance_rate": a_chance,
        "a_ratio": (total_a_resel / total_a_slots) / a_chance
        if total_a_slots and a_chance
        else float("inf"),
        "binom_a_p": binom_a_p,
        "chi2_stat": chi2_stat,
        "chi2_p": chi2_p,
        "chi2_dof": int(chi2_dof),
        "n_editions": n_ed,
        "w_neg_ed": w_neg_ed,
        "a_pos_ed": a_pos_ed,
        "sign_w_p": sign_w.pvalue,
        "sign_a_p": sign_a.pvalue,
        "wilcox_stat": wilcox_stat,
        "wilcox_p": wilcox_p,
    }


# ── Output ────────────────────────────────────────────────────────────────────


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def print_results(
    stats: pd.DataFrame,
    results: dict,
    n_editions: int,
    window: int,
) -> None:
    n_non_first = len(stats)
    final = stats.iloc[-1]

    print(f"===== POOL SIZES (birth-year adjusted, window={window}) =====")
    print(f"  Distinct works   (final-edition pool): {int(final['pool_works']):>5,}")
    print(f"  Distinct authors (final-edition pool): {int(final['pool_authors']):>5,}")
    print(f"  Total editions: {n_editions}  ({n_non_first} non-first)")
    print()
    print("===== RESELECTION SUMMARY =====")
    col_w = "Works"
    col_a = "Authors"
    print(f"{'':35} {col_w:>10} {col_a:>10}")
    print(
        f"{'Slots (N)':35} {results['total_w_slots']:>10,} {results['total_a_slots']:>10,}"
    )
    print(
        f"{'Observed reselections':35} {results['total_w_resel']:>10,} {results['total_a_resel']:>10,}"
    )
    print(
        f"{'Observed rate':35} {fmt_pct(results['w_obs_rate']):>10} {fmt_pct(results['a_obs_rate']):>10}"
    )
    print(
        f"{'Chance rate (null model)':35} {fmt_pct(results['w_chance_rate']):>10}"
        f" {fmt_pct(results['a_chance_rate']):>10}"
    )
    print(
        f"{'Obs / Expected':35} {results['w_ratio']:>10.2f} {results['a_ratio']:>10.2f}"
    )
    print(f"{'Binomial p (works < chance)':35} {fmt_p(results['binom_w_p']):>10}")
    print(
        f"{'Binomial p (authors > chance)':35} {'':10} {fmt_p(results['binom_a_p']):>10}"
    )
    print()
    print("===== WORKS vs AUTHORS (CHI-SQUARE) =====")
    print(
        f"  Works:   {results['total_w_resel']:,} reselected / {results['total_w_slots']:,} slots"
        f"  ({fmt_pct(results['w_obs_rate'])})"
    )
    print(
        f"  Authors: {results['total_a_resel']:,} reselected / {results['total_a_slots']:,} slots"
        f"  ({fmt_pct(results['a_obs_rate'])})"
    )
    print(
        f"  chi2 = {results['chi2_stat']:.2f}  "
        f"dof = {results['chi2_dof']}  "
        f"p = {fmt_p(results['chi2_p'])}"
    )
    print()
    n = results["n_editions"]
    print(f"===== PER-EDITION SIGN TESTS ({n} non-first editions) =====")
    print(
        f"  Works below chance:   {results['w_neg_ed']:2d} / {n}"
        f"  (p = {fmt_p(results['sign_w_p'])}, binomial sign test)"
    )
    print(
        f"  Authors above chance: {results['a_pos_ed']:2d} / {n}"
        f"  (p = {fmt_p(results['sign_a_p'])}, binomial sign test)"
    )
    print()
    print("===== PAIRED WILCOXON (author delta − work delta, per edition) =====")
    print(
        f"  statistic = {results['wilcox_stat']:.1f}  "
        f"p = {fmt_p(results['wilcox_p'])}  (alternative: author delta > work delta)"
    )


# ── Charts ───────────────────────────────────────────────────────────────────


def _x_labels(stats: pd.DataFrame) -> list[str]:
    """Year labels; same-year editions get a/b/c suffix."""
    from collections import Counter

    counts = Counter(stats["year"])
    idx: dict[int, int] = {}
    labels = []
    for yr in stats["year"]:
        if counts[yr] > 1:
            i = idx.get(yr, 0)
            labels.append(f"{yr}{chr(ord('a') + i)}")
            idx[yr] = i + 1
        else:
            labels.append(str(yr))
    return labels


def plot_deltas(stats: pd.DataFrame, window: int, out_path: Path) -> None:
    """Save two line charts of (observed − expected) reselection rate per edition."""
    w_obs = stats["w_resel"] / stats["w_slots"].replace(0, float("nan"))
    a_obs = stats["a_resel"] / stats["a_slots"].replace(0, float("nan"))
    w_delta = (w_obs - stats["p_w_chance"]) * 100
    a_delta = (a_obs - stats["p_a_chance"]) * 100

    x = list(range(len(stats)))
    labels = _x_labels(stats)

    fig, (ax_w, ax_a) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, delta, title, color in [
        (ax_w, w_delta, "Works", "#2196F3"),
        (ax_a, a_delta, "Authors", "#E64A19"),
    ]:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
        pos = delta.clip(lower=0)
        neg = delta.clip(upper=0)
        ax.fill_between(x, pos, 0, alpha=0.18, color=color, label="_nolegend_")
        ax.fill_between(x, neg, 0, alpha=0.18, color="#888888", label="_nolegend_")
        ax.plot(
            x, delta, color=color, linewidth=1.8, marker="o", markersize=5, zorder=3
        )
        ax.set_ylabel("Δ reselection rate (pp)", fontsize=9)
        ax.set_title(
            f"{title}: Observed − Expected reselection rate per edition", fontsize=10
        )
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}pp"))
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.set_xlim(-0.5, len(x) - 0.5)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax_a.set_xlabel("Anthology edition (chronological)", fontsize=9)

    fig.suptitle(
        f"Author vs. work reselection: observed − expected rate by anthology edition\n"
        f"(birth-year-adjusted pool, window = {window} yrs)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-root-works",
        action="store_true",
        help="Restrict to works without a parent work (exclude excerpts/selections).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=BIRTH_YEAR_WINDOW,
        metavar="N",
        help=f"Birth-year window for contemporaneous pool (default: {BIRTH_YEAR_WINDOW}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR / "reselection_vs_chance_delta.png",
        metavar="FILE",
        help="Output path for the delta line charts (default: reselection_vs_chance_delta.png).",
    )
    args = parser.parse_args()

    df = load_data(only_root_works=args.only_root_works)

    work_max_by, author_by = precompute_birth_years(df)
    ed_authors = expand_authors(df)
    sorted_editions = sorted_edition_ids(df)

    stats = compute_per_edition_stats(
        df, ed_authors, sorted_editions, work_max_by, author_by, args.window
    )

    results = run_stats(stats)
    print_results(stats, results, len(sorted_editions), args.window)
    plot_deltas(stats, args.window, args.out)


if __name__ == "__main__":
    main()
