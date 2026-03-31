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


DATA_FILE = (
    Path(__file__).parent.parent / "data" / "2026-03-13 works per afam anthology.csv"
)
BIRTH_YEAR_WINDOW = 5


# ── Edition-key logic ─────────────────────────────────────────────────────────


def assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["edition_key"] = df.apply(
        lambda r: (
            f"{r['series_id']}|{r['anthology_edition']}"
            if r["series_id"].strip()
            else r["anthology_id"]
        ),
        axis=1,
    )
    return df


# ── Birth-year precomputation ─────────────────────────────────────────────────


def precompute_birth_years(
    df: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (work_max_birth_year, author_birth_year).

    work_max_birth_year: work_id → max author birth year for that work.
    author_birth_year:   author_id → birth year.
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
        # Map author_id → birth_year (first mapping wins)
        for aid, yr in zip(aids, years):
            if aid not in author_by:
                author_by[aid] = yr
        # work_max_birth_year: take max across all appearances
        if wid and years:
            existing = work_max_by.get(wid)
            new_max = max(years)
            if existing is None or new_max > existing:
                work_max_by[wid] = new_max

    return work_max_by, author_by


# ── Author expansion ──────────────────────────────────────────────────────────


def expand_authors(df: pd.DataFrame) -> dict[str, set[str]]:
    """Return {edition_key: set of author_ids}."""
    ed_authors: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        aids_raw = row["author_ids"].strip()
        if not aids_raw:
            continue
        ek = row["edition_key"]
        aids = {a.strip() for a in aids_raw.split(",") if a.strip()}
        ed_authors.setdefault(ek, set()).update(aids)
    return ed_authors


# ── Edition ordering ──────────────────────────────────────────────────────────


def sorted_edition_keys(df: pd.DataFrame) -> list[str]:
    ed_year = df.groupby("edition_key")["anthology_publication_year"].min()
    return [ek for ek, _ in sorted(ed_year.items(), key=lambda x: (x[1], x[0]))]


# ── Per-edition statistics ────────────────────────────────────────────────────


def compute_per_edition_stats(
    df: pd.DataFrame,
    ed_authors: dict[str, set[str]],
    sorted_editions: list[str],
    work_max_by: dict[str, int],
    author_by: dict[str, int],
    window: int,
) -> pd.DataFrame:
    """Return DataFrame with one row per non-first edition."""
    all_work_ids: set[str] = set(w.strip() for w in df["work_id"] if w.strip())
    all_author_ids: set[str] = set()
    for aids in ed_authors.values():
        all_author_ids.update(aids)

    global_max_by = max(work_max_by.values()) if work_max_by else 1950
    ed_year = df.groupby("edition_key")["anthology_publication_year"].min()

    rows: list[dict] = []
    seen_works: set[str] = set()
    seen_authors: set[str] = set()

    for i, ek in enumerate(sorted_editions):
        ed_df = df[df["edition_key"] == ek]
        ed_works = set(w.strip() for w in ed_df["work_id"] if w.strip())
        ed_auths = ed_authors.get(ek, set())

        if i == 0:
            seen_works.update(ed_works)
            seen_authors.update(ed_auths)
            continue

        # Cutoff: max birth year of authors in this edition + window
        by_in_edition = [author_by[a] for a in ed_auths if a in author_by]
        cutoff = (max(by_in_edition) if by_in_edition else global_max_by) + window

        # Contemporaneous pools
        pool_works = {w for w in all_work_ids if work_max_by.get(w, cutoff) <= cutoff}
        pool_authors = {a for a in all_author_ids if author_by.get(a, cutoff) <= cutoff}

        # Slots: edition's works/authors that fall within the contemporaneous pool
        w_slots = len(ed_works & pool_works)
        a_slots = len(ed_auths & pool_authors)

        # Seen-before sets restricted to the contemporaneous pool
        seen_works_filt = seen_works & pool_works
        seen_authors_filt = seen_authors & pool_authors

        # Reselections
        w_resel = len(ed_works & seen_works_filt)
        a_resel = len(ed_auths & seen_authors_filt)

        # Chance rates
        p_w_chance = len(seen_works_filt) / len(pool_works) if pool_works else 0.0
        p_a_chance = len(seen_authors_filt) / len(pool_authors) if pool_authors else 0.0

        rows.append(
            {
                "edition_key": ek,
                "year": int(ed_year[ek]),
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

        seen_works.update(ed_works)
        seen_authors.update(ed_auths)

    return pd.DataFrame(rows)


# ── Statistical tests ─────────────────────────────────────────────────────────


def run_stats(stats: pd.DataFrame) -> dict:
    total_w_slots = int(stats["w_slots"].sum())
    total_w_resel = int(stats["w_resel"].sum())
    total_a_slots = int(stats["a_slots"].sum())
    total_a_resel = int(stats["a_resel"].sum())

    # Weighted-average chance rate (weights = slot count per edition)
    w_chance = float((stats["p_w_chance"] * stats["w_slots"]).sum() / total_w_slots)
    a_chance = float((stats["p_a_chance"] * stats["a_slots"]).sum() / total_a_slots)

    # Binomial tests (direction-appropriate one-sided)
    # Works: observed < expected → test whether below chance
    binom_w = binomtest(total_w_resel, total_w_slots, p=w_chance, alternative="less")
    # Authors: observed > expected → test whether above chance
    binom_a = binomtest(total_a_resel, total_a_slots, p=a_chance, alternative="greater")

    # Chi-square 2×2: works vs authors reselection rates
    table = [
        [total_w_resel, total_w_slots - total_w_resel],
        [total_a_resel, total_a_slots - total_a_resel],
    ]
    chi2_stat, chi2_p, chi2_dof, _ = chi2_contingency(table, correction=False)

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
    wilcox_stat, wilcox_p = wilcoxon(paired_diff, alternative="greater")

    return {
        "total_w_slots": total_w_slots,
        "total_w_resel": total_w_resel,
        "w_obs_rate": total_w_resel / total_w_slots,
        "w_chance_rate": w_chance,
        "w_ratio": (total_w_resel / total_w_slots) / w_chance
        if w_chance
        else float("inf"),
        "binom_w_p": binom_w.pvalue,
        "total_a_slots": total_a_slots,
        "total_a_resel": total_a_resel,
        "a_obs_rate": total_a_resel / total_a_slots,
        "a_chance_rate": a_chance,
        "a_ratio": (total_a_resel / total_a_slots) / a_chance
        if a_chance
        else float("inf"),
        "binom_a_p": binom_a.pvalue,
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
        default=Path(__file__).parent.parent / "viz" / "reselection_vs_chance_delta.png",
        metavar="FILE",
        help="Output path for the delta line charts (default: reselection_vs_chance_delta.png).",
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_FILE, dtype=str, na_filter=False)
    df["anthology_publication_year"] = df["anthology_publication_year"].astype(int)

    if args.only_root_works:
        df = df[
            (df["parent_work_id"].str.strip() == "")
            & (df["parent_work_title"].str.strip() == "")
        ]

    df = assign_edition_key(df)
    work_max_by, author_by = precompute_birth_years(df)
    ed_authors = expand_authors(df)
    sorted_editions = sorted_edition_keys(df)

    stats = compute_per_edition_stats(
        df, ed_authors, sorted_editions, work_max_by, author_by, args.window
    )

    results = run_stats(stats)
    print_results(stats, results, len(sorted_editions), args.window)
    plot_deltas(stats, args.window, args.out)


if __name__ == "__main__":
    main()
