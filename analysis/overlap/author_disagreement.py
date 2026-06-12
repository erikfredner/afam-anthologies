"""
author_disagreement.py
----------------------
Quantifies the claim: "On an author-by-author basis, editors usually
disagree about who to anthologize."

The claim is tested under five distinct operationalizations, each with an
explicit definition, threshold, and SUPPORTS / DOES NOT SUPPORT verdict:

  1. Pairwise roster Jaccard — |A∩B| / |A∪B| per cross-series edition pair.
     Supported if the median is below 0.5 and most pairs fall below 0.5.
  2. Overlap coefficient — |A∩B| / min(|A|,|B|) per cross-series pair.
     Guards against Jaccard being low merely because rosters differ in size.
  3. Per-author majority inclusion — among editorial units (series) whose
     latest edition postdates the author's corpus debut, what share selected
     the author? Supported if most authors are passed over by a majority of
     eligible units.
  4. Cohen's kappa — chance-corrected agreement per cross-series pair, over a
     pool of authors who had debuted by the earlier edition's year. Supported
     if the median kappa falls below 0.4 (Landis–Koch "moderate").
  5. Single-series authors — share of authors appearing in exactly one
     editorial unit (series or standalone) ever. Supported if above 50%.

Within-series edition pairs (e.g. NAAAL 1997 vs NAAAL 2004) are excluded
from the headline pairwise metrics — successive editions of the same series
inherit prior selections, so within-series overlap reflects path-dependence
rather than independent agreement. All-pairs numbers are printed as a
sensitivity check. Editions without a series are treated as singleton series.

Sign tests across pairs overstate significance (pairs sharing an edition are
not independent), so verdicts rest on descriptive thresholds; a conservative
edition-level sign test is printed alongside. A full permutation null
(shuffling rosters within contemporaneous pools) is left as future work —
analysis/overlap/simulate_author_work_overlap.py covers the Monte Carlo angle.

Usage:
    uv run python analysis/overlap/author_disagreement.py
    uv run python analysis/overlap/author_disagreement.py --include-excerpts
    uv run python analysis/overlap/author_disagreement.py --save-csv
"""

from __future__ import annotations

import argparse
import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from afam import DATA_DIR
from afam.cli import add_root_works_flag, add_save_csv_flag
from afam.db import query as query_db
from afam.sql import query_path

OUT_CSV_PAIRS = DATA_DIR / "author_disagreement_pairs.csv"
OUT_CSV_AUTHORS = DATA_DIR / "author_disagreement_authors.csv"

MIN_POOL = 10  # minimum kappa pool size; smaller pools yield nan
KAPPA_MODERATE = 0.4  # Landis–Koch lower bound for "moderate" agreement


def _series_identity(series_id, edition_id) -> str:
    if pd.notna(series_id):
        return f"series_{int(series_id)}"
    return f"standalone_{int(edition_id)}"


def prepare_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw query rows to one row per (edition, author).

    Expected columns: edition_id, anthology_publication_year, series_id,
    author_id, author_name. Rows with null author_id are dropped.
    Returns columns: edition_id, series_identity, year, author_id, author_name.
    """
    df = raw[raw["author_id"].notna()].copy()
    df["series_identity"] = [
        _series_identity(s, e) for s, e in zip(df["series_id"], df["edition_id"])
    ]
    df["edition_id"] = df["edition_id"].astype(int)
    df["author_id"] = df["author_id"].astype(int)
    df["year"] = df["anthology_publication_year"].astype(int)
    long = df[
        ["edition_id", "series_identity", "year", "author_id", "author_name"]
    ].drop_duplicates(subset=["edition_id", "author_id"])
    return long.reset_index(drop=True)


def build_rosters(
    long: pd.DataFrame,
) -> tuple[dict[int, frozenset], dict[int, str], dict[int, int], dict[int, int]]:
    """Return (ed_to_authors, ed_to_series, ed_to_year, author_debut_year)."""
    ed_to_authors = {
        eid: frozenset(grp["author_id"]) for eid, grp in long.groupby("edition_id")
    }
    ed_to_series = long.groupby("edition_id")["series_identity"].first().to_dict()
    ed_to_year = long.groupby("edition_id")["year"].first().to_dict()
    debut_year = long.groupby("author_id")["year"].min().to_dict()
    return ed_to_authors, ed_to_series, ed_to_year, debut_year


def cohen_kappa(
    in_a: frozenset, in_b: frozenset, pool: frozenset
) -> tuple[float, float, float]:
    """Cohen's kappa for two in/out classifications over a candidate pool.

    Returns (kappa, p_o, p_e); kappa is nan when the pool is empty or
    expected agreement is 1 (no variation to correct for).
    """
    n = len(pool)
    if n == 0:
        return math.nan, math.nan, math.nan
    a = in_a & pool
    b = in_b & pool
    both = len(a & b)
    only_a = len(a - b)
    only_b = len(b - a)
    neither = n - both - only_a - only_b
    p_o = (both + neither) / n
    p_e = ((both + only_a) / n) * ((both + only_b) / n) + ((only_b + neither) / n) * (
        (only_a + neither) / n
    )
    if p_e == 1.0:
        return math.nan, p_o, p_e
    return (p_o - p_e) / (1 - p_e), p_o, p_e


def compute_pairwise(long: pd.DataFrame, min_pool: int = MIN_POOL) -> pd.DataFrame:
    """One row per unordered edition pair with all pairwise agreement metrics.

    The kappa pool for a pair is every corpus author whose debut year is at
    or before the earlier edition's year — authors demonstrably available to
    both editorial teams.
    """
    ed_to_authors, ed_to_series, ed_to_year, debut = build_rosters(long)
    editions = sorted(ed_to_authors, key=lambda e: (ed_to_year[e], e))

    rows: list[dict] = []
    for ea, eb in combinations(editions, 2):
        a_set, b_set = ed_to_authors[ea], ed_to_authors[eb]
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        smaller = min(len(a_set), len(b_set))

        pool_year = min(ed_to_year[ea], ed_to_year[eb])
        pool = frozenset(aid for aid, dy in debut.items() if dy <= pool_year)
        if len(pool) >= min_pool:
            kappa, p_o, p_e = cohen_kappa(a_set, b_set, pool)
            prevalence = (len(a_set & pool) + len(b_set & pool)) / (2 * len(pool))
        else:
            kappa = p_o = p_e = prevalence = math.nan

        rows.append(
            {
                "edition_a": ea,
                "edition_b": eb,
                "year_a": ed_to_year[ea],
                "year_b": ed_to_year[eb],
                "series_a": ed_to_series[ea],
                "series_b": ed_to_series[eb],
                "cross_series": ed_to_series[ea] != ed_to_series[eb],
                "size_a": len(a_set),
                "size_b": len(b_set),
                "n_intersection": inter,
                "n_union": union,
                "jaccard": inter / union if union else math.nan,
                "overlap_coef": inter / smaller if smaller else math.nan,
                "pool_size": len(pool),
                "kappa": kappa,
                "p_o": p_o,
                "p_e": p_e,
                "prevalence": prevalence,
            }
        )
    return pd.DataFrame(rows)


def compute_author_inclusion(long: pd.DataFrame, level: str = "series") -> pd.DataFrame:
    """Per-author eligibility and inclusion counts.

    A unit (edition or series) is eligible for an author iff its latest
    edition year is >= the author's corpus debut year. Every unit that
    includes the author is eligible by construction, so n_included is simply
    the number of distinct units containing the author.
    """
    if level not in ("series", "edition"):
        raise ValueError(f"unknown level: {level!r}")
    unit_col = "series_identity" if level == "series" else "edition_id"

    debut = long.groupby("author_id")["year"].min()
    names = long.groupby("author_id")["author_name"].first()
    unit_max_year = long.groupby(unit_col)["year"].max()
    included = long.groupby("author_id")[unit_col].nunique()

    unit_years = np.sort(unit_max_year.to_numpy())
    rows: list[dict] = []
    for aid, dy in debut.items():
        # units with max year >= debut
        n_eligible = int(len(unit_years) - np.searchsorted(unit_years, dy, side="left"))
        n_included = int(included[aid])
        share = n_included / n_eligible
        rows.append(
            {
                "author_id": int(aid),
                "author_name": names[aid],
                "debut_year": int(dy),
                "n_eligible": n_eligible,
                "n_included": n_included,
                "inclusion_share": share,
                "majority_included": share > 0.5,
            }
        )
    return pd.DataFrame(rows)


def compute_single_series_share(long: pd.DataFrame) -> dict:
    """Share of authors appearing in exactly one editorial unit (series)."""
    n_series = long.groupby("author_id")["series_identity"].nunique()
    n_authors = len(n_series)
    n_single = int((n_series == 1).sum())
    return {
        "n_authors": n_authors,
        "n_single_series": n_single,
        "share": n_single / n_authors if n_authors else math.nan,
    }


def summarize_pairs(values: pd.Series, threshold: float = 0.5) -> dict:
    """Distribution summary plus a below-threshold sign test for pair metrics."""
    vals = values.dropna()
    n = len(vals)
    n_below = int((vals < threshold).sum())
    return {
        "n": n,
        "median": float(vals.median()) if n else math.nan,
        "q1": float(vals.quantile(0.25)) if n else math.nan,
        "q3": float(vals.quantile(0.75)) if n else math.nan,
        "mean": float(vals.mean()) if n else math.nan,
        "min": float(vals.min()) if n else math.nan,
        "max": float(vals.max()) if n else math.nan,
        "n_below": n_below,
        "pct_below": n_below / n if n else math.nan,
        "sign_p": (
            binomtest(n_below, n, p=0.5, alternative="greater").pvalue
            if n
            else math.nan
        ),
    }


def edition_level_sign_test(
    pairs: pd.DataFrame, metric: str, threshold: float = 0.5
) -> dict:
    """Conservative sign test on per-edition medians of a pair metric.

    Each edition contributes one value — the median of `metric` across its
    cross-series pairs — sidestepping the non-independence of overlapping
    pairs (n drops from ~300 pairs to ~26 editions).
    """
    cross = pairs[pairs["cross_series"]]
    melted = pd.concat(
        [
            cross[["edition_a", metric]].rename(columns={"edition_a": "edition"}),
            cross[["edition_b", metric]].rename(columns={"edition_b": "edition"}),
        ]
    ).dropna(subset=[metric])
    medians = melted.groupby("edition")[metric].median()
    n = len(medians)
    n_below = int((medians < threshold).sum())
    return {
        "n_editions": n,
        "n_below": n_below,
        "sign_p": (
            binomtest(n_below, n, p=0.5, alternative="greater").pvalue
            if n
            else math.nan
        ),
    }


# ── Report ────────────────────────────────────────────────────────────────────


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_p(p: float) -> str:
    if pd.isna(p):
        return "nan"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _print_distribution(label: str, s: dict) -> None:
    print(
        f"  {label:28} n={s['n']:>3}  median={s['median']:.3f}  "
        f"IQR=[{s['q1']:.3f}, {s['q3']:.3f}]  mean={s['mean']:.3f}  "
        f"range=[{s['min']:.3f}, {s['max']:.3f}]"
    )


def print_report(
    pairs: pd.DataFrame,
    incl_series: pd.DataFrame,
    incl_edition: pd.DataFrame,
    single: dict,
    long: pd.DataFrame,
) -> list[dict]:
    """Print the five-metric report; return summary rows for the verdict table."""
    cross = pairs[pairs["cross_series"]]
    verdicts: list[dict] = []

    print("===== CORPUS =====")
    print(f"  Editions: {long['edition_id'].nunique()}")
    print(
        f"  Editorial units (series + standalones): {long['series_identity'].nunique()}"
    )
    print(f"  Distinct authors: {long['author_id'].nunique()}")
    print(f"  Edition pairs: {len(pairs):,} ({len(cross):,} cross-series)")

    # ── Metric 1: Jaccard ──
    jac_cross = summarize_pairs(cross["jaccard"])
    jac_all = summarize_pairs(pairs["jaccard"])
    jac_ed = edition_level_sign_test(pairs, "jaccard")
    print("\n===== METRIC 1: PAIRWISE ROSTER JACCARD =====")
    print("Definition: |A∩B| / |A∪B| over the author rosters of two editions.")
    _print_distribution("Cross-series (HEADLINE):", jac_cross)
    print(
        f"    Pairs below 0.5: {jac_cross['n_below']}/{jac_cross['n']} "
        f"({_fmt_pct(jac_cross['pct_below'])}), sign test p={_fmt_p(jac_cross['sign_p'])}"
    )
    print(
        f"    Edition-level (conservative): {jac_ed['n_below']}/{jac_ed['n_editions']} "
        f"editions with median < 0.5, p={_fmt_p(jac_ed['sign_p'])}"
    )
    _print_distribution("All pairs (sensitivity):", jac_all)
    print("Caveat: pairs share editions and are not independent; verdicts rest on")
    print("        the descriptive thresholds, not the pair-level p-values.")
    supports = jac_cross["median"] < 0.5 and jac_cross["pct_below"] > 0.5
    verdicts.append(
        {
            "metric": "1. Roster Jaccard",
            "headline": f"median={jac_cross['median']:.3f}, "
            f"{_fmt_pct(jac_cross['pct_below'])} of pairs < 0.5",
            "threshold": "median < 0.5 and majority of pairs < 0.5",
            "supports": supports,
        }
    )
    print(f"VERDICT: {'SUPPORTS' if supports else 'DOES NOT SUPPORT'} the claim")

    # ── Metric 2: overlap coefficient ──
    ovl_cross = summarize_pairs(cross["overlap_coef"])
    ovl_all = summarize_pairs(pairs["overlap_coef"])
    ovl_ed = edition_level_sign_test(pairs, "overlap_coef")
    print("\n===== METRIC 2: OVERLAP COEFFICIENT =====")
    print("Definition: |A∩B| / min(|A|,|B|) — of the smaller roster, what share did")
    print("the other editor also pick? Guards against Jaccard being low merely")
    print("because rosters differ in size.")
    _print_distribution("Cross-series (HEADLINE):", ovl_cross)
    print(
        f"    Pairs below 0.5: {ovl_cross['n_below']}/{ovl_cross['n']} "
        f"({_fmt_pct(ovl_cross['pct_below'])}), sign test p={_fmt_p(ovl_cross['sign_p'])}"
    )
    print(
        f"    Edition-level (conservative): {ovl_ed['n_below']}/{ovl_ed['n_editions']} "
        f"editions with median < 0.5, p={_fmt_p(ovl_ed['sign_p'])}"
    )
    _print_distribution("All pairs (sensitivity):", ovl_all)
    supports = ovl_cross["median"] < 0.5
    verdicts.append(
        {
            "metric": "2. Overlap coefficient",
            "headline": f"median={ovl_cross['median']:.3f}",
            "threshold": "median < 0.5",
            "supports": supports,
        }
    )
    print(f"VERDICT: {'SUPPORTS' if supports else 'DOES NOT SUPPORT'} the claim")

    # ── Metric 3: per-author majority inclusion ──
    print("\n===== METRIC 3: PER-AUTHOR MAJORITY INCLUSION =====")
    print("Definition: an editorial unit is eligible for an author iff its latest")
    print("edition postdates (>=) the author's corpus debut year. What share of")
    print("authors were selected by a majority of their eligible units?")
    rows3 = {}
    for label, table, headline in (
        ("Series level (HEADLINE)", incl_series, True),
        ("Edition level (sensitivity)", incl_edition, False),
    ):
        filt = table[table["n_eligible"] >= 2]
        maj_all = table["majority_included"].mean()
        maj_filt = filt["majority_included"].mean()
        med_share = filt["inclusion_share"].median()
        print(f"  {label}:")
        print(
            f"    Authors majority-included (n_eligible >= 2): "
            f"{int(filt['majority_included'].sum())}/{len(filt)} ({_fmt_pct(maj_filt)})"
        )
        print(f"    Median inclusion share (n_eligible >= 2): {med_share:.3f}")
        print(
            f"    Unfiltered (incl. single-eligible-unit authors): {_fmt_pct(maj_all)} "
            f"majority-included of {len(table)}"
        )
        if headline:
            rows3 = {"maj_filt": maj_filt, "n": len(filt)}
    print("Caveat: an author's debut counts as an inclusion (debut is defined from")
    print("        the corpus), biasing shares upward — conservative for the claim.")
    supports = rows3["maj_filt"] < 0.5
    verdicts.append(
        {
            "metric": "3. Majority inclusion",
            "headline": f"{_fmt_pct(rows3['maj_filt'])} of authors majority-included "
            f"(series level, n={rows3['n']})",
            "threshold": "< 50% majority-included",
            "supports": supports,
        }
    )
    print(f"VERDICT: {'SUPPORTS' if supports else 'DOES NOT SUPPORT'} the claim")

    # ── Metric 4: Cohen's kappa ──
    kap_cross = summarize_pairs(cross["kappa"], threshold=KAPPA_MODERATE)
    kap_all = summarize_pairs(pairs["kappa"], threshold=KAPPA_MODERATE)
    kap_ed = edition_level_sign_test(pairs, "kappa", threshold=KAPPA_MODERATE)
    n_pos = int((cross["kappa"] > 0).sum())
    n_kap = int(cross["kappa"].notna().sum())
    print("\n===== METRIC 4: CHANCE-CORRECTED AGREEMENT (COHEN'S KAPPA) =====")
    print("Definition: each pair of editions = two raters classifying every author")
    print("in a shared pool as in/out; pool = corpus authors debuting at or before")
    print(f"the earlier edition's year (pools < {MIN_POOL} excluded).")
    _print_distribution("Cross-series (HEADLINE):", kap_cross)
    print(
        f"    Pairs with kappa > 0: {n_pos}/{n_kap} ({_fmt_pct(n_pos / n_kap)})"
        if n_kap
        else "    No pairs with a valid kappa"
    )
    print(
        f"    Pairs below {KAPPA_MODERATE} ('moderate'): {kap_cross['n_below']}/"
        f"{kap_cross['n']} ({_fmt_pct(kap_cross['pct_below'])}), "
        f"sign test p={_fmt_p(kap_cross['sign_p'])}"
    )
    print(
        f"    Edition-level (conservative): {kap_ed['n_below']}/{kap_ed['n_editions']} "
        f"editions with median < {KAPPA_MODERATE}, p={_fmt_p(kap_ed['sign_p'])}"
    )
    _print_distribution("All pairs (sensitivity):", kap_all)
    print(
        f"  Observed/expected agreement: median p_o={cross['p_o'].median():.3f}, "
        f"median p_e={cross['p_e'].median():.3f}, "
        f"median prevalence={cross['prevalence'].median():.3f}"
    )
    print("Caveat: with large pools, p_e is dominated by joint negatives (the")
    print("        prevalence paradox); read kappa alongside metric 2.")
    supports = kap_cross["median"] < KAPPA_MODERATE
    if kap_cross["median"] > 0:
        print("NUANCE: median kappa > 0 — editors agree more than chance even while")
        print("        disagreeing in absolute terms.")
    verdicts.append(
        {
            "metric": "4. Cohen's kappa",
            "headline": f"median={kap_cross['median']:.3f}",
            "threshold": f"median < {KAPPA_MODERATE} (Landis–Koch 'moderate')",
            "supports": supports,
        }
    )
    print(f"VERDICT: {'SUPPORTS' if supports else 'DOES NOT SUPPORT'} the claim")

    # ── Metric 5: single-series authors ──
    print("\n===== METRIC 5: SINGLE-SERIES AUTHORS =====")
    print("Definition: share of all anthologized authors appearing in exactly one")
    print("editorial unit (series or standalone) ever — no second editorial team")
    print("ever agreed they belonged.")
    print(
        f"  Single-unit authors: {single['n_single_series']:,}/{single['n_authors']:,} "
        f"({_fmt_pct(single['share'])})"
    )
    supports = single["share"] > 0.5
    verdicts.append(
        {
            "metric": "5. Single-series authors",
            "headline": f"{_fmt_pct(single['share'])} appear in one unit only",
            "threshold": "> 50%",
            "supports": supports,
        }
    )
    print(f"VERDICT: {'SUPPORTS' if supports else 'DOES NOT SUPPORT'} the claim")

    # ── Summary ──
    print("\n===== SUMMARY OF VERDICTS =====")
    width = max(len(v["metric"]) for v in verdicts)
    for v in verdicts:
        verdict = "SUPPORTS" if v["supports"] else "DOES NOT SUPPORT"
        print(f"  {v['metric']:<{width}}  {verdict:<16}  {v['headline']}")
        print(f"  {'':<{width}}  {'':<16}  (threshold: {v['threshold']})")
    n_support = sum(v["supports"] for v in verdicts)
    print(f"\n  {n_support}/{len(verdicts)} operationalizations support the claim.")
    return verdicts


def save_csvs(
    pairs: pd.DataFrame, incl_series: pd.DataFrame, incl_edition: pd.DataFrame
) -> None:
    OUT_CSV_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT_CSV_PAIRS, index=False)
    print(f"\nSaved → {OUT_CSV_PAIRS}")

    authors = incl_series.merge(
        incl_edition.drop(columns=["author_name", "debut_year"]),
        on="author_id",
        suffixes=("_series", "_edition"),
    )
    authors.to_csv(OUT_CSV_AUTHORS, index=False)
    print(f"Saved → {OUT_CSV_AUTHORS}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_root_works_flag(parser)
    add_save_csv_flag(parser)
    args = parser.parse_args()

    raw = query_db(query_path("work-selection-divergence"))
    if args.only_root_works:
        raw = raw[raw["parent_id"].isna()]

    long = prepare_long(raw)
    pairs = compute_pairwise(long)
    incl_series = compute_author_inclusion(long, level="series")
    incl_edition = compute_author_inclusion(long, level="edition")
    single = compute_single_series_share(long)

    print_report(pairs, incl_series, incl_edition, single, long)

    if args.save_csv:
        save_csvs(pairs, incl_series, incl_edition)


if __name__ == "__main__":
    main()
