"""
vernacular_reselection_test.py
------------------------------
Test whether reselection rates for vernacular works differ from those of
non-vernacular works, among the anthology editions that contain any
editor-designated vernacular material.

Groups
------
vernacular (V)     : distinct works whose start page falls inside a vernacular
                     page range (``data/vernacular_pages.csv``) in any edition
non-vernacular (NV): works appearing in at least one vernacular-containing
                     edition but never inside a vernacular range

Debuts and reselections are computed over the full AFAM corpus (a vernacular
work reselected by an edition without a marked vernacular section still
counts). Works debuting with no subsequent edition in scope are excluded.

Two tests
---------
1. Pooled comparison — V vs. NV on two metrics (ever-reselected after debut,
   and per-opportunity reselection rate), each in "all", "cross-series", and
   "cross-series, vernacular-containing editions only" scope. The last
   restricts the opportunity set further than plain cross-series: a later
   cross-series edition only counts as an opportunity (and a reselection
   only counts) if that edition itself contains a non-zero number of
   vernacular works — so an editor's cross-series pickup of a V/NV work in an
   edition with zero vernacular content never enters the comparison. Every
   scope excludes later editions of the debut's own series, so NAAAL's
   recurring vernacular sections can't masquerade as cross-editor agreement.
   Chi-square, Fisher exact, and two-proportion z tests. A dedicated report
   section compares the vernacular-editions-only estimate against the full
   cross-series estimate for the same metric.
2. Size-matched Monte Carlo — the vernacular rate against the null
   distribution of |V|-sized random samples of NV works, drawn either
   uniformly ("unmatched") or matching V's debut-edition distribution
   ("stratified", which equalizes reselection opportunities). Runs for every
   metric/scope in Test 1, including the vernacular-editions-only scope.

Usage:
    uv run python analysis/vernacular/vernacular_reselection_test.py
    uv run python analysis/vernacular/vernacular_reselection_test.py --include-excerpts
    uv run python analysis/vernacular/vernacular_reselection_test.py --n 1000 --seed 0
    uv run python analysis/vernacular/vernacular_reselection_test.py --save-csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportions_ztest

from afam import DATA_DIR, REPO_ROOT
from afam.cli import add_root_works_flag, add_save_csv_flag
from afam.vernacular import load_vernacular_ranges

from vernacular_works import load_data as load_page_data
from vernacular_works import match_vernacular_works

sys.path.insert(0, str(REPO_ROOT / "analysis" / "reselection"))

from author_vs_work_debut_reselection import (  # noqa: E402
    _clean_id,
    _rate_stats,
    _safe_div,
    build_work_rows,
    compute_work_records,
    fmt_p,
    fmt_pct,
)
from author_vs_work_debut_reselection import (  # noqa: E402
    load_data as load_reselection_data,
)

OUT_SUMMARY = DATA_DIR / "vernacular_reselection_test_summary.csv"
OUT_WORKS = DATA_DIR / "vernacular_reselection_works.csv"

DEFAULT_N = 10_000
DEFAULT_SEED = 42


# ── Metric specifications ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricSpec:
    """One reselection metric × scope combination.

    ``kind`` is ``"ever"`` (binary: reselected by >=1 later edition, one
    observation per debut) or ``"opportunity"`` (pooled later-edition slots).
    ``eligible_col`` defines the opportunity denominator that must be positive
    for a debut to enter the comparison.
    """

    name: str
    description: str
    kind: str
    eligible_col: str
    flag_col: str = ""
    num_col: str = ""
    den_col: str = ""


SPECS = [
    MetricSpec(
        "ever_all",
        "Work was reselected at least once in any later edition",
        "ever",
        "subsequent_count",
        flag_col="is_reselected",
    ),
    MetricSpec(
        "ever_cross_series",
        "Work was reselected at least once in a later different series/standalone group",
        "ever",
        "cross_series_subsequent_count",
        flag_col="is_reselected_cross_series",
    ),
    MetricSpec(
        "opportunity_all",
        "Later edition opportunities that reselect the debut work",
        "opportunity",
        "subsequent_count",
        num_col="reselection_count",
        den_col="subsequent_count",
    ),
    MetricSpec(
        "opportunity_cross_series",
        "Later different-series opportunities that reselect the debut work",
        "opportunity",
        "cross_series_subsequent_count",
        num_col="cross_series_reselection_count",
        den_col="cross_series_subsequent_count",
    ),
    MetricSpec(
        "ever_cross_series_vern_eds",
        "Work was reselected at least once in a later different-series edition "
        "that itself contains a non-zero number of vernacular works",
        "ever",
        "vern_cross_series_subsequent_count",
        flag_col="is_reselected_vern_cross_series",
    ),
    MetricSpec(
        "opportunity_cross_series_vern_eds",
        "Later different-series opportunities, restricted to editions "
        "containing a non-zero number of vernacular works, that reselect the "
        "debut work",
        "opportunity",
        "vern_cross_series_subsequent_count",
        num_col="vern_cross_series_reselection_count",
        den_col="vern_cross_series_subsequent_count",
    ),
]

# (full cross-series metric, vernacular-editions-only restricted metric)
RESTRICTION_PAIRS = [
    ("ever_cross_series", "ever_cross_series_vern_eds"),
    ("opportunity_cross_series", "opportunity_cross_series_vern_eds"),
]


# ── Group classification ──────────────────────────────────────────────────────


def build_groups(
    page_df: pd.DataFrame, ranges: pd.DataFrame, work_records: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Classify per-work debut/reselection records into V and NV pools.

    ``page_df`` should be the *unfiltered* appearance table (excerpts
    included) so the edition universe doesn't depend on the root-works
    filter; root-vs-excerpt scope is applied through ``work_records``,
    whose membership gates every classified row.

    Returns (classified, info): ``classified`` is the subset of
    ``work_records`` for works appearing in vernacular-containing editions,
    with a ``group`` column (``vernacular`` / ``non_vernacular``) and a
    normalized ``work_key``; ``info`` carries corpus/pool counts.
    """
    matched = match_vernacular_works(page_df, ranges)
    vern_ids = (
        {_clean_id(w) for w in matched["work_id"].unique()}
        if not matched.empty
        else set()
    )

    vol_to_ed = (
        page_df[["volume_id", "edition_id"]]
        .drop_duplicates("volume_id")
        .set_index("volume_id")["edition_id"]
    )
    mapped_editions = ranges["volume_id"].map(vol_to_ed)
    if mapped_editions.isna().any():
        missing = sorted(
            int(v) for v in ranges.loc[mapped_editions.isna(), "volume_id"].unique()
        )
        print(f"WARNING: vernacular volumes not found in AFAM corpus: {missing}")
    vern_editions = set(mapped_editions.dropna().unique())
    works_in_vern_eds = {
        _clean_id(w)
        for w in page_df.loc[
            page_df["edition_id"].isin(vern_editions), "work_id"
        ].unique()
    }

    out = work_records.copy()
    out["work_key"] = out["work_id"].map(_clean_id)
    out = out[out["work_key"].isin(works_in_vern_eds | vern_ids)].copy()
    out["group"] = np.where(
        out["work_key"].isin(vern_ids), "vernacular", "non_vernacular"
    )

    info = {
        "n_editions_total": int(page_df["edition_id"].nunique()),
        "n_editions_vernacular": len(vern_editions),
        "n_vernacular_works": int((out["group"] == "vernacular").sum()),
        "n_non_vernacular_works": int((out["group"] == "non_vernacular").sum()),
        "vern_edition_ids": {_clean_id(e) for e in vern_editions},
    }
    return out.reset_index(drop=True), info


def compute_vern_cross_series_records(
    raw: pd.DataFrame, all_editions: pd.DataFrame, vern_edition_ids: set[str]
) -> pd.DataFrame:
    """Work-level cross-series opportunity/reselection counts restricted to
    later editions that themselves contain a non-zero number of vernacular
    works (``vern_edition_ids``, from ``build_groups``' ``info``).

    Mirrors the cross-series columns ``author_vs_work_debut_reselection``'s
    ``compute_entity_records`` produces, but a later cross-series edition
    only contributes an opportunity — and only counts as a reselection — if
    it belongs to ``vern_edition_ids``. An edition with zero vernacular
    selections drops out of both numerator and denominator entirely.
    """
    empty_cols = [
        "work_key",
        "vern_cross_series_subsequent_count",
        "vern_cross_series_reselection_count",
        "is_reselected_vern_cross_series",
    ]

    work_rows = build_work_rows(raw)
    ed_lookup = all_editions[["edition_id", "edition_order", "entry_group"]].copy()
    ed_lookup["edition_id_key"] = ed_lookup["edition_id"].map(_clean_id)
    vern_orders = set(
        ed_lookup.loc[
            ed_lookup["edition_id_key"].isin(vern_edition_ids), "edition_order"
        ]
    )

    appearances = (
        work_rows[["work_id", "edition_id", "entry_group"]]
        .dropna(subset=["work_id", "edition_id"])
        .drop_duplicates(["work_id", "edition_id"])
        .merge(ed_lookup[["edition_id", "edition_order"]], on="edition_id", how="inner")
        .sort_values(["work_id", "edition_order", "edition_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if appearances.empty:
        return pd.DataFrame(columns=empty_cols)

    debut = (
        appearances.groupby("work_id", sort=False)
        .first()
        .reset_index()
        .rename(columns={"edition_order": "debut_order", "entry_group": "debut_group"})[
            ["work_id", "debut_order", "debut_group"]
        ]
    )

    appearance_orders = (
        appearances.groupby("work_id")["edition_order"].apply(list).to_dict()
    )
    appearance_groups = (
        appearances.groupby("work_id")["entry_group"].apply(list).to_dict()
    )
    all_orders = all_editions["edition_order"].to_numpy(dtype=int)
    all_groups_by_order = all_editions.set_index("edition_order")[
        "entry_group"
    ].to_dict()

    records: list[dict] = []
    for _, row in debut.iterrows():
        work_id = row["work_id"]
        debut_order = int(row["debut_order"])
        debut_group = row["debut_group"]

        opportunity_orders = {
            o
            for o in all_orders
            if o > debut_order
            and o in vern_orders
            and all_groups_by_order.get(o) != debut_group
        }

        orders = appearance_orders.get(work_id, [])
        groups = appearance_groups.get(work_id, [])
        reselected_orders = {
            order
            for order, group in zip(orders, groups, strict=False)
            if order > debut_order and order in vern_orders and group != debut_group
        }

        records.append(
            {
                "work_key": _clean_id(work_id),
                "vern_cross_series_subsequent_count": len(opportunity_orders),
                "vern_cross_series_reselection_count": len(reselected_orders),
                "is_reselected_vern_cross_series": len(reselected_orders) > 0,
            }
        )

    return pd.DataFrame(records)


def attach_vern_cross_series_records(
    classified: pd.DataFrame, raw: pd.DataFrame, all_editions: pd.DataFrame, info: dict
) -> pd.DataFrame:
    """Merge ``compute_vern_cross_series_records`` output onto ``classified``.

    A left join: every classified work has an appearance row in ``raw``, so
    missing matches only arise in degenerate/test fixtures, where they're
    filled to "no opportunity" rather than left as NaN.
    """
    records = compute_vern_cross_series_records(
        raw, all_editions, info["vern_edition_ids"]
    )
    out = classified.merge(records, on="work_key", how="left")
    out["vern_cross_series_subsequent_count"] = (
        out["vern_cross_series_subsequent_count"].fillna(0).astype(int)
    )
    out["vern_cross_series_reselection_count"] = (
        out["vern_cross_series_reselection_count"].fillna(0).astype(int)
    )
    out["is_reselected_vern_cross_series"] = (
        out["is_reselected_vern_cross_series"].fillna(False).astype(bool)
    )
    return out


def metric_counts(frame: pd.DataFrame, spec: MetricSpec) -> tuple[int, int]:
    """Return (k, n) for one group under a metric spec (frame pre-filtered
    to ``eligible_col > 0``)."""
    if spec.kind == "ever":
        return int(frame[spec.flag_col].sum()), len(frame)
    return int(frame[spec.num_col].sum()), int(frame[spec.den_col].sum())


# ── Test 1: pooled comparison ─────────────────────────────────────────────────


def comparison_row(
    metric: str,
    description: str,
    vern_k: int,
    vern_n: int,
    nonvern_k: int,
    nonvern_n: int,
) -> dict:
    vern_rate, vern_lo, vern_hi = _rate_stats(vern_k, vern_n)
    nonvern_rate, nonvern_lo, nonvern_hi = _rate_stats(nonvern_k, nonvern_n)
    table = [[vern_k, vern_n - vern_k], [nonvern_k, nonvern_n - nonvern_k]]

    try:
        chi2, chi2_p, _, _ = chi2_contingency(table, correction=False)
    except ValueError:
        chi2, chi2_p = float("nan"), float("nan")

    try:
        _, fisher_p = fisher_exact(table, alternative="two-sided")
    except ValueError:
        fisher_p = float("nan")

    try:
        _, z_p = proportions_ztest([vern_k, nonvern_k], [vern_n, nonvern_n])
    except (ValueError, ZeroDivisionError):
        z_p = float("nan")

    vern_odds = _safe_div(vern_rate, 1 - vern_rate)
    nonvern_odds = _safe_div(nonvern_rate, 1 - nonvern_rate)

    return {
        "test": "pooled",
        "metric": metric,
        "description": description,
        "vernacular_n": vern_n,
        "vernacular_k": vern_k,
        "vernacular_rate": vern_rate,
        "vernacular_ci_lo": vern_lo,
        "vernacular_ci_hi": vern_hi,
        "non_vernacular_n": nonvern_n,
        "non_vernacular_k": nonvern_k,
        "non_vernacular_rate": nonvern_rate,
        "non_vernacular_ci_lo": nonvern_lo,
        "non_vernacular_ci_hi": nonvern_hi,
        "rate_difference": vern_rate - nonvern_rate,
        "risk_ratio": _safe_div(vern_rate, nonvern_rate),
        "odds_ratio": _safe_div(vern_odds, nonvern_odds),
        "chi2": chi2,
        "chi2_p": chi2_p,
        "fisher_p": fisher_p,
        "two_proportion_z_p": z_p,
    }


def build_comparison(classified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in SPECS:
        eligible = classified[classified[spec.eligible_col] > 0]
        vern = eligible[eligible["group"] == "vernacular"]
        nonvern = eligible[eligible["group"] == "non_vernacular"]
        rows.append(
            comparison_row(
                spec.name,
                spec.description,
                *metric_counts(vern, spec),
                *metric_counts(nonvern, spec),
            )
        )
    return pd.DataFrame(rows)


# ── Test 2: size-matched Monte Carlo ──────────────────────────────────────────


def sample_indices(
    rng: np.random.Generator,
    pools: dict[object, np.ndarray],
    need: dict[object, int],
) -> np.ndarray:
    """Draw ``need[stratum]`` positions without replacement from each stratum's
    pool, capped at pool size."""
    parts = []
    for stratum, count in need.items():
        pool = pools.get(stratum)
        if pool is None or len(pool) == 0 or count == 0:
            continue
        take = min(count, len(pool))
        parts.append(rng.choice(pool, size=take, replace=False))
    return np.concatenate(parts) if parts else np.empty(0, dtype=int)


def empirical_two_sided_p(rates: np.ndarray, observed: float) -> float:
    """Two-sided empirical p with the (1+count)/(N+1) correction.

    Only finite draws count as evidence: NaN rates (empty/degenerate draws)
    are excluded from N, and the p is NaN when no finite draw or observation
    exists — never spuriously significant.
    """
    valid = rates[~np.isnan(rates)]
    if valid.size == 0 or np.isnan(observed):
        return float("nan")
    n = len(valid)
    p_hi = (1 + int((valid >= observed).sum())) / (n + 1)
    p_lo = (1 + int((valid <= observed).sum())) / (n + 1)
    return min(1.0, 2 * min(p_hi, p_lo))


def monte_carlo(
    vern: pd.DataFrame,
    nonvern: pd.DataFrame,
    spec: MetricSpec,
    n_draws: int,
    rng: np.random.Generator,
    stratify: bool,
) -> dict:
    """Compare the vernacular rate to |V|-sized random NV samples.

    ``vern``/``nonvern`` must already be filtered to ``eligible_col > 0``.
    With ``stratify``, each draw matches V's debut-edition distribution
    (strata short of NV works are capped at pool size).
    """
    vern_k, vern_n = metric_counts(vern, spec)
    observed = _safe_div(vern_k, vern_n)

    if spec.kind == "ever":
        num = nonvern[spec.flag_col].to_numpy(dtype=float)
        den = np.ones(len(nonvern))
    else:
        num = nonvern[spec.num_col].to_numpy(dtype=float)
        den = nonvern[spec.den_col].to_numpy(dtype=float)

    if stratify:
        need = {k: int(v) for k, v in vern["debut_edition_id"].value_counts().items()}
        positions = pd.Series(np.arange(len(nonvern)), index=nonvern.index)
        pools = {
            k: positions.loc[grp.index].to_numpy()
            for k, grp in nonvern.groupby("debut_edition_id")
        }
    else:
        need = {None: len(vern)}
        pools = {None: np.arange(len(nonvern))}

    effective = sum(min(count, len(pools.get(k, ()))) for k, count in need.items())
    shortfall = len(vern) - effective

    rates = np.empty(n_draws)
    for i in range(n_draws):
        idx = sample_indices(rng, pools, need)
        rates[i] = _safe_div(num[idx].sum(), den[idx].sum())

    # Null stats over finite draws only; a fully-degenerate null (e.g.
    # vernacular strata absent from the NV pool) yields NaN across the board.
    valid = rates[~np.isnan(rates)]
    null_mean = float(valid.mean()) if valid.size else float("nan")
    null_sd = float(valid.std(ddof=1)) if valid.size > 1 else float("nan")
    null_lo, null_hi = (
        (float(q) for q in np.quantile(valid, [0.025, 0.975]))
        if valid.size
        else (float("nan"), float("nan"))
    )
    return {
        "test": "monte_carlo",
        "metric": spec.name,
        "mode": "stratified" if stratify else "unmatched",
        "vernacular_n": vern_n,
        "vernacular_k": vern_k,
        "vernacular_rate": observed,
        "sample_size": effective,
        "sample_shortfall": shortfall,
        "n_draws": n_draws,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_lo": null_lo,
        "null_hi": null_hi,
        "z_score": _safe_div(observed - null_mean, null_sd),
        "empirical_p": empirical_two_sided_p(rates, observed),
    }


def run_rng(seed: int, spec_name: str, mode: str) -> np.random.Generator:
    """An independent generator per metric×mode run, keyed by name.

    Deriving each run's stream from (seed, spec_name, mode) rather than
    threading one generator through the loop keeps every run's results
    stable under the same --seed when SPECS is reordered or extended.
    """
    digest = hashlib.sha256(f"{spec_name}:{mode}".encode()).digest()
    return np.random.default_rng([seed, int.from_bytes(digest[:8], "big")])


def build_monte_carlo(
    classified: pd.DataFrame, n_draws: int, seed: int
) -> pd.DataFrame:
    rows = []
    for spec in SPECS:
        eligible = classified[classified[spec.eligible_col] > 0]
        vern = eligible[eligible["group"] == "vernacular"]
        nonvern = eligible[eligible["group"] == "non_vernacular"]
        if vern.empty or nonvern.empty:
            continue
        for stratify in (False, True):
            mode = "stratified" if stratify else "unmatched"
            rng = run_rng(seed, spec.name, mode)
            rows.append(monte_carlo(vern, nonvern, spec, n_draws, rng, stratify))
    return pd.DataFrame(rows)


# ── Output ────────────────────────────────────────────────────────────────────


def fmt_z(z: float) -> str:
    return "n/a" if pd.isna(z) else f"{z:+.2f}"


def print_groups(info: dict, only_root_works: bool) -> None:
    scope = "root works only" if only_root_works else "including excerpts"
    print("\n===== GROUPS =====")
    print(
        f"  Vernacular-containing editions: {info['n_editions_vernacular']} "
        f"of {info['n_editions_total']} AFAM editions"
    )
    print(
        f"  Vernacular works (V): {info['n_vernacular_works']:,}   "
        f"Non-vernacular pool (NV): {info['n_non_vernacular_works']:,}   ({scope})"
    )


def print_comparison(comparison: pd.DataFrame) -> None:
    print("\n===== TEST 1: POOLED COMPARISON (V vs NV) =====")
    print(
        "  Note: per-opportunity rows pool a work's later-edition slots, which\n"
        "  are not independent observations; Test 2 is the robust version."
    )
    for _, row in comparison.iterrows():
        print(f"\n  {row['metric']}: {row['description']}")
        print(
            f"    Vernacular:     n={int(row['vernacular_n']):5d}  "
            f"k={int(row['vernacular_k']):5d}  "
            f"rate={fmt_pct(row['vernacular_rate'])}  "
            f"95% CI [{fmt_pct(row['vernacular_ci_lo'])}, "
            f"{fmt_pct(row['vernacular_ci_hi'])}]"
        )
        print(
            f"    Non-vernacular: n={int(row['non_vernacular_n']):5d}  "
            f"k={int(row['non_vernacular_k']):5d}  "
            f"rate={fmt_pct(row['non_vernacular_rate'])}  "
            f"95% CI [{fmt_pct(row['non_vernacular_ci_lo'])}, "
            f"{fmt_pct(row['non_vernacular_ci_hi'])}]"
        )
        print(
            f"    Diff={fmt_pct(row['rate_difference'])}  "
            f"RR={row['risk_ratio']:.2f}x  "
            f"OR={row['odds_ratio']:.2f}x  "
            f"chi2 p={fmt_p(row['chi2_p'])}  "
            f"Fisher p={fmt_p(row['fisher_p'])}  "
            f"z-test p={fmt_p(row['two_proportion_z_p'])}"
        )


def print_restriction_comparison(comparison: pd.DataFrame) -> None:
    """Report each vernacular-editions-only restricted metric against its
    full cross-series counterpart, and the change in the V-NV rate
    difference between the two."""
    print("\n===== TEST 1b: VERNACULAR-EDITIONS-ONLY vs FULL CROSS-SERIES =====")
    print(
        "  Vernacular-editions-only: opportunities (and reselections) only\n"
        "  count when the later cross-series edition itself contains a\n"
        "  non-zero number of vernacular works."
    )
    by_metric = comparison.set_index("metric")
    for full_name, restricted_name in RESTRICTION_PAIRS:
        if full_name not in by_metric.index or restricted_name not in by_metric.index:
            continue
        full = by_metric.loc[full_name]
        restricted = by_metric.loc[restricted_name]
        print(f"\n  {restricted_name} vs {full_name}")
        print(
            f"    Full cross-series:        "
            f"V={fmt_pct(full['vernacular_rate'])} (n={int(full['vernacular_n'])})  "
            f"NV={fmt_pct(full['non_vernacular_rate'])} (n={int(full['non_vernacular_n'])})  "
            f"diff={fmt_pct(full['rate_difference'])}  "
            f"chi2 p={fmt_p(full['chi2_p'])}"
        )
        print(
            f"    Vernacular-editions-only:  "
            f"V={fmt_pct(restricted['vernacular_rate'])} (n={int(restricted['vernacular_n'])})  "
            f"NV={fmt_pct(restricted['non_vernacular_rate'])} (n={int(restricted['non_vernacular_n'])})  "
            f"diff={fmt_pct(restricted['rate_difference'])}  "
            f"chi2 p={fmt_p(restricted['chi2_p'])}"
        )
        delta = restricted["rate_difference"] - full["rate_difference"]
        print(
            f"    Change in (V-NV) diff from restricting to vernacular-containing "
            f"editions: {fmt_pct(delta)}"
        )
    print()


def print_monte_carlo(mc: pd.DataFrame, n_draws: int, seed: int) -> None:
    print(
        f"\n===== TEST 2: SIZE-MATCHED MONTE CARLO "
        f"(N={n_draws:,} draws, seed={seed}) ====="
    )
    print(
        "  Null: random NV samples of |V| works — 'unmatched' draws uniformly,\n"
        "  'stratified' matches V's debut-edition distribution."
    )
    for _, row in mc.iterrows():
        note = (
            f"  [{int(row['sample_shortfall'])} short of |V|]"
            if row["sample_shortfall"]
            else ""
        )
        print(
            f"  {row['metric']:<34} {row['mode']:<11} "
            f"obs={fmt_pct(row['vernacular_rate'])}  "
            f"null={fmt_pct(row['null_mean'])} ± {fmt_pct(row['null_sd'])}  "
            f"z={fmt_z(row['z_score'])}  "
            f"p={fmt_p(row['empirical_p'])}{note}"
        )
    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_root_works_flag(parser)
    add_save_csv_flag(parser)
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        metavar="N",
        help=f"Monte Carlo draws per metric/mode (default: {DEFAULT_N}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="S",
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    args = parser.parse_args()

    # The page table stays unfiltered so the edition universe is stable;
    # root-vs-excerpt scope enters through work_records (see build_groups).
    page_df = load_page_data(only_root_works=False)
    ranges = load_vernacular_ranges()
    raw, all_editions = load_reselection_data(args.only_root_works)
    work_records = compute_work_records(raw, all_editions)

    classified, info = build_groups(page_df, ranges, work_records)
    classified = attach_vern_cross_series_records(classified, raw, all_editions, info)
    comparison = build_comparison(classified)
    mc = build_monte_carlo(classified, args.n, args.seed)

    print_groups(info, args.only_root_works)
    print_comparison(comparison)
    print_restriction_comparison(comparison)
    print_monte_carlo(mc, args.n, args.seed)

    if args.save_csv:
        summary = pd.concat([comparison, mc], ignore_index=True, sort=False)
        summary.to_csv(OUT_SUMMARY, index=False)

        titles = (
            page_df.assign(work_key=page_df["work_id"].map(_clean_id))
            .groupby("work_key")["work_title"]
            .first()
        )
        works_out = classified.copy()
        works_out["work_title"] = works_out["work_key"].map(titles)
        works_out[
            [
                "work_id",
                "work_title",
                "group",
                "debut_year",
                "debut_edition_id",
                "subsequent_count",
                "reselection_count",
                "is_reselected",
                "cross_series_subsequent_count",
                "cross_series_reselection_count",
                "is_reselected_cross_series",
                "vern_cross_series_subsequent_count",
                "vern_cross_series_reselection_count",
                "is_reselected_vern_cross_series",
                "n_appearances",
            ]
        ].to_csv(OUT_WORKS, index=False)
        print(f"Saved → {OUT_SUMMARY}")
        print(f"Saved → {OUT_WORKS}")


if __name__ == "__main__":
    main()
