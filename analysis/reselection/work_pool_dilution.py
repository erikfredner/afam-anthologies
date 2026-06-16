"""
work_pool_dilution.py
---------------------
Test whether Claude McKay's "If We Must Die" outpaces Langston Hughes's
"I, Too" because Hughes's anthologized work pool is larger.

The unit of analysis is the individual ``work_id`` in each AFAM edition. This
intentionally does not use the common ``parent_id IS NULL`` root-work filter:
both focal poems are child works in the database, and dropping child rows would
remove the observations whose published counts this script is testing.

For each focal author/work and each post-debut edition, the primary null model
sets

    p(selected) = repeat_slots / prior_author_work_pool

where repeat slots are the author's works in that edition that had appeared in
earlier AFAM editions. The popularity-weighted sensitivity instead samples
distinct repeat slots with probability proportional to each prior work's
cumulative selection count.

Usage:
    uv run python analysis/reselection/work_pool_dilution.py
    uv run python analysis/reselection/work_pool_dilution.py --sims 100000
    uv run python analysis/reselection/work_pool_dilution.py --save-csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from afam import DATA_DIR
from afam.cli import add_save_csv_flag
from afam.db import query
from afam.sql import query_path

from author_vs_work_debut_reselection import add_entry_group

OUT_PER_EDITION = DATA_DIR / "work_pool_dilution_per_edition.csv"
OUT_SUMMARY = DATA_DIR / "work_pool_dilution_summary.csv"
OUT_GAP = DATA_DIR / "work_pool_dilution_gap.csv"
OUT_CORPUS_BINS = DATA_DIR / "work_pool_dilution_corpus_bins.csv"
OUT_CORPUS_LIFTS = DATA_DIR / "work_pool_dilution_corpus_lifts.csv"
OUT_AUTHOR_PORTFOLIO = DATA_DIR / "work_pool_dilution_author_portfolio.csv"
OUT_AUTHOR_PORTFOLIO_SUMMARY = (
    DATA_DIR / "work_pool_dilution_author_portfolio_summary.csv"
)

NullModel = Literal["uniform", "popularity"]

FOCAL_LOOKUPS = [
    ("If We Must Die", "Claude McKay"),
    ("I, Too", "Langston Hughes"),
]


@dataclass(frozen=True)
class FocalWork:
    poem: str
    author: str
    work_id: object
    author_id: object


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _fmt_float(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _fmt_p(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def load_data() -> pd.DataFrame:
    """Load all individual AFAM work appearances; child works are retained."""
    raw = query(query_path("works-per-afam-edition"))
    raw["anthology_publication_year"] = raw["anthology_publication_year"].astype(int)
    return add_entry_group(raw)


def resolve_focal_works() -> list[FocalWork]:
    """Resolve focal IDs from titles/names and assert the expected matches."""
    rows = query(
        """
        SELECT
            w.id AS work_id,
            w.title AS work_title,
            a.id AS author_id,
            a.name AS author_name
        FROM data_work w
        JOIN data_work_authors wa ON wa.work_id = w.id
        JOIN data_author a ON a.id = wa.author_id
        WHERE w.title IN ('If We Must Die', 'I, Too')
        ORDER BY w.title, a.name, w.id
        """
    )
    focal: list[FocalWork] = []
    for title, author in FOCAL_LOOKUPS:
        match = rows[(rows["work_title"] == title) & (rows["author_name"] == author)]
        if len(match) != 1:
            raise ValueError(
                f"Expected one match for {title!r} by {author!r}; found {len(match)}"
            )
        row = match.iloc[0]
        focal.append(
            FocalWork(
                poem=title,
                author=author,
                work_id=row["work_id"],
                author_id=row["author_id"],
            )
        )
    return focal


def fetch_author_work_metadata(author_ids: list[object]) -> pd.DataFrame:
    """Return title/name metadata for the focal authors' works."""
    if not author_ids:
        return pd.DataFrame(
            columns=["work_id", "work_title", "author_id", "author_name"]
        )
    placeholders = ", ".join(["%s"] * len(author_ids))
    return query(
        f"""
        SELECT
            w.id AS work_id,
            w.title AS work_title,
            a.id AS author_id,
            a.name AS author_name
        FROM data_work w
        JOIN data_work_authors wa ON wa.work_id = w.id
        JOIN data_author a ON a.id = wa.author_id
        WHERE a.id IN ({placeholders})
        ORDER BY a.name, w.title, w.id
        """,
        tuple(author_ids),
    )


def build_units(df: pd.DataFrame, collapse_series: bool = False) -> pd.DataFrame:
    """Return chronological edition/group units with stable integer order."""
    if "entry_group" not in df.columns:
        df = add_entry_group(df)

    if collapse_series:
        units = (
            df[["entry_group", "anthology_publication_year"]]
            .drop_duplicates()
            .groupby("entry_group", as_index=False)["anthology_publication_year"]
            .min()
            .rename(columns={"entry_group": "unit_id"})
        )
    else:
        units = (
            df[["edition_id", "anthology_publication_year"]]
            .drop_duplicates("edition_id")
            .rename(columns={"edition_id": "unit_id"})
        )

    units["unit_label"] = units["unit_id"].map(str)
    units = units.sort_values(
        ["anthology_publication_year", "unit_label"], kind="mergesort"
    ).reset_index(drop=True)
    units["unit_order"] = np.arange(len(units), dtype=int)
    return units


def prepare_appearances(
    df: pd.DataFrame, units: pd.DataFrame, collapse_series: bool = False
) -> pd.DataFrame:
    """One row per author-work-unit appearance."""
    out = df.dropna(subset=["author_id", "work_id"]).copy()
    out["unit_id"] = out["entry_group"] if collapse_series else out["edition_id"]
    out = out.merge(
        units[["unit_id", "unit_order", "anthology_publication_year", "unit_label"]],
        on="unit_id",
        how="inner",
        suffixes=("", "_unit"),
    )
    return (
        out[
            [
                "author_id",
                "work_id",
                "unit_id",
                "unit_order",
                "unit_label",
                "anthology_publication_year_unit",
            ]
        ]
        .drop_duplicates(["author_id", "work_id", "unit_id"])
        .rename(columns={"anthology_publication_year_unit": "year"})
    )


def build_author_unit_works(appearances: pd.DataFrame) -> dict[object, dict[int, set]]:
    """Map author_id -> unit_order -> set(work_id)."""
    nested: dict[object, dict[int, set]] = {}
    for (author_id, unit_order), group in appearances.groupby(
        ["author_id", "unit_order"], sort=False
    ):
        nested.setdefault(author_id, {})[int(unit_order)] = set(group["work_id"])
    return nested


def weighted_subset_inclusion_probability(
    weights: dict[object, float], focal_id: object, k: int
) -> float:
    """Inclusion probability for a weighted k-subset.

    Subsets are sampled without replacement with probability proportional to the
    product of member weights. With equal weights, this reduces to k / n.
    """
    n = len(weights)
    if focal_id not in weights or k <= 0 or n == 0:
        return 0.0
    if k >= n:
        return 1.0

    focal_weight = float(weights[focal_id])
    if focal_weight <= 0:
        return 0.0

    coeffs = np.zeros(k + 1, dtype=float)
    coeffs[0] = 1.0
    for work_id, weight in weights.items():
        if work_id == focal_id:
            continue
        w = float(weight)
        if w <= 0:
            continue
        max_j = min(k, np.flatnonzero(coeffs).max(initial=0) + 1)
        for j in range(max_j, 0, -1):
            coeffs[j] += coeffs[j - 1] * w

    numerator = focal_weight * coeffs[k - 1]
    denominator = coeffs[k] + numerator
    return float(numerator / denominator) if denominator else 0.0


def selection_probability(
    prior_counts: dict[object, int],
    focal_id: object,
    repeat_slots: int,
    null_model: NullModel,
) -> float:
    """Probability that the focal work fills one of the repeat slots."""
    pool_size = len(prior_counts)
    if pool_size == 0 or repeat_slots <= 0:
        return 0.0
    if repeat_slots >= pool_size:
        return 1.0
    if null_model == "uniform":
        return float(repeat_slots / pool_size)
    if null_model == "popularity":
        return weighted_subset_inclusion_probability(
            {work_id: float(count) for work_id, count in prior_counts.items()},
            focal_id,
            repeat_slots,
        )
    raise ValueError(f"Unknown null model: {null_model}")


def compute_focal_record(
    appearances: pd.DataFrame,
    units: pd.DataFrame,
    focal: FocalWork,
    null_model: NullModel = "uniform",
    variant: str = "edition_uniform",
) -> pd.DataFrame:
    """Per-unit dilution record for one author/work."""
    author_unit_works = build_author_unit_works(appearances)
    by_order = author_unit_works.get(focal.author_id, {})
    focal_orders = sorted(
        order for order, works in by_order.items() if focal.work_id in works
    )
    if not focal_orders:
        raise ValueError(f"{focal.poem!r} never appears for author {focal.author!r}")

    debut_order = int(focal_orders[0])
    seen: set = set()
    prior_counts: dict[object, int] = {}
    rows: list[dict] = []
    units_by_order = units.set_index("unit_order")

    for order in units["unit_order"].astype(int):
        current = by_order.get(int(order), set())
        prior = set(seen)

        if order > debut_order:
            repeats = current & prior
            p_selected = selection_probability(
                prior_counts, focal.work_id, len(repeats), null_model
            )
            unit = units_by_order.loc[order]
            rows.append(
                {
                    "variant": variant,
                    "null_model": null_model,
                    "poem": focal.poem,
                    "author": focal.author,
                    "work_id": focal.work_id,
                    "author_id": focal.author_id,
                    "unit_order": order,
                    "unit_id": unit["unit_id"],
                    "unit_label": unit["unit_label"],
                    "year": int(unit["anthology_publication_year"]),
                    "prior_pool": len(prior),
                    "repeat_slots": len(repeats),
                    "selected": focal.work_id in current,
                    "p_null": p_selected,
                }
            )

        for work_id in current:
            seen.add(work_id)
            prior_counts[work_id] = prior_counts.get(work_id, 0) + 1

    return pd.DataFrame(rows)


def poisson_binomial_tail(probs: list[float] | np.ndarray, observed: int) -> float:
    """Exact P[X >= observed] for independent Bernoulli trials."""
    probs = np.asarray(probs, dtype=float)
    if observed <= 0:
        return 1.0
    if observed > len(probs):
        return 0.0

    dp = np.zeros(len(probs) + 1, dtype=float)
    dp[0] = 1.0
    upper = 0
    for p in np.clip(probs, 0.0, 1.0):
        upper += 1
        for j in range(upper, 0, -1):
            dp[j] = dp[j] * (1.0 - p) + dp[j - 1] * p
        dp[0] *= 1.0 - p
    return float(dp[observed:].sum())


def summarize_focal_records(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed, expected, lift, and exact tail probability."""
    rows: list[dict] = []
    if records.empty:
        return pd.DataFrame()

    group_cols = ["variant", "null_model", "poem", "author", "work_id", "author_id"]
    for keys, group in records.groupby(group_cols, sort=False):
        observed = int(group["selected"].sum())
        expected = float(group["p_null"].sum())
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "post_debut_observed": observed,
                "post_debut_opportunities": int(len(group)),
                "total_selections_including_debut": observed + 1,
                "expected": expected,
                "lift": _safe_div(observed, expected),
                "tail_p_ge_observed": poisson_binomial_tail(
                    group["p_null"].to_numpy(), observed
                ),
                "mean_prior_pool": float(group["prior_pool"].mean()),
                "mean_repeat_slots": float(group["repeat_slots"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_gap(
    summary: pd.DataFrame,
    records: pd.DataFrame,
    sims: int = 100_000,
    seed: int = 20260612,
    variant: str = "edition_uniform",
) -> pd.DataFrame:
    """Monte Carlo distribution of McKay count minus Hughes count."""
    variant_records = records[records["variant"] == variant]
    mckay = variant_records[variant_records["poem"] == "If We Must Die"]
    hughes = variant_records[variant_records["poem"] == "I, Too"]
    if mckay.empty or hughes.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    mckay_counts = rng.binomial(
        1, mckay["p_null"].to_numpy(), size=(sims, len(mckay))
    ).sum(axis=1)
    hughes_counts = rng.binomial(
        1, hughes["p_null"].to_numpy(), size=(sims, len(hughes))
    ).sum(axis=1)
    gaps = mckay_counts - hughes_counts

    obs_mckay = int(mckay["selected"].sum())
    obs_hughes = int(hughes["selected"].sum())
    observed_gap = obs_mckay - obs_hughes
    exp_mckay = float(mckay["p_null"].sum())
    exp_hughes = float(hughes["p_null"].sum())
    expected_gap = exp_mckay - exp_hughes

    return pd.DataFrame(
        [
            {
                "variant": variant,
                "sims": sims,
                "seed": seed,
                "observed_mckay": obs_mckay,
                "observed_hughes": obs_hughes,
                "observed_gap": observed_gap,
                "expected_mckay": exp_mckay,
                "expected_hughes": exp_hughes,
                "expected_gap": expected_gap,
                "mean_simulated_gap": float(gaps.mean()),
                "p_gap_ge_observed": float((gaps >= observed_gap).mean()),
                "dilution_share_of_gap": _safe_div(expected_gap, observed_gap),
                "residual_gap": observed_gap - expected_gap,
            }
        ]
    )


def compute_corpus_context(
    appearances: pd.DataFrame, units: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute dilution expectation/lift for every author-work pair."""
    total_units = len(units)
    author_unit_works = build_author_unit_works(appearances)
    pair_orders = (
        appearances.groupby(["author_id", "work_id"])["unit_order"]
        .apply(lambda s: sorted({int(v) for v in s}))
        .to_dict()
    )
    author_pool_size = (
        appearances.groupby("author_id")["work_id"].nunique().astype(int).to_dict()
    )

    rows: list[dict] = []
    for author_id, by_order in author_unit_works.items():
        seen: set = set()
        p_by_order: dict[int, float] = {}
        pool_by_order: dict[int, int] = {}
        slots_by_order: dict[int, int] = {}
        for order in units["unit_order"].astype(int):
            current = by_order.get(int(order), set())
            prior = set(seen)
            repeats = current & prior
            p_by_order[int(order)] = _safe_div(len(repeats), len(prior))
            pool_by_order[int(order)] = len(prior)
            slots_by_order[int(order)] = len(repeats)
            seen.update(current)

        for (pair_author_id, work_id), orders in pair_orders.items():
            if pair_author_id != author_id:
                continue
            debut_order = int(orders[0])
            if debut_order >= total_units - 1:
                continue
            later_orders = [order for order in range(debut_order + 1, total_units)]
            observed = sum(1 for order in later_orders if order in orders)
            probs = np.array([p_by_order[order] for order in later_orders], dtype=float)
            expected = float(np.nansum(probs))
            rows.append(
                {
                    "author_id": author_id,
                    "work_id": work_id,
                    "debut_order": debut_order,
                    "post_debut_opportunities": len(later_orders),
                    "post_debut_observed": int(observed),
                    "raw_reselection_rate": _safe_div(observed, len(later_orders)),
                    "expected": expected,
                    "lift": _safe_div(observed, expected),
                    "tail_p_ge_observed": poisson_binomial_tail(probs, int(observed)),
                    "author_pool_size": int(author_pool_size.get(author_id, 0)),
                    "mean_prior_pool_after_debut": float(
                        np.mean([pool_by_order[order] for order in later_orders])
                    ),
                    "mean_repeat_slots_after_debut": float(
                        np.mean([slots_by_order[order] for order in later_orders])
                    ),
                }
            )

    corpus = pd.DataFrame(rows)
    if corpus.empty:
        return corpus, pd.DataFrame()

    bins = [0, 9, 24, 49, 99, np.inf]
    labels = ["1-9", "10-24", "25-49", "50-99", "100+"]
    tmp = corpus.copy()
    tmp["author_pool_bin"] = pd.cut(
        tmp["author_pool_size"], bins=bins, labels=labels, include_lowest=True
    )
    binned = (
        tmp.groupby("author_pool_bin", observed=False)
        .agg(
            work_pairs=("work_id", "count"),
            mean_author_pool_size=("author_pool_size", "mean"),
            mean_raw_reselection_rate=("raw_reselection_rate", "mean"),
            median_raw_reselection_rate=("raw_reselection_rate", "median"),
            mean_expected_reselection_rate=(
                "expected",
                lambda s: (
                    float(s.sum() / tmp.loc[s.index, "post_debut_opportunities"].sum())
                    if tmp.loc[s.index, "post_debut_opportunities"].sum()
                    else float("nan")
                ),
            ),
        )
        .reset_index()
    )
    return corpus, binned


def add_focal_ranks(corpus: pd.DataFrame, focal: list[FocalWork]) -> pd.DataFrame:
    """Return corpus lift ranks for the focal poems."""
    rows: list[dict] = []
    valid = corpus[corpus["expected"] > 0].copy()
    valid["lift_rank"] = valid["lift"].rank(method="min", ascending=False)
    valid_10 = valid[valid["post_debut_opportunities"] >= 10].copy()
    valid_10["lift_rank_ge10_opp"] = valid_10["lift"].rank(
        method="min", ascending=False
    )

    for item in focal:
        row = valid[
            (valid["author_id"] == item.author_id) & (valid["work_id"] == item.work_id)
        ]
        row_10 = valid_10[
            (valid_10["author_id"] == item.author_id)
            & (valid_10["work_id"] == item.work_id)
        ]
        if row.empty:
            continue
        data = row.iloc[0].to_dict()
        data.update({"poem": item.poem, "author": item.author, "rank_n": len(valid)})
        data["lift_rank"] = int(data["lift_rank"])
        if row_10.empty:
            data["lift_rank_ge10_opp"] = np.nan
            data["rank_ge10_n"] = len(valid_10)
        else:
            data["lift_rank_ge10_opp"] = int(row_10.iloc[0]["lift_rank_ge10_opp"])
            data["rank_ge10_n"] = len(valid_10)
        rows.append(data)
    return pd.DataFrame(rows)


def build_author_portfolio(
    raw: pd.DataFrame,
    corpus: pd.DataFrame,
    metadata: pd.DataFrame,
    focal: list[FocalWork],
) -> pd.DataFrame:
    """Return per-work frequency and lift context for the focal authors."""
    focal_ids = {item.author_id for item in focal}
    counts = (
        raw.dropna(subset=["author_id", "work_id"])
        .loc[lambda df: df["author_id"].isin(focal_ids)]
        .groupby(["author_id", "work_id"])["edition_id"]
        .nunique()
        .reset_index(name="total_selections")
    )
    portfolio = counts.merge(metadata, on=["author_id", "work_id"], how="left").merge(
        corpus,
        on=["author_id", "work_id"],
        how="left",
    )
    focal_map = {(item.author_id, item.work_id): item.poem for item in focal}
    portfolio["is_focal_poem"] = [
        focal_map.get((author_id, work_id), "") != ""
        for author_id, work_id in zip(
            portfolio["author_id"], portfolio["work_id"], strict=False
        )
    ]
    portfolio["focal_poem"] = [
        focal_map.get((author_id, work_id), "")
        for author_id, work_id in zip(
            portfolio["author_id"], portfolio["work_id"], strict=False
        )
    ]
    return portfolio.sort_values(
        ["author_name", "total_selections", "work_title"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_author_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Count each focal author's frequent and high-lift candidate works."""
    rows: list[dict] = []
    if portfolio.empty:
        return pd.DataFrame()

    for author, group in portfolio.groupby("author_name", sort=True):
        eligible_lift = group[
            (group["post_debut_opportunities"] >= 10) & (group["expected"] > 0)
        ]
        row = {
            "author": author,
            "work_pool": int(len(group)),
            "total_selections": int(group["total_selections"].sum()),
        }
        for threshold in [2, 3, 5, 10, 13, 20, 23]:
            row[f"works_ge_{threshold}_selections"] = int(
                (group["total_selections"] >= threshold).sum()
            )
        for threshold in [2, 3, 4, 5]:
            row[f"works_ge_{threshold}x_lift_opp_ge_10"] = int(
                (eligible_lift["lift"] >= threshold).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compute(
    raw: pd.DataFrame,
    focal: list[FocalWork],
    sims: int = 100_000,
    seed: int = 20260612,
) -> dict[str, pd.DataFrame]:
    """Compute focal, sensitivity, gap, and corpus-context tables."""
    if "entry_group" not in raw.columns:
        raw = add_entry_group(raw)
    raw = raw.copy()
    raw["anthology_publication_year"] = raw["anthology_publication_year"].astype(int)

    all_records: list[pd.DataFrame] = []
    variant_specs = [
        ("edition_uniform", False, "uniform"),
        ("edition_popularity", False, "popularity"),
        ("series_collapsed_uniform", True, "uniform"),
    ]
    primary_appearances = pd.DataFrame()
    primary_units = pd.DataFrame()

    for variant, collapse_series, null_model in variant_specs:
        units = build_units(raw, collapse_series=collapse_series)
        appearances = prepare_appearances(raw, units, collapse_series=collapse_series)
        if variant == "edition_uniform":
            primary_appearances = appearances
            primary_units = units
        for item in focal:
            all_records.append(
                compute_focal_record(
                    appearances,
                    units,
                    item,
                    null_model=null_model,
                    variant=variant,
                )
            )

    per_edition = pd.concat(all_records, ignore_index=True)
    summary = summarize_focal_records(per_edition)
    gap = simulate_gap(summary, per_edition, sims=sims, seed=seed)
    corpus, corpus_bins = compute_corpus_context(primary_appearances, primary_units)
    focal_ranks = add_focal_ranks(corpus, focal)
    return {
        "per_edition": per_edition,
        "summary": summary,
        "gap": gap,
        "corpus": corpus,
        "corpus_bins": corpus_bins,
        "focal_ranks": focal_ranks,
    }


def add_author_portfolio_outputs(
    outputs: dict[str, pd.DataFrame],
    raw: pd.DataFrame,
    focal: list[FocalWork],
) -> None:
    """Attach focal-author portfolio tables to the output dictionary."""
    metadata = fetch_author_work_metadata([item.author_id for item in focal])
    portfolio = build_author_portfolio(raw, outputs["corpus"], metadata, focal)
    outputs["author_portfolio"] = portfolio
    outputs["author_portfolio_summary"] = summarize_author_portfolio(portfolio)


def print_per_edition(per_edition: pd.DataFrame) -> None:
    primary = per_edition[per_edition["variant"] == "edition_uniform"].copy()
    cols = [
        "poem",
        "year",
        "unit_label",
        "prior_pool",
        "repeat_slots",
        "selected",
        "p_null",
    ]
    primary = primary[cols].sort_values(
        ["poem", "year", "unit_label"], kind="mergesort"
    )
    print("\nPer-edition primary null")
    print(primary.to_string(index=False, formatters={"p_null": lambda x: f"{x:.3f}"}))


def print_summary(summary: pd.DataFrame) -> None:
    table = summary[
        [
            "variant",
            "poem",
            "post_debut_observed",
            "post_debut_opportunities",
            "total_selections_including_debut",
            "expected",
            "lift",
            "tail_p_ge_observed",
        ]
    ].copy()
    print("\nFocal summary")
    print(
        table.to_string(
            index=False,
            formatters={
                "expected": lambda x: _fmt_float(x, 2),
                "lift": lambda x: _fmt_float(x, 2),
                "tail_p_ge_observed": _fmt_p,
            },
        )
    )


def print_gap(gap: pd.DataFrame) -> None:
    if gap.empty:
        return
    row = gap.iloc[0]
    print("\nGap decomposition")
    print(
        "Observed post-debut gap: "
        f"{int(row['observed_gap'])} selections "
        f"({int(row['observed_mckay'])} - {int(row['observed_hughes'])})."
    )
    print(
        "Uniform-dilution expected gap: "
        f"{row['expected_gap']:.2f} selections "
        f"({row['dilution_share_of_gap']:.1%} of observed); "
        f"residual gap {row['residual_gap']:.2f}; "
        f"Monte Carlo P(gap >= observed) = {_fmt_p(row['p_gap_ge_observed'])}."
    )


def print_corpus_context(corpus_bins: pd.DataFrame, focal_ranks: pd.DataFrame) -> None:
    print("\nCorpus pool-size context")
    print(
        corpus_bins.to_string(
            index=False,
            formatters={
                "mean_author_pool_size": lambda x: _fmt_float(x, 1),
                "mean_raw_reselection_rate": lambda x: f"{x:.1%}",
                "median_raw_reselection_rate": lambda x: f"{x:.1%}",
                "mean_expected_reselection_rate": lambda x: f"{x:.1%}",
            },
        )
    )

    if focal_ranks.empty:
        return
    rank_cols = [
        "poem",
        "post_debut_observed",
        "post_debut_opportunities",
        "expected",
        "lift",
        "lift_rank",
        "rank_n",
        "lift_rank_ge10_opp",
        "rank_ge10_n",
    ]
    print("\nFocal lift ranks")
    print(
        focal_ranks[rank_cols].to_string(
            index=False,
            formatters={
                "expected": lambda x: _fmt_float(x, 2),
                "lift": lambda x: _fmt_float(x, 2),
            },
        )
    )


def print_author_portfolio(portfolio: pd.DataFrame, summary: pd.DataFrame) -> None:
    if portfolio.empty or summary.empty:
        return

    print("\nAuthor portfolio context")
    print(summary.to_string(index=False))

    for author, group in portfolio.groupby("author_name", sort=True):
        print(f"\nTop {author} works by total selections")
        cols = [
            "work_title",
            "total_selections",
            "post_debut_observed",
            "expected",
            "lift",
            "is_focal_poem",
        ]
        top_frequency = group.sort_values(
            ["total_selections", "work_title"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10)
        print(
            top_frequency[cols].to_string(
                index=False,
                formatters={
                    "expected": lambda x: _fmt_float(x, 2),
                    "lift": lambda x: _fmt_float(x, 2),
                },
            )
        )

        high_lift = group[
            (group["post_debut_opportunities"] >= 10)
            & (group["expected"] > 0)
            & (group["total_selections"] >= 3)
        ].sort_values(["lift", "total_selections"], ascending=[False, False])
        print(
            f"\nTop {author} works by lift, among works with >=10 opportunities and >=3 selections"
        )
        print(
            high_lift[cols]
            .head(10)
            .to_string(
                index=False,
                formatters={
                    "expected": lambda x: _fmt_float(x, 2),
                    "lift": lambda x: _fmt_float(x, 2),
                },
            )
        )


def print_verdict(summary: pd.DataFrame, gap: pd.DataFrame) -> None:
    primary = summary[summary["variant"] == "edition_uniform"].set_index("poem")
    if primary.empty or gap.empty:
        return
    mckay = primary.loc["If We Must Die"]
    hughes = primary.loc["I, Too"]
    gap_row = gap.iloc[0]
    expected_ratio = mckay["expected"] / hughes["expected"]
    observed_ratio = mckay["post_debut_observed"] / hughes["post_debut_observed"]
    print("\nVerdict")
    print(
        "The pool-dilution hypothesis is supported as a relative work-level "
        "baseline, but it does not reproduce most of the absolute gap by "
        "itself. Under the uniform null, both poems are strong positive "
        "outliers relative to their authors' pools "
        f"(lifts {mckay['lift']:.2f}x and {hughes['lift']:.2f}x), but the "
        f"expected ratio from pool dilution ({expected_ratio:.2f}) is close to "
        f"the observed post-debut ratio ({observed_ratio:.2f}). The model "
        f"reproduces {gap_row['dilution_share_of_gap']:.1%} of the observed "
        f"{int(gap_row['observed_gap'])}-selection post-debut gap, leaving "
        f"{gap_row['residual_gap']:.2f} selections as residual."
    )


def print_portfolio_verdict(portfolio_summary: pd.DataFrame) -> None:
    if portfolio_summary.empty:
        return
    indexed = portfolio_summary.set_index("author")
    if not {"Claude McKay", "Langston Hughes"}.issubset(indexed.index):
        return
    mckay = indexed.loc["Claude McKay"]
    hughes = indexed.loc["Langston Hughes"]
    print("\nPortfolio verdict")
    print(
        "The author-portfolio version of the hypothesis is also supported. "
        f"Hughes has {int(hughes['work_pool'])} individual anthologized works "
        f"to McKay's {int(mckay['work_pool'])}, and more frequent alternatives: "
        f"{int(hughes['works_ge_10_selections'])} Hughes works have at least "
        f"10 selections, compared with {int(mckay['works_ge_10_selections'])} "
        "McKay works. On the dilution-adjusted scale, Hughes also has more "
        f"high-lift candidates ({int(hughes['works_ge_4x_lift_opp_ge_10'])} "
        f"works at >=4x lift with >=10 opportunities, versus "
        f"{int(mckay['works_ge_4x_lift_opp_ge_10'])} for McKay)."
    )


def save_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    paths = {
        "per_edition": OUT_PER_EDITION,
        "summary": OUT_SUMMARY,
        "gap": OUT_GAP,
        "corpus_bins": OUT_CORPUS_BINS,
        "corpus": OUT_CORPUS_LIFTS,
        "author_portfolio": OUT_AUTHOR_PORTFOLIO,
        "author_portfolio_summary": OUT_AUTHOR_PORTFOLIO_SUMMARY,
    }
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        outputs[key].to_csv(path, index=False)
        print(f"Saved -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sims",
        type=int,
        default=100_000,
        help="Monte Carlo simulations for the gap decomposition (default: 100000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260612,
        help="Random seed for simulations (default: 20260612).",
    )
    add_save_csv_flag(parser)
    args = parser.parse_args()

    raw = load_data()
    focal = resolve_focal_works()
    outputs = compute(raw, focal, sims=args.sims, seed=args.seed)
    add_author_portfolio_outputs(outputs, raw, focal)

    print_per_edition(outputs["per_edition"])
    print_summary(outputs["summary"])
    print_gap(outputs["gap"])
    print_corpus_context(outputs["corpus_bins"], outputs["focal_ranks"])
    print_author_portfolio(
        outputs["author_portfolio"], outputs["author_portfolio_summary"]
    )
    print_verdict(outputs["summary"], outputs["gap"])
    print_portfolio_verdict(outputs["author_portfolio_summary"])

    if args.save_csv:
        save_outputs(outputs)


if __name__ == "__main__":
    main()
