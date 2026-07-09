"""
logistic_reselection.py
-----------------------
Logistic regression predicting author or work reselection in AFAM anthologies.

For a given target edition year, models P(in_target | prior_count) using prior
appearances in all earlier AFAM editions. Editions are counted by edition_id
(volumes collapsed into their parent edition). Root works only by default.

Usage:
    uv run python analysis/logistic_reselection.py
    uv run python analysis/logistic_reselection.py --year 1996 --mode authors
    uv run python analysis/logistic_reselection.py --year 2025 --mode both
    uv run python analysis/logistic_reselection.py --year 2025 --mode works --include-excerpts
    # Decontaminate: drop prior NAAAL editions (series_id=3) from the features
    uv run python analysis/logistic_reselection.py --year 2025 --mode both --exclude-prior-series 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError

from afam.db import query
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

OUT_DIR = OUTPUT_DIR
LINE_COLOR = "#1f77b4"
POINT_COLOR = "#d62728"


def load_data(include_excerpts: bool = False) -> pd.DataFrame:
    df = query(query_path("works-per-afam-edition"))
    if not include_excerpts:
        df = df[df["parent_id"].isna()]
    return df


# ── Target / prior split ──────────────────────────────────────────────────────


def resolve_target_year(df: pd.DataFrame, year_arg: int | None) -> int:
    if year_arg is None:
        return int(df["anthology_publication_year"].max())
    return year_arg


# ── Feature frames ────────────────────────────────────────────────────────────


def build_author_frame(
    df: pd.DataFrame,
    target_year: int,
    min_prior: int = 1,
    exclude_prior_series: set[int] | None = None,
) -> pd.DataFrame:
    prior = df[df["anthology_publication_year"] < target_year]
    if exclude_prior_series:
        prior = prior[~prior["series_id"].isin(exclude_prior_series)]
    target = df[df["anthology_publication_year"] == target_year]
    prior_counts = prior.groupby("author_id")["edition_id"].nunique()
    target_authors = set(target["author_id"].dropna())
    all_authors = pd.DataFrame({"author_id": df["author_id"].dropna().unique()})
    all_authors["prior_count"] = (
        all_authors["author_id"].map(prior_counts).fillna(0).astype(int)
    )
    all_authors["in_target"] = all_authors["author_id"].isin(target_authors).astype(int)
    return all_authors[all_authors["prior_count"] >= min_prior].reset_index(drop=True)


def build_work_frame(
    df: pd.DataFrame,
    target_year: int,
    min_prior: int = 1,
    exclude_prior_series: set[int] | None = None,
) -> pd.DataFrame:
    prior = df[df["anthology_publication_year"] < target_year]
    if exclude_prior_series:
        prior = prior[~prior["series_id"].isin(exclude_prior_series)]
    target = df[df["anthology_publication_year"] == target_year]
    prior_counts = prior.groupby("work_id")["edition_id"].nunique()
    target_works = set(target["work_id"].dropna())
    all_works = pd.DataFrame({"work_id": df["work_id"].dropna().unique()})
    all_works["prior_count"] = (
        all_works["work_id"].map(prior_counts).fillna(0).astype(int)
    )
    all_works["in_target"] = all_works["work_id"].isin(target_works).astype(int)
    return all_works[all_works["prior_count"] >= min_prior].reset_index(drop=True)


# ── Model helpers ─────────────────────────────────────────────────────────────


def fit_logit_safe(y: pd.Series, x: pd.Series) -> Any:
    X = sm.add_constant(x.rename("prior_count"), has_constant="add")
    model = sm.Logit(y, X)
    try:
        return model.fit(disp=False)
    except (LinAlgError, ValueError):
        return model.fit_regularized(alpha=1e-6, L1_wt=0.0)


def empirical_rates(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("prior_count")["in_target"]
        .agg(probability="mean", n="size")
        .reset_index()
        .sort_values("prior_count")
    )


def cumulative_buckets(frame: pd.DataFrame, kmax: int | None = None) -> pd.DataFrame:
    """For each threshold k, the selection rate among entities with prior_count
    >= k. Reproduces the cumulative ">= k priors" method (formerly in the
    CSV-backed freq_bucket_predictability.py): pools all higher-prior entities
    rather than reading a single per-point rate. Pass the min_prior=0 frame so
    every entity is represented.
    """
    if kmax is None:
        kmax = int(frame["prior_count"].max())
    rows = []
    for k in range(1, kmax + 1):
        sub = frame[frame["prior_count"] >= k]
        n = len(sub)
        selected = int(sub["in_target"].sum())
        rows.append(
            {
                "threshold": k,
                "n": n,
                "selected": selected,
                "pct": (100.0 * selected / n) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def predicted_probabilities(result: Any, max_count: int) -> pd.DataFrame:
    counts = np.arange(0, max_count + 1, dtype=int)
    X = sm.add_constant(pd.Series(counts, name="prior_count"), has_constant="add")
    probs = np.asarray(result.predict(X), dtype=float)
    return pd.DataFrame({"prior_count": counts, "probability": probs})


# ── Exclusion labelling ───────────────────────────────────────────────────────


def series_suffix(ids: set[int]) -> str:
    """Short filename token for an excluded-series set ({3} → 'naaal')."""
    if ids == {3}:
        return "naaal"
    return "series-" + "-".join(str(i) for i in sorted(ids))


# ── Figures ───────────────────────────────────────────────────────────────────


def plot_probability_curve(
    frame: pd.DataFrame,
    result: Any,
    out_path: Path,
    singular: str,
    target_year: int,
    exclude_note: str = "",
) -> None:
    bucket_df = empirical_rates(frame)
    curve_df = predicted_probabilities(result, int(frame["prior_count"].max()))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_top.plot(
        curve_df["prior_count"],
        curve_df["probability"],
        color=LINE_COLOR,
        linewidth=2.0,
        label="Logistic model",
        zorder=2,
    )
    ax_top.scatter(
        bucket_df["prior_count"],
        bucket_df["probability"],
        s=36,
        color=POINT_COLOR,
        alpha=0.75,
        label="Observed rate",
        zorder=3,
    )

    ax_top.set_ylabel(f"Probability of selection ({target_year})")
    subtitle = f"(target year: {target_year}{exclude_note})"
    ax_top.set_title(
        f"{singular.capitalize()} selection probability for AFAM anthologies\n"
        f"{subtitle}"
    )
    ax_top.set_xlim(0, int(frame["prior_count"].max()) + 0.5)
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_top.grid(True, alpha=0.25, linestyle=":")
    ax_top.legend(frameon=False)

    n_label = f"N {singular}s"
    ax_bot.bar(
        bucket_df["prior_count"],
        bucket_df["n"],
        color=POINT_COLOR,
        alpha=0.6,
        width=0.5,
    )
    ax_bot.set_xlabel(f"Number of prior anthologies selecting a {singular}")
    ax_bot.set_ylabel(n_label)
    ax_bot.grid(True, alpha=0.25, linestyle=":", axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Per-mode analysis ─────────────────────────────────────────────────────────


def run_mode(
    df: pd.DataFrame,
    mode: str,
    target_year: int,
    out_dir: Path,
    exclude_prior_series: set[int] | None = None,
) -> None:
    singular = "author" if mode == "authors" else "work"
    build = build_author_frame if mode == "authors" else build_work_frame
    frame = build(df, target_year, exclude_prior_series=exclude_prior_series)
    full_frame = build(
        df, target_year, min_prior=0, exclude_prior_series=exclude_prior_series
    )

    if frame.empty:
        print(
            f"\n[{mode}] No {mode} with prior_count >= 1 for target year {target_year}."
        )
        return

    result = fit_logit_safe(frame["in_target"], frame["prior_count"])
    curve_df = predicted_probabilities(result, int(frame["prior_count"].max()))

    if exclude_prior_series:
        suffix = f"_excl-{series_suffix(exclude_prior_series)}"
        ids = sorted(exclude_prior_series)
        exclude_note = f", excl. prior series {ids}"
        header_note = f", excluding prior series {ids}"
    else:
        suffix = ""
        exclude_note = ""
        header_note = ""

    fig_path = out_dir / f"logit_{mode}_{target_year}{suffix}.png"
    plot_probability_curve(
        frame, result, fig_path, singular, target_year, exclude_note=exclude_note
    )

    zero_prior_in_target = int(
        full_frame.query("prior_count == 0 and in_target == 1").shape[0]
    )

    print(f"\n===== {mode.upper()} (target year: {target_year}{header_note}) =====")
    print(f"In fitted sample (prior_count >= 1)  : {len(frame)}")
    print(f"Selected in target year              : {int(frame['in_target'].sum())}")
    print(f"Zero-prior target-only {mode:<13}: {zero_prior_in_target}")
    print(f"Logit intercept                      : {float(result.params['const']):.3f}")
    print(
        f"Logit slope (prior_count)            : {float(result.params['prior_count']):.3f}"
    )
    print(
        f"Predicted probability range          : "
        f"{curve_df['probability'].min():.3f} to {curve_df['probability'].max():.3f}"
    )
    print(f"Saved figure                         : {fig_path}")

    buckets = cumulative_buckets(full_frame)
    print(f"Cumulative '>= k prior anthologies' selection rate ({mode}):")
    for _, r in buckets.iterrows():
        print(
            f"  >= {int(r['threshold']):>2} priors : "
            f"{int(r['selected']):>4}/{int(r['n']):>4}  = {r['pct']:5.1f}%"
        )
    crossed = buckets[buckets["pct"] > 50]
    if not crossed.empty:
        k = int(crossed.iloc[0]["threshold"])
        print(f"  -> first threshold with >50% selected: >= {k} prior anthologies")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["authors", "works", "both"],
        default="both",
        help="Analyze authors, works, or both (default: both).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        metavar="YEAR",
        help="Target edition publication year (default: latest in DB).",
    )
    parser.add_argument(
        "--include-excerpts",
        action="store_true",
        help="Include works with a parent work (excerpts/selections). Default: root works only.",
    )
    parser.add_argument(
        "--exclude-prior-series",
        type=int,
        nargs="+",
        default=None,
        metavar="SERIES_ID",
        help=(
            "Drop these series_ids from the prior-count features (e.g. 3 = NAAAL) to "
            "avoid within-series contamination. Default: include all series."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        metavar="DIR",
        help=f"Output directory for PNG figures (default: {OUT_DIR}).",
    )
    args = parser.parse_args()

    df = load_data(include_excerpts=args.include_excerpts)
    target_year = resolve_target_year(df, args.year)
    exclude = set(args.exclude_prior_series) if args.exclude_prior_series else None

    modes = ["authors", "works"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_mode(df, mode, target_year, args.out_dir, exclude_prior_series=exclude)


if __name__ == "__main__":
    main()
