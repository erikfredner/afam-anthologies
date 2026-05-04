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
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
import statsmodels.api as sm
from dotenv import dotenv_values
from numpy.linalg import LinAlgError


ENV_FILE = Path(__file__).parent.parent / ".env"
QUERIES = Path(__file__).parent.parent / "queries"
OUT_DIR = Path(__file__).parent.parent / "viz"
LINE_COLOR = "#1f77b4"
POINT_COLOR = "#d62728"


# ── DB helpers ────────────────────────────────────────────────────────────────


def parse_db_params(env_file: Path) -> dict[str, str]:
    env = dotenv_values(env_file)
    raw = env["DATABASE_URL"]
    return {
        "host": re.search(r"-h\s+(\S+)", raw).group(1),
        "user": re.search(r"-U\s+(\S+)", raw).group(1),
        "password": re.search(r"PGPASSWORD=(\S+)", raw).group(1),
        "dbname": raw.split()[-1],
    }


def query_db(params: dict[str, str], sql_file: Path) -> pd.DataFrame:
    sql = sql_file.read_text()
    with psycopg.connect(**params) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc.name for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def load_data(params: dict[str, str], include_excerpts: bool = False) -> pd.DataFrame:
    df = query_db(params, QUERIES / "works-per-afam-edition.sql")
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
    df: pd.DataFrame, target_year: int, min_prior: int = 1
) -> pd.DataFrame:
    prior = df[df["anthology_publication_year"] < target_year]
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
    df: pd.DataFrame, target_year: int, min_prior: int = 1
) -> pd.DataFrame:
    prior = df[df["anthology_publication_year"] < target_year]
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


def empirical_rates_cumulative(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in sorted(frame["prior_count"].unique()):
        subset = frame[frame["prior_count"] >= k]
        rows.append(
            {
                "prior_count": k,
                "probability": subset["in_target"].mean(),
                "n": len(subset),
            }
        )
    return pd.DataFrame(rows)


def predicted_probabilities(result: Any, max_count: int) -> pd.DataFrame:
    counts = np.arange(0, max_count + 1, dtype=int)
    X = sm.add_constant(pd.Series(counts, name="prior_count"), has_constant="add")
    probs = np.asarray(result.predict(X), dtype=float)
    return pd.DataFrame({"prior_count": counts, "probability": probs})


# ── Figures ───────────────────────────────────────────────────────────────────


def plot_probability_curve(
    frame: pd.DataFrame,
    result: Any,
    out_path: Path,
    singular: str,
    target_year: int,
    cumulative: bool = False,
) -> None:
    bucket_df = (
        empirical_rates_cumulative(frame) if cumulative else empirical_rates(frame)
    )
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
    ax_top.set_title(
        f"{singular.capitalize()} selection probability for AFAM anthologies\n"
        f"(target year: {target_year})"
    )
    ax_top.set_xlim(0, int(frame["prior_count"].max()) + 0.5)
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_top.grid(True, alpha=0.25, linestyle=":")
    ax_top.legend(frameon=False)

    n_label = f"N {singular}s (≥ k)" if cumulative else f"N {singular}s"
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


def run_mode(df: pd.DataFrame, mode: str, target_year: int, out_dir: Path) -> None:
    singular = "author" if mode == "authors" else "work"
    if mode == "authors":
        frame = build_author_frame(df, target_year)
        full_frame = build_author_frame(df, target_year, min_prior=0)
    else:
        frame = build_work_frame(df, target_year)
        full_frame = build_work_frame(df, target_year, min_prior=0)

    if frame.empty:
        print(
            f"\n[{mode}] No {mode} with prior_count >= 1 for target year {target_year}."
        )
        return

    result = fit_logit_safe(frame["in_target"], frame["prior_count"])
    curve_df = predicted_probabilities(result, int(frame["prior_count"].max()))

    fig_path = out_dir / f"logit_{mode}_{target_year}.png"
    fig_cumulative_path = out_dir / f"logit_{mode}_{target_year}_cumulative.png"
    plot_probability_curve(frame, result, fig_path, singular, target_year)
    plot_probability_curve(
        frame, result, fig_cumulative_path, singular, target_year, cumulative=True
    )

    zero_prior_in_target = int(
        full_frame.query("prior_count == 0 and in_target == 1").shape[0]
    )

    print(f"\n===== {mode.upper()} (target year: {target_year}) =====")
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
    print(f"Saved figure (cumulative)            : {fig_cumulative_path}")


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
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        metavar="DIR",
        help=f"Output directory for PNG figures (default: {OUT_DIR}).",
    )
    args = parser.parse_args()

    params = parse_db_params(ENV_FILE)
    df = load_data(params, include_excerpts=args.include_excerpts)
    target_year = resolve_target_year(df, args.year)

    modes = ["authors", "works"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_mode(df, mode, target_year, args.out_dir)


if __name__ == "__main__":
    main()
