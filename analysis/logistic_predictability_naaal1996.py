"""
Predicting inclusion in *The Norton Anthology of African American Literature*
(first edition 1996) from an author's earlier anthology circulation.

Console output
--------------
* sample overview
* Model 1 – binary flag (≥ 1 prior appearance)
* Model 2 – continuous prior probability

CSV output
----------
* data/naaal1996_threshold_tests.csv :
  thresholds k = 1 … K* (largest k with at least ONE ‘No’ among ≥ k authors)
  and for each: contingency counts, odds ratio, Fisher p, and logit β.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from scipy.stats import fisher_exact

# -------------------------------------------------------------------- #
# Configuration
# -------------------------------------------------------------------- #
DATA_FILE: Final = Path(__file__).parent.parent / "data" / "202505121539 authors works.csv"
CSV_OUT: Final = Path(__file__).parent.parent / "data" / "naaal1996_threshold_tests.csv"
FIG_OUT: Final = Path(__file__).parent.parent / "viz" / "naaal1996_author_selection_logit.png"
FIG_OUT_CUMULATIVE: Final = Path(__file__).parent.parent / "viz" / "naaal1996_author_selection_logit_cumulative.png"
TARGET_SERIES: Final = "The Norton Anthology of African American Literature"
TARGET_YEAR: Final = 1996
TARGET_EDITION: Final = "1"
LINE_COLOR = "#1f77b4"
POINT_COLOR = "#d62728"

# -------------------------------------------------------------------- #
# Data helpers
# -------------------------------------------------------------------- #
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, na_filter=False)
    df["anthology_year"] = df["anthology_year"].astype(int)
    return df


def authors_in_target(df: pd.DataFrame) -> set[str]:
    mask = (
        (df["anthology_series"] == TARGET_SERIES)
        & (df["anthology_edition"] == TARGET_EDITION)
        & (df["anthology_year"] == TARGET_YEAR)
    )
    return set(df.loc[mask, "work_author"])


def build_author_frame(df: pd.DataFrame) -> pd.DataFrame:
    target_set = authors_in_target(df)

    pre1996 = df[df["anthology_year"] < TARGET_YEAR]
    prior_counts = (
        pre1996.groupby("work_author")["anthology_id"].nunique().rename("prior_count")
    )
    total_ants = pre1996["anthology_id"].nunique()
    if total_ants == 0:
        raise ValueError("No pre‑1996 anthologies in dataset.")

    first_year = df.groupby("work_author")["anthology_year"].min()

    authors = (
        pd.DataFrame({"author": df["work_author"].unique()})
        .merge(prior_counts, how="left", left_on="author", right_index=True)
        .merge(first_year.rename("first_year"), how="left", left_on="author", right_index=True)
        .fillna({"prior_count": 0})
    )

    authors["prior_count"] = authors["prior_count"].astype(int)
    authors["has_prior"] = (authors["prior_count"] > 0).astype(int)
    authors["prior_prob"] = authors["prior_count"] / total_ants
    authors["in_naaal"] = authors["author"].isin(target_set).astype(int)

    return authors[authors["first_year"] <= TARGET_YEAR].reset_index(drop=True)


# -------------------------------------------------------------------- #
# Logit with robust convergence
# -------------------------------------------------------------------- #
def fit_logit_safe(y: pd.Series, x: pd.Series) -> sm.Logit:
    """Fit logit; fall back to tiny‑ridge regularised fit on failure."""
    X = sm.add_constant(x, has_constant="add")
    model = sm.Logit(y, X)
    try:
        return model.fit(disp=False)
    except (LinAlgError, ValueError):
        # Ridge (L1_wt=0) always converges
        return model.fit_regularized(alpha=1e-6, L1_wt=0.0)


def pval_safe(result: sm.Logit, param: str) -> str:
    return f"{result.pvalues[param]:.3g}" if hasattr(result, "pvalues") else "n/a"


# -------------------------------------------------------------------- #
# Figure helpers
# -------------------------------------------------------------------- #
def empirical_rates(authors: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed selection rates by prior-count bucket."""
    summary = (
        authors.groupby("prior_count")["in_naaal"]
        .agg(probability="mean", n="size")
        .reset_index()
        .sort_values("prior_count")
    )
    return summary


def empirical_rates_cumulative(authors: pd.DataFrame) -> pd.DataFrame:
    """Summarize selection rates for authors with prior_count >= k."""
    thresholds = sorted(authors["prior_count"].unique())
    rows = []
    for k in thresholds:
        subset = authors[authors["prior_count"] >= k]
        rows.append({"prior_count": k, "probability": subset["in_naaal"].mean(), "n": len(subset)})
    return pd.DataFrame(rows)


def predicted_probabilities(result: Any, max_prior_count: int, min_prior_count: int = 0) -> pd.DataFrame:
    """Return fitted probabilities for each integer prior count."""
    counts = np.arange(min_prior_count, max_prior_count + 1, dtype=int)
    X = sm.add_constant(pd.Series(counts, name="prior_count"), has_constant="add")
    probs = np.asarray(result.predict(X), dtype=float)
    return pd.DataFrame({"prior_count": counts, "probability": probs})


def plot_probability_curve(authors: pd.DataFrame, result: Any, out_path: Path, cumulative: bool = False) -> None:
    """Plot empirical bucket rates and the fitted logistic probability curve."""
    bucket_df = empirical_rates_cumulative(authors) if cumulative else empirical_rates(authors)
    curve_df = predicted_probabilities(result, int(authors["prior_count"].max()), min_prior_count=0)

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

    ax_top.set_ylabel("Probability of selection for NAAAL 1996")
    ax_top.set_title(
        "Author selection probability for the 1996 Norton Anthology\n"
        "of African American Literature (1st edition)"
    )
    ax_top.set_xlim(0, int(authors["prior_count"].max()) + 0.5)
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_top.grid(True, alpha=0.25, linestyle=":")
    ax_top.legend(frameon=False)

    ax_bot.bar(bucket_df["prior_count"], bucket_df["n"], color=POINT_COLOR, alpha=0.6, width=0.5)
    ax_bot.set_xlabel("Number of prior anthologies selecting an author")
    ax_bot.set_ylabel("N authors (≥ k)" if cumulative else "N authors")
    ax_bot.grid(True, alpha=0.25, linestyle=":", axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- #
# Threshold sweep
# -------------------------------------------------------------------- #
def sweep_thresholds(authors: pd.DataFrame, out_path: Path) -> None:
    """Write threshold table, stopping once no_ge_k == 0 (no negatives in ≥k group)."""
    rows: List[Dict[str, Any]] = []
    for k in range(1, authors["prior_count"].max() + 1):
        flag_name = f"ge_{k}"
        flag = (authors["prior_count"] >= k).astype(int).rename(flag_name)

        ct = (
            authors.assign(flag=flag)
            .groupby(["flag", "in_naaal"])
            .size()
            .unstack(fill_value=0)
        )

        yes_ge, no_ge = ct.loc[1, 1], ct.loc[1, 0]
        yes_lt, no_lt = ct.loc[0, 1], ct.loc[0, 0]

        # Stop sweep when ≥k group has no negative cases (no_ge == 0)
        if no_ge == 0:
            break

        or_fisher, p_fisher = fisher_exact([[yes_ge, no_ge], [yes_lt, no_lt]])

        res = fit_logit_safe(authors["in_naaal"], flag)
        beta = res.params[flag_name]
        exp_beta = np.exp(beta)

        rows.append(
            {
                "threshold_k": k,
                "yes_ge_k": yes_ge,
                "no_ge_k": no_ge,
                "yes_lt_k": yes_lt,
                "no_lt_k": no_lt,
                "odds_ratio": round(or_fisher, 4),
                "fisher_p": p_fisher,
                "logit_beta": beta,
                "exp_beta": exp_beta,
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Threshold sweep saved → {out_path}")


# -------------------------------------------------------------------- #
# Main analysis
# -------------------------------------------------------------------- #
def main() -> None:
    df = load_data(DATA_FILE)
    authors = build_author_frame(df)

    # -------- Descriptive overview -----------------------------------
    tot = len(authors)
    prior_yes = authors["has_prior"].sum()
    in_naaal = authors["in_naaal"].sum()
    print("=========== SAMPLE OVERVIEW ===========")
    print(f"Total authors                    : {tot}")
    print(f"Authors with ≥1 prior anthology   : {prior_yes}")
    print(f"Authors with 0  prior anthology   : {tot - prior_yes}")
    print(f"Authors in NAAAL 1996             : {in_naaal}\n")

    # -------- Model 1 -------------------------------------------------
    print("=========== MODEL 1: has_prior (0/1) ===========")
    m1 = fit_logit_safe(authors["in_naaal"], authors["has_prior"])
    beta1 = m1.params["has_prior"]
    print(f"β (has_prior)      : {beta1:.3f}")
    print(f"p‑value            : {pval_safe(m1, 'has_prior')}")
    print(f"Odds ratio (e^β)   : {np.exp(beta1):.2f}")
    print(f"McFadden pseudo‑R² : {1 - m1.llf / m1.llnull:.3f}\n")

    # -------- Model 2 -------------------------------------------------
    print("=========== MODEL 2: prior_prob (0‒1) ===========")
    m2 = fit_logit_safe(authors["in_naaal"], authors["prior_prob"])
    beta2 = m2.params["prior_prob"]
    print(f"β (prior_prob)     : {beta2:.3f}")
    print(f"p‑value            : {pval_safe(m2, 'prior_prob')}")
    print(f"Odds ratio (e^β)   : {np.exp(beta2):.2f}")
    print(f"McFadden pseudo‑R² : {1 - m2.llf / m2.llnull:.3f}\n")

    # -------- Threshold sweep ----------------------------------------
    sweep_thresholds(authors, CSV_OUT)

    # -------- Interpretation -----------------------------------------
    print("=========== INTERPRETATION ===========")
    print(
        f"*Any* prior appearance multiplies the odds of selection by ≈ {np.exp(beta1):.1f}. "
        f"Moving from zero selections to inclusion in *all* earlier anthologies multiplies "
        f"the odds by ≈ {np.exp(beta2):.0f}.  A threshold‑by‑threshold table is saved "
        f"to '{CSV_OUT.name}'."
    )

    # -------- Prior-count logit and figures --------------------------
    authors_fitted = authors[authors["prior_count"] >= 1]
    result_count = fit_logit_safe(authors_fitted["in_naaal"], authors_fitted["prior_count"])
    plot_probability_curve(authors_fitted, result_count, FIG_OUT)
    plot_probability_curve(authors_fitted, result_count, FIG_OUT_CUMULATIVE, cumulative=True)
    print(f"Saved figure                     : {FIG_OUT}")
    print(f"Saved figure (cumulative)        : {FIG_OUT_CUMULATIVE}")


if __name__ == "__main__":
    main()
