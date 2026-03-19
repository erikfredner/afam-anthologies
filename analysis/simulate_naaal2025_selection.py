"""
simulate_naaal2025_selection.py
-------------------------------
Trains logistic regression + Bayesian (Laplace) models on pre-2025
anthology appearances to predict NAAAL 2025 inclusion.

Produces:
  • console model comparison table
  • viz/naaal2025_predicted_prob_curves.png
  • top-N surprising absences tables (stdout; optionally CSV with --save-csv)

Usage:
    uv run python analysis/simulate_naaal2025_selection.py
    uv run python analysis/simulate_naaal2025_selection.py --only-root-works
    uv run python analysis/simulate_naaal2025_selection.py --save-csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE = (
    Path(__file__).parent.parent / "data" / "2026-03-13 works per afam anthology.csv"
)
FIG_OUT = Path(__file__).parent.parent / "viz" / "naaal2025_predicted_prob_curves.png"
FIG_OUT_LR = Path(__file__).parent.parent / "viz" / "naaal2025_logit_curves.png"
CSV_OUT_W = (
    Path(__file__).parent.parent / "data" / "naaal2025_surprising_absences_works.csv"
)
CSV_OUT_A = (
    Path(__file__).parent.parent / "data" / "naaal2025_surprising_absences_authors.csv"
)

NAAAL_KEY = "3|4"
NAAAL_YEAR = 2025
WORK_COLOR = "#1f77b4"
AUTHOR_COLOR = "#d62728"
TOP_N = 20


# ── Helpers (copied verbatim from naaal1996_prior_selection.py) ───────────────


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


def expand_authors(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form table with one row per (author_id, edition_key, author_name)."""
    records: list[dict] = []
    for _, row in df[df["author_ids"] != ""].iterrows():
        ids = [i.strip() for i in row["author_ids"].split(",")]
        names = [n.strip() for n in row["author_names"].split(";")]
        names += [""] * (len(ids) - len(names))
        for aid, name in zip(ids, names):
            records.append(
                {
                    "author_id": aid,
                    "author_name": name,
                    "edition_key": row["edition_key"],
                }
            )
    return pd.DataFrame(records)


# ── Data frames ───────────────────────────────────────────────────────────────


def build_work_frame(df: pd.DataFrame, only_root: bool) -> pd.DataFrame:
    """One row per work. Includes zero-prior works (debut in NAAAL)."""
    if only_root:
        df = df[(df["parent_work_id"] == "") & (df["parent_work_title"] == "")]

    naaal_works = set(df.loc[df["edition_key"] == NAAAL_KEY, "work_id"])
    other = df[df["edition_key"] != NAAAL_KEY]
    prior_counts = other.groupby("work_id")["edition_key"].nunique()

    # Metadata per work_id: take first occurrence for each work
    meta = (
        df[["work_id", "work_title", "author_names", "parent_work_title"]]
        .drop_duplicates("work_id")
        .set_index("work_id")
    )

    wdf = pd.DataFrame({"work_id": df["work_id"].unique()})
    wdf["prior_count"] = wdf["work_id"].map(prior_counts).fillna(0).astype(int)
    wdf["in_naaal"] = wdf["work_id"].isin(naaal_works).astype(int)
    wdf = wdf.join(meta, on="work_id")
    wdf = wdf[wdf["prior_count"] >= 1]
    return wdf.reset_index(drop=True)


def build_author_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per author. Includes zero-prior authors (debut in NAAAL)."""
    expanded = expand_authors(df)
    naaal_authors = set(expanded.loc[expanded["edition_key"] == NAAAL_KEY, "author_id"])
    other = expanded[expanded["edition_key"] != NAAAL_KEY]
    prior_counts = other.groupby("author_id")["edition_key"].nunique()

    # Canonical name: most common name for each author_id
    name_map = expanded.groupby("author_id")["author_name"].agg(
        lambda s: s.value_counts().index[0]
    )

    adf = pd.DataFrame({"author_id": expanded["author_id"].unique()})
    adf["prior_count"] = adf["author_id"].map(prior_counts).fillna(0).astype(int)
    adf["in_naaal"] = adf["author_id"].isin(naaal_authors).astype(int)
    adf["author_name"] = adf["author_id"].map(name_map).fillna("")
    adf = adf[adf["prior_count"] >= 1]
    return adf.reset_index(drop=True)


# ── Models ────────────────────────────────────────────────────────────────────


def fit_logit_safe(
    y: pd.Series, x: pd.Series
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit logit on prior_count; fall back to regularised fit on failure."""
    X = sm.add_constant(x.rename("prior_count"), has_constant="add")
    model = sm.Logit(y, X)
    try:
        return model.fit(disp=False)
    except (LinAlgError, ValueError):
        return model.fit_regularized(alpha=1e-6, L1_wt=0.0)


@dataclass
class BayesResult:
    map_intercept: float
    map_slope: float
    slope_sd: float
    slope_ci_lo: float
    slope_ci_hi: float
    _map_params: np.ndarray = field(repr=False)
    _cov: np.ndarray = field(repr=False)

    def predict(self, x_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean_prob, lo_band, hi_band) using delta-method CI."""
        eta_mean = self._map_params[0] + self._map_params[1] * x_vals
        var_eta = (
            self._cov[0, 0] + x_vals**2 * self._cov[1, 1] + 2 * x_vals * self._cov[0, 1]
        )
        sd_eta = np.sqrt(np.maximum(var_eta, 0))
        mean_prob = expit(eta_mean)
        lo_band = expit(eta_mean - 1.96 * sd_eta)
        hi_band = expit(eta_mean + 1.96 * sd_eta)
        return mean_prob, lo_band, hi_band


def fit_bayes_laplace(y: pd.Series, x: pd.Series) -> BayesResult:
    """MAP estimate + Laplace approximation (pure scipy, no extra deps).

    Prior: β_j ~ Normal(0, 10²).
    Hessian at MAP computed analytically from the logistic likelihood.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    sigma = 10.0

    def neg_log_post(beta: np.ndarray) -> float:
        eta = beta[0] + beta[1] * x_arr
        p = np.clip(expit(eta), 1e-10, 1 - 1e-10)
        ll = np.sum(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p))
        log_prior = -0.5 * np.sum(beta**2) / sigma**2
        return -(ll + log_prior)

    result = minimize(neg_log_post, x0=np.zeros(2), method="BFGS")
    beta_map = result.x

    # Analytical Hessian at MAP: H = X^T W X + (1/σ²) I
    eta = beta_map[0] + beta_map[1] * x_arr
    p = expit(eta)
    w = p * (1 - p)
    X_mat = np.column_stack([np.ones_like(x_arr), x_arr])
    H = X_mat.T @ (w[:, None] * X_mat) + (1.0 / sigma**2) * np.eye(2)
    cov = np.linalg.inv(H)

    slope_sd = float(np.sqrt(cov[1, 1]))
    slope_ci_lo = float(beta_map[1] - 1.96 * slope_sd)
    slope_ci_hi = float(beta_map[1] + 1.96 * slope_sd)

    return BayesResult(
        map_intercept=float(beta_map[0]),
        map_slope=float(beta_map[1]),
        slope_sd=slope_sd,
        slope_ci_lo=slope_ci_lo,
        slope_ci_hi=slope_ci_hi,
        _map_params=beta_map,
        _cov=cov,
    )


# ── AUC helpers ───────────────────────────────────────────────────────────────


def _auc(y: pd.Series, probs: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probs))


# ── Output helpers ────────────────────────────────────────────────────────────


def print_model_comparison(
    lr_w: sm.regression.linear_model.RegressionResultsWrapper,
    bayes_w: BayesResult,
    auc_lr_w: float,
    auc_b_w: float,
    lr_a: sm.regression.linear_model.RegressionResultsWrapper,
    bayes_a: BayesResult,
    auc_lr_a: float,
    auc_b_a: float,
) -> None:
    def _ci_str(lo: float, hi: float) -> str:
        return f"[{lo:.3f}, {hi:.3f}]"

    def _pseudo_r2(r: sm.regression.linear_model.RegressionResultsWrapper) -> str:
        try:
            return f"{1 - r.llf / r.llnull:.3f}"
        except Exception:
            return "n/a"

    def _lr_slope(r: sm.regression.linear_model.RegressionResultsWrapper) -> float:
        return float(r.params["prior_count"])

    def _lr_ci(
        r: sm.regression.linear_model.RegressionResultsWrapper,
    ) -> tuple[float, float]:
        try:
            ci = r.conf_int()
            return float(ci.loc["prior_count", 0]), float(ci.loc["prior_count", 1])
        except Exception:
            return float("nan"), float("nan")

    lrw_slope = _lr_slope(lr_w)
    lrw_ci = _lr_ci(lr_w)
    lra_slope = _lr_slope(lr_a)
    lra_ci = _lr_ci(lr_a)

    rows = [
        (
            "β slope",
            f"{lrw_slope:.3f}",
            f"{bayes_w.map_slope:.3f}",
            f"{lra_slope:.3f}",
            f"{bayes_a.map_slope:.3f}",
        ),
        (
            "exp(β)",
            f"{np.exp(lrw_slope):.2f}",
            f"{np.exp(bayes_w.map_slope):.2f}",
            f"{np.exp(lra_slope):.2f}",
            f"{np.exp(bayes_a.map_slope):.2f}",
        ),
        (
            "95% CI slope",
            _ci_str(*lrw_ci),
            _ci_str(bayes_w.slope_ci_lo, bayes_w.slope_ci_hi),
            _ci_str(*lra_ci),
            _ci_str(bayes_a.slope_ci_lo, bayes_a.slope_ci_hi),
        ),
        (
            "McFadden R² / post-SD",
            _pseudo_r2(lr_w),
            f"SD={bayes_w.slope_sd:.3f}",
            _pseudo_r2(lr_a),
            f"SD={bayes_a.slope_sd:.3f}",
        ),
        (
            "ROC-AUC",
            f"{auc_lr_w:.3f}",
            f"{auc_b_w:.3f}",
            f"{auc_lr_a:.3f}",
            f"{auc_b_a:.3f}",
        ),
    ]

    col_label = 26
    col_m = 18
    header = (
        f"{'':>{col_label}}"
        f"  {'Works (logit)':>{col_m}}"
        f"  {'Works (Bayes)':>{col_m}}"
        f"  {'Authors (logit)':>{col_m}}"
        f"  {'Authors (Bayes)':>{col_m}}"
    )
    sep = "-" * len(header)

    print("\n===== MODEL COMPARISON: WORKS vs AUTHORS =====")
    print(header)
    print(sep)
    for label, wl, wb, al, ab in rows:
        print(
            f"{label:>{col_label}}"
            f"  {wl:>{col_m}}"
            f"  {wb:>{col_m}}"
            f"  {al:>{col_m}}"
            f"  {ab:>{col_m}}"
        )
    print()


def plot_probability_curves(
    lr_w: sm.regression.linear_model.RegressionResultsWrapper,
    bayes_w: BayesResult,
    n_works: int,
    lr_a: sm.regression.linear_model.RegressionResultsWrapper,
    bayes_a: BayesResult,
    n_authors: int,
    out: Path,
) -> None:
    w_max = float(lr_w.model.exog[:, 1].max())
    a_max = float(lr_a.model.exog[:, 1].max())
    w_xs = np.linspace(0, w_max, 300)
    a_xs = np.linspace(0, a_max, 300)

    # Logit predicted probabilities
    w_lr_probs = expit(
        float(lr_w.params["const"]) + float(lr_w.params["prior_count"]) * w_xs
    )
    a_lr_probs = expit(
        float(lr_a.params["const"]) + float(lr_a.params["prior_count"]) * a_xs
    )

    # Bayes MAP + delta-method CI
    w_b_mean, w_b_lo, w_b_hi = bayes_w.predict(w_xs)
    a_b_mean, a_b_lo, a_b_hi = bayes_a.predict(a_xs)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(
        w_xs, w_b_lo, w_b_hi, color=WORK_COLOR, alpha=0.15, label="Works — Bayes 95% CI"
    )
    ax.plot(
        w_xs,
        w_lr_probs,
        color=WORK_COLOR,
        linewidth=2,
        label=f"Works — logistic fit (N={n_works:,})",
        zorder=3,
    )

    ax.fill_between(
        a_xs,
        a_b_lo,
        a_b_hi,
        color=AUTHOR_COLOR,
        alpha=0.15,
        label="Authors — Bayes 95% CI",
    )
    ax.plot(
        a_xs,
        a_lr_probs,
        color=AUTHOR_COLOR,
        linewidth=2,
        label=f"Authors — logistic fit (N={n_authors:,})",
        zorder=3,
    )

    ax.set_xlabel(
        "Number of pre-2025 non-NAAAL anthologies selecting item", fontsize=11
    )
    ax.set_ylabel("Predicted P(included in NAAAL 2025)", fontsize=11)
    ax.set_title(
        "Logistic + Bayesian predicted-probability curves: works vs authors",
        fontsize=12,
    )
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def plot_logit_curves(
    lr_w: sm.regression.linear_model.RegressionResultsWrapper,
    n_works: int,
    lr_a: sm.regression.linear_model.RegressionResultsWrapper,
    n_authors: int,
    out: Path,
) -> None:
    w_max = int(lr_w.model.exog[:, 1].max())
    a_max = int(lr_a.model.exog[:, 1].max())
    x_max = max(w_max, a_max)
    xs = np.arange(0, x_max + 1)

    w_probs = expit(
        float(lr_w.params["const"]) + float(lr_w.params["prior_count"]) * xs
    )
    a_probs = expit(
        float(lr_a.params["const"]) + float(lr_a.params["prior_count"]) * xs
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        xs,
        w_probs,
        color=WORK_COLOR,
        linewidth=2,
        marker="o",
        markersize=5,
        label=f"Works (N={n_works:,})",
    )
    ax.plot(
        xs,
        a_probs,
        color=AUTHOR_COLOR,
        linewidth=2,
        marker="s",
        markersize=5,
        label=f"Authors (N={n_authors:,})",
    )

    ax.set_xlabel("Prior anthology appearances (non-NAAAL, ≤2025)", fontsize=11)
    ax.set_ylabel("P(selected for NAAAL 2025)", fontsize=11)
    ax.set_title(
        "Predicted probability of NAAAL 2025 inclusion by prior appearances",
        fontsize=12,
    )
    ax.set_xlim(-0.5, x_max + 0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def build_surprising_absences(
    frame: pd.DataFrame,
    lr_result: sm.regression.linear_model.RegressionResultsWrapper,
    id_col: str,
    name_cols: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Items NOT in NAAAL with predicted_p >= threshold, sorted by predicted P descending."""
    predicted_p = lr_result.predict()
    frame = frame.copy()
    frame["predicted_p"] = predicted_p
    absent = frame[frame["in_naaal"] == 0].sort_values("predicted_p", ascending=False)
    absent = absent[absent["predicted_p"] >= threshold]
    cols = [id_col] + name_cols + ["prior_count", "predicted_p"]
    return absent[cols].reset_index(drop=True)


def print_surprising_absences(
    works_df: pd.DataFrame,
    authors_df: pd.DataFrame,
    save_csv: bool,
    threshold: float = 0.5,
) -> None:
    print(f"\n===== TOP {TOP_N} SURPRISING ABSENCES — WORKS =====")
    print(works_df.head(TOP_N).to_string(index=False))

    print(f"\n===== TOP {TOP_N} SURPRISING ABSENCES — AUTHORS =====")
    print(authors_df.head(TOP_N).to_string(index=False))

    if save_csv:
        works_df.to_csv(CSV_OUT_W, index=False)
        authors_df.to_csv(CSV_OUT_A, index=False)
        print(f"\nSaved → {CSV_OUT_W}  ({len(works_df)} works, predicted_p ≥ {threshold})")
        print(f"Saved → {CSV_OUT_A}  ({len(authors_df)} authors, predicted_p ≥ {threshold})")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate NAAAL 2025 selection with logistic + Bayesian models"
    )
    parser.add_argument(
        "--only-root-works",
        action="store_true",
        help="Restrict to works without a parent work (exclude excerpts/selections)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Write surprising-absences tables to data/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="P",
        help="Minimum predicted probability for surprising-absences output (default: 0.5)",
    )
    args = parser.parse_args()

    # 1. Load + assign edition keys + filter to NAAAL_YEAR
    df = pd.read_csv(DATA_FILE, dtype=str, na_filter=False)
    df["anthology_publication_year"] = df["anthology_publication_year"].astype(int)
    df = df[df["anthology_publication_year"] <= NAAAL_YEAR]
    df = assign_edition_key(df)

    n_pre_editions = int(df[df["edition_key"] != NAAAL_KEY]["edition_key"].nunique())

    # 2. Build frames
    work_frame = build_work_frame(df, only_root=args.only_root_works)
    author_frame = build_author_frame(df)

    print("\n===== DATA OVERVIEW =====")
    print(f"Pre-2025 non-NAAAL editions counted : {n_pre_editions}")
    suffix = "  (root-works filter active)" if args.only_root_works else ""
    print(f"Works in sample                     : {len(work_frame)}{suffix}")
    print(f"Authors in sample                   : {len(author_frame)}")
    print(f"Works in NAAAL 2025                 : {int(work_frame['in_naaal'].sum())}")
    print(
        f"Authors in NAAAL 2025               : {int(author_frame['in_naaal'].sum())}"
    )

    # 3. Fit models
    lr_w = fit_logit_safe(work_frame["in_naaal"], work_frame["prior_count"])
    bayes_w = fit_bayes_laplace(work_frame["in_naaal"], work_frame["prior_count"])
    lr_a = fit_logit_safe(author_frame["in_naaal"], author_frame["prior_count"])
    bayes_a = fit_bayes_laplace(author_frame["in_naaal"], author_frame["prior_count"])

    # 4. AUC
    x_w = work_frame["prior_count"].to_numpy(dtype=float)
    x_a = author_frame["prior_count"].to_numpy(dtype=float)
    auc_lr_w = _auc(work_frame["in_naaal"], lr_w.predict())
    auc_lr_a = _auc(author_frame["in_naaal"], lr_a.predict())
    auc_b_w = _auc(work_frame["in_naaal"], bayes_w.predict(x_w)[0])
    auc_b_a = _auc(author_frame["in_naaal"], bayes_a.predict(x_a)[0])

    # 5. Print model comparison
    print_model_comparison(
        lr_w,
        bayes_w,
        auc_lr_w,
        auc_b_w,
        lr_a,
        bayes_a,
        auc_lr_a,
        auc_b_a,
    )

    # 6. Plot
    plot_probability_curves(
        lr_w,
        bayes_w,
        len(work_frame),
        lr_a,
        bayes_a,
        len(author_frame),
        FIG_OUT,
    )
    plot_logit_curves(lr_w, len(work_frame), lr_a, len(author_frame), FIG_OUT_LR)

    # 7. Surprising absences
    sa_works = build_surprising_absences(
        work_frame,
        lr_w,
        "work_id",
        ["work_title", "author_names", "parent_work_title"],
        threshold=args.threshold,
    )
    sa_authors = build_surprising_absences(
        author_frame,
        lr_a,
        "author_id",
        ["author_name"],
        threshold=args.threshold,
    )
    print_surprising_absences(sa_works, sa_authors, save_csv=args.save_csv, threshold=args.threshold)


if __name__ == "__main__":
    main()
