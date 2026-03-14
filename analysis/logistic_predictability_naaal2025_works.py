"""
Estimate work-selection probability for the 2025 Norton Anthology of African
American Literature (4th edition) from prior anthology appearances.

The model treats multi-volume editions as a single anthology when counting
previous selections. Because works with zero prior appearances are present in
the data only if they were selected for the 2025 Norton, they are excluded from
the fitted model to avoid a tautological zero-prior bucket.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError

DATA_FILE = Path(__file__).parent.parent / "data" / "2026-03-13 work ids in afam anthologies.csv"
FIG_OUT = Path(__file__).parent.parent / "viz" / "naaal2025_work_selection_logit.png"
FIG_OUT_CUMULATIVE = Path(__file__).parent.parent / "viz" / "naaal2025_work_selection_logit_cumulative.png"
TARGET_YEAR = "2025"
TARGET_EDITION = "4"
TARGET_KEY = f"untitled|{TARGET_YEAR}|{TARGET_EDITION}"
LINE_COLOR = "#1f77b4"
POINT_COLOR = "#d62728"


def load_data(path: Path) -> pd.DataFrame:
    """Load the work-level anthology CSV as strings and add edition keys."""
    df = pd.read_csv(path, dtype=str, na_filter=False)
    return assign_edition_key(df)


def _strip_volume(title: str) -> str:
    return re.sub(r",?\s+[Vv]ol\.?\s+\d+\s*$", "", title).strip()


def _clean_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-volume anthologies into edition-level keys."""
    meta_rows: list[dict[str, str]] = []
    meta = df[
        [
            "anthology_id",
            "anthology_title",
            "edition_number",
            "volume",
            "publication_year",
        ]
    ].drop_duplicates()
    for _, row in meta.iterrows():
        anthology_id = _clean_str(row["anthology_id"])
        title = _clean_str(row["anthology_title"])
        edition = _clean_str(row["edition_number"])
        volume = _clean_str(row["volume"])
        year = _clean_str(row["publication_year"])

        if title and volume:
            key = f"{_strip_volume(title)}|{edition}"
        elif not title and year and edition:
            key = f"untitled|{year}|{edition}"
        else:
            key = anthology_id

        meta_rows.append({"anthology_id": anthology_id, "edition_key": key})

    return df.merge(pd.DataFrame(meta_rows), on="anthology_id", how="left")


def find_target_key(df: pd.DataFrame) -> str:
    """Return the collapsed edition key for the 2025 Norton 4th edition rows."""
    keys = (
        df.loc[
            (df["publication_year"] == TARGET_YEAR) & (df["edition_number"] == TARGET_EDITION),
            "edition_key",
        ]
        .drop_duplicates()
        .tolist()
    )
    if len(keys) != 1:
        raise ValueError(
            f"Expected exactly one target edition key for {TARGET_YEAR} edition {TARGET_EDITION}, "
            f"found {keys!r}"
        )
    return keys[0]


def build_work_frame(df: pd.DataFrame, target_key: str, min_prior: int = 1) -> pd.DataFrame:
    """Return one row per work with prior anthology count and target inclusion."""
    target_works = set(df.loc[df["edition_key"] == target_key, "work_id"])
    prior_counts = (
        df.loc[df["edition_key"] != target_key].groupby("work_id")["edition_key"].nunique()
    )

    works = pd.DataFrame({"work_id": df["work_id"].drop_duplicates().sort_values()})
    works["prior_count"] = works["work_id"].map(prior_counts).fillna(0).astype(int)
    works["in_target"] = works["work_id"].isin(target_works).astype(int)

    return works[works["prior_count"] >= min_prior].reset_index(drop=True)


def fit_logit_safe(y: pd.Series, x: pd.Series) -> Any:
    """Fit a one-predictor logistic regression with a regularized fallback."""
    X = sm.add_constant(x.rename("prior_count"), has_constant="add")
    model = sm.Logit(y, X)
    try:
        return model.fit(disp=False)
    except (LinAlgError, ValueError):
        return model.fit_regularized(alpha=1e-6, L1_wt=0.0)


def empirical_rates(works: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed selection rates by prior-count bucket."""
    summary = (
        works.groupby("prior_count")["in_target"]
        .agg(probability="mean", n="size")
        .reset_index()
        .sort_values("prior_count")
    )
    return summary


def empirical_rates_cumulative(works: pd.DataFrame) -> pd.DataFrame:
    """Summarize selection rates for works with prior_count >= k."""
    thresholds = sorted(works["prior_count"].unique())
    rows = []
    for k in thresholds:
        subset = works[works["prior_count"] >= k]
        rows.append({"prior_count": k, "probability": subset["in_target"].mean(), "n": len(subset)})
    return pd.DataFrame(rows)


def predicted_probabilities(result: Any, max_prior_count: int, min_prior_count: int = 0) -> pd.DataFrame:
    """Return fitted probabilities for each integer prior count."""
    counts = np.arange(min_prior_count, max_prior_count + 1, dtype=int)
    X = sm.add_constant(pd.Series(counts, name="prior_count"), has_constant="add")
    probs = np.asarray(result.predict(X), dtype=float)
    return pd.DataFrame({"prior_count": counts, "probability": probs})


def plot_probability_curve(works: pd.DataFrame, result: Any, out_path: Path, cumulative: bool = False) -> None:
    """Plot empirical bucket rates and the fitted logistic probability curve."""
    bucket_df = empirical_rates_cumulative(works) if cumulative else empirical_rates(works)
    curve_df = predicted_probabilities(result, int(works["prior_count"].max()), min_prior_count=0)

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

    ax_top.set_ylabel("Probability of selection for NAAAL 2025")
    ax_top.set_title(
        "Work selection probability for the 2025 Norton Anthology\n"
        "of African American Literature (4th edition)"
    )
    ax_top.set_xlim(0, int(works["prior_count"].max()) + 0.5)
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_top.grid(True, alpha=0.25, linestyle=":")
    ax_top.legend(frameon=False)

    ax_bot.bar(bucket_df["prior_count"], bucket_df["n"], color=POINT_COLOR, alpha=0.6, width=0.5)
    ax_bot.set_xlabel("Number of prior anthologies selecting a work")
    ax_bot.set_ylabel("N works (≥ k)" if cumulative else "N works")
    ax_bot.grid(True, alpha=0.25, linestyle=":", axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--out", type=Path, default=FIG_OUT)
    parser.add_argument("--out-cumulative", type=Path, default=FIG_OUT_CUMULATIVE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.data_file)
    target_key = find_target_key(df)
    works = build_work_frame(df, target_key=target_key, min_prior=1)
    if works.empty:
        raise ValueError("No works remain after filtering to prior_count >= 1.")

    result = fit_logit_safe(works["in_target"], works["prior_count"])
    curve_df = predicted_probabilities(result, int(works["prior_count"].max()), min_prior_count=0)

    plot_probability_curve(works, result, args.out)
    plot_probability_curve(works, result, args.out_cumulative, cumulative=True)

    zero_prior_in_target = int(
        (
            build_work_frame(df, target_key=target_key, min_prior=0)
            .query("prior_count == 0 and in_target == 1")
            .shape[0]
        )
    )
    print(f"Target edition key               : {target_key}")
    print(f"Works in fitted sample           : {len(works)}")
    print(f"Works selected for NAAAL 2025    : {int(works['in_target'].sum())}")
    print(f"Zero-prior target-only works     : {zero_prior_in_target}")
    print(f"Logit intercept                  : {float(result.params['const']):.3f}")
    print(f"Logit slope (prior_count)        : {float(result.params['prior_count']):.3f}")
    print(
        "Predicted probability range      : "
        f"{curve_df['probability'].min():.3f} to {curve_df['probability'].max():.3f}"
    )
    print(f"Saved figure                     : {args.out}")
    print(f"Saved figure (cumulative)        : {args.out_cumulative}")


if __name__ == "__main__":
    main()
