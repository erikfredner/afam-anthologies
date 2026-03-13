"""
Estimate author-selection probability for the 2025 Norton Anthology of African
American Literature (4th edition) from prior anthology appearances.

The model treats multi-volume editions as a single anthology when counting
previous selections. Because authors with zero prior appearances are present in
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

DATA_FILE = (
    Path(__file__).parent.parent / "data" / "2026-03-13 author ids in afam anthologies.csv"
)
FIG_OUT = Path(__file__).parent.parent / "viz" / "naaal2025_author_selection_logit.png"
TARGET_YEAR = "2025"
TARGET_EDITION = "4"
TARGET_KEY = f"untitled|{TARGET_YEAR}|{TARGET_EDITION}"
LINE_COLOR = "#1f77b4"
POINT_COLOR = "#d62728"


def load_data(path: Path) -> pd.DataFrame:
    """Load the author-level anthology CSV as strings and add edition keys."""
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


def build_author_frame(
    df: pd.DataFrame, target_key: str, min_prior: int = 1
) -> pd.DataFrame:
    """Return one row per author with prior anthology count and target inclusion."""
    target_authors = set(df.loc[df["edition_key"] == target_key, "author_id"])
    prior_counts = (
        df.loc[df["edition_key"] != target_key].groupby("author_id")["edition_key"].nunique()
    )

    authors = pd.DataFrame({"author_id": df["author_id"].drop_duplicates().sort_values()})
    authors["prior_count"] = authors["author_id"].map(prior_counts).fillna(0).astype(int)
    authors["in_target"] = authors["author_id"].isin(target_authors).astype(int)

    return authors[authors["prior_count"] >= min_prior].reset_index(drop=True)


def fit_logit_safe(y: pd.Series, x: pd.Series) -> Any:
    """Fit a one-predictor logistic regression with a regularized fallback."""
    X = sm.add_constant(x.rename("prior_count"), has_constant="add")
    model = sm.Logit(y, X)
    try:
        return model.fit(disp=False)
    except (LinAlgError, ValueError):
        return model.fit_regularized(alpha=1e-6, L1_wt=0.0)


def empirical_rates(authors: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed selection rates by prior-count bucket."""
    summary = (
        authors.groupby("prior_count")["in_target"]
        .agg(probability="mean", n="size")
        .reset_index()
        .sort_values("prior_count")
    )
    return summary


def predicted_probabilities(result: Any, max_prior_count: int, min_prior_count: int = 0) -> pd.DataFrame:
    """Return fitted probabilities for each integer prior count."""
    counts = np.arange(min_prior_count, max_prior_count + 1, dtype=int)
    X = sm.add_constant(pd.Series(counts, name="prior_count"), has_constant="add")
    probs = np.asarray(result.predict(X), dtype=float)
    return pd.DataFrame({"prior_count": counts, "probability": probs})


def plot_probability_curve(authors: pd.DataFrame, result: Any, out_path: Path) -> None:
    """Plot empirical bucket rates and the fitted logistic probability curve."""
    bucket_df = empirical_rates(authors)
    curve_df = predicted_probabilities(result, int(authors["prior_count"].max()), min_prior_count=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        curve_df["prior_count"],
        curve_df["probability"],
        color=LINE_COLOR,
        linewidth=2.0,
        label="Logistic model",
        zorder=2,
    )
    ax.scatter(
        bucket_df["prior_count"],
        bucket_df["probability"],
        s=np.sqrt(bucket_df["n"]) * 18,
        color=POINT_COLOR,
        alpha=0.75,
        label="Observed rate",
        zorder=3,
    )

    ax.set_xlabel("Number of prior anthologies selecting an author")
    ax.set_ylabel("Probability of selection for NAAAL 2025")
    ax.set_title(
        "Author selection probability for the 2025 Norton Anthology\n"
        "of African American Literature (4th edition)"
    )
    ax.set_xlim(0, int(authors["prior_count"].max()) + 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--out", type=Path, default=FIG_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.data_file)
    target_key = find_target_key(df)
    authors = build_author_frame(df, target_key=target_key, min_prior=1)
    if authors.empty:
        raise ValueError("No authors remain after filtering to prior_count >= 1.")

    result = fit_logit_safe(authors["in_target"], authors["prior_count"])
    curve_df = predicted_probabilities(result, int(authors["prior_count"].max()), min_prior_count=0)

    plot_probability_curve(authors, result, args.out)

    zero_prior_in_target = int(
        (
            build_author_frame(df, target_key=target_key, min_prior=0)
            .query("prior_count == 0 and in_target == 1")
            .shape[0]
        )
    )
    print(f"Target edition key               : {target_key}")
    print(f"Authors in fitted sample         : {len(authors)}")
    print(f"Authors selected for NAAAL 2025  : {int(authors['in_target'].sum())}")
    print(f"Zero-prior target-only authors   : {zero_prior_in_target}")
    print(f"Logit intercept                  : {float(result.params['const']):.3f}")
    print(f"Logit slope (prior_count)        : {float(result.params['prior_count']):.3f}")
    print(
        "Predicted probability range      : "
        f"{curve_df['probability'].min():.3f} to {curve_df['probability'].max():.3f}"
    )
    print(f"Saved figure                     : {args.out}")


if __name__ == "__main__":
    main()
