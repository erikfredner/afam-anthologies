"""
predictability_new_focus.py
---------------------------
Mirror of viz/predictability/predictability_repeat_focus.py.  Where that
script shows the share of each AFAM edition drawn from *previously*
anthologized authors and works, this one shows the inverse: the share
that is appearing in the tradition for the *first* time.

Reads the CSVs written by analysis/predictability/predictability_over_time.py
(`data/predictability_over_time_{authors,works}.csv`) and derives:

    fraction_new_slot  = 1 - fraction_repeat_slot
    fraction_new_page  = 1 - fraction_repeat_page

Produces a single 2x2 figure: authors on top, works on bottom; slot-weighted
in the left column, page-weighted in the right.  Each panel shows the
overall trend (subgroup_dim == "all") with an OLS fit and Spearman rho
annotation.

Usage:
    uv run python viz/predictability/predictability_new_focus.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from afam import DATA_DIR
from afam.viz_style import OUTPUT_DIR

DEFAULT_CSV_DIR = DATA_DIR
DEFAULT_OUT = OUTPUT_DIR / "predictability_new_focus.png"

POINT_COLOR = "#111111"
TREND_COLOR = "#1f5fa8"
OUTLIER_COLOR = "#c0392b"

EDITION_LABELS = {
    72: "Calverton",
    71: "Cromwell (Readings)",
    68: "Negro Caravan",
    69: "Dreer",
    55: "Black Voices",
    63: "Intro to Black Lit.",
    64: "Dark Symphony",
    54: "Cavalcade",
    57: "Black Insights",
    59: "Black Lit. in America",
    61: "Blackamerican Lit.",
    21: "Afro-Am. Writing ed.1",
    58: "Black Writers of Am.",
    20: "Afro-Am. Writing ed.2",
    19: "New Cavalcade",
    56: "AAL: Brief Intro.",
    70: "Cornerstones",
    16: "NAAAL ed.1",
    32: "AAL ed.2",
    60: "Call & Response",
    53: "Prentice Hall",
    17: "NAAAL ed.2",
    73: "AAL (2004)",
    13: "NAAAL ed.3",
    18: "Wiley Blackwell",
    43: "NAAAL ed.4",
}

PANELS = [
    (
        "authors",
        "fraction_new_slot",
        "Authors: share of edition never previously anthologized",
    ),
    (
        "authors",
        "fraction_new_page",
        "Authors: share of edition pages from never-before-anthologized authors",
    ),
    (
        "works",
        "fraction_new_slot",
        "Works: share of edition never previously anthologized",
    ),
    (
        "works",
        "fraction_new_page",
        "Works: share of edition pages from never-before-anthologized works",
    ),
]

Y_LABELS = {
    (
        "authors",
        "fraction_new_slot",
    ): "Share of authors in edition\n(by count, not page span)",
    ("authors", "fraction_new_page"): "Share of edition pages\nattributable to authors",
    (
        "works",
        "fraction_new_slot",
    ): "Share of works in edition\n(by count, not page span)",
    ("works", "fraction_new_page"): "Share of edition pages\nattributable to works",
}


def add_new_columns(df: pd.DataFrame) -> pd.DataFrame:
    for weight in ("slot", "page"):
        df[f"fraction_new_{weight}"] = 1 - df[f"fraction_repeat_{weight}"]
    return df


def annotate_trend(ax: plt.Axes, years: np.ndarray, values: np.ndarray):
    if len(years) < 4:
        return None
    lr = linregress(years, values)
    xs = np.array([years.min(), years.max()])
    ax.plot(
        xs,
        lr.intercept + lr.slope * xs,
        color=TREND_COLOR,
        linestyle="-",
        linewidth=2.8,
        alpha=1.0,
        label="OLS trend",
        zorder=4,
    )
    rho, rho_p = spearmanr(years, values)
    ax.text(
        0.98,
        0.97,
        f"OLS slope = {lr.slope:+.4f}/yr\n"
        f"Spearman ρ = {rho:+.2f}  (p = {rho_p:.1e})\n"
        f"n = {len(years)} editions",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8.5,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="#dddddd", alpha=0.85, pad=4),
    )
    return lr


def annotate_outlier(
    ax: plt.Axes,
    years: np.ndarray,
    values: np.ndarray,
    edition_ids: np.ndarray,
    lr,
) -> None:
    predicted = lr.intercept + lr.slope * years
    residuals = values - predicted
    idx = int(np.argmax(np.abs(residuals)))
    eid = int(edition_ids[idx])
    label = EDITION_LABELS.get(eid, f"edition {eid}")
    x, y = float(years[idx]), float(values[idx])

    ax.plot(
        x,
        y,
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor=OUTLIER_COLOR,
        markeredgewidth=1.8,
        zorder=6,
    )

    year_mid = (years.min() + years.max()) / 2
    # Default offset: up-right; flip to avoid the top-right annotation box
    # and the panel edges.
    if x > year_mid:
        ha, dx = "right", -10
    else:
        ha, dx = "left", 10
    if y > 0.55:
        va, dy = "top", -10
    else:
        va, dy = "bottom", 10

    ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.5,
        color=OUTLIER_COLOR,
        ha=ha,
        va=va,
        fontweight="bold",
        zorder=7,
    )


def draw_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str) -> None:
    overall = (
        df[df["subgroup_dim"] == "all"].sort_values("year").dropna(subset=[metric])
    )

    years = overall["year"].to_numpy(dtype=float)
    values = overall[metric].to_numpy(dtype=float)
    edition_ids = overall["edition_id"].to_numpy()

    ax.plot(
        years,
        values,
        color=POINT_COLOR,
        linestyle="none",
        marker="o",
        markersize=6,
        alpha=0.85,
        label="observed",
        zorder=5,
    )
    lr = annotate_trend(ax, years, values)
    if lr is not None:
        annotate_outlier(ax, years, values, edition_ids, lr)

    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(1925, 2030)
    ax.grid(True, alpha=0.25, linestyle=":")


def build_figure(authors_df: pd.DataFrame, works_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    frames = {"authors": authors_df, "works": works_df}

    for ax, (entity, metric, title) in zip(axes.flat, PANELS):
        draw_panel(ax, frames[entity], metric, title)
        ax.set_ylabel(Y_LABELS[(entity, metric)])

    for ax in axes[1, :]:
        ax.set_xlabel("Anthology year")

    seen: dict[str, plt.Line2D] = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen[label] = handle
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="lower center",
        ncol=len(seen),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Each new AFAM edition draws a shrinking share from previously "
        "unanthologized authors and works (1929-2025)",
        fontsize=13,
        y=0.99,
    )
    fig.text(
        0.5,
        0.945,
        "Left column slot-weighted (each entity = 1); "
        "right column page-weighted (TOC page span in the edition).",
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help=f"Where to read the predictability CSVs (default: {DEFAULT_CSV_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output PNG path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    authors = add_new_columns(
        pd.read_csv(args.csv_dir / "predictability_over_time_authors.csv")
    )
    works = add_new_columns(
        pd.read_csv(args.csv_dir / "predictability_over_time_works.csv")
    )
    fig = build_figure(authors, works)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
