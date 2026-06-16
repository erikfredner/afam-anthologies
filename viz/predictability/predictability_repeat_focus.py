"""
predictability_repeat_focus.py
------------------------------
Focused viz of the two strongest predictability signals from
analysis/predictability_over_time.py:

    fraction_repeat_*          — share of an edition drawn from any
                                  prior AFAM-tagged anthology
    fraction_frequent_canon_*  — share of an edition from entities with
                                  >= 3 prior appearances

Produces a single 2×2 figure (authors on top, works on bottom; the two
metrics across the columns).  Each panel shows the overall trend with an
OLS fit and Spearman ρ annotation.

Use --weight slot (default) for count-based shares or --weight page for
page-span-weighted shares.

Usage:
    uv run python viz/predictability_repeat_focus.py
    uv run python viz/predictability_repeat_focus.py --weight page
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

POINT_COLOR = "#111111"
TREND_COLOR = "#c0392b"


def build_panels(weight: str) -> list[tuple[str, str, str]]:
    return [
        (
            "authors",
            f"fraction_repeat_{weight}",
            "Authors: share of edition appearing in any prior anthology",
        ),
        (
            "authors",
            f"fraction_frequent_canon_{weight}",
            "Authors: share of edition with ≥3 prior anthology appearances",
        ),
        (
            "works",
            f"fraction_repeat_{weight}",
            "Works: share of edition appearing in any prior anthology",
        ),
        (
            "works",
            f"fraction_frequent_canon_{weight}",
            "Works: share of edition with ≥3 prior anthology appearances",
        ),
    ]


WEIGHT_LABELS = {
    "slot": {
        "y_authors": "Share of authors in edition\n(by count, not page span)",
        "y_works": "Share of works in edition\n(by count, not page span)",
        "subtitle": "Slot-weighted: each author or work in the edition counts "
        "as one — not weighted by pages.",
        "default_out": "predictability_repeat_focus.png",
    },
    "page": {
        "y_authors": "Share of edition pages\nattributable to authors",
        "y_works": "Share of edition pages\nattributable to works",
        "subtitle": "Page-weighted: each author/work contributes in proportion "
        "to their TOC page span in the edition.",
        "default_out": "predictability_repeat_focus_pages.png",
    },
}


def annotate_trend(ax: plt.Axes, years: np.ndarray, values: np.ndarray) -> None:
    mask = ~np.isnan(values)
    if mask.sum() < 4:
        return
    y_clean = years[mask]
    v_clean = values[mask]
    lr = linregress(y_clean, v_clean)
    xs = np.array([y_clean.min(), y_clean.max()])
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
    rho, rho_p = spearmanr(y_clean, v_clean)
    ax.text(
        0.02,
        0.97,
        f"OLS slope = {lr.slope:+.4f}/yr\n"
        f"Spearman ρ = {rho:+.2f}  (p = {rho_p:.1e})\n"
        f"n = {mask.sum()} editions",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="#dddddd", alpha=0.85, pad=4),
    )


def draw_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str) -> None:
    overall = (
        df[df["subgroup_dim"] == "all"].sort_values("year").dropna(subset=[metric])
    )

    ax.plot(
        overall["year"],
        overall[metric],
        color=POINT_COLOR,
        linestyle="none",
        marker="o",
        markersize=6,
        alpha=0.85,
        label="observed",
        zorder=5,
    )
    annotate_trend(
        ax, overall["year"].to_numpy(), overall[metric].to_numpy(dtype=float)
    )

    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(1925, 2030)
    ax.grid(True, alpha=0.25, linestyle=":")


def build_figure(
    authors_df: pd.DataFrame, works_df: pd.DataFrame, weight: str
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    frames = {"authors": authors_df, "works": works_df}
    panels = build_panels(weight)
    labels = WEIGHT_LABELS[weight]

    for ax, (entity, metric, title) in zip(axes.flat, panels):
        draw_panel(ax, frames[entity], metric, title)

    axes[0, 0].set_ylabel(labels["y_authors"])
    axes[1, 0].set_ylabel(labels["y_works"])
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
        "Prior-selection predicts new selection: rising over time in AFAM anthologies (1929–2025)",
        fontsize=13,
        y=0.99,
    )
    fig.text(
        0.5,
        0.945,
        labels["subtitle"],
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
        "--weight",
        choices=["slot", "page"],
        default="slot",
        help="Use slot-weighted (count) or page-weighted shares (default: slot).",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help=f"Where to read the predictability CSVs (default: {DEFAULT_CSV_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default depends on --weight).",
    )
    args = parser.parse_args()

    out_path = args.out or (OUTPUT_DIR / WEIGHT_LABELS[args.weight]["default_out"])

    authors = pd.read_csv(args.csv_dir / "predictability_over_time_authors.csv")
    works = pd.read_csv(args.csv_dir / "predictability_over_time_works.csv")
    fig = build_figure(authors, works, args.weight)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
