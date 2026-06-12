"""
debut_reselection_forest.py
---------------------------
Forest/dumbbell chart of debut reselection rates for authors vs works across
African-American Literature anthology editions, with 95% Wilson CIs.

Reuses the estimands computed by
analysis/reselection/author_vs_work_debut_reselection.py — each debuting
author or work is counted exactly once per metric:

  * ever reselected after debut (all later editions / cross-series only)
  * per-opportunity reselection rate (all later editions / cross-series only)

The cross-series rows exclude later editions from the debut's own series, so
they guard against within-series inertia inflating both rates.

Usage:
    uv run python viz/reselection/debut_reselection_forest.py
    uv run python viz/reselection/debut_reselection_forest.py --include-excerpts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from afam.cli import add_root_works_flag
from afam.viz_style import AUTHOR_COLOR, OUTPUT_DIR, WORK_COLOR

sys.path.insert(0, str(Path(__file__).parents[2] / "analysis" / "reselection"))
from author_vs_work_debut_reselection import (  # noqa: E402
    build_summary,
    compute_author_records,
    compute_work_records,
    load_data,
)

OUT_FILE = OUTPUT_DIR / "debut_reselection_forest.png"
GRID_KW = dict(alpha=0.25, linestyle=":")

METRIC_LABELS = {
    "ever_all": "Ever reselected\n(all later editions)",
    "ever_cross_series": "Ever reselected\n(cross-series only)",
    "opportunity_all": "Per-opportunity reselection\n(all later editions)",
    "opportunity_cross_series": "Per-opportunity reselection\n(cross-series only)",
}


def plot(summary: pd.DataFrame, out: Path) -> None:
    rows = (
        summary.set_index("metric")
        .loc[list(METRIC_LABELS)]
        .reset_index()
        .iloc[::-1]  # first metric on top
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    for y, row in rows.iterrows():
        work_pct = 100 * row["work_rate"]
        auth_pct = 100 * row["author_rate"]

        ax.plot(
            [work_pct, auth_pct],
            [y, y],
            color="gray",
            linewidth=1.2,
            alpha=0.6,
            zorder=2,
        )
        ax.errorbar(
            work_pct,
            y,
            xerr=[
                [work_pct - 100 * row["work_ci_lo"]],
                [100 * row["work_ci_hi"] - work_pct],
            ],
            fmt="o",
            color=WORK_COLOR,
            capsize=3,
            markersize=7,
            zorder=3,
        )
        ax.errorbar(
            auth_pct,
            y,
            xerr=[
                [auth_pct - 100 * row["author_ci_lo"]],
                [100 * row["author_ci_hi"] - auth_pct],
            ],
            fmt="s",
            color=AUTHOR_COLOR,
            capsize=3,
            markersize=7,
            zorder=3,
        )

        ax.annotate(
            f"{work_pct:.1f}%",
            (work_pct, y),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
            color=WORK_COLOR,
        )
        ax.annotate(
            f"{auth_pct:.1f}%",
            (auth_pct, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=AUTHOR_COLOR,
        )
        ax.annotate(
            f"RR {row['risk_ratio_author_over_work']:.2f}×",
            (1.01, y),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, 1),
            fontsize=9,
            va="bottom",
        )
        ax.annotate(
            f"authors n={int(row['author_n']):,}\nworks n={int(row['work_n']):,}",
            (1.01, y),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, -1),
            fontsize=7,
            color="gray",
            va="top",
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([METRIC_LABELS[m] for m in rows["metric"]], fontsize=10)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Reselection rate (%, 95% Wilson CI)", fontsize=11)
    ax.set_title("Debut reselection rates: authors vs works", fontsize=12)
    ax.grid(True, axis="x", **GRID_KW)

    handles = [
        plt.Line2D([], [], color=WORK_COLOR, marker="o", linestyle="", label="Works"),
        plt.Line2D(
            [], [], color=AUTHOR_COLOR, marker="s", linestyle="", label="Authors"
        ),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forest plot of author vs work debut reselection rates."
    )
    add_root_works_flag(parser)
    args = parser.parse_args()

    raw, all_editions = load_data(args.only_root_works)
    works = compute_work_records(raw, all_editions)
    authors = compute_author_records(raw, all_editions)
    summary = build_summary(works, authors)

    plot(summary, OUT_FILE)


if __name__ == "__main__":
    main()
