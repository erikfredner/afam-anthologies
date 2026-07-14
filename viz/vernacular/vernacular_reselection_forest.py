"""
vernacular_reselection_forest.py
--------------------------------
Forest plot of vernacular work reselection rates against size-matched
non-vernacular Monte Carlo nulls.

Reuses the estimates computed by
analysis/vernacular/vernacular_reselection_test.py: for each of the six
metrics (ever-reselected / per-opportunity × all / cross-series / cross-series
restricted to later editions that themselves contain vernacular works) the
observed vernacular rate (95% Wilson CI) is drawn over the null distribution
of |V|-sized random non-vernacular samples (95% empirical interval), under
both sampling modes — 'unmatched' (uniform draws) and 'debut-stratified'
(draws matching the vernacular group's debut-edition distribution, which
equalizes reselection opportunities).

Usage:
    uv run python viz/vernacular/vernacular_reselection_forest.py
    uv run python viz/vernacular/vernacular_reselection_forest.py --include-excerpts
    uv run python viz/vernacular/vernacular_reselection_forest.py --n 1000 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afam.cli import add_root_works_flag
from afam.vernacular import load_vernacular_ranges
from afam.viz_style import OUTPUT_DIR, WORK_COLOR

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "vernacular"))
sys.path.insert(0, str(REPO_ROOT / "analysis" / "reselection"))

from author_vs_work_debut_reselection import _rate_stats, fmt_p  # noqa: E402
from author_vs_work_debut_reselection import (  # noqa: E402
    compute_work_records,
)
from author_vs_work_debut_reselection import (  # noqa: E402
    load_data as load_reselection_data,
)
from vernacular_reselection_test import (  # noqa: E402
    DEFAULT_N,
    DEFAULT_SEED,
    attach_vern_cross_series_records,
    build_groups,
    build_monte_carlo,
    fmt_z,
)
from vernacular_works import load_data as load_page_data  # noqa: E402

OUT_FILE = OUTPUT_DIR / "vernacular_reselection_forest.png"
GRID_KW = dict(alpha=0.25, linestyle=":")

NULL_BAND_COLOR = "#c8c8c8"
NULL_MEAN_COLOR = "#666666"

METRIC_LABELS = {
    "ever_all": "Ever reselected\n(all later editions)",
    "ever_cross_series": "Ever reselected\n(cross-series only)",
    "ever_cross_series_vern_eds": "Ever reselected\n(cross-series, vernacular editions only)",
    "opportunity_all": "Per-opportunity\n(all later editions)",
    "opportunity_cross_series": "Per-opportunity\n(cross-series only)",
    "opportunity_cross_series_vern_eds": "Per-opportunity\n(cross-series, vernacular editions only)",
}
MODE_LABELS = {"unmatched": "unmatched null", "stratified": "debut-stratified null"}
GROUP_GAP = 0.8  # extra vertical space between metric groups


def plot(mc: pd.DataFrame, vern_works: int, n_draws: int, out: Path) -> None:
    if mc.empty:
        print("No Monte Carlo results to plot (a group is empty in every metric).")
        return

    fig, ax = plt.subplots(figsize=(11, 1.75 * len(METRIC_LABELS)))

    rows = mc.set_index(["metric", "mode"])
    y_positions: list[float] = []
    y_labels: list[str] = []
    y = 0.0
    for metric in METRIC_LABELS:
        group_ys = []
        for mode in MODE_LABELS:
            # build_monte_carlo skips a metric when either group is empty for
            # that scope; render whatever rows exist rather than crashing.
            if (metric, mode) not in rows.index:
                continue
            row = rows.loc[(metric, mode)]
            obs = 100 * row["vernacular_rate"]
            _, lo_ci, hi_ci = _rate_stats(
                int(row["vernacular_k"]), int(row["vernacular_n"])
            )

            # Null: 95% empirical interval band + mean marker
            ax.plot(
                [100 * row["null_lo"], 100 * row["null_hi"]],
                [y, y],
                linewidth=7,
                color=NULL_BAND_COLOR,
                solid_capstyle="round",
                zorder=2,
            )
            ax.plot(
                100 * row["null_mean"],
                y,
                marker="D",
                markersize=6,
                color=NULL_MEAN_COLOR,
                zorder=3,
            )

            # Observed vernacular rate + Wilson CI
            ax.errorbar(
                obs,
                y,
                xerr=[[obs - 100 * lo_ci], [100 * hi_ci - obs]],
                fmt="o",
                color=WORK_COLOR,
                capsize=3,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=4,
            )

            ax.annotate(
                f"z={fmt_z(row['z_score'])}  p={fmt_p(row['empirical_p'])}",
                (1.01, y),
                xycoords=("axes fraction", "data"),
                textcoords="offset points",
                xytext=(0, 1),
                fontsize=9,
                va="bottom",
            )
            shortfall = int(row["sample_shortfall"])
            n_note = f"n={int(row['vernacular_n']):,}" + (
                f"  [null {shortfall} short of |V|]" if shortfall else ""
            )
            ax.annotate(
                n_note,
                (1.01, y),
                xycoords=("axes fraction", "data"),
                textcoords="offset points",
                xytext=(0, -1),
                fontsize=7,
                color="gray",
                va="top",
            )

            y_positions.append(y)
            y_labels.append(MODE_LABELS[mode])
            group_ys.append(y)
            y -= 1.0

        if group_ys:
            ax.annotate(
                METRIC_LABELS[metric],
                (-0.155, float(np.mean(group_ys))),
                xycoords=("axes fraction", "data"),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
            y -= GROUP_GAP

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylim(min(y_positions) - 0.7, 0.7)
    ax.set_xlim(left=0)
    ax.set_xlabel("Reselection rate (%)", fontsize=11)
    ax.set_title(
        f"Vernacular works ({vern_works}) vs. size-matched non-vernacular nulls\n"
        f"(observed rate with 95% Wilson CI over the null's 95% interval, "
        f"{n_draws:,} draws each)",
        fontsize=12,
    )
    ax.grid(True, axis="x", **GRID_KW)

    handles = [
        plt.Line2D(
            [],
            [],
            color=WORK_COLOR,
            marker="o",
            markeredgecolor="white",
            linestyle="",
            label="Observed vernacular rate (95% Wilson CI)",
        ),
        plt.Line2D(
            [],
            [],
            color=NULL_MEAN_COLOR,
            marker="D",
            linestyle="",
            label="Null mean",
        ),
        plt.Line2D(
            [],
            [],
            color=NULL_BAND_COLOR,
            linewidth=7,
            solid_capstyle="round",
            label="Null 95% interval",
        ),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")

    fig.text(
        0.01,
        0.005,
        "Per-opportunity rows pool work×edition slots; their Wilson CIs treat "
        "correlated slots as independent and are anti-conservative.",
        fontsize=7.5,
        color="gray",
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forest plot of vernacular vs. non-vernacular reselection rates."
    )
    add_root_works_flag(parser)
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        metavar="N",
        help=f"Monte Carlo draws per metric/mode (default: {DEFAULT_N}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="S",
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_FILE,
        metavar="FILE",
        help=f"Output PNG path (default: {OUT_FILE}).",
    )
    args = parser.parse_args()

    # The page table stays unfiltered so the edition universe is stable;
    # root-vs-excerpt scope enters through work_records (see build_groups).
    page_df = load_page_data(only_root_works=False)
    ranges = load_vernacular_ranges()
    raw, all_editions = load_reselection_data(args.only_root_works)
    work_records = compute_work_records(raw, all_editions)

    classified, info = build_groups(page_df, ranges, work_records)
    classified = attach_vern_cross_series_records(classified, raw, all_editions, info)
    mc = build_monte_carlo(classified, args.n, args.seed)

    plot(mc, info["n_vernacular_works"], args.n, args.out)


if __name__ == "__main__":
    main()
