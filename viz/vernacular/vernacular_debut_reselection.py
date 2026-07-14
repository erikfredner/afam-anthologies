"""
vernacular_debut_reselection.py
--------------------------------
Jittered boxplot of later-edition reselection counts, by debut edition and
vernacular / non-vernacular status.

For every work appearing in a vernacular-containing AFAM edition (the V/NV
pool built by analysis/vernacular/vernacular_reselection_test.py), plots
how many later editions reselected it (``reselection_count``), grouped by
debut edition (chronological x-axis) and colored by group. Each debut
edition gets a pair of boxes — one per group — with that group's works
jittered on top, so the two groups' distributions can be compared directly
within each edition. An annotation box gives the pooled overall V-vs-NV
rates (ever-reselected and per-opportunity, matching Test 1 in
vernacular_reselection_test.py) as context for the per-edition spread.

Later debut editions mechanically have fewer subsequent editions available
to reselect them, so counts trend down toward the right regardless of
group — this is the same debut-edition confound that
vernacular_reselection_test's debut-stratified Monte Carlo null controls
for.

Usage:
    uv run python viz/vernacular/vernacular_debut_reselection.py
    uv run python viz/vernacular/vernacular_debut_reselection.py --include-excerpts
    uv run python viz/vernacular/vernacular_debut_reselection.py --cross-series
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afam.cli import add_root_works_flag
from afam.editions import EDITION_LABELS
from afam.vernacular import load_vernacular_ranges
from afam.viz_style import AUTHOR_COLOR, DPI, OUTPUT_DIR, WORK_COLOR

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "vernacular"))
sys.path.insert(0, str(REPO_ROOT / "analysis" / "reselection"))

from author_vs_work_debut_reselection import (  # noqa: E402
    compute_work_records,
    fmt_p,
    fmt_pct,
)
from author_vs_work_debut_reselection import (  # noqa: E402
    load_data as load_reselection_data,
)
from vernacular_reselection_test import (  # noqa: E402
    attach_vern_cross_series_records,
    build_comparison,
    build_groups,
)
from vernacular_works import load_data as load_page_data  # noqa: E402

OUT_FILE = OUTPUT_DIR / "vernacular_debut_reselection.png"

VERN_COLOR = WORK_COLOR
NONVERN_COLOR = AUTHOR_COLOR
GROUP_OFFSET = 0.18
BOX_WIDTH = 0.32
GROUPS = (
    ("vernacular", "Vernacular", VERN_COLOR, -GROUP_OFFSET),
    ("non_vernacular", "Non-vernacular", NONVERN_COLOR, GROUP_OFFSET),
)


def summary_text(comparison: pd.DataFrame, cross_series: bool) -> str:
    """Pooled V-vs-NV overall rates (analysis/vernacular/vernacular_reselection_test.py
    Test 1), formatted for an in-plot annotation matching the chart's scope."""
    rows = comparison.set_index("metric")
    ever = rows.loc["ever_cross_series" if cross_series else "ever_all"]
    opp = rows.loc["opportunity_cross_series" if cross_series else "opportunity_all"]

    def fmt_rate(row: pd.Series, prefix: str) -> str:
        return (
            f"{fmt_pct(row[f'{prefix}_rate'])} "
            f"({int(row[f'{prefix}_k'])}/{int(row[f'{prefix}_n'])})"
        )

    return (
        "Pooled overall rates (V vs. NV)\n"
        f"Ever reselected:      {fmt_rate(ever, 'vernacular')}  vs.  "
        f"{fmt_rate(ever, 'non_vernacular')}\n"
        f"                       diff={fmt_pct(ever['rate_difference'])}  "
        f"p={fmt_p(ever['two_proportion_z_p'])}\n"
        f"Per-opportunity rate:  {fmt_rate(opp, 'vernacular')}  vs.  "
        f"{fmt_rate(opp, 'non_vernacular')}\n"
        f"                       diff={fmt_pct(opp['rate_difference'])}  "
        f"p={fmt_p(opp['two_proportion_z_p'])}"
    )


def plot(
    classified: pd.DataFrame,
    count_col: str,
    comparison: pd.DataFrame,
    cross_series: bool,
    out: Path,
) -> None:
    ed_years = classified.groupby("debut_edition_id")["debut_year"].first()
    editions = sorted(ed_years.index, key=lambda e: (ed_years[e], e))
    labels = [
        f"{EDITION_LABELS.get(e, str(e))}\n({int(ed_years[e])})" for e in editions
    ]

    x = np.arange(1, len(editions) + 1)

    fig, ax = plt.subplots(figsize=(16, 7))
    rng = np.random.default_rng(0)

    for group, _, color, dx in GROUPS:
        positions = []
        data = []
        for xi, ed in zip(x, editions):
            vals = classified.loc[
                (classified["debut_edition_id"] == ed) & (classified["group"] == group),
                count_col,
            ].to_numpy()
            if len(vals) == 0:
                continue
            positions.append(xi + dx)
            data.append(vals)

        if data:
            ax.boxplot(
                data,
                positions=positions,
                widths=BOX_WIDTH,
                showfliers=False,
                patch_artist=True,
                medianprops=dict(color="#222222", linewidth=1.3),
                boxprops=dict(facecolor=color, alpha=0.3, edgecolor=color),
                whiskerprops=dict(color=color, alpha=0.9),
                capprops=dict(color=color, alpha=0.9),
                zorder=2,
            )

        for pos, vals in zip(positions, data):
            jitter = rng.uniform(
                -BOX_WIDTH / 2 * 0.85, BOX_WIDTH / 2 * 0.85, size=len(vals)
            )
            ax.scatter(
                pos + jitter,
                vals,
                s=16,
                color=color,
                alpha=0.55,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

    scope = (
        "cross-series later editions"
        if count_col.startswith("cross")
        else "later editions"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=60, ha="right")
    ax.set_xlim(0.3, len(editions) + 0.7)
    y_max = classified[count_col].max()
    ax.set_ylim(-0.4, y_max * 1.32)  # headroom for the summary annotation
    ax.set_ylabel(f"N reselections after debut ({scope})")
    ax.set_xlabel("Debut edition")
    ax.set_title(
        "Reselection counts after debut, by debut edition\n"
        "(vernacular vs. non-vernacular works in vernacular-containing editions)"
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color,
            alpha=0.3,
            edgecolor=color,
            label=f"{label} (box = IQR, line = median; dots = individual works)",
        )
        for _, label, color, _ in GROUPS
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")

    ax.text(
        0.01,
        0.985,
        summary_text(comparison, cross_series),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(
            facecolor="white", edgecolor="#cccccc", alpha=0.9, boxstyle="round,pad=0.5"
        ),
        zorder=5,
    )

    fig.text(
        0.01,
        0.005,
        "Later debut editions have fewer subsequent editions in which to be reselected, "
        "so counts trend down toward the right regardless of group.",
        fontsize=7.5,
        color="gray",
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_root_works_flag(parser)
    parser.add_argument(
        "--cross-series",
        action="store_true",
        help="Count only cross-series later reselections (default: all later editions).",
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
    comparison = build_comparison(classified)

    count_col = (
        "cross_series_reselection_count" if args.cross_series else "reselection_count"
    )
    plot(classified, count_col, comparison, args.cross_series, args.out)


if __name__ == "__main__":
    main()
