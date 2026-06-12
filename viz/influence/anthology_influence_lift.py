"""
anthology_influence_lift.py
---------------------------
Figure ranking anthology editions by their influence lift on subsequent
editions: how much more often each edition's selections appear in later
editions than expected from the corpus baseline (see
analysis/influence/anthology_influence.py for the metric).

Two panels share the y-axis: left = all selections ("taste confirmation"),
right = debuts only ("agenda setting"). Each row is an edition; a horizontal
dumbbell connects its author lift and work lift. The x-axis is log-scaled
(lift is a ratio, so 0.5 and 2.0 sit symmetrically around the dashed lift=1
reference). Markers are filled when the binomial p-value is < 0.05, hollow
otherwise. The chronologically last edition is omitted: it has no subsequent
editions.

Figure: output/anthology_influence_lift.png

Usage:
    uv run python viz/influence/anthology_influence_lift.py
    uv run python viz/influence/anthology_influence_lift.py --exclude-within-series
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from afam.cli import add_root_works_flag
from afam.viz_style import AUTHOR_COLOR, DPI, OUTPUT_DIR, WORK_COLOR

sys.path.insert(0, str(Path(__file__).parents[2] / "analysis" / "influence"))
from anthology_influence import compute_all_variants, load_data  # noqa: E402

OUT_FILE = OUTPUT_DIR / "anthology_influence_lift.png"
ALPHA = 0.05


def build_panel_frames(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot the long results into one frame per variant, ranked for plotting.

    Rows are editions with valid lifts (drops the last edition, which has no
    subsequent editions); ordering is by all-selections author lift,
    descending, shared by both panels.
    """
    frames: dict[str, pd.DataFrame] = {}
    for variant in ["all", "debut"]:
        block = results[results["variant"] == variant]
        wide = block.pivot(
            index=["edition_id", "label", "year"],
            columns="entity",
            values=["lift", "p_value"],
        )
        wide.columns = [f"{stat}_{entity}" for stat, entity in wide.columns]
        frames[variant] = wide.reset_index()

    order_source = frames["all"].dropna(subset=["lift_authors"])
    ranked_ids = order_source.sort_values("lift_authors")["edition_id"].tolist()

    for variant, frame in frames.items():
        frame = frame[frame["edition_id"].isin(ranked_ids)].copy()
        frame["rank"] = frame["edition_id"].map(
            {eid: i for i, eid in enumerate(ranked_ids)}
        )
        frames[variant] = frame.sort_values("rank").reset_index(drop=True)
    return frames


def plot_panel(ax: plt.Axes, frame: pd.DataFrame, title: str, floor: float) -> None:
    for _, row in frame.iterrows():
        y = row["rank"]
        la, lw_ = row["lift_authors"], row["lift_works"]
        if pd.notna(la) and pd.notna(lw_):
            ax.plot(
                [max(la, floor), max(lw_, floor)],
                [y, y],
                color="grey",
                linewidth=1.0,
                zorder=1,
            )
        for lift, p, color in [
            (la, row["p_value_authors"], AUTHOR_COLOR),
            (lw_, row["p_value_works"], WORK_COLOR),
        ]:
            if pd.isna(lift):
                continue
            significant = pd.notna(p) and p < ALPHA
            # Zero lifts can't sit on a log axis: clip to the floor and mark
            # with a left-pointing triangle.
            ax.scatter(
                max(lift, floor),
                y,
                s=45,
                marker="<" if lift < floor else "o",
                color=color if significant else "white",
                edgecolors=color,
                linewidths=1.4,
                zorder=3,
            )

    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lift (observed ÷ expected pickup rate, log scale)", fontsize=9)
    ax.grid(axis="x", alpha=0.3, zorder=0)


def plot_figure(frames: dict[str, pd.DataFrame], out_path: Path) -> None:
    base = frames["all"]
    labels = [f"{row['label']} ({int(row['year'])})" for _, row in base.iterrows()]

    fig, (ax_all, ax_deb) = plt.subplots(
        1, 2, figsize=(12, 9), sharey=True, sharex=True
    )

    lifts = pd.concat(
        [f[["lift_authors", "lift_works"]] for f in frames.values()]
    ).stack()
    floor = float(lifts[lifts > 0].min()) * 0.6

    plot_panel(ax_all, frames["all"], "All selections\n(taste confirmation)", floor)
    plot_panel(ax_deb, frames["debut"], "Debuts only\n(agenda setting)", floor)

    ax_all.set_yticks(base["rank"])
    ax_all.set_yticklabels(labels, fontsize=8)
    ax_all.set_ylim(-0.7, len(base) - 0.3)

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color=AUTHOR_COLOR, label="Authors"
        ),
        plt.Line2D([], [], marker="o", linestyle="", color=WORK_COLOR, label="Works"),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            color="grey",
            label=f"Hollow: p ≥ {ALPHA}",
        ),
        plt.Line2D(
            [],
            [],
            marker="<",
            linestyle="",
            color="grey",
            label="◄ lift = 0 (clipped)",
        ),
    ]
    ax_deb.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)

    fig.suptitle(
        "Anthology influence on subsequent anthologies\n"
        "(forward pickup rate of each edition's selections ÷ corpus baseline)",
        fontsize=12,
    )
    fig.text(
        0.99,
        0.005,
        "Chronologically last edition omitted: no subsequent editions.",
        ha="right",
        fontsize=7,
        color="grey",
    )
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_works_flag(parser)
    parser.add_argument(
        "--exclude-within-series",
        action="store_true",
        help=(
            "Skip source/target pairs within the same anthology series and "
            "drop a series' own prior selections from its baseline pool."
        ),
    )
    parser.add_argument(
        "--stratify-prior-count",
        action="store_true",
        help=(
            "Standardize expectations by per-item prior selection count "
            "(popularity-adjusted baseline; conservative lower bound on influence)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help=f"Output figure path (default: {OUT_FILE})",
    )
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        suffix = "_stratified" if args.stratify_prior_count else ""
        out_path = OUTPUT_DIR / f"anthology_influence_lift{suffix}.png"

    df = load_data(only_root_works=args.only_root_works)
    results = compute_all_variants(
        df,
        exclude_within_series=args.exclude_within_series,
        stratify_prior_count=args.stratify_prior_count,
    )
    frames = build_panel_frames(results)
    plot_figure(frames, out_path)


if __name__ == "__main__":
    main()
