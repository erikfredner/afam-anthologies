"""
vernacular_share.py
-------------------
Bar chart of how much of each anthology edition is editor-designated vernacular
material, measured two ways:

  * % of selections — distinct vernacular works ÷ distinct works in the edition;
  * % of pages — vernacular page count ÷ full book extent.

The two shares diverge: vernacular pieces (spirituals, blues, folktales) are
short, so they make up a far larger share of *selections* than of *pages*. The
metric comes from analysis/vernacular/vernacular_works.py.

Figure: output/vernacular_share.png

Usage:
    uv run python viz/vernacular/vernacular_share.py
    uv run python viz/vernacular/vernacular_share.py --include-excerpts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from afam.cli import add_root_works_flag
from afam.viz_style import AUTHOR_COLOR, DPI, OUTPUT_DIR, WORK_COLOR

sys.path.insert(0, str(Path(__file__).parents[2] / "analysis" / "vernacular"))
from vernacular_works import compute_edition_shares, load_data  # noqa: E402
from afam.vernacular import load_vernacular_ranges  # noqa: E402

OUT_FILE = OUTPUT_DIR / "vernacular_share.png"


def plot_figure(shares, out_path: Path) -> None:
    labels = [f"{e} ({y})" for e, y in zip(shares["edition"], shares["year"])]
    x = np.arange(len(shares))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(
        x - width / 2,
        shares["pct_selections"],
        width,
        color=WORK_COLOR,
        label="% of selections",
    )
    ax.bar(
        x + width / 2,
        shares["pct_pages"],
        width,
        color=AUTHOR_COLOR,
        label="% of pages",
    )

    ax.set_ylabel("Share vernacular (%)")
    ax.set_title(
        "Vernacular material as a share of each anthology edition\n"
        "(editor-designated spirituals, blues, folktales, sermons, ...)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_works_flag(parser)
    args = parser.parse_args()

    df = load_data(only_root_works=args.only_root_works)
    ranges = load_vernacular_ranges()
    shares = compute_edition_shares(df, ranges)
    plot_figure(shares, OUT_FILE)


if __name__ == "__main__":
    main()
