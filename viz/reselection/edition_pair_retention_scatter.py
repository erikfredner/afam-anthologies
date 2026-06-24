"""
edition_pair_retention_scatter.py
---------------------------------
For every ordered pair of African-American Literature anthology editions
(earlier → later), computes the share of the earlier edition's works and
authors that reappear in the later edition, then scatters work retention (x)
against author retention (y) with the 45° identity line.

Each point compares two concrete rosters, counting every work and author at
most once per pair — so the consistent cloud above the diagonal shows the
author-over-work reselection gap holding edition by edition, without the
cumulative ">= k" counting of the survival-curve view. Same-series pairs
(e.g. successive NAAAL editions) are marked separately from cross-series
pairs, since within-series inertia inflates retention for both.

Usage:
    uv run python viz/reselection/edition_pair_retention_scatter.py
    uv run python viz/reselection/edition_pair_retention_scatter.py --include-excerpts
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

from afam.cli import add_root_works_flag
from afam.editions import EDITION_LABELS
from afam.viz_style import OUTPUT_DIR

sys.path.insert(0, str(Path(__file__).parents[2] / "analysis" / "reselection"))
from author_vs_work_debut_reselection import load_data  # noqa: E402

# Render text (including mathtext italics) in Helvetica Now Micro, falling back
# to plain Helvetica then Arial. Keep the family as "sans-serif" so the listed
# fallbacks are consulted; the ←/→ arrows in the axis labels use mathtext.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Helvetica Now Micro",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Helvetica Now Micro"
plt.rcParams["mathtext.it"] = "Helvetica Now Micro:italic"
plt.rcParams["mathtext.bf"] = "Helvetica Now Micro:bold"

OUT_FILE = OUTPUT_DIR / "edition_pair_retention_scatter.png"
# Vector variants emitted alongside the PNG for print/typesetting. PDF (not
# EPS) so the points' alpha transparency is preserved — the EPS/PostScript
# backend renders partially transparent artists opaque.
OUT_FORMATS = ("png", "svg", "pdf")
# 600 dpi keeps the raster (PNG) crisp at print sizes.
SAVE_DPI = 600
GRID_KW = dict(alpha=0.25, linestyle=":")
POINT_COLOR = "black"

# Arrows use mathtext so they render independent of Helvetica's glyph coverage
# (Helvetica lacks ←/→ and per-glyph fallback does not engage for it).
X_AXIS_LABEL = (
    r"$\leftarrow$ fewer works reselected          "
    r"more works reselected $\rightarrow$   (%)"
)
Y_AXIS_LABEL = (
    r"$\leftarrow$ fewer authors reselected          "
    r"more authors reselected $\rightarrow$   (%)"
)

# Corner pairs to call out: the point nearest each (work, author) retention
# target. Final label placement is solved by adjustText, so no manual offsets.
CALLOUT_TARGETS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (1.0, 0.0),
]

# Mathtext special characters that must be escaped to render literally.
_MATHTEXT_SPECIAL = "\\#$%&_{}~^"
# Trailing edition designator (e.g. "ed.2") kept upright, outside the italics.
_ED_SUFFIX = re.compile(r"\s+(ed\.\d+)$")


def _entity_sets(df: pd.DataFrame, id_col: str) -> dict[object, set]:
    rows = df.dropna(subset=[id_col, "edition_id"])
    return rows.groupby("edition_id")[id_col].apply(set).to_dict()


def compute_pair_retention(raw: pd.DataFrame, editions: pd.DataFrame) -> pd.DataFrame:
    """One row per ordered edition pair with work and author retention shares.

    ``editions`` must carry edition_id, anthology_publication_year,
    edition_order, and entry_group (see build_edition_table). Pairs whose
    earlier edition contributes no works or no authors are dropped.
    """
    works_by_ed = _entity_sets(raw, "work_id")
    authors_by_ed = _entity_sets(raw, "author_id")

    ordered = editions.sort_values("edition_order").reset_index(drop=True)
    rows = []
    for i in range(len(ordered)):
        earlier = ordered.iloc[i]
        w_i = works_by_ed.get(earlier["edition_id"], set())
        a_i = authors_by_ed.get(earlier["edition_id"], set())
        if not w_i or not a_i:
            continue
        for j in range(i + 1, len(ordered)):
            later = ordered.iloc[j]
            w_j = works_by_ed.get(later["edition_id"], set())
            a_j = authors_by_ed.get(later["edition_id"], set())
            rows.append(
                {
                    "earlier_edition_id": earlier["edition_id"],
                    "later_edition_id": later["edition_id"],
                    "earlier_year": int(earlier["anthology_publication_year"]),
                    "later_year": int(later["anthology_publication_year"]),
                    "year_gap": int(later["anthology_publication_year"])
                    - int(earlier["anthology_publication_year"]),
                    "same_series": earlier["entry_group"] == later["entry_group"],
                    "work_retention": len(w_i & w_j) / len(w_i),
                    "author_retention": len(a_i & a_j) / len(a_i),
                }
            )
    return pd.DataFrame(rows)


def _above_share(pairs: pd.DataFrame) -> tuple[int, int]:
    above = int((pairs["author_retention"] > pairs["work_retention"]).sum())
    return above, len(pairs)


def nearest_pair_index(pairs: pd.DataFrame, target: tuple[float, float]) -> int:
    """Index of the pair whose (work, author) retention is closest to target."""
    dist2 = (pairs["work_retention"] - target[0]) ** 2 + (
        pairs["author_retention"] - target[1]
    ) ** 2
    return int(dist2.idxmin())


def _italic_label(label: str) -> str:
    """Render an edition label with its anthology title in mathtext italics.

    A trailing edition designator (e.g. "ed.2") is kept upright outside the
    italics so it reads like a citation: $NAAAL$ ed.2.
    """
    match = _ED_SUFFIX.search(label)
    title = label[: match.start()] if match else label
    suffix = f" {match.group(1)}" if match else ""
    # U+2010 renders as an ordinary glyph; a plain "-" becomes a spaced minus.
    title = title.replace("-", "‐")
    escaped = "".join(
        f"\\{ch}" if ch in _MATHTEXT_SPECIAL else ch for ch in title
    ).replace(" ", r"\ ")
    return rf"$\mathit{{{escaped}}}$" + suffix


def _pair_label(row: pd.Series) -> str:
    earlier = EDITION_LABELS.get(int(row["earlier_edition_id"]), "?")
    later = EDITION_LABELS.get(int(row["later_edition_id"]), "?")
    return (
        f"{_italic_label(earlier)} ({row['earlier_year']}) &\n"
        f"{_italic_label(later)} ({row['later_year']})"
    )


def plot(pairs: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    cross = pairs[~pairs["same_series"]]
    same = pairs[pairs["same_series"]]

    ax.scatter(
        100 * cross["work_retention"],
        100 * cross["author_retention"],
        color=POINT_COLOR,
        marker="o",
        s=28,
        alpha=0.55,
        label=f"Cross-series pairs (N={len(cross)})",
        zorder=3,
    )
    ax.scatter(
        100 * same["work_retention"],
        100 * same["author_retention"],
        facecolors="none",
        edgecolors=POINT_COLOR,
        marker="s",
        s=34,
        alpha=0.9,
        label=f"Same-series pairs (N={len(same)})",
        zorder=3,
    )

    lim = 100 * max(pairs["work_retention"].max(), pairs["author_retention"].max())
    lim = min(105.0, lim * 1.08 + 2)
    ax.plot(
        [0, lim],
        [0, lim],
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        zorder=1,
    )

    # Indices of the pairs to label: nearest each corner of the retention
    # square, plus the cross-series pair with the highest work retention.
    callout_idxs = [nearest_pair_index(pairs, target) for target in CALLOUT_TARGETS]
    if not cross.empty:
        callout_idxs.append(int(cross["work_retention"].idxmax()))

    texts = []
    for idx in dict.fromkeys(callout_idxs):  # de-dup, preserve order
        row = pairs.loc[idx]
        texts.append(
            ax.text(
                100 * row["work_retention"],
                100 * row["author_retention"],
                _pair_label(row),
                fontsize=7.5,
                va="center",
                zorder=4,
            )
        )

    ax.set_xlim(-2, lim)
    ax.set_ylim(-2, lim)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=11)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=11)
    ax.set_title("Author vs. work retention across all edition pairs", fontsize=12)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, **GRID_KW)
    ax.set_aspect("equal")

    # Solve label positions so they avoid every point, each other, and the
    # legend, drawing a thin leader back to each point.
    adjust_text(
        texts,
        x=(100 * pairs["work_retention"]).to_numpy(),
        y=(100 * pairs["author_retention"]).to_numpy(),
        objects=[legend],
        ax=ax,
        expand=(1.4, 1.6),
        arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.7, shrinkA=0),
    )

    fig.tight_layout()
    for fmt in OUT_FORMATS:
        path = out.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter author vs work retention for every ordered edition pair."
    )
    add_root_works_flag(parser)
    args = parser.parse_args()

    raw, editions = load_data(args.only_root_works)
    pairs = compute_pair_retention(raw, editions)

    above_all, n_all = _above_share(pairs)
    print(
        f"{n_all} edition pairs; author retention exceeds work retention in "
        f"{above_all} ({100 * above_all / n_all:.1f}%)"
    )

    plot(pairs, OUT_FILE)


if __name__ == "__main__":
    main()
