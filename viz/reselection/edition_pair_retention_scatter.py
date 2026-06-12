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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from afam.cli import add_root_works_flag
from afam.editions import EDITION_LABELS
from afam.viz_style import AUTHOR_COLOR, OUTPUT_DIR

sys.path.insert(0, str(Path(__file__).parents[2] / "analysis" / "reselection"))
from author_vs_work_debut_reselection import load_data  # noqa: E402

OUT_FILE = OUTPUT_DIR / "edition_pair_retention_scatter.png"
GRID_KW = dict(alpha=0.25, linestyle=":")

# Corner pairs to call out: nearest point to each (work, author) retention
# target, with the text offset (points) used to keep labels out of the cloud.
CALLOUT_TARGETS: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((0.0, 0.0), (30, -30)),
    ((0.0, 1.0), (18, 8)),
    ((1.0, 1.0), (-12, 8)),
    ((1.0, 0.0), (12, -22)),
]


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


def _pair_label(row: pd.Series) -> str:
    earlier = EDITION_LABELS.get(int(row["earlier_edition_id"]), "?")
    later = EDITION_LABELS.get(int(row["later_edition_id"]), "?")
    return f"{earlier} ({row['earlier_year']}) →\n{later} ({row['later_year']})"


def plot(pairs: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    cross = pairs[~pairs["same_series"]]
    same = pairs[pairs["same_series"]]

    ax.scatter(
        100 * cross["work_retention"],
        100 * cross["author_retention"],
        color=AUTHOR_COLOR,
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
        edgecolors=AUTHOR_COLOR,
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
    ax.text(
        lim * 0.97,
        lim * 0.97,
        "authors = works",
        fontsize=8,
        color="gray",
        ha="right",
        va="bottom",
        rotation=45,
        rotation_mode="anchor",
    )

    seen: set[int] = set()
    for target, offset in CALLOUT_TARGETS:
        idx = nearest_pair_index(pairs, target)
        if idx in seen:
            continue
        seen.add(idx)
        row = pairs.loc[idx]
        ax.annotate(
            _pair_label(row),
            (100 * row["work_retention"], 100 * row["author_retention"]),
            textcoords="offset points",
            xytext=offset,
            fontsize=7.5,
            ha="left" if offset[0] >= 0 else "right",
            va="center",
            arrowprops=dict(
                arrowstyle="-", color="gray", linewidth=0.7, shrinkA=0, shrinkB=4
            ),
            zorder=4,
        )

    above_all, n_all = _above_share(pairs)
    above_cross, n_cross = _above_share(cross)
    ax.text(
        0.03,
        0.97,
        f"Authors retained at a higher rate than works in\n"
        f"{above_all} of {n_all} edition pairs ({100 * above_all / n_all:.1f}%); "
        f"{above_cross} of {n_cross} cross-series ({100 * above_cross / n_cross:.1f}%)",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )

    ax.set_xlim(-2, lim)
    ax.set_ylim(-2, lim)
    ax.set_xlabel("% of earlier edition's works selected by later edition", fontsize=11)
    ax.set_ylabel(
        "% of earlier edition's authors selected by later edition", fontsize=11
    )
    ax.set_title("Author vs. work retention across all edition pairs", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, **GRID_KW)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
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
