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

Editions published in the same year (e.g. the three 1968 anthologies) have no
real editorial "earlier -> later" relationship, so neither ordering is picked
via an arbitrary edition_id tiebreak. Both permutations are computed instead
(each edition's own roster as its own denominator) and plotted as a distinct
"same-year" category — excluded from the directional retention statistics
and the labeled callouts, since those describe genuine chronological
reselection. Six specific edition pairs are labeled directly on the plot
(see CALLOUT_PAIRS), rather than choosing points algorithmically.

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

# Colorblind-safe categorical palette (validated: worst-pair CVD deltaE 47,
# well above the 12 target for protan/deutan/tritan). Assigned in fixed order
# and doubled with marker shape, so identity never rides on color alone.
CROSS_SERIES_COLOR = "#2a78d6"  # slot 1, blue
SAME_SERIES_COLOR = "#1baf7a"  # slot 2, aqua
SAME_YEAR_COLOR = "#eda100"  # slot 3, yellow

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

# Specific edition pairs to call out, as (earlier_edition_id, later_edition_id)
# — the direction each anthology's roster is normalized against. Final label
# placement is solved by adjustText, so no manual offsets.
CALLOUT_PAIRS: list[tuple[int, int]] = [
    (21, 20),  # Afro-Am. Writing ed.1 & ed.2
    (13, 18),  # NAAAL ed.3 vs. Wiley Blackwell (both 2014)
    (16, 60),  # NAAAL ed.1 vs. Call & Response (1997/1998)
    (54, 59),  # Cavalcade vs. Black Lit. in America (both 1971)
    (54, 19),  # Cavalcade & New Cavalcade
    (68, 54),  # Negro Caravan & Cavalcade
]

# Mathtext special characters that must be escaped to render literally.
# "&" is deliberately excluded: matplotlib's mathtext has no "\&" escape (it
# rejects the whole expression), but a bare "&" renders fine inside \mathit.
_MATHTEXT_SPECIAL = "\\#$%_{}~^"
# Trailing edition designator (e.g. "ed.2") kept upright, outside the italics.
_ED_SUFFIX = re.compile(r"\s+(ed\.\d+)$")


def _entity_sets(df: pd.DataFrame, id_col: str) -> dict[object, set]:
    rows = df.dropna(subset=[id_col, "edition_id"])
    return rows.groupby("edition_id")[id_col].apply(set).to_dict()


def _pair_row(
    earlier: pd.Series,
    later: pd.Series,
    w_e: set,
    a_e: set,
    w_l: set,
    a_l: set,
    same_year: bool,
) -> dict:
    return {
        "earlier_edition_id": earlier["edition_id"],
        "later_edition_id": later["edition_id"],
        "earlier_year": int(earlier["anthology_publication_year"]),
        "later_year": int(later["anthology_publication_year"]),
        "year_gap": int(later["anthology_publication_year"])
        - int(earlier["anthology_publication_year"]),
        "same_series": earlier["entry_group"] == later["entry_group"],
        "same_year": same_year,
        "work_retention": len(w_e & w_l) / len(w_e),
        "author_retention": len(a_e & a_l) / len(a_e),
    }


def compute_pair_retention(raw: pd.DataFrame, editions: pd.DataFrame) -> pd.DataFrame:
    """One row per ordered edition pair with work and author retention shares.

    ``editions`` must carry edition_id, anthology_publication_year,
    edition_order, and entry_group (see build_edition_table). A pair is
    dropped for a given direction if that direction's earlier edition
    contributes no works or no authors.

    Same-year editions have no genuine "earlier -> later" relationship, so
    both permutations are emitted (each flagged ``same_year=True``, each
    normalized by its own roster) instead of picking one direction via the
    edition_id tiebreak that ``edition_order`` otherwise uses to sequence
    same-year editions.
    """
    works_by_ed = _entity_sets(raw, "work_id")
    authors_by_ed = _entity_sets(raw, "author_id")

    ordered = editions.sort_values("edition_order").reset_index(drop=True)
    rows = []
    for i in range(len(ordered)):
        e1 = ordered.iloc[i]
        w1 = works_by_ed.get(e1["edition_id"], set())
        a1 = authors_by_ed.get(e1["edition_id"], set())
        for j in range(i + 1, len(ordered)):
            e2 = ordered.iloc[j]
            w2 = works_by_ed.get(e2["edition_id"], set())
            a2 = authors_by_ed.get(e2["edition_id"], set())
            same_year = int(e1["anthology_publication_year"]) == int(
                e2["anthology_publication_year"]
            )
            if w1 and a1:
                rows.append(_pair_row(e1, e2, w1, a1, w2, a2, same_year))
            if same_year and w2 and a2:
                rows.append(_pair_row(e2, e1, w2, a2, w1, a1, same_year))
    return pd.DataFrame(rows)


def _above_share(pairs: pd.DataFrame) -> tuple[int, int]:
    above = int((pairs["author_retention"] > pairs["work_retention"]).sum())
    return above, len(pairs)


def callout_pair_index(pairs: pd.DataFrame, edition_ids: tuple[int, int]) -> int:
    """Index of the row for a specific (earlier, later) edition-id pair.

    Falls back to the reverse direction (relevant for same-year pairs, which
    carry both permutations) before raising if neither direction is present.
    """
    earlier, later = edition_ids
    match = pairs[
        (pairs["earlier_edition_id"] == earlier) & (pairs["later_edition_id"] == later)
    ]
    if match.empty:
        match = pairs[
            (pairs["earlier_edition_id"] == later)
            & (pairs["later_edition_id"] == earlier)
        ]
    if match.empty:
        raise ValueError(f"No pair found for edition ids {edition_ids}")
    return int(match.index[0])


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

    # Same-year pairs carry both permutations (see compute_pair_retention) and
    # have no genuine chronological direction, so they get their own marker
    # and are kept out of the cross-/same-series chronological groups.
    chrono = pairs[~pairs["same_year"]]
    cross = chrono[~chrono["same_series"]]
    same = chrono[chrono["same_series"]]
    same_year = pairs[pairs["same_year"]]

    ax.scatter(
        100 * cross["work_retention"],
        100 * cross["author_retention"],
        color=CROSS_SERIES_COLOR,
        marker="o",
        s=28,
        alpha=0.65,
        label=f"Cross-series pairs (N={len(cross)})",
        zorder=3,
    )
    ax.scatter(
        100 * same["work_retention"],
        100 * same["author_retention"],
        facecolors="none",
        edgecolors=SAME_SERIES_COLOR,
        linewidths=1.4,
        marker="s",
        s=34,
        alpha=0.95,
        label=f"Same-series pairs (N={len(same)})",
        zorder=3,
    )
    ax.scatter(
        100 * same_year["work_retention"],
        100 * same_year["author_retention"],
        facecolors="none",
        edgecolors=SAME_YEAR_COLOR,
        linewidths=1.4,
        marker="^",
        s=40,
        alpha=0.9,
        linestyle="--",
        label=f"Same-year pairs, both directions (N={len(same_year)})",
        zorder=2,
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

    # Indices of the specific edition pairs to label (see CALLOUT_PAIRS).
    callout_idxs = [callout_pair_index(pairs, ids) for ids in CALLOUT_PAIRS]

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

    chrono = pairs[~pairs["same_year"]]
    above_all, n_all = _above_share(chrono)
    print(
        f"{n_all} chronological edition pairs; author retention exceeds work "
        f"retention in {above_all} ({100 * above_all / n_all:.1f}%)"
    )
    n_same_year = int(pairs["same_year"].sum())
    if n_same_year:
        print(
            f"{n_same_year} same-year pairs (both directions) plotted "
            "separately; excluded from the stat above."
        )

    plot(pairs, OUT_FILE)


if __name__ == "__main__":
    main()
