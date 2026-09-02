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
"same-year" category, excluded from the directional retention statistic
printed to stdout, since that describes genuine chronological reselection.
Seven specific edition pairs are labeled directly on the plot (see
CALLOUT_PAIRS), rather than choosing points algorithmically. Three of the
labels are same-year comparisons: both permutations of the 2014 NAAAL ed.3 /
Wiley Blackwell pair are labeled, so the gap between reading that pair in
either direction is visible; the 1971 Cavalcade / Black Lit. in America pair
is labeled in one direction only, its twin sitting unlabeled nearby. Labels
name the direction as "denominator -> comparison", since for a same-year pair
that is the only thing separating the two points.

A variant that drops works with no author on record (anonymous spirituals,
folk material, unsigned periodical pieces) is available via
--authored-works-only, written to a separate output file. Those works can
never contribute to author retention — they have no author to reselect — so
including them puts entities in the work denominator that are structurally
absent from the author denominator. The variant is the sensitivity check on
whether the author-over-work gap survives restricting both axes to the same
authored material.

Alongside the static figures, an interactive plotly version is written to
docs/ (tracked, served by GitHub Pages) where hovering any point names both
anthologies, the direction, and the counts behind each share. That makes every
point identifiable rather than just the seven callouts, so the HTML carries no
annotations at all. Only the default scope is published.

Usage:
    uv run python viz/reselection/edition_pair_retention_scatter.py
    uv run python viz/reselection/edition_pair_retention_scatter.py --include-excerpts
    uv run python viz/reselection/edition_pair_retention_scatter.py --authored-works-only
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from adjustText import adjust_text
from matplotlib.ticker import PercentFormatter

from afam import DOCS_DIR
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
# --authored-works-only writes here instead, so both variants coexist.
OUT_FILE_AUTHORED = OUTPUT_DIR / "edition_pair_retention_scatter_authored_only.png"
# Vector variants emitted alongside the PNG for print/typesetting. PDF (not
# EPS) so the points' alpha transparency is preserved — the EPS/PostScript
# backend renders partially transparent artists opaque.
OUT_FORMATS = ("png", "svg", "pdf")
# 600 dpi keeps the raster (PNG) crisp at print sizes.
SAVE_DPI = 600
# The interactive sibling lives in docs/ (tracked, served by GitHub Pages) rather
# than the gitignored output/. Only the default scope is published: the
# --authored-works-only variant is a sensitivity check on the static figure, not
# something a reader browses.
OUT_HTML = DOCS_DIR / "edition_pair_retention_scatter.html"
GRID_KW = dict(alpha=0.25, linestyle=":")

# Colorblind-safe categorical palette (validated: worst-pair CVD deltaE 47,
# well above the 12 target for protan/deutan/tritan). Assigned in fixed order
# and doubled with marker shape, so identity never rides on color alone.
CROSS_SERIES_COLOR = "#2a78d6"  # slot 1, blue
SAME_SERIES_COLOR = "#1baf7a"  # slot 2, aqua
SAME_YEAR_COLOR = "#eda100"  # slot 3, yellow

# Category is encoded by color + shape only: every series uses the same solid
# fill, marker area, and opacity, so fill/outline/size carry no meaning. The
# hairline white edge is applied identically to all three and exists only to
# keep opaque markers separable where the cloud is dense.
MARKER_SIZE = 32
MARKER_KW = dict(alpha=1.0, linewidths=0.4, edgecolors="white")
CROSS_SERIES_MARKER = "o"
SAME_SERIES_MARKER = "s"
SAME_YEAR_MARKER = "^"

# Arrows use mathtext so they render independent of Helvetica's glyph coverage
# (Helvetica lacks ←/→ and per-glyph fallback does not engage for it).
X_AXIS_LABEL = (
    r"$\leftarrow$ fewer works reselected          "
    r"more works reselected $\rightarrow$"
)
Y_AXIS_LABEL = (
    r"$\leftarrow$ fewer authors reselected          "
    r"more authors reselected $\rightarrow$"
)
# Plotly renders axis titles as HTML, where the arrows are ordinary glyphs and
# the mathtext escaping above would render literally.
X_AXIS_LABEL_HTML = "← fewer works reselected          more works reselected →"
Y_AXIS_LABEL_HTML = "← fewer authors reselected          more authors reselected →"

# Specific edition pairs to call out, as (earlier_edition_id, later_edition_id)
# — the direction each anthology's roster is normalized against. Final label
# placement is solved by adjustText, so no manual offsets.
CALLOUT_PAIRS: list[tuple[int, int]] = [
    (21, 20),  # Afro-Am. Writing ed.1 -> ed.2
    (13, 18),  # NAAAL ed.3 -> Wiley Blackwell (both 2014)
    (18, 13),  # Wiley Blackwell -> NAAAL ed.3 (the reverse permutation)
    (16, 60),  # NAAAL ed.1 -> Call & Response (1997/1998)
    (54, 59),  # Cavalcade -> Black Lit. in America (both 1971)
    (54, 19),  # Cavalcade -> New Cavalcade
    (68, 54),  # Negro Caravan -> Cavalcade
]

# Mathtext special characters that must be escaped to render literally.
# "&" is deliberately excluded: matplotlib's mathtext has no "\&" escape (it
# rejects the whole expression), but a bare "&" renders fine inside \mathit.
_MATHTEXT_SPECIAL = "\\#$%_{}~^"
# Trailing edition designator (e.g. "ed.2") kept upright, outside the italics.
_ED_SUFFIX = re.compile(r"\s+(ed\.\d+)$")


def drop_unauthored_works(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop appearances of works with no author on record.

    ``raw`` carries one row per (work, author, edition) from a LEFT JOIN onto
    the authors tables, so a work with no author appears as a single row with a
    null ``author_id``. Dropping null-author rows therefore removes exactly the
    unauthored works and leaves every authored work's rows intact.
    """
    return raw[raw["author_id"].notna()].copy()


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
        # The ratios' numerators and denominators, kept so the interactive
        # tooltip can report the counts a retention share is computed from.
        "earlier_n_works": len(w_e),
        "earlier_n_authors": len(a_e),
        "works_retained": len(w_e & w_l),
        "authors_retained": len(a_e & a_l),
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


def _axis_limit(pairs: pd.DataFrame) -> float:
    """Upper bound (in percentage points) shared by both renderers.

    Kept as one function so the static figure and its interactive sibling
    cannot drift onto different scales.
    """
    lim = 100 * max(pairs["work_retention"].max(), pairs["author_retention"].max())
    return min(105.0, lim * 1.08 + 2)


def _above_share(pairs: pd.DataFrame) -> tuple[int, int]:
    above = int((pairs["author_retention"] > pairs["work_retention"]).sum())
    return above, len(pairs)


def page_summary(pairs: pd.DataFrame) -> dict:
    """The counts the published page's prose quotes back to the reader.

    Kept out of plot_html so the numbers in the text are derived from the same
    frame the markers are, and can be checked without invoking plotly.
    """
    chrono = pairs[~pairs["same_year"]]
    above, n_chrono = _above_share(chrono)
    return {
        "n_points": len(pairs),
        "n_chrono": n_chrono,
        "n_cross_series": int((~chrono["same_series"]).sum()),
        "n_same_series": int(chrono["same_series"].sum()),
        "n_same_year": int(pairs["same_year"].sum()),
        "above": above,
        "above_pct": 100 * above / n_chrono if n_chrono else float("nan"),
        # Chronological pairs that do *not* sit above the identity line. Named
        # individually on the page rather than left as a residual, since there
        # are currently one or two of them.
        "exceptions": chrono[chrono["author_retention"] <= chrono["work_retention"]],
    }


def _exception_sentence(exceptions: pd.DataFrame) -> str:
    """Name the chronological pairs where authors were not the better-kept set."""
    if exceptions.empty:
        return "No chronological pair falls on or below it."
    named = [
        f"{_html_label(EDITION_LABELS.get(int(row['earlier_edition_id']), '?'))}"
        f" ({int(row['earlier_year'])}) → "
        f"{_html_label(EDITION_LABELS.get(int(row['later_edition_id']), '?'))}"
        f" ({int(row['later_year'])}), which kept"
        f" {100 * row['work_retention']:.0f}% of the works against"
        f" {100 * row['author_retention']:.0f}% of the authors"
        for _, row in exceptions.iterrows()
    ]
    if len(named) == 1:
        return f"The one exception is {named[0]}."
    return "The exceptions are " + "; ".join(named) + "."


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
    """Label a called-out pair as "denominator -> comparison".

    The arrow (not "&") because both permutations of a same-year pair can be
    labeled at once: the two points differ only in which roster is the
    denominator, so a symmetric conjunction would leave them indistinguishable.
    """
    earlier = EDITION_LABELS.get(int(row["earlier_edition_id"]), "?")
    later = EDITION_LABELS.get(int(row["later_edition_id"]), "?")
    return (
        f"{_italic_label(earlier)} ({row['earlier_year']}) "
        f"$\\rightarrow$\n"
        f"{_italic_label(later)} ({row['later_year']})"
    )


def _html_label(label: str) -> str:
    """The HTML counterpart of _italic_label, for plotly hover text.

    Same convention — anthology title italicized, a trailing edition designator
    kept upright — expressed in the small HTML subset plotly renders. The label
    is escaped first: "Call & Response" must not reach the browser as a bare
    ampersand.
    """
    match = _ED_SUFFIX.search(label)
    title = label[: match.start()] if match else label
    suffix = f" {match.group(1)}" if match else ""
    return f"<i>{html.escape(title)}</i>{suffix}"


def _pair_hover(row: pd.Series) -> str:
    """Tooltip naming the two anthologies a point compares, and in which order.

    Direction is spelled out as denominator → comparison for the same reason
    _pair_label uses an arrow: a same-year pair contributes both permutations,
    and the two points differ only in whose roster is the denominator.
    """
    earlier = _html_label(EDITION_LABELS.get(int(row["earlier_edition_id"]), "?"))
    later = _html_label(EDITION_LABELS.get(int(row["later_edition_id"]), "?"))
    if row["same_year"]:
        category = "Same-year pair"
    elif row["same_series"]:
        category = "Same-series pair"
    else:
        category = "Cross-series pair"
    gap = int(row["year_gap"])
    gap_text = "same year" if gap == 0 else f"{gap}-year gap"
    return (
        f"<b>{earlier} ({int(row['earlier_year'])})"
        f" → {later} ({int(row['later_year'])})</b><br>"
        f"{category} · {gap_text}<br><br>"
        f"Works reselected: {100 * row['work_retention']:.1f}%"
        f" ({int(row['works_retained'])} of {int(row['earlier_n_works'])})<br>"
        f"Authors reselected: {100 * row['author_retention']:.1f}%"
        f" ({int(row['authors_retained'])} of {int(row['earlier_n_authors'])})"
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

    # Rarer categories are drawn last so the 300-odd opaque cross-series points
    # cannot bury them; z-order carries no meaning beyond that. Handles are
    # kept so the legend can stay in the logical order regardless.
    h_cross = ax.scatter(
        100 * cross["work_retention"],
        100 * cross["author_retention"],
        color=CROSS_SERIES_COLOR,
        marker=CROSS_SERIES_MARKER,
        s=MARKER_SIZE,
        label=f"Cross-series pairs (N={len(cross)})",
        zorder=2,
        **MARKER_KW,
    )
    h_same_year = ax.scatter(
        100 * same_year["work_retention"],
        100 * same_year["author_retention"],
        color=SAME_YEAR_COLOR,
        marker=SAME_YEAR_MARKER,
        s=MARKER_SIZE,
        label=f"Same-year pairs, both directions (N={len(same_year)})",
        zorder=3,
        **MARKER_KW,
    )
    h_same = ax.scatter(
        100 * same["work_retention"],
        100 * same["author_retention"],
        color=SAME_SERIES_COLOR,
        marker=SAME_SERIES_MARKER,
        s=MARKER_SIZE,
        label=f"Same-series pairs (N={len(same)})",
        zorder=4,
        **MARKER_KW,
    )

    lim = _axis_limit(pairs)
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
    # Units live on the tick labels ("100%") rather than in the axis titles.
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    legend = ax.legend(
        handles=[h_cross, h_same, h_same_year],
        frameon=False,
        fontsize=9,
        loc="lower right",
    )
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


# The published page's styling mirrors docs/index.html so the site reads as one
# thing; the additions are a wider body (the square chart needs more than the
# 42rem prose measure) and the .figure panel, which keeps its own white ground
# because plotly's "plotly_white" template does not follow prefers-color-scheme.
PAGE_CSS = """  :root { color-scheme: light dark; }
  body {
    max-width: 54rem;
    margin: 0 auto;
    padding: 3rem 1.25rem 4rem;
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
  }
  .prose { max-width: 42rem; margin: 0 auto; }
  h1 { font-size: 1.6rem; line-height: 1.25; margin: 0 0 0.75rem; }
  h2 { font-size: 1.1rem; margin: 2.5rem 0 0.5rem; }
  p { margin: 0 0 1rem; }
  .lede { color: #555; }
  a { color: #2a78d6; }
  .note { font-size: 0.9rem; color: #555; }
  .figure {
    max-width: 900px;
    margin: 2rem auto;
    padding: 0.5rem;
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
  }
  @media (prefers-color-scheme: dark) {
    .lede, .note { color: #aaa; }
    a { color: #6ea8e8; }
    .figure { border-color: #333; }
  }"""

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Author vs. work retention, by edition pair</title>
<style>
{css}
</style>
</head>
<body>

<div class="prose">

<h1>Author vs. work retention, by edition pair</h1>

<p class="lede">
  Every ordered pair of the twenty-six anthologies, earlier to later. A point's
  horizontal position is the share of the earlier anthology's <b>works</b> that
  reappear in the later one; its vertical position is the share of its
  <b>authors</b>. Hover a point to name both anthologies, the direction, and the
  counts behind each share.
</p>

</div>

<div class="figure">
{figure}
</div>

<div class="prose">

<h2>How to read it</h2>

<p>
  The dashed line is parity. A point above it means the later anthology kept more
  of the earlier one's authors than of its works &mdash; it reprinted the writer,
  but chose something else by them. {above} of the {n_chrono} chronological pairs
  sit above the line ({above_pct:.1f}%). {exception_sentence}
</p>

<p>
  Each work and each author is counted at most once per pair, and both shares are
  normalized against the <i>earlier</i> anthology's roster. A &rarr; B is
  therefore a different point from B &rarr; A, and the direction is the first
  thing the tooltip names.
</p>

<p>
  Marker shape separates three kinds of pair.
  <b>Cross-series pairs</b> ({n_cross_series}) compare anthologies from different
  series. <b>Same-series pairs</b> ({n_same_series}) compare successive editions of
  one anthology, such as the four editions of <i>The Norton Anthology of African
  American Literature</i>; inertia within a series lifts both axes, so these are
  marked apart rather than mixed into the cloud. <b>Same-year pairs</b>
  ({n_same_year} points) compare anthologies published in the same year &mdash; the
  three of 1968, for instance &mdash; which have no genuine earlier-and-later
  relationship: both directions are plotted, and neither is counted in the figure
  above.
</p>

<p class="note">
  <a href="index.html">All figures and tables</a>
</p>

</div>

</body>
</html>
"""

# Fixed rather than a fresh UUID per run: this page is tracked, and a random div
# id would rewrite the whole file on every regeneration.
HTML_DIV_ID = "retention-scatter"


def plot_html(pairs: pd.DataFrame, out: Path) -> None:
    """Interactive sibling of plot(), with every point identifiable on hover.

    The static figure can only name the seven CALLOUT_PAIRS, leaving the other
    ~330 points anonymous. Here the callouts are dropped entirely: hover names
    both anthologies for any point, which is what the annotations existed to do.

    The chart is written into a page carrying the reading context a bare plotly
    dump cannot: what a point is, what the diagonal means, and why same-series
    and same-year pairs are marked apart. The prose quotes page_summary(), so a
    regeneration cannot leave the text describing an older frame.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    chrono = pairs[~pairs["same_year"]]
    groups = [
        (
            chrono[~chrono["same_series"]],
            "Cross-series pairs",
            CROSS_SERIES_COLOR,
            "circle",
        ),
        (
            pairs[pairs["same_year"]],
            "Same-year pairs, both directions",
            SAME_YEAR_COLOR,
            "triangle-up",
        ),
        (
            chrono[chrono["same_series"]],
            "Same-series pairs",
            SAME_SERIES_COLOR,
            "square",
        ),
    ]

    lim = _axis_limit(pairs)
    fig = go.Figure()
    # Identity line added first so the markers sit on top of it.
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=lim,
        y1=lim,
        line=dict(color="gray", width=1, dash="dash"),
    )
    # Same order as plot(): rarer categories drawn last so the cross-series
    # cloud cannot bury them.
    for group, label, color, symbol in groups:
        fig.add_trace(
            go.Scatter(
                x=100 * group["work_retention"],
                y=100 * group["author_retention"],
                mode="markers",
                name=f"{label} (N={len(group)})",
                marker=dict(
                    color=color,
                    symbol=symbol,
                    size=9,
                    line=dict(width=0.5, color="white"),
                ),
                text=[_pair_hover(row) for _, row in group.iterrows()],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_xaxes(
        title_text=X_AXIS_LABEL_HTML,
        range=[-2, lim],
        ticksuffix="%",
        gridcolor="#e6e6e6",
    )
    fig.update_yaxes(
        title_text=Y_AXIS_LABEL_HTML,
        range=[-2, lim],
        ticksuffix="%",
        gridcolor="#e6e6e6",
        # Plotly's equivalent of matplotlib's set_aspect("equal"): the diagonal
        # must read as 45° for the above/below-the-line comparison to hold.
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        template="plotly_white",
        # No fixed pixel width: the page sizes the chart, so it stays readable
        # on a narrow screen. Height is fixed because scaleanchor squares the
        # axes, and a square chart in a short box would waste the width.
        autosize=True,
        hovermode="closest",
        legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
        margin=dict(l=80, r=30, t=30, b=70),
    )
    # CDN rather than an inlined bundle: this file is committed, and inlining
    # plotly.js would add ~3 MB to the repository on every regeneration.
    figure = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        default_width="100%",
        default_height="820px",
        div_id=HTML_DIV_ID,
    )
    summary = page_summary(pairs)
    out.write_text(
        PAGE_TEMPLATE.format(
            css=PAGE_CSS,
            figure=figure,
            exception_sentence=_exception_sentence(summary["exceptions"]),
            **{k: v for k, v in summary.items() if k != "exceptions"},
        ),
        encoding="utf-8",
    )
    print(f"Saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter author vs work retention for every ordered edition pair."
    )
    add_root_works_flag(parser)
    parser.add_argument(
        "--authored-works-only",
        action="store_true",
        help=(
            "Exclude works with no author on record, and write the variant "
            f"figure to {OUT_FILE_AUTHORED.name}."
        ),
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=OUT_HTML,
        help=(
            "Where to write the interactive HTML version "
            f"(default: {OUT_HTML}). Not written under --authored-works-only."
        ),
    )
    args = parser.parse_args()

    raw, editions = load_data(args.only_root_works)
    if args.authored_works_only:
        before = raw["work_id"].nunique()
        raw = drop_unauthored_works(raw)
        after = raw["work_id"].nunique()
        print(
            f"Authored-works-only scope: dropped {before - after} of {before} "
            f"works with no author on record ({after} remain)."
        )
    out_file = OUT_FILE_AUTHORED if args.authored_works_only else OUT_FILE

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

    plot(pairs, out_file)
    if args.authored_works_only:
        print("Interactive HTML not written: published for the default scope only.")
    else:
        plot_html(pairs, args.html_out)


if __name__ == "__main__":
    main()
