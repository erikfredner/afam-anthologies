"""author_page_share_reselection.py
--------------------------------------
For each author in the AFAM-tagged anthology corpus, combine two signals:

  • reselection_prob — share of editions strictly after their debut in which
    they reappear (selections ÷ available subsequent opportunities)
  • page_share      — pages allocated to the author in each edition they
    appear in, expressed as a fraction of that edition's total pages

Two figures land in output/:

  1. author_page_share_reselection.png         (chart 1)
     One boxplot per author at x = reselection_prob.
     y = author's share of edition pages.

  2. author_page_share_reselection_by_form.png (chart 2)
     One boxplot per (author, form) entity (so Hughes-poetry and
     Hughes-nonfiction become distinct boxes at the same x).
     y = (author's pages in form within edition) ÷ (form's pages in edition).

Page splitting conventions:
  - Multi-author works split pages equally among authors.
  - Multi-form works split pages equally among forms (chart 2 only).
  - Works without a form are labeled "Unknown".

CLI:
    uv run python viz/reselection/author_page_share_reselection.py
    uv run python viz/reselection/author_page_share_reselection.py --include-excerpts
    uv run python viz/reselection/author_page_share_reselection.py --save-csv
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afam import DATA_DIR
from afam.cli import add_root_works_flag, add_save_csv_flag
from afam.db import query as query_db
from afam.sql import query_path
from afam.viz_style import DPI, OUTPUT_DIR

CHART1_PNG = "author_page_share_reselection.png"
CHART2_PNG = "author_page_share_reselection_by_form.png"

PER_EDITION_CSV = "author_page_share_reselection_per_edition.csv"
PER_FORM_CSV = "author_page_share_reselection_per_form_edition.csv"
SUMMARY_CSV = "author_page_share_reselection_summary.csv"

# Highlight the top-N authors (by total edition appearances) with text labels
TOP_N_LABEL = 6
RNG_SEED = 0

# Author filters: keep only durable, repeatedly-anthologized authors
MIN_RESELECTION_PROB = 0.5
MIN_APPEARANCES = 10
X_AXIS_MIN = 0.48
X_AXIS_MAX = 1.02


# ── Page spans (verbatim from analysis/predictability/predictability_over_time.py) ─


def compute_work_volume_spans(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (work_id, volume_id) with the work's page span in that volume.

    Span = toc_next - toc_page when both present (and positive).  When toc_next
    is NULL, falls back to last_toc_page - toc_page + 1 (treating the work as
    the last entry in its volume).
    """
    spans = raw[
        ["work_id", "volume_id", "toc_page", "toc_next", "last_toc_page"]
    ].drop_duplicates(["work_id", "volume_id"])
    span = spans["toc_next"] - spans["toc_page"]
    fallback = spans["last_toc_page"] - spans["toc_page"] + 1
    span = span.where(spans["toc_next"].notna(), fallback)
    span = span.clip(lower=0).fillna(0).astype(int)
    return pd.DataFrame(
        {
            "work_id": spans["work_id"].values,
            "volume_id": spans["volume_id"].values,
            "span": span.values,
        }
    )


# ── Per-(work, edition) span (sum across volumes of the same edition) ─────────


def build_work_per_edition(
    raw: pd.DataFrame, work_volume_spans: pd.DataFrame
) -> pd.DataFrame:
    """One row per (work_id, edition_id) with year and summed page span."""
    base = raw[["work_id", "volume_id", "edition_id", "edition_year"]].drop_duplicates(
        ["work_id", "volume_id", "edition_id"]
    )
    base = base.merge(work_volume_spans, on=["work_id", "volume_id"], how="left")
    grouped = (
        base.groupby(["work_id", "edition_id", "edition_year"], dropna=False)
        .agg(span=("span", "sum"))
        .reset_index()
    )
    return grouped


# ── Per-author and per-(author, form) page allocations ────────────────────────


def build_author_form_counts(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per work_id with author_count and form_count.

    Used to split page spans equally among co-authors and among co-forms.
    Works with no form get form_count=1 and a synthetic form_name='Unknown'.
    """
    authors_per_work = (
        raw.dropna(subset=["author_id"])
        .groupby("work_id")["author_id"]
        .nunique()
        .rename("author_count")
    )
    forms_per_work = (
        raw.dropna(subset=["form_name"])
        .groupby("work_id")["form_name"]
        .nunique()
        .rename("form_count")
    )
    all_works = raw[["work_id"]].drop_duplicates().set_index("work_id")
    counts = all_works.join(authors_per_work).join(forms_per_work).reset_index()
    counts["author_count"] = counts["author_count"].fillna(0).astype(int)
    counts["form_count"] = counts["form_count"].fillna(1).astype(int).clip(lower=1)
    return counts


def build_author_per_edition(
    raw: pd.DataFrame,
    work_per_edition: pd.DataFrame,
    counts: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (author_id, edition_id) with year, author_name, pages.

    Each work's span is divided equally among its co-authors before being
    summed at the author level.
    """
    author_links = raw.dropna(subset=["author_id"])[
        ["work_id", "author_id", "author_name"]
    ].drop_duplicates(["work_id", "author_id"])
    author_links = author_links.merge(counts[["work_id", "author_count"]], on="work_id")
    merged = author_links.merge(work_per_edition, on="work_id")
    merged["author_work_pages"] = merged["span"] / merged["author_count"].clip(lower=1)
    grouped = (
        merged.groupby(["author_id", "edition_id", "edition_year"], dropna=False)
        .agg(
            author_pages=("author_work_pages", "sum"),
            author_name=("author_name", "first"),
        )
        .reset_index()
    )
    return grouped


def build_author_form_per_edition(
    raw: pd.DataFrame,
    work_per_edition: pd.DataFrame,
    counts: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (author_id, form_name, edition_id) with pages.

    Each work's span is divided equally first among its co-authors and then
    among its co-forms.  Works lacking any form contribute a single
    form_name='Unknown' row.
    """
    # Build per-work author × form cartesian product (with Unknown fallback)
    awa = raw.dropna(subset=["author_id"])[
        ["work_id", "author_id", "author_name"]
    ].drop_duplicates(["work_id", "author_id"])
    forms = raw[["work_id", "form_name"]].drop_duplicates(["work_id", "form_name"])
    # Works with no form: insert a single 'Unknown' row
    works_without_form = (
        forms.groupby("work_id")["form_name"]
        .apply(lambda s: s.notna().any())
        .rename("has_form")
        .reset_index()
    )
    unknown_rows = works_without_form.loc[~works_without_form["has_form"], ["work_id"]]
    unknown_rows = unknown_rows.assign(form_name="Unknown")
    forms = pd.concat(
        [forms.dropna(subset=["form_name"]), unknown_rows], ignore_index=True
    )

    cartesian = awa.merge(forms, on="work_id")
    cartesian = cartesian.merge(
        counts[["work_id", "author_count", "form_count"]], on="work_id"
    )
    merged = cartesian.merge(work_per_edition, on="work_id")
    merged["author_form_work_pages"] = merged["span"] / (
        merged["author_count"].clip(lower=1) * merged["form_count"].clip(lower=1)
    )
    grouped = (
        merged.groupby(
            ["author_id", "form_name", "edition_id", "edition_year"], dropna=False
        )
        .agg(
            author_form_pages=("author_form_work_pages", "sum"),
            author_name=("author_name", "first"),
        )
        .reset_index()
    )
    return grouped


# ── Per-edition denominators ──────────────────────────────────────────────────


def edition_total_pages(work_per_edition: pd.DataFrame) -> pd.DataFrame:
    """Sum of work page spans per edition."""
    return (
        work_per_edition.groupby(["edition_id", "edition_year"], dropna=False)
        .agg(edition_pages=("span", "sum"))
        .reset_index()
    )


def edition_form_total_pages(
    raw: pd.DataFrame,
    work_per_edition: pd.DataFrame,
    counts: pd.DataFrame,
) -> pd.DataFrame:
    """Sum of form-split work page spans per (edition, form).

    Each work contributes span / form_count to each of its forms.  Works with
    no form contribute their full span to form_name='Unknown'.
    """
    forms = raw[["work_id", "form_name"]].drop_duplicates(["work_id", "form_name"])
    works_without_form = (
        forms.groupby("work_id")["form_name"]
        .apply(lambda s: s.notna().any())
        .rename("has_form")
        .reset_index()
    )
    unknown_rows = works_without_form.loc[~works_without_form["has_form"], ["work_id"]]
    unknown_rows = unknown_rows.assign(form_name="Unknown")
    forms = pd.concat(
        [forms.dropna(subset=["form_name"]), unknown_rows], ignore_index=True
    )
    merged = forms.merge(counts[["work_id", "form_count"]], on="work_id")
    merged = merged.merge(work_per_edition, on="work_id")
    merged["form_work_pages"] = merged["span"] / merged["form_count"].clip(lower=1)
    return (
        merged.groupby(["edition_id", "edition_year", "form_name"], dropna=False)
        .agg(form_pages=("form_work_pages", "sum"))
        .reset_index()
    )


# ── Reselection rate per author ───────────────────────────────────────────────


def compute_reselection_rates(
    author_per_edition: pd.DataFrame,
    all_edition_years: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-author reselection_prob from raw appearances.

    Adapted from analysis/reselection/author_first_selection_success.py:compute.
    Uses edition_id as the unique edition key.  An author's debut is their
    earliest edition_year (ties broken by edition_id).  reselection_prob =
    (subsequent editions in which the author appears) ÷ (AFAM editions
    strictly after debut year).  Authors with no subsequent edition are
    dropped (subsequent_count == 0).
    """
    pairs = author_per_edition[
        ["author_id", "author_name", "edition_id", "edition_year"]
    ].drop_duplicates(["author_id", "edition_id"])
    sorted_pairs = pairs.sort_values(["author_id", "edition_year", "edition_id"])

    debut = (
        sorted_pairs.groupby("author_id", sort=False)
        .first()
        .reset_index()[["author_id", "author_name", "edition_id", "edition_year"]]
        .rename(
            columns={"edition_id": "debut_edition_id", "edition_year": "debut_year"}
        )
    )

    author_eds: dict = pairs.groupby("author_id")["edition_id"].apply(set).to_dict()
    appearance_count: dict = (
        pairs.groupby("author_id")["edition_id"].nunique().to_dict()
    )

    records: list[dict] = []
    for _, row in debut.iterrows():
        aid = row["author_id"]
        debut_yr = int(row["debut_year"])
        subsequent = all_edition_years[all_edition_years["edition_year"] > debut_yr]
        subsequent_count = len(subsequent)
        if subsequent_count == 0:
            continue
        reselection_count = int(subsequent["edition_id"].isin(author_eds[aid]).sum())
        records.append(
            {
                "author_id": aid,
                "author_name": row["author_name"],
                "debut_edition_id": row["debut_edition_id"],
                "debut_year": debut_yr,
                "subsequent_count": subsequent_count,
                "reselection_count": reselection_count,
                "reselection_prob": reselection_count / subsequent_count,
                "n_appearances": appearance_count[aid],
            }
        )
    return pd.DataFrame(records)


# ── Plotting ──────────────────────────────────────────────────────────────────


def _jittered_scatter(
    ax: plt.Axes,
    positions: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
    jitter: float = 0.004,
    color: str = "#1f77b4",
    alpha: float = 0.35,
    size: int = 10,
) -> None:
    if len(positions) == 0:
        return
    noise = rng.uniform(-jitter, jitter, size=len(positions))
    ax.scatter(positions + noise, values, s=size, color=color, alpha=alpha, zorder=2)


def _plot_box_groups(
    ax: plt.Axes,
    groups: list[tuple[str, float, np.ndarray]],
    *,
    rng: np.random.Generator,
    box_width: float = 0.012,
    scatter_color: str = "#1f77b4",
) -> None:
    """For each (label, x_position, y_values) group, draw a small boxplot at
    x_position and overlay jittered scatter points.

    Groups with only one y value get a scatter point only (no box).
    """
    boxable = [(lbl, x, ys) for (lbl, x, ys) in groups if len(ys) >= 2]
    singletons = [(lbl, x, ys) for (lbl, x, ys) in groups if len(ys) < 2]

    if boxable:
        positions = [x for _, x, _ in boxable]
        data = [ys for _, _, ys in boxable]
        ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            showfliers=False,
            manage_ticks=False,
            patch_artist=False,
            medianprops=dict(color="black", linewidth=1.2),
            boxprops=dict(color="black", linewidth=0.8),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
            zorder=3,
        )

    # Scatter all points (including singletons) so every entity is visible
    all_positions = np.array([x for _, x, ys in groups for _ in ys])
    all_values = np.array([y for _, _, ys in groups for y in ys])
    _jittered_scatter(
        ax, all_positions, all_values, rng=rng, color=scatter_color, alpha=0.35
    )

    if singletons:
        # Highlight singleton points with an open marker so they stand out
        s_positions = np.array([x for _, x, _ in singletons])
        s_values = np.array([ys[0] for _, _, ys in singletons])
        ax.scatter(
            s_positions,
            s_values,
            s=18,
            facecolors="none",
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
        )


def _annotate_top(
    ax: plt.Axes,
    groups: list[tuple[str, float, np.ndarray]],
    *,
    top_n: int,
) -> None:
    """Label the top-N entities by appearance count (len(ys))."""
    ranked = sorted(groups, key=lambda g: len(g[2]), reverse=True)[:top_n]
    for label, x, ys in ranked:
        y = float(np.max(ys))
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=8,
            color="black",
            zorder=5,
        )


def plot_chart1(
    per_edition: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    title_suffix: str,
    out_path,
) -> None:
    """Per-author boxplot of edition page share vs. reselection probability."""
    rng = np.random.default_rng(RNG_SEED)
    df = per_edition.merge(
        rates[["author_id", "reselection_prob"]], on="author_id", how="inner"
    )

    groups: list[tuple[str, float, np.ndarray]] = []
    for aid, grp in df.groupby("author_id"):
        x = float(grp["reselection_prob"].iloc[0])
        ys = grp["page_share"].to_numpy(dtype=float)
        label = str(grp["author_name"].iloc[0])
        groups.append((label, x, ys))

    fig, ax = plt.subplots(figsize=(11, 6))
    _plot_box_groups(ax, groups, rng=rng, scatter_color="#1f77b4")
    _annotate_top(ax, groups, top_n=TOP_N_LABEL)

    ax.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
    ax.set_xlabel(
        "Reselection probability (subsequent appearances ÷ available editions)",
        fontsize=10,
    )
    ax.set_ylabel("Share of edition pages", fontsize=10)
    ax.set_title(
        f"Author page share vs. reselection probability\n{title_suffix}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_chart2(
    per_form_edition: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    title_suffix: str,
    out_path,
) -> None:
    """Faceted boxplots: one subplot per form, each plotting per-author
    distribution of (share of form pages within edition) vs. reselection
    probability.  Authors with multiple forms appear in multiple subplots,
    each labeled with just the author name (no form qualifier).
    """
    df = per_form_edition.merge(
        rates[["author_id", "reselection_prob"]], on="author_id", how="inner"
    )
    forms = sorted(f for f in df["form_name"].dropna().unique())
    if not forms:
        return

    n = len(forms)
    ncols = 3 if n > 4 else 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharex=True, sharey=True
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, form in zip(axes_flat, forms, strict=False):
        sub = df[df["form_name"] == form]
        groups: list[tuple[str, float, np.ndarray]] = []
        for _, grp in sub.groupby("author_id"):
            x = float(grp["reselection_prob"].iloc[0])
            ys = grp["page_share_within_form"].to_numpy(dtype=float)
            label = str(grp["author_name"].iloc[0])
            groups.append((label, x, ys))

        rng = np.random.default_rng(RNG_SEED)
        _plot_box_groups(ax, groups, rng=rng, scatter_color="#d62728")
        _annotate_top(ax, groups, top_n=TOP_N_LABEL)

        ax.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
        ax.set_title(f"{form}  (n={len(groups)})", fontsize=10)
        ax.grid(True, alpha=0.25, linestyle=":")

    # Hide any unused trailing axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.supxlabel(
        "Reselection probability (subsequent appearances ÷ available editions)",
        fontsize=10,
    )
    fig.supylabel("Share of form pages within edition", fontsize=10)
    fig.suptitle(
        f"Author page share by form vs. reselection probability\n{title_suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_root_works_flag(parser)
    add_save_csv_flag(parser)
    args = parser.parse_args()

    raw = query_db(query_path("author-page-share-reselection"))
    if args.only_root_works:
        raw = raw[raw["parent_id"].isna()].reset_index(drop=True)

    work_spans = compute_work_volume_spans(raw)
    work_per_edition = build_work_per_edition(raw, work_spans)
    counts = build_author_form_counts(raw)

    author_per_edition = build_author_per_edition(raw, work_per_edition, counts)
    author_form_per_edition = build_author_form_per_edition(
        raw, work_per_edition, counts
    )

    ed_totals = edition_total_pages(work_per_edition)
    ed_form_totals = edition_form_total_pages(raw, work_per_edition, counts)

    per_edition = author_per_edition.merge(
        ed_totals[["edition_id", "edition_pages"]], on="edition_id", how="left"
    )
    per_edition["page_share"] = per_edition["author_pages"] / per_edition[
        "edition_pages"
    ].replace(0, np.nan)
    per_edition = per_edition.dropna(subset=["page_share"]).reset_index(drop=True)

    per_form_edition = author_form_per_edition.merge(
        ed_form_totals[["edition_id", "form_name", "form_pages"]],
        on=["edition_id", "form_name"],
        how="left",
    )
    per_form_edition["page_share_within_form"] = per_form_edition[
        "author_form_pages"
    ] / per_form_edition["form_pages"].replace(0, np.nan)
    per_form_edition = per_form_edition.dropna(
        subset=["page_share_within_form"]
    ).reset_index(drop=True)

    all_edition_years = raw[["edition_id", "edition_year"]].drop_duplicates()
    rates = compute_reselection_rates(author_per_edition, all_edition_years)
    rates = rates[
        (rates["reselection_prob"] >= MIN_RESELECTION_PROB)
        & (rates["n_appearances"] >= MIN_APPEARANCES)
    ].reset_index(drop=True)

    years = sorted(int(y) for y in all_edition_years["edition_year"].unique())
    mode = "root works only" if args.only_root_works else "incl. excerpts"
    title_suffix = (
        f"AFAM-tagged anthologies, {years[0]}–{years[-1]} ({mode}); "
        f"n={len(rates)} authors with reselection ≥ {MIN_RESELECTION_PROB:g} "
        f"and ≥ {MIN_APPEARANCES} appearances"
    )

    plot_chart1(
        per_edition,
        rates,
        title_suffix=title_suffix,
        out_path=OUTPUT_DIR / CHART1_PNG,
    )
    plot_chart2(
        per_form_edition,
        rates,
        title_suffix=title_suffix,
        out_path=OUTPUT_DIR / CHART2_PNG,
    )
    print(f"Wrote {OUTPUT_DIR / CHART1_PNG}")
    print(f"Wrote {OUTPUT_DIR / CHART2_PNG}")

    if args.save_csv:
        per_edition_out = per_edition.merge(
            rates[["author_id", "reselection_prob"]], on="author_id", how="inner"
        )[
            [
                "author_id",
                "author_name",
                "edition_id",
                "edition_year",
                "author_pages",
                "edition_pages",
                "page_share",
                "reselection_prob",
            ]
        ].sort_values(["author_name", "edition_year"])
        per_form_out = per_form_edition.merge(
            rates[["author_id", "reselection_prob"]], on="author_id", how="inner"
        )[
            [
                "author_id",
                "author_name",
                "form_name",
                "edition_id",
                "edition_year",
                "author_form_pages",
                "form_pages",
                "page_share_within_form",
                "reselection_prob",
            ]
        ].sort_values(["author_name", "form_name", "edition_year"])

        median_share = (
            per_edition.groupby("author_id")["page_share"]
            .median()
            .rename("median_page_share")
        )
        median_form_share = (
            per_form_edition.groupby("author_id")["page_share_within_form"]
            .median()
            .rename("median_page_share_within_form")
        )
        summary = (
            rates.merge(median_share, on="author_id", how="left")
            .merge(median_form_share, on="author_id", how="left")
            .sort_values(
                ["reselection_prob", "n_appearances"], ascending=[False, False]
            )
        )

        per_edition_out.to_csv(DATA_DIR / PER_EDITION_CSV, index=False)
        per_form_out.to_csv(DATA_DIR / PER_FORM_CSV, index=False)
        summary.to_csv(DATA_DIR / SUMMARY_CSV, index=False)
        print(f"Wrote {DATA_DIR / PER_EDITION_CSV}")
        print(f"Wrote {DATA_DIR / PER_FORM_CSV}")
        print(f"Wrote {DATA_DIR / SUMMARY_CSV}")


if __name__ == "__main__":
    main()
