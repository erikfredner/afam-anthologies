"""first_selection_success.py
----------------------------
Two figures examining how well debut anthologies' author picks were validated
by subsequent anthologies.

Figure 1: viz/first_selection_success_bar.png
    Horizontal bar chart — mean reselection probability per debut anthology,
    with a diamond marker for ever-reselected rate.

Figure 2: viz/first_selection_success_scatter.png
    Bubble scatter — debut year vs. ever-reselected rate, bubble sized by
    number of debut authors, with a linear trend line.

Usage: uv run python viz/first_selection_success.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Paths ───────────────────────────────────────────────────────────────────────

from afam.db import query
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

OUT_DIR = OUTPUT_DIR
sys.path.insert(0, str(Path(__file__).parents[2] / "analysis" / "reselection"))
from author_first_selection_success import compute as compute_author_success  # noqa: E402


# ── Edition-key logic (private copy, mirrors selection_frequency_decay.py) ──────


def _strip_volume(title: str) -> str:
    return re.sub(r",?\s+[Vv]ol\.?\s+\d+\s*$", "", title).strip()


def _assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    meta_rows: list[dict] = []
    for _, r in (
        df[["anthology_id", "anthology_title", "series", "edition_number", "volume"]]
        .drop_duplicates()
        .iterrows()
    ):
        series = r["series"].strip()
        edition = r["edition_number"].strip()
        volume = r["volume"].strip()
        title = r["anthology_title"].strip()
        aid = r["anthology_id"]

        if series:
            key = f"{series}|{edition}"
        elif volume:
            key = f"{_strip_volume(title)}|{edition}"
        else:
            key = aid

        meta_rows.append({"anthology_id": aid, "edition_key": key})

    return df.merge(pd.DataFrame(meta_rows), on="anthology_id")


# ── Labels ──────────────────────────────────────────────────────────────────────

SERIES_ABBREV: dict[str, str] = {
    "The Norton Anthology of African American Literature": "NAAAL",
    "Afro-American Writing: An Anthology of Prose and Poetry": "Afro-Am. Writing",
    "African American Literature": "AAL Anthology",
    "The Wiley Blackwell Anthology of African American Literature": "Wiley Blackwell AAL",
}


def _short_label(edition_key: str, year: int, ek_to_title: dict[str, str]) -> str:
    if "|" in edition_key:
        series_or_title, edition = edition_key.rsplit("|", 1)
        abbrev = SERIES_ABBREV.get(series_or_title, series_or_title[:18])
        return f"{abbrev} ed.{edition}\n({year})"
    title = ek_to_title.get(edition_key, edition_key)
    return f"{title[:22]}\n({year})"


# ── Core computation (private, self-contained) ───────────────────────────────────


def _compute(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Return edition-level aggregate DataFrame."""
    df = df_raw.copy()
    df["publication_year"] = df["publication_year"].astype(int)
    df = _assign_edition_key(df)

    ek_to_title: dict[str, str] = (
        df.groupby("edition_key")["anthology_title"].first().to_dict()
    )

    pairs = (
        df[["edition_key", "author_id", "publication_year"]]
        .drop_duplicates(subset=["edition_key", "author_id"])
        .copy()
    )

    edition_year = (
        df.groupby("edition_key")["publication_year"]
        .min()
        .rename("edition_year")
        .reset_index()
    )
    pairs = pairs.merge(edition_year, on="edition_key")
    all_editions = edition_year.copy()

    sorted_pairs = pairs.sort_values(["author_id", "edition_year", "edition_key"])
    debut = (
        sorted_pairs.groupby("author_id", sort=False)
        .first()
        .reset_index()[["author_id", "edition_key", "edition_year"]]
        .rename(
            columns={"edition_key": "debut_edition_key", "edition_year": "debut_year"}
        )
    )

    author_ek_set: dict[str, set] = (
        pairs.groupby("author_id")["edition_key"].apply(set).to_dict()
    )

    records = []
    for _, row in debut.iterrows():
        aid = row["author_id"]
        debut_ek = row["debut_edition_key"]
        debut_yr = int(row["debut_year"])

        subsequent = all_editions[all_editions["edition_year"] > debut_yr]
        subsequent_count = len(subsequent)
        author_eks = author_ek_set.get(aid, set())
        reselection_count = int(subsequent["edition_key"].isin(author_eks).sum())

        if subsequent_count == 0:
            reselection_prob = float("nan")
            ever_reselected = float("nan")
        else:
            reselection_prob = reselection_count / subsequent_count
            ever_reselected = 1.0 if reselection_count > 0 else 0.0

        records.append(
            {
                "debut_edition_key": debut_ek,
                "debut_year": debut_yr,
                "author_id": aid,
                "reselection_prob": reselection_prob,
                "ever_reselected": ever_reselected,
                "subsequent_count": subsequent_count,
            }
        )

    detail_df = pd.DataFrame(records)

    agg_records = []
    for (debut_ek, debut_yr), grp in detail_df.groupby(
        ["debut_edition_key", "debut_year"], sort=False
    ):
        n_debut_authors = len(grp)
        n_with_subsequent = int((grp["subsequent_count"] > 0).sum())

        valid_prob = grp["reselection_prob"].dropna()
        mean_reselection_prob = (
            float(valid_prob.mean()) if len(valid_prob) > 0 else float("nan")
        )

        valid_ever = grp["ever_reselected"].dropna()
        ever_reselected_rate = (
            float(valid_ever.mean()) if len(valid_ever) > 0 else float("nan")
        )

        short_lbl = _short_label(debut_ek, int(debut_yr), ek_to_title)

        agg_records.append(
            {
                "debut_edition_key": debut_ek,
                "debut_year": int(debut_yr),
                "n_debut_authors": n_debut_authors,
                "n_with_subsequent": n_with_subsequent,
                "mean_reselection_prob": mean_reselection_prob,
                "ever_reselected_rate": ever_reselected_rate,
                "short_label": short_lbl,
            }
        )

    return (
        pd.DataFrame(agg_records)
        .sort_values(["debut_year", "debut_edition_key"])
        .reset_index(drop=True)
    )


# ── Figure 1: Horizontal bar chart ──────────────────────────────────────────────


def fig1_bar(edition_df: pd.DataFrame, out_dir: Path) -> None:
    # Exclude anthologies with no subsequent opportunities
    df = edition_df[edition_df["n_with_subsequent"] > 0].copy()
    if df.empty:
        print("Figure 1: no editions with subsequent opportunities — skipping")
        return

    # Sort ascending by year so oldest is at top after invert_yaxis
    df = df.sort_values(["debut_year", "debut_edition_key"]).reset_index(drop=True)
    n = len(df)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.5 * n + 2)))

    ax.barh(
        y_pos,
        df["mean_reselection_prob"],
        color="#1f77b4",
        alpha=0.75,
        height=0.6,
        label="Mean reselection prob. (p̄)",
    )
    ax.scatter(
        df["ever_reselected_rate"],
        y_pos,
        marker="D",
        color="#d62728",
        s=60,
        zorder=3,
        label="Ever-reselected rate",
    )

    # Right-side annotations
    for i, row in df.iterrows():
        prob = row["mean_reselection_prob"]
        prob_str = f"{prob:.2f}" if not np.isnan(prob) else "N/A"
        ax.text(
            min(max(prob, row["ever_reselected_rate"]) + 0.03, 1.0),
            i,
            f"N={row['n_with_subsequent']}, p̄={prob_str}",
            va="center",
            fontsize=7,
            color="#444444",
        )

    ax.axvline(0.5, color="gray", linestyle="--", lw=1.0, alpha=0.6, label="x = 0.5")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["short_label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Rate", fontsize=11)
    ax.set_title(
        "Author first-selection success by debut anthology\n"
        "(scaled by subsequent anthology opportunities; most recent excluded)",
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = out_dir / "first_selection_success_bar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 2: Scatter/bubble ─────────────────────────────────────────────────────


def fig2_scatter(edition_df: pd.DataFrame, out_dir: Path) -> None:
    df = edition_df.dropna(subset=["ever_reselected_rate"]).copy()
    if df.empty:
        print("Figure 2: no editions with ever_reselected_rate — skipping")
        return

    x = df["debut_year"].values.astype(float)
    y = df["ever_reselected_rate"].values
    sizes = df["n_debut_authors"].values

    # Bubble area proportional to n_debut_authors
    bubble_area = sizes * 60

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(
        x,
        y,
        s=bubble_area,
        color="#1f77b4",
        alpha=0.65,
        edgecolors="white",
        linewidth=0.5,
        zorder=3,
    )

    # Linear trend line
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(
            x_line,
            np.polyval(coeffs, x_line),
            color="gray",
            lw=1.5,
            linestyle="--",
            label="Linear trend",
            zorder=2,
        )

    # Annotate each bubble
    for _, row in df.iterrows():
        ax.annotate(
            row["short_label"],
            xy=(row["debut_year"], row["ever_reselected_rate"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
            color="#333333",
        )

    ax.axhline(0.5, color="gray", linestyle=":", lw=1.0, alpha=0.7, label="y = 0.5")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Debut year", fontsize=11)
    ax.set_ylabel(
        "Ever-reselected rate\n(fraction of debut authors ever reselected)", fontsize=11
    )
    ax.set_title(
        "How often were debut anthology authors reselected later?\n"
        "(bubble size = number of debut authors)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)

    # Bubble size legend — combined with trend/reference labels, placed below plot
    for n_label in [1, 5, 20]:
        ax.scatter(
            [],
            [],
            s=n_label * 60,
            color="#1f77b4",
            alpha=0.65,
            label=f"N={n_label} debut authors",
        )
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    out = out_dir / "first_selection_success_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    df_raw = query(query_path("work-selection-divergence"))
    _, edition_df = compute_author_success(df_raw)
    edition_df = edition_df.rename(columns={"anthology_label": "short_label"})

    OUT_DIR.mkdir(exist_ok=True)
    fig1_bar(edition_df, OUT_DIR)
    fig2_scatter(edition_df, OUT_DIR)


if __name__ == "__main__":
    main()
