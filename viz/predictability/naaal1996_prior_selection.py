"""
naaal1996_prior_selection.py
----------------------------
For every work and author appearing in any anthology published in 1996 or
earlier, counts how many anthology editions other than the 1996 Norton
Anthology of African American Literature (NAAAL) selected them, then plots
the percentage that were included in NAAAL 1996 at each prior-selection count.

X-axis : number of non-NAAAL ≤1996 editions selecting an item
Y-axis : % of items at that count that were selected for NAAAL 1996
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from afam.db import query as query_db
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

OUT_FILE = OUTPUT_DIR / "naaal1996_prior_selection.png"
OUT_FILE_2 = OUTPUT_DIR / "naaal1996_prior_selection_twopanel.png"

# Target NAAAL edition: series_id=3 (NAAAL), edition_number="1" (first edition).
NAAAL_SERIES_ID = 3
NAAAL_EDITION = "1"

WORK_COLOR = "#1f77b4"
AUTHOR_COLOR = "#d62728"


# ── Works analysis ────────────────────────────────────────────────────────────


def build_work_frame(df: pd.DataFrame, target_id: int) -> pd.DataFrame:
    naaal_works = set(df.loc[df["edition_id"] == target_id, "work_id"])
    other = df[df["edition_id"] != target_id]
    prior_counts = other.groupby("work_id")["edition_id"].nunique()

    all_works = df["work_id"].unique()
    wdf = pd.DataFrame({"work_id": all_works})
    wdf["prior_count"] = wdf["work_id"].map(prior_counts).fillna(0).astype(int)
    wdf["in_naaal"] = wdf["work_id"].isin(naaal_works)
    return wdf[wdf["prior_count"] >= 1].reset_index(drop=True)


# ── Authors analysis ──────────────────────────────────────────────────────────


def build_author_frame(df: pd.DataFrame, target_id: int) -> pd.DataFrame:
    a = df.dropna(subset=["author_id"])
    naaal_authors = set(a.loc[a["edition_id"] == target_id, "author_id"])
    other = a[a["edition_id"] != target_id]
    prior_counts = other.groupby("author_id")["edition_id"].nunique()

    all_authors = a["author_id"].unique()
    adf = pd.DataFrame({"author_id": all_authors})
    adf["prior_count"] = adf["author_id"].map(prior_counts).fillna(0).astype(int)
    adf["in_naaal"] = adf["author_id"].isin(naaal_authors)
    return adf[adf["prior_count"] >= 1].reset_index(drop=True)


# ── Inclusion curve ───────────────────────────────────────────────────────────


def inclusion_curve(
    frame: pd.DataFrame, count_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each observed prior_count k, return:
      ks    : the k values
      pcts  : % of items with prior_count == k that are in NAAAL 1996
      ns    : number of items at each k (for context)
    """
    rows = []
    for k, grp in frame.groupby(count_col):
        pct = 100.0 * grp["in_naaal"].sum() / len(grp)
        rows.append({"k": k, "pct": pct, "n": len(grp)})
    result = pd.DataFrame(rows).sort_values("k")
    return result["k"].to_numpy(), result["pct"].to_numpy(), result["n"].to_numpy()


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot(
    work_ks: np.ndarray,
    work_pcts: np.ndarray,
    work_ns: np.ndarray,
    n_works: int,
    auth_ks: np.ndarray,
    auth_pcts: np.ndarray,
    auth_ns: np.ndarray,
    n_authors: int,
    out: Path,
) -> None:
    work_label = f"Works (N={n_works:,})"
    author_label = f"Authors (N={n_authors:,})"

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(work_ks, work_pcts, color=WORK_COLOR, linewidth=0.8, alpha=0.4, zorder=2)
    ax.scatter(
        work_ks,
        work_pcts,
        s=np.sqrt(work_ns) * 10,
        color=WORK_COLOR,
        marker="o",
        alpha=0.6,
        zorder=3,
        label=work_label,
    )

    ax.plot(auth_ks, auth_pcts, color=AUTHOR_COLOR, linewidth=0.8, alpha=0.4, zorder=2)
    ax.scatter(
        auth_ks,
        auth_pcts,
        s=np.sqrt(auth_ns) * 10,
        color=AUTHOR_COLOR,
        marker="s",
        alpha=0.6,
        zorder=3,
        label=author_label,
    )

    ax.set_xlabel(
        "Number of non-NAAAL anthologies (≤1996) selecting an item", fontsize=11
    )
    ax.set_ylabel(
        "% selected for the 1996 Norton Anthology of\nAfrican American Literature",
        fontsize=11,
    )
    ax.set_title(
        "Prior anthology selection as a predictor of inclusion in NAAAL 1996\n"
        "(point size ∝ N items in frequency band)",
        fontsize=12,
    )
    ax.set_xlim(0.5, max(work_ks.max(), auth_ks.max()) + 0.5)
    ax.set_ylim(-2, 105)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ── Two-panel plot ────────────────────────────────────────────────────────────


def plot_two_panel(
    work_ks,
    work_pcts,
    work_ns,
    n_works,
    auth_ks,
    auth_pcts,
    auth_ns,
    n_authors,
    out: Path,
) -> None:
    work_label = f"Works (N={n_works:,})"
    author_label = f"Authors (N={n_authors:,})"
    max_k = max(work_ks.max(), auth_ks.max())

    fig, (ax_pct, ax_n) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )

    # ── Top panel: % inclusion curves ────────────────────────────────────────
    ax_pct.plot(
        work_ks, work_pcts, color=WORK_COLOR, linewidth=1.2, alpha=0.5, zorder=2
    )
    ax_pct.scatter(
        work_ks,
        work_pcts,
        s=40,
        color=WORK_COLOR,
        marker="o",
        alpha=0.6,
        zorder=3,
        label=work_label,
    )
    ax_pct.plot(
        auth_ks, auth_pcts, color=AUTHOR_COLOR, linewidth=1.2, alpha=0.5, zorder=2
    )
    ax_pct.scatter(
        auth_ks,
        auth_pcts,
        s=40,
        color=AUTHOR_COLOR,
        marker="s",
        alpha=0.6,
        zorder=3,
        label=author_label,
    )

    ax_pct.set_ylabel(
        "% selected for the 1996 Norton Anthology of\nAfrican American Literature",
        fontsize=11,
    )
    ax_pct.set_title(
        "Prior anthology selection as a predictor of inclusion in NAAAL 1996",
        fontsize=12,
    )
    ax_pct.set_ylim(-2, 105)
    ax_pct.legend(frameon=False, fontsize=9)
    ax_pct.grid(True, alpha=0.25, linestyle=":")

    # ── Bottom panel: N per band ──────────────────────────────────────────────
    bar_width = 0.35
    ax_n.bar(
        work_ks - bar_width / 2, work_ns, width=bar_width, color=WORK_COLOR, alpha=0.7
    )
    ax_n.bar(
        auth_ks + bar_width / 2, auth_ns, width=bar_width, color=AUTHOR_COLOR, alpha=0.7
    )
    ax_n.set_yscale("log")
    ax_n.set_ylabel("N items\nin band (log scale)", fontsize=9)
    ax_n.set_xlabel(
        "Number of non-NAAAL anthologies (≤1996) selecting an item", fontsize=11
    )
    ax_n.grid(True, alpha=0.25, linestyle=":")

    ax_n.set_xlim(0.5, max_k + 0.5)
    ax_n.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    df = query_db(query_path("works-authors-per-afam-edition"))
    target = df[
        (df["series_id"] == NAAAL_SERIES_ID) & (df["edition_number"] == NAAAL_EDITION)
    ]
    target_id = int(target["edition_id"].iloc[0])
    target_year = int(target["anthology_publication_year"].iloc[0])
    df = df[
        (df["edition_id"] == target_id)
        | (df["anthology_publication_year"] < target_year)
    ]

    work_frame = build_work_frame(df, target_id)
    author_frame = build_author_frame(df, target_id)

    work_ks, work_pcts, work_ns = inclusion_curve(work_frame, "prior_count")
    auth_ks, auth_pcts, auth_ns = inclusion_curve(author_frame, "prior_count")

    OUT_FILE.parent.mkdir(exist_ok=True)
    plot(
        work_ks,
        work_pcts,
        work_ns,
        len(work_frame),
        auth_ks,
        auth_pcts,
        auth_ns,
        len(author_frame),
        OUT_FILE,
    )
    plot_two_panel(
        work_ks,
        work_pcts,
        work_ns,
        len(work_frame),
        auth_ks,
        auth_pcts,
        auth_ns,
        len(author_frame),
        OUT_FILE_2,
    )


if __name__ == "__main__":
    main()
