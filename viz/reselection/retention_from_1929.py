"""retention_from_1929.py
-----------------------
Bar chart showing how much of the 1929 "Anthology of American Negro Literature"
(anthology_id=67) was retained — by author and by work — in each subsequent anthology.

Figure: viz/retention_from_1929.png

Usage: uv run python viz/retention_from_1929.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────────────

from afam.db import query as query_db
from afam.editions import EDITION_LABELS
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

OUT_DIR = OUTPUT_DIR

REF_EDITION_KEY = "72"  # 1929 Anthology of American Negro Literature (Calverton)


# ── Style constants (match work_selection_divergence.py) ───────────────────────

C_BLUE = "#1f77b4"
C_RED = "#d62728"
GRID_KW = dict(alpha=0.25, linestyle=":")


# ── Labels ─────────────────────────────────────────────────────────────────────


def make_label(edition_key: str, year: int) -> str:
    short = EDITION_LABELS.get(int(edition_key), str(edition_key))
    return f"{short}\n{year}"


# ── Title normalization (private copy) ─────────────────────────────────────────


def _normalize_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"^(from |excerpt from |selections? from )", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip()


# ── Load and prepare ───────────────────────────────────────────────────────────


def _load_and_prepare() -> pd.DataFrame:
    """Long (author × edition) table from the live DB.

    Columns match the legacy CSV shape consumed by compute_retention:
    edition_key (= edition_id as string), ek_year, author_id, norm_title.
    Works without an author are dropped (retention is keyed on authors).
    """
    raw = query_db(query_path("works-authors-per-afam-edition"))
    raw = raw.dropna(subset=["author_id"]).copy()
    long = pd.DataFrame(
        {
            "edition_key": raw["edition_id"].astype(int).astype(str),
            "ek_year": raw["anthology_publication_year"].astype(int),
            "author_id": raw["author_id"].astype(int).astype(str),
            "norm_title": raw["work_title"].apply(_normalize_title),
        }
    )
    print(
        f"Loaded {len(long)} long rows, "
        f"{long['edition_key'].nunique()} editions, "
        f"{long['author_id'].nunique()} authors"
    )
    return long


# ── Core computation ───────────────────────────────────────────────────────────


def compute_retention(long: pd.DataFrame, ref_key: str) -> pd.DataFrame:
    """Compute per-edition retention of a reference anthology's authors and works.

    Parameters
    ----------
    long:
        DataFrame with columns: edition_key, ek_year, author_id, norm_title.
    ref_key:
        Edition key of the reference anthology (excluded from output).

    Returns
    -------
    DataFrame with columns: edition_key, year, author_pct, work_pct.
    Sorted by year ascending. Reference edition is excluded.
    """
    authors_ref = frozenset(
        long.loc[long["edition_key"] == ref_key, "author_id"].unique()
    )
    works_ref = frozenset(
        long.loc[long["edition_key"] == ref_key, "norm_title"].unique()
    )

    records = []
    for ek, grp in long.groupby("edition_key"):
        if ek == ref_key:
            continue
        ek_authors = frozenset(grp["author_id"].unique())
        ek_works = frozenset(grp["norm_title"].unique())
        author_pct = (
            100 * len(ek_authors & authors_ref) / len(authors_ref)
            if authors_ref
            else 0.0
        )
        work_pct = (
            100 * len(ek_works & works_ref) / len(works_ref) if works_ref else 0.0
        )
        year = int(grp["ek_year"].iloc[0])
        records.append(
            {
                "edition_key": ek,
                "year": year,
                "author_pct": author_pct,
                "work_pct": work_pct,
            }
        )

    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


# ── Figure ─────────────────────────────────────────────────────────────────────


def plot_retention(df: pd.DataFrame, out_path: Path) -> None:
    """Render the retention grouped bar chart."""
    labels = [make_label(row["edition_key"], row["year"]) for _, row in df.iterrows()]

    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - width / 2, df["author_pct"], width=width, color=C_BLUE, label="Authors")
    ax.bar(x + width / 2, df["work_pct"], width=width, color=C_RED, label="Works")

    ax.axhline(50, color="gray", linestyle="--", lw=1.0, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of 1929 selections retained", fontsize=11)
    ax.set_title(
        "Retention of 1929 anthology selections in subsequent anthologies",
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis="y", **GRID_KW)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    long = _load_and_prepare()
    df = compute_retention(long, REF_EDITION_KEY)

    print(f"\nRetention rows: {len(df)} subsequent editions")
    if not df.empty:
        print(
            df[["edition_key", "year", "author_pct", "work_pct"]].to_string(index=False)
        )

    out_path = OUT_DIR / "retention_from_1929.png"
    plot_retention(df, out_path)


if __name__ == "__main__":
    main()
