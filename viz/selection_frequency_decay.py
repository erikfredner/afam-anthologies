"""
selection_frequency_decay.py
----------------------------
For every work and author in the anthology dataset, count how many unique
anthology editions include them (volumes of the same edition are merged).
Then plot the cumulative survival curve: for each integer k, what percentage
of all works / authors appear in at least k editions?

Starting point is always (1, 100) by definition.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKS_FILE   = Path(__file__).parent.parent / "data" / "2026-03-13 work ids in afam anthologies.csv"
AUTHORS_FILE = Path(__file__).parent.parent / "data" / "2026-03-13 author ids in afam anthologies.csv"
OUT_FILE     = Path(__file__).parent / "selection_frequency_decay.png"

# ── Edition-key logic (mirrors heatmap scripts) ───────────────────────────────

def _strip_volume(title: str) -> str:
    return re.sub(r",?\s+[Vv]ol\.?\s+\d+\s*$", "", title).strip()


def assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    meta_rows: list[dict] = []
    for _, r in (
        df[["anthology_id", "anthology_title", "series", "edition_number", "volume"]]
        .drop_duplicates()
        .iterrows()
    ):
        series  = r["series"].strip()
        edition = r["edition_number"].strip()
        volume  = r["volume"].strip()
        title   = r["anthology_title"].strip()
        aid     = r["anthology_id"]

        if series:
            key = f"{series}|{edition}"
        elif volume:
            key = f"{_strip_volume(title)}|{edition}"
        else:
            key = aid

        meta_rows.append({"anthology_id": aid, "edition_key": key})

    return df.merge(pd.DataFrame(meta_rows), on="anthology_id")


# ── Counting and survival curve ───────────────────────────────────────────────

def edition_counts(df: pd.DataFrame, id_col: str) -> np.ndarray:
    """Number of unique edition_keys each item appears in."""
    return df.groupby(id_col)["edition_key"].nunique().values


def survival_curve(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For k = 1 … max(counts), return the percentage of items with count >= k.
    Always starts at (1, 100.0).
    """
    max_k = int(counts.max())
    ks    = np.arange(1, max_k + 1)
    pcts  = np.array([100.0 * np.sum(counts >= k) / len(counts) for k in ks])
    return ks, pcts


# ── Plot ──────────────────────────────────────────────────────────────────────

WORK_COLOR   = "#1f77b4"   # blue
AUTHOR_COLOR = "#d62728"   # red


def plot(
    work_ks:   np.ndarray, work_pcts:   np.ndarray, n_works:   int,
    auth_ks:   np.ndarray, auth_pcts:   np.ndarray, n_authors: int,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    scale = 1.5   # pct × scale → scatter area in points²

    work_label   = f"Works (N={n_works})"
    author_label = f"Authors (N={n_authors})"

    ax.plot(work_ks, work_pcts, color=WORK_COLOR, linewidth=0.8, alpha=0.4, zorder=2)
    ax.scatter(work_ks, work_pcts, s=work_pcts * scale,
               color=WORK_COLOR, marker="o", zorder=3, label=work_label)

    ax.plot(auth_ks, auth_pcts, color=AUTHOR_COLOR, linewidth=0.8, alpha=0.4, zorder=2)
    ax.scatter(auth_ks, auth_pcts, s=auth_pcts * scale,
               color=AUTHOR_COLOR, marker="s", zorder=3, label=author_label)

    ax.set_xlabel("Number of anthology editions selecting an item", fontsize=11)
    ax.set_ylabel("% of items selected in at least this many editions", fontsize=11)
    ax.set_title(
        "Selection frequency decay across African American literature anthology editions",
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    works_df   = assign_edition_key(pd.read_csv(WORKS_FILE,   dtype=str, na_filter=False))
    authors_df = assign_edition_key(pd.read_csv(AUTHORS_FILE, dtype=str, na_filter=False))

    work_counts   = edition_counts(works_df,   "work_id")
    author_counts = edition_counts(authors_df, "author_id")

    work_ks,   work_pcts   = survival_curve(work_counts)
    author_ks, author_pcts = survival_curve(author_counts)

    OUT_FILE.parent.mkdir(exist_ok=True)
    plot(work_ks, work_pcts, len(work_counts),
         author_ks, author_pcts, len(author_counts),
         OUT_FILE)


if __name__ == "__main__":
    main()
