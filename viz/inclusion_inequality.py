"""
inclusion_inequality.py
-----------------------
Compute inclusion-frequency distributions for authors and works (unique
anthology editions, volumes merged via edition_key).  Report Gini coefficients
and plot Lorenz curves for both.

Two variants are produced:
  - inclusion_inequality.png          all works
  - inclusion_inequality_root.png     root works only (parent_work_id == "")
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKS_FILE   = Path(__file__).parent.parent / "data" / "2026-03-13 works per afam anthology.csv"
AUTHORS_FILE = Path(__file__).parent.parent / "data" / "2026-03-13 author ids in afam anthologies.csv"
OUT_FILE      = Path(__file__).parent / "inclusion_inequality.png"
OUT_FILE_ROOT = Path(__file__).parent / "inclusion_inequality_root.png"

# ── Edition-key logic (mirrors heatmap / decay scripts) ──────────────────────

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


def load_works() -> pd.DataFrame:
    """
    Load the works-per-anthology file, renaming columns to match the names
    expected by assign_edition_key.
    """
    df = pd.read_csv(WORKS_FILE, dtype=str, na_filter=False)
    return df.rename(columns={
        "anthology_series": "series",
        "anthology_edition": "edition_number",
        "anthology_volume":  "volume",
    })


def edition_counts(df: pd.DataFrame, id_col: str) -> np.ndarray:
    """Number of unique edition_keys each item appears in."""
    return df.groupby(id_col)["edition_key"].nunique().values


# ── Gini and Lorenz ───────────────────────────────────────────────────────────

def gini(counts: np.ndarray) -> float:
    x = np.sort(counts).astype(float)
    n = len(x)
    ranks = np.arange(1, n + 1)
    return float((2 * np.dot(ranks, x)) / (n * x.sum()) - (n + 1) / n)


def lorenz(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (population_share, inclusion_share), each starting at 0 and
    ending at 1, suitable for plotting.
    """
    x = np.sort(counts).astype(float)
    pop = np.concatenate([[0.0], np.arange(1, len(x) + 1) / len(x)])
    inc = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
    return pop, inc


# ── Plot ──────────────────────────────────────────────────────────────────────

WORK_COLOR   = "#1f77b4"   # blue
AUTHOR_COLOR = "#d62728"   # red


def plot(
    work_counts:   np.ndarray,
    author_counts: np.ndarray,
    out: Path,
    title: str,
) -> None:
    work_g   = gini(work_counts)
    author_g = gini(author_counts)

    work_pop,   work_inc   = lorenz(work_counts)
    author_pop, author_inc = lorenz(author_counts)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8,
            linestyle="--", label="Perfect equality")

    ax.plot(work_pop, work_inc, color=WORK_COLOR, linewidth=1.8,
            label=f"Works (N={len(work_counts)}, Gini={work_g:.3f})")
    ax.plot(author_pop, author_inc, color=AUTHOR_COLOR, linewidth=1.8,
            label=f"Authors (N={len(author_counts)}, Gini={author_g:.3f})")

    ax.fill_between(work_pop,   work_inc,   work_pop,   alpha=0.08, color=WORK_COLOR)
    ax.fill_between(author_pop, author_inc, author_pop, alpha=0.08, color=AUTHOR_COLOR)

    ax.set_xlabel("Cumulative share of items (sorted by inclusion frequency)", fontsize=11)
    ax.set_ylabel("Cumulative share of anthology inclusions", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    print(f"  Works  Gini: {work_g:.4f}")
    print(f"  Author Gini: {author_g:.4f}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    works_df   = assign_edition_key(load_works())
    authors_df = assign_edition_key(pd.read_csv(AUTHORS_FILE, dtype=str, na_filter=False))

    author_counts = edition_counts(authors_df, "author_id")

    OUT_FILE.parent.mkdir(exist_ok=True)

    # All works
    all_work_counts = edition_counts(works_df, "work_id")
    plot(
        all_work_counts, author_counts,
        OUT_FILE,
        "Inequality in anthology inclusion\nacross African American literature",
    )

    # Root works only (exclude excerpts / selections)
    root_works_df   = works_df[works_df["parent_work_id"] == ""]
    root_work_counts = edition_counts(root_works_df, "work_id")
    plot(
        root_work_counts, author_counts,
        OUT_FILE_ROOT,
        "Inequality in anthology inclusion\nacross African American literature (root works only)",
    )


if __name__ == "__main__":
    main()
