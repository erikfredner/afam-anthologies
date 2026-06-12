"""
selection_frequency_distribution.py
-----------------------------------
For every work and author selected in an African-American Literature anthology
edition, counts how many distinct editions include them, then plots the
distribution of *exact* edition counts: the % of works and authors appearing
in exactly k editions, side by side.

Unlike selection_frequency_decay.py (which plots the cumulative ">= k"
survival curve, where an author in ten editions contributes to all ten bins),
every work and author here is counted exactly once, in its own k bin.

Usage:
    uv run python viz/inequality/selection_frequency_distribution.py
    uv run python viz/inequality/selection_frequency_distribution.py --include-excerpts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afam.cli import add_root_works_flag
from afam.db import query
from afam.sql import query_path
from afam.viz_style import AUTHOR_COLOR, OUTPUT_DIR, WORK_COLOR

OUT_FILE = OUTPUT_DIR / "selection_frequency_distribution.png"
GRID_KW = dict(alpha=0.25, linestyle=":")
SMALL_N = 10


def edition_counts(df: pd.DataFrame, id_col: str) -> np.ndarray:
    """Distinct-edition count per entity (one value per work/author)."""
    return (
        df.dropna(subset=[id_col, "edition_id"])
        .drop_duplicates([id_col, "edition_id"])
        .groupby(id_col)["edition_id"]
        .size()
        .to_numpy(dtype=int)
    )


def exact_k_stats(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ks, ns (entities in exactly k editions), pcts for k = 1 … max."""
    max_k = int(counts.max())
    ks = np.arange(1, max_k + 1)
    ns = np.array([int(np.sum(counts == k)) for k in ks])
    pcts = 100.0 * ns / len(counts)
    return ks, ns, pcts


def plot(
    work_ks: np.ndarray,
    work_ns: np.ndarray,
    work_pcts: np.ndarray,
    n_works: int,
    auth_ks: np.ndarray,
    auth_ns: np.ndarray,
    auth_pcts: np.ndarray,
    n_authors: int,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_w = 0.4
    ax.bar(
        work_ks - bar_w / 2,
        work_pcts,
        width=bar_w,
        color=WORK_COLOR,
        alpha=0.8,
        label=f"Works (N={n_works:,})",
    )
    ax.bar(
        auth_ks + bar_w / 2,
        auth_pcts,
        width=bar_w,
        color=AUTHOR_COLOR,
        alpha=0.8,
        label=f"Authors (N={n_authors:,})",
    )

    for ks, ns, pcts, offset in [
        (work_ks, work_ns, work_pcts, -bar_w / 2),
        (auth_ks, auth_ns, auth_pcts, bar_w / 2),
    ]:
        for k, n, pct in zip(ks, ns, pcts, strict=True):
            if 0 < n < SMALL_N:
                ax.annotate(
                    f"{n}",
                    (k + offset, pct),
                    textcoords="offset points",
                    xytext=(0, 2),
                    ha="center",
                    fontsize=7,
                    color="gray",
                )

    ax.set_yscale("log")
    ax.set_xlabel("Number of anthology editions selecting an item (k)", fontsize=11)
    ax.set_ylabel("% of items selected in exactly k editions", fontsize=11)
    ax.set_title(
        "Distribution of author and work selection counts across anthologies",
        fontsize=12,
    )
    max_k = max(int(work_ks.max()), int(auth_ks.max()))
    ax.set_xlim(0.5, max_k + 0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, **GRID_KW)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the exact-k distribution of author and work selection counts."
    )
    add_root_works_flag(parser)
    args = parser.parse_args()

    raw = query(query_path("works-per-afam-edition"))
    if args.only_root_works:
        raw = raw[raw["parent_id"].isna()].copy()

    work_counts = edition_counts(raw, "work_id")
    author_counts = edition_counts(raw, "author_id")

    work_ks, work_ns, work_pcts = exact_k_stats(work_counts)
    auth_ks, auth_ns, auth_pcts = exact_k_stats(author_counts)

    plot(
        work_ks,
        work_ns,
        work_pcts,
        len(work_counts),
        auth_ks,
        auth_ns,
        auth_pcts,
        len(author_counts),
        OUT_FILE,
    )


if __name__ == "__main__":
    main()
