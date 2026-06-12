"""
continuation_probability.py
---------------------------
Conditional continuation curve for author and work selections across
African-American Literature anthology editions: among items already selected
in at least k editions, what fraction were selected at least once more?

    p(k) = N(selected in >= k+1 editions) / N(selected in >= k editions)

Each point is a conditional probability with an explicit denominator, so —
unlike the cumulative ">= k" survival curve in selection_frequency_decay.py —
no item inflates the comparison by being counted across every bin. A higher
author curve at every k articulates the claim directly: at each level of
prior inclusion, authors are more likely than works to be selected again.

Points whose denominator N(>= k) falls below MIN_N (5) are dropped rather
than plotted with uninformative confidence intervals. 95% CIs are Wilson.

Usage:
    uv run python viz/reselection/continuation_probability.py
    uv run python viz/reselection/continuation_probability.py --include-excerpts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from afam.cli import add_root_works_flag
from afam.db import query
from afam.sql import query_path
from afam.viz_style import AUTHOR_COLOR, OUTPUT_DIR, WORK_COLOR

OUT_FILE = OUTPUT_DIR / "continuation_probability.png"
GRID_KW = dict(alpha=0.25, linestyle=":")
MIN_N = 5


def edition_counts(df: pd.DataFrame, id_col: str) -> np.ndarray:
    """Distinct-edition count per entity (one value per work/author)."""
    return (
        df.dropna(subset=[id_col, "edition_id"])
        .drop_duplicates([id_col, "edition_id"])
        .groupby(id_col)["edition_id"]
        .size()
        .to_numpy(dtype=int)
    )


def continuation_stats(counts: np.ndarray, min_n: int = MIN_N) -> pd.DataFrame:
    """Per-k continuation probabilities with Wilson 95% CIs.

    Rows cover k = 1 … max-1 where the denominator N(>= k) is at least
    ``min_n``; p_continue = N(>= k+1) / N(>= k).
    """
    max_k = int(counts.max())
    rows = []
    for k in range(1, max_k):
        n_k = int(np.sum(counts >= k))
        if n_k < min_n:
            break
        n_k1 = int(np.sum(counts >= k + 1))
        lo, hi = proportion_confint(n_k1, n_k, alpha=0.05, method="wilson")
        rows.append(
            {
                "k": k,
                "n_at_least_k": n_k,
                "n_at_least_k_plus_1": n_k1,
                "p_continue": n_k1 / n_k,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
            }
        )
    return pd.DataFrame(rows)


def plot(work_stats: pd.DataFrame, auth_stats: pd.DataFrame, out: Path) -> None:
    fig, (ax, ax_n) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    for stats, color, marker, label in [
        (work_stats, WORK_COLOR, "o", "Works"),
        (auth_stats, AUTHOR_COLOR, "s", "Authors"),
    ]:
        ax.fill_between(
            stats["k"],
            100 * stats["ci_lo"],
            100 * stats["ci_hi"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            stats["k"],
            100 * stats["p_continue"],
            color=color,
            marker=marker,
            linewidth=1.2,
            label=label,
        )

    ax.set_ylabel("P(selected in ≥ k+1 editions | selected in ≥ k)  (%)", fontsize=11)
    ax.set_title(
        "Conditional continuation probability of author and work selections",
        fontsize=12,
    )
    ax.set_ylim(-2, 105)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, **GRID_KW)

    bar_w = 0.4
    ax_n.bar(
        work_stats["k"] - bar_w / 2,
        work_stats["n_at_least_k"],
        width=bar_w,
        color=WORK_COLOR,
        alpha=0.7,
        label="Works",
    )
    ax_n.bar(
        auth_stats["k"] + bar_w / 2,
        auth_stats["n_at_least_k"],
        width=bar_w,
        color=AUTHOR_COLOR,
        alpha=0.7,
        label="Authors",
    )
    ax_n.set_yscale("log")
    ax_n.set_xlabel("Prior selections (k editions)", fontsize=11)
    ax_n.set_ylabel("N (≥ k editions)", fontsize=9)
    ax_n.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_n.grid(True, **GRID_KW)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot conditional continuation probabilities for authors vs works."
    )
    add_root_works_flag(parser)
    args = parser.parse_args()

    raw = query(query_path("works-per-afam-edition"))
    if args.only_root_works:
        raw = raw[raw["parent_id"].isna()].copy()

    work_stats = continuation_stats(edition_counts(raw, "work_id"))
    auth_stats = continuation_stats(edition_counts(raw, "author_id"))

    plot(work_stats, auth_stats, OUT_FILE)


if __name__ == "__main__":
    main()
