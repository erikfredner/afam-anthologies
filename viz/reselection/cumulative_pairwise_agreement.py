"""cumulative_pairwise_agreement.py
-----------------------------------
Track how pairwise author-roster and work-selection agreement evolve
cumulatively as new anthology editions enter the corpus.

Figure: viz/cumulative_pairwise_agreement.png

Usage: uv run python viz/cumulative_pairwise_agreement.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────────────

from afam.db import query
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

OUT_DIR = OUTPUT_DIR


# ── Style constants (match work_selection_divergence.py) ───────────────────────

C_BLUE = "#1f77b4"
C_RED = "#d62728"
GRID_KW = dict(alpha=0.25, linestyle=":")


def _clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _load_and_prepare_db(only_root_works: bool = True) -> pd.DataFrame:
    df = query(query_path("work-selection-divergence"))
    print(f"Loaded {len(df)} DB rows, {df['edition_id'].nunique()} editions")
    if only_root_works:
        df = df[df["parent_id"].isna()].copy()
        print(f"  Root-work rows: {len(df)}")

    df = df.dropna(subset=["author_id", "work_id"]).copy()
    df["edition_key"] = [
        f"{_clean_id(series_id)}|{_clean_id(edition_number)}"
        if _clean_id(series_id)
        else _clean_id(edition_id)
        for series_id, edition_number, edition_id in zip(
            df["series_id"], df["edition_number"], df["edition_id"], strict=False
        )
    ]
    df["ek_year"] = df["anthology_publication_year"].astype(int)
    df["author_id"] = df["author_id"].map(_clean_id)
    df["work_id"] = df["work_id"].map(_clean_id)
    long = df[["edition_key", "ek_year", "author_id", "work_id"]].drop_duplicates()
    print(f"  Unique edition_keys: {long['edition_key'].nunique()}")
    print(f"  Unique authors: {long['author_id'].nunique()}")
    print(f"  Unique works: {long['work_id'].nunique()}")
    print(f"  Total long rows: {len(long)}")
    return long


# ── Core metric ────────────────────────────────────────────────────────────────


def jaccard(set_a: frozenset, set_b: frozenset) -> float:
    """Jaccard similarity of two sets.

    Returns nan when both sets are empty.
    """
    union = set_a | set_b
    if not union:
        return math.nan
    return len(set_a & set_b) / len(union)


# ── Snapshot computation ───────────────────────────────────────────────────────


def compute_snapshots(long: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative pairwise Jaccard snapshots.

    Parameters
    ----------
    long:
        DataFrame with columns: edition_key, ek_year, author_id, and either
        work_id or norm_title.

    Returns
    -------
    DataFrame with columns:
        year, n_editions, n_pairs,
        author_median, author_p25, author_p75,
        work_median, work_p25, work_p75
    """
    # Precompute per-edition author and work sets
    ek_to_authors: dict[str, frozenset] = {
        ek: frozenset(grp["author_id"].unique())
        for ek, grp in long.groupby("edition_key")
    }
    work_col = "work_id" if "work_id" in long.columns else "norm_title"
    ek_to_works: dict[str, frozenset] = {
        ek: frozenset(grp[work_col].unique()) for ek, grp in long.groupby("edition_key")
    }

    # Edition year lookup
    ek_to_year: dict[str, int] = (
        long.groupby("edition_key")["ek_year"].first().to_dict()
    )

    sorted_years = sorted(set(ek_to_year.values()))

    editions_so_far: list[str] = []
    running_author: list[float] = []
    running_work: list[float] = []
    snapshots: list[dict] = []

    for year in sorted_years:
        new_eks = [ek for ek, yr in ek_to_year.items() if yr == year]

        # Add pairs: each new edition vs. all prior editions
        for new_ek in new_eks:
            for prior_ek in editions_so_far:
                running_author.append(
                    jaccard(ek_to_authors[new_ek], ek_to_authors[prior_ek])
                )
                running_work.append(jaccard(ek_to_works[new_ek], ek_to_works[prior_ek]))

        # Add pairs within this year's new editions
        for i, ek_a in enumerate(new_eks):
            for ek_b in new_eks[i + 1 :]:
                running_author.append(jaccard(ek_to_authors[ek_a], ek_to_authors[ek_b]))
                running_work.append(jaccard(ek_to_works[ek_a], ek_to_works[ek_b]))

        editions_so_far.extend(new_eks)

        if not running_author:
            continue  # first edition, no pairs yet

        a_arr = np.array(running_author, dtype=float)
        w_arr = np.array(running_work, dtype=float)

        snapshots.append(
            {
                "year": year,
                "n_editions": len(editions_so_far),
                "n_pairs": len(running_author),
                "author_median": float(np.nanmedian(a_arr)),
                "author_p25": float(np.nanpercentile(a_arr, 25)),
                "author_p75": float(np.nanpercentile(a_arr, 75)),
                "work_median": float(np.nanmedian(w_arr)),
                "work_p25": float(np.nanpercentile(w_arr, 25)),
                "work_p75": float(np.nanpercentile(w_arr, 75)),
            }
        )

    return pd.DataFrame(snapshots)


# ── Figure ─────────────────────────────────────────────────────────────────────


def plot_snapshots(df: pd.DataFrame, out_path: Path) -> None:
    """Render the cumulative pairwise agreement figure."""
    fig, ax = plt.subplots(figsize=(11, 6))

    years = df["year"].values

    # Author Jaccard — blue
    ax.plot(
        years,
        df["author_median"],
        color=C_BLUE,
        lw=2,
        label="Author Jaccard (median ± IQR)",
    )
    ax.fill_between(years, df["author_p25"], df["author_p75"], color=C_BLUE, alpha=0.18)

    # Work Jaccard — red
    ax.plot(
        years, df["work_median"], color=C_RED, lw=2, label="Work Jaccard (median ± IQR)"
    )
    ax.fill_between(years, df["work_p25"], df["work_p75"], color=C_RED, alpha=0.18)

    # Reference line
    ax.axhline(0.5, color="gray", linestyle="--", lw=1.0, zorder=0)

    # Annotate n_pairs at first, middle, last snapshots
    annot_indices = [0, len(df) // 2, len(df) - 1]
    seen = set()
    for idx in annot_indices:
        if idx in seen:
            continue
        seen.add(idx)
        row = df.iloc[idx]
        ax.annotate(
            f"n={int(row['n_pairs'])}",
            xy=(row["year"], row["author_median"]),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=7,
            color="gray",
            ha="center",
        )

    ax.set_xlim(years[0] - 2, years[-1] + 2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Jaccard similarity", fontsize=11)
    ax.set_title(
        "Pairwise anthology agreement over time\n(cumulative; all pairs up to each year)",
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, **GRID_KW)
    fig.tight_layout()

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    long = _load_and_prepare_db(only_root_works=True)
    snapshots = compute_snapshots(long)

    print(f"\nSnapshots: {len(snapshots)} years with ≥1 pair")
    if not snapshots.empty:
        last = snapshots.iloc[-1]
        print(
            f"Final year {int(last['year'])}: {int(last['n_editions'])} editions, "
            f"{int(last['n_pairs'])} pairs"
        )
        print(
            f"  Author Jaccard median={last['author_median']:.3f}  "
            f"Work Jaccard median={last['work_median']:.3f}"
        )

    out_path = OUT_DIR / "cumulative_pairwise_agreement.png"
    plot_snapshots(snapshots, out_path)


if __name__ == "__main__":
    main()
