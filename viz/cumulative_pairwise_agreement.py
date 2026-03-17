"""cumulative_pairwise_agreement.py
-----------------------------------
Track how pairwise author-roster and work-selection agreement evolve
cumulatively as new anthology editions enter the corpus.

Figure: viz/cumulative_pairwise_agreement.png

Usage: uv run python viz/cumulative_pairwise_agreement.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_FILE = (
    Path(__file__).parent.parent / "data" / "2026-03-13 works per afam anthology.csv"
)
OUT_DIR = Path(__file__).parent


# ── Style constants (match work_selection_divergence.py) ───────────────────────

C_BLUE = "#1f77b4"
C_RED = "#d62728"
GRID_KW = dict(alpha=0.25, linestyle=":")


# ── Title normalization (private copy) ─────────────────────────────────────────


def _normalize_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"^(from |excerpt from |selections? from )", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip()


# ── Load and prepare (private copy, adapted from work_selection_divergence.py) ─


def _load_and_prepare(data_file: Path) -> pd.DataFrame:
    df = pd.read_csv(data_file, dtype=str, na_filter=False)
    print(f"Loaded {len(df)} rows, {df['anthology_id'].nunique()} raw anthologies")

    # Build edition_key: series_id|edition when series, else anthology_id
    df["edition_key"] = df.apply(
        lambda r: (
            f"{r['series_id']}|{r['anthology_edition']}"
            if r["series_id"].strip()
            else r["anthology_id"]
        ),
        axis=1,
    )

    # Year per edition_key (minimum year across volumes of same edition)
    year_by_key = (
        df.groupby("edition_key")["anthology_publication_year"].min().astype(int)
    )
    df["ek_year"] = df["edition_key"].map(year_by_key)

    # Normalize work titles
    df["norm_title"] = df["work_title"].apply(_normalize_title)

    # Explode multi-author rows — split author_ids on ","
    df["_author_list"] = df["author_ids"].apply(
        lambda s: [i.strip() for i in s.split(",") if i.strip()]
    )
    df_auth = df[df["_author_list"].apply(len) > 0].copy()
    df_auth = df_auth.explode("_author_list")
    df_auth = df_auth.rename(columns={"_author_list": "author_id"})

    long = df_auth.drop(
        columns=[c for c in df_auth.columns if c.startswith("_")],
        errors="ignore",
    )

    print(f"  Unique edition_keys: {long['edition_key'].nunique()}")
    print(f"  Unique authors: {long['author_id'].nunique()}")
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
        DataFrame with columns: edition_key, ek_year, author_id, norm_title.

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
    ek_to_works: dict[str, frozenset] = {
        ek: frozenset(grp["norm_title"].unique())
        for ek, grp in long.groupby("edition_key")
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
                running_author.append(jaccard(ek_to_authors[new_ek], ek_to_authors[prior_ek]))
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
    ax.plot(years, df["author_median"], color=C_BLUE, lw=2, label="Author Jaccard (median ± IQR)")
    ax.fill_between(years, df["author_p25"], df["author_p75"], color=C_BLUE, alpha=0.18)

    # Work Jaccard — red
    ax.plot(years, df["work_median"], color=C_RED, lw=2, label="Work Jaccard (median ± IQR)")
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
    long = _load_and_prepare(DATA_FILE)
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
