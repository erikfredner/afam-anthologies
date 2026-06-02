"""
author_work_ratio_heatmap.py
----------------------------
For every pair of anthology editions, compute:

    cell (row, col) = |authors in common| / |works in common|

Values > 1 mean the two editions share more unique authors than unique works —
i.e. the same author appears in both via *different* works.  Values < 1 mean
shared works span fewer unique authors (some authors have multiple shared works).
Diagonal = total unique authors / total unique works for that edition.

Cells where works-in-common = 0 are masked (shown in grey).
Within-series pairs (e.g. NAAAL ed.1 vs NAAAL ed.2) are masked by default;
use --include-within-series to show them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afam.db import query as query_db
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

OUT_FILE = OUTPUT_DIR / "author_work_ratio_heatmap.png"

SERIES_ID_ABBREV: dict[str, str] = {
    "3": "NAAAL",
    "8": "Afro-Am. Writing",
    "12": "AAL Anthology",
    "17": "Wiley Blackwell AAL",
    "19": "Cavalcade",
}

STANDALONE_SHORT: dict[str, str] = {
    "67": "Amer. Negro Lit.",
    "66": "Readings fr. Negro Authors",
    "62": "Negro Caravan",
    "63": "Amer. Lit. by Negro Authors",
    "56": "Intro to Black Lit.",
    "43": "Black Voices",
    "46": "Black Insights",
    "48": "Black Lit. in America",
    "47": "Black Writers of America",
    "109": "Black Culture",
    "64": "Cornerstones",
    "44": "AAL Brief Intro",
    "49": "Call & Response",
    "39": "Prentice Hall AAL",
    "86": "Afr. Am. Lit.",
    "50": "Blackamerican Lit.",
}


# ── Load and prepare ──────────────────────────────────────────────────────────


def load_and_prepare_db() -> tuple[dict[str, set], dict[str, set], dict[str, int]]:
    df = query_db(query_path("work-selection-divergence"))

    df["edition_key"] = df.apply(
        lambda r: (
            f"{int(r['series_id'])}|{r['edition_number']}"
            if pd.notna(r["series_id"])
            else str(r["edition_id"])
        ),
        axis=1,
    )

    work_sets: dict[str, set] = {
        key: set(grp["work_id"].dropna().astype(str))
        for key, grp in df.groupby("edition_key")
    }
    author_sets: dict[str, set] = {
        key: set(grp["author_id"].dropna().astype(str))
        for key, grp in df.groupby("edition_key")
    }
    year_map: dict[str, int] = (
        df.groupby("edition_key")["anthology_publication_year"]
        .min()
        .astype(int)
        .to_dict()
    )
    return work_sets, author_sets, year_map


# ── Labels ────────────────────────────────────────────────────────────────────


def make_label(edition_key: str, year: int) -> str:
    if "|" in edition_key:
        series_id, edition = edition_key.rsplit("|", 1)
        abbrev = SERIES_ID_ABBREV.get(series_id, series_id)
        return f"{abbrev} ed.{edition}\n{year}"
    short = STANDALONE_SHORT.get(edition_key, edition_key[:20])
    return f"{short}\n{year}"


# ── Series helpers ────────────────────────────────────────────────────────────


def _same_series(ki: str, kj: str) -> bool:
    """True when both keys belong to the same named series."""
    if "|" not in ki or "|" not in kj:
        return False
    return ki.rsplit("|", 1)[0] == kj.rsplit("|", 1)[0]


# ── Ratio matrix ──────────────────────────────────────────────────────────────


def ratio_matrix(
    keys: list[str],
    work_sets: dict[str, set],
    author_sets: dict[str, set],
    cross_series_only: bool = True,
) -> np.ndarray:
    """Cell (i, j) = |authors_i ∩ authors_j| / |works_i ∩ works_j|.

    NaN when works-in-common = 0 or the pair is filtered by cross_series_only.
    """
    n = len(keys)
    mat = np.full((n, n), np.nan)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if cross_series_only and ki != kj and _same_series(ki, kj):
                continue
            works_common = len(work_sets.get(ki, set()) & work_sets.get(kj, set()))
            authors_common = len(
                author_sets.get(ki, set()) & author_sets.get(kj, set())
            )
            if works_common > 0:
                mat[i, j] = authors_common / works_common
    return mat


# ── Jaccard matrix ────────────────────────────────────────────────────────────


def jaccard_matrix(
    keys: list[str],
    sets: dict[str, set],
    cross_series_only: bool = True,
) -> np.ndarray:
    """Cell (i, j) = |sets_i ∩ sets_j| / |sets_i ∪ sets_j|. NaN when union = 0 or filtered."""
    n = len(keys)
    mat = np.full((n, n), np.nan)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if cross_series_only and ki != kj and _same_series(ki, kj):
                continue
            inter = len(sets.get(ki, set()) & sets.get(kj, set()))
            union = len(sets.get(ki, set()) | sets.get(kj, set()))
            if union > 0:
                mat[i, j] = inter / union
    return mat


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot(
    mat: np.ndarray,
    labels: list[str],
    out: Path,
    cmap: str = "RdBu_r",
) -> None:
    n = len(labels)

    valid = mat[~np.isnan(mat)]
    vmin = float(np.min(valid))
    vmax = float(np.max(valid))
    norm = mcolors.TwoSlopeNorm(
        vcenter=1.0,
        vmin=min(vmin, 0.95),
        vmax=max(vmax, 1.05),
    )

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("lightgrey")
    masked = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Authors in common / Works in common", fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)

    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5)

    ax.set_title(
        "Authors in common vs. works in common between anthology editions\n"
        "(cell = authors shared / works shared; grey = no shared works or within-series pair; "
        "blue > 1 = more authors than works shared)",
        fontsize=11,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")


def plot_jaccard(
    mat: np.ndarray,
    labels: list[str],
    out: Path,
    cmap: str = "viridis",
    title: str = "",
) -> None:
    n = len(labels)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("lightgrey")
    masked = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(masked, cmap=cmap_obj, vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Jaccard similarity", fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)

    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5)

    ax.set_title(
        title
        + "\n(grey = no shared items or within-series pair; diagonal = edition vs. itself = 1.0)",
        fontsize=11,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")


# ── Argparse ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heatmap of authors-in-common / works-in-common for AFAM anthology pairs"
    )
    parser.add_argument(
        "--include-within-series",
        action="store_true",
        help=(
            "Include within-series pairs (e.g. NAAAL ed.1 vs NAAAL ed.2). "
            "Excluded by default to avoid inflating aggregate statistics."
        ),
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    cross_series_only = not args.include_within_series

    work_sets, author_sets, year_map = load_and_prepare_db()

    all_keys = sorted(
        set(work_sets) | set(author_sets),
        key=lambda k: year_map.get(k, 0),
    )
    labels = [make_label(k, year_map.get(k, 0)) for k in all_keys]

    OUT_FILE.parent.mkdir(exist_ok=True)

    mat = ratio_matrix(
        all_keys, work_sets, author_sets, cross_series_only=cross_series_only
    )

    # ── Summary statistic: % of distinct pairings where authors > works ───────
    n = len(all_keys)
    total_pairs = 0
    pairs_authors_gt_works = 0
    for i in range(n):
        for j in range(i + 1, n):  # upper triangle only — each pair counted once
            if cross_series_only and _same_series(all_keys[i], all_keys[j]):
                continue
            v = mat[i, j]
            if not np.isnan(v):
                total_pairs += 1
                if v > 1.0:
                    pairs_authors_gt_works += 1
    pct = 100.0 * pairs_authors_gt_works / total_pairs if total_pairs else float("nan")
    print(
        f"{pairs_authors_gt_works} of {total_pairs} pairings ({pct:.1f}%) have "
        f"more authors in common than works in common."
    )

    # ── Jaccard similarity ────────────────────────────────────────────────────
    author_jacc = jaccard_matrix(all_keys, author_sets, cross_series_only)
    work_jacc = jaccard_matrix(all_keys, work_sets, cross_series_only)

    # Collect upper-triangle values for summary
    aj_vals, wj_vals, pairs_author_jacc_gt_work_jacc = [], [], 0
    for i in range(n):
        for j in range(i + 1, n):
            if cross_series_only and _same_series(all_keys[i], all_keys[j]):
                continue
            aj = author_jacc[i, j]
            wj = work_jacc[i, j]
            if not np.isnan(aj):
                aj_vals.append(aj)
            if not np.isnan(wj):
                wj_vals.append(wj)
            if not np.isnan(aj) and not np.isnan(wj) and aj > wj:
                pairs_author_jacc_gt_work_jacc += 1

    aj_arr = np.array(aj_vals)
    wj_arr = np.array(wj_vals)
    n_both = min(len(aj_vals), len(wj_vals))
    pct_aj_gt_wj = (
        100.0 * pairs_author_jacc_gt_work_jacc / n_both if n_both else float("nan")
    )
    print(
        f"Author Jaccard: mean={np.mean(aj_arr):.3f}, median={np.median(aj_arr):.3f} "
        f"across {len(aj_arr)} pairs."
    )
    print(
        f"Work Jaccard:   mean={np.mean(wj_arr):.3f}, median={np.median(wj_arr):.3f} "
        f"across {len(wj_arr)} pairs."
    )
    print(
        f"{pairs_author_jacc_gt_work_jacc} of {n_both} pairings ({pct_aj_gt_wj:.1f}%) have "
        f"higher author Jaccard than work Jaccard."
    )

    plot(mat, labels, OUT_FILE)
    plot(mat, labels, OUT_FILE.with_stem(OUT_FILE.stem + "_bw"), cmap="bwr")
    plot_jaccard(
        author_jacc,
        labels,
        OUT_FILE.with_stem(OUT_FILE.stem + "_jaccard_authors"),
        title="Jaccard similarity — authors shared between anthology editions",
    )
    plot_jaccard(
        work_jacc,
        labels,
        OUT_FILE.with_stem(OUT_FILE.stem + "_jaccard_works"),
        title="Jaccard similarity — works shared between anthology editions",
    )


if __name__ == "__main__":
    main()
