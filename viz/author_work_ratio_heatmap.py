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
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORK_DATA_FILE = (
    Path(__file__).parent.parent / "data" / "2026-03-13 work ids in afam anthologies.csv"
)
AUTHOR_DATA_FILE = (
    Path(__file__).parent.parent / "data" / "2026-03-13 author ids in afam anthologies.csv"
)
OUT_FILE = Path(__file__).parent / "author_work_ratio_heatmap.png"

SERIES_ABBREV: dict[str, str] = {
    "The Norton Anthology of African American Literature":          "NAAAL",
    "Afro-American Writing: An Anthology of Prose and Poetry":     "Afro-Am. Writing",
    "African American Literature":                                  "AAL Anthology",
    "The Wiley Blackwell Anthology of African American Literature": "Wiley Blackwell AAL",
}

STANDALONE_SHORT: dict[str, str] = {
    "67":  "Amer. Negro Lit.",
    "66":  "Readings fr. Negro Authors",
    "62":  "Negro Caravan",
    "63":  "Amer. Lit. by Negro Authors",
    "56":  "Intro to Black Lit.",
    "43":  "Black Voices",
    "42":  "Cavalcade",
    "46":  "Black Insights",
    "48":  "Black Lit. in America",
    "47":  "Black Writers of America",
    "109": "Black Culture",
    "64":  "Cornerstones",
    "44":  "AAL Brief Intro",
    "49":  "Call & Response",
    "39":  "Prentice Hall AAL",
    "86":  "Afr. Am. Lit.",
}

MULTIVOL_SHORT: dict[str, str] = {
    "Blackamerican Literature, 1760-Present|1":                                        "Blackamerican Lit.",
    "The New Cavalcade: African American Writing from 1760 to the Present|1":         "New Cavalcade",
}


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_works   = pd.read_csv(WORK_DATA_FILE,   dtype=str, na_filter=False)
    df_authors = pd.read_csv(AUTHOR_DATA_FILE, dtype=str, na_filter=False)
    return df_works, df_authors


# ── 2. Assign edition keys ────────────────────────────────────────────────────

def _strip_volume(title: str) -> str:
    """Remove trailing ', vol. N' / ', Volume N' markers."""
    return re.sub(r",?\s+[Vv]ol\.?\s+\d+\s*$", "", title).strip()


def assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    meta_rows: list[dict] = []
    for _, r in (
        df[["anthology_id", "anthology_title", "series",
            "edition_number", "volume", "publication_year"]]
        .drop_duplicates()
        .iterrows()
    ):
        series  = r["series"].strip()
        edition = r["edition_number"].strip()
        volume  = r["volume"].strip()
        title   = r["anthology_title"].strip()
        year    = int(r["publication_year"])
        aid     = r["anthology_id"]

        if series:
            key = f"{series}|{edition}"
        elif volume:
            key = f"{_strip_volume(title)}|{edition}"
        else:
            key = aid

        meta_rows.append({"anthology_id": aid, "edition_key": key, "sort_year": year})

    return df.merge(pd.DataFrame(meta_rows), on="anthology_id")


# ── 3. Build per-edition sets ─────────────────────────────────────────────────

def build_work_sets(df: pd.DataFrame) -> dict[str, set[str]]:
    return {key: set(grp["work_id"]) for key, grp in df.groupby("edition_key")}


def build_author_sets(df: pd.DataFrame) -> dict[str, set[str]]:
    return {key: set(grp["author_id"]) for key, grp in df.groupby("edition_key")}


# ── 4. Ordering and labels ────────────────────────────────────────────────────

def sort_year_for(key: str, df: pd.DataFrame) -> int:
    return df.loc[df["edition_key"] == key, "sort_year"].min()


def make_label(key: str, df: pd.DataFrame) -> str:
    year = sort_year_for(key, df)
    if "|" in key:
        head, edition = key.rsplit("|", 1)
        if head in SERIES_ABBREV:
            return f"{SERIES_ABBREV[head]} ed.{edition}\n{year}"
        if key in MULTIVOL_SHORT:
            return f"{MULTIVOL_SHORT[key]}\n{year}"
        return f"{head[:20]}\n{year}"
    short = STANDALONE_SHORT.get(key, key[:20])
    return f"{short}\n{year}"


# ── 5. Ratio matrix ───────────────────────────────────────────────────────────

def ratio_matrix(
    keys: list[str],
    work_sets: dict[str, set[str]],
    author_sets: dict[str, set[str]],
) -> np.ndarray:
    """Cell (i, j) = |authors_i ∩ authors_j| / |works_i ∩ works_j|.

    NaN when works-in-common = 0.
    """
    n = len(keys)
    mat = np.full((n, n), np.nan)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            works_common   = len(work_sets.get(ki, set())   & work_sets.get(kj, set()))
            authors_common = len(author_sets.get(ki, set()) & author_sets.get(kj, set()))
            if works_common > 0:
                mat[i, j] = authors_common / works_common
    return mat


# ── 6. Plot ───────────────────────────────────────────────────────────────────

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
    # Ensure the diverging norm has room on both sides of 1.0
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
        "(cell = authors shared / works shared; grey = no shared works; "
        "blue > 1 = more authors than works shared)",
        fontsize=11,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")


# ── 7. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    df_works, df_authors = load()
    df_works   = assign_edition_key(df_works)
    df_authors = assign_edition_key(df_authors)

    work_sets   = build_work_sets(df_works)
    author_sets = build_author_sets(df_authors)

    # Order by publication year, using the work file as authority
    all_keys = sorted(
        set(work_sets) | set(author_sets),
        key=lambda k: sort_year_for(k, df_works),
    )
    labels = [make_label(k, df_works) for k in all_keys]

    OUT_FILE.parent.mkdir(exist_ok=True)

    mat = ratio_matrix(all_keys, work_sets, author_sets)

    # ── Summary statistic: % of distinct pairings where authors > works ───────
    n = len(all_keys)
    total_pairs = 0
    pairs_authors_gt_works = 0
    for i in range(n):
        for j in range(i + 1, n):  # upper triangle only — each pair counted once
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

    plot(mat, labels, OUT_FILE)
    plot(mat, labels, OUT_FILE.with_stem(OUT_FILE.stem + "_bw"), cmap="bwr")


if __name__ == "__main__":
    main()
