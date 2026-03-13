"""
author_overlap_heatmap.py
-------------------------
For every pair of anthology editions, compute the percentage of the row
edition's author IDs that also appear in the column edition.  Editions that
span multiple volumes are merged before comparison.  Renders a viridis
heatmap ordered chronologically by publication year.

Cell (row, col) = |authors_in_row ∩ authors_in_col| / |authors_in_row| × 100
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = Path(__file__).parent.parent / "data" / "2026-03-13 author ids in afam anthologies.csv"
OUT_FILE = Path(__file__).parent / "author_overlap_heatmap.png"

# Abbreviated display names for the heatmap axes
SERIES_ABBREV: dict[str, str] = {
    "The Norton Anthology of African American Literature":         "NAAAL",
    "Afro-American Writing: An Anthology of Prose and Poetry":    "Afro-Am. Writing",
    "African American Literature":                                 "AAL Anthology",
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

# Multi-volume standalones: stripped title|edition → short label
MULTIVOL_SHORT: dict[str, str] = {
    "Blackamerican Literature, 1760-Present|1":                                            "Blackamerican Lit.",
    "The New Cavalcade: African American Writing from 1760 to the Present|1":             "New Cavalcade",
}


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE, dtype=str, na_filter=False)


# ── 2. Assign edition keys ────────────────────────────────────────────────────

def _strip_volume(title: str) -> str:
    """Remove trailing ', vol. N' / ', Volume N' markers."""
    return re.sub(r",?\s+[Vv]ol\.?\s+\d+\s*$", "", title).strip()


def assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'edition_key' (str) and 'sort_year' (int) columns.

    Grouping logic:
      - series non-empty  →  "{series}|{edition_number}"
      - series empty, volume non-empty  →  "{stripped_title}|{edition_number}"
      - series empty, volume empty  →  anthology_id (unique standalone)
    """
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


# ── 3. Build work-sets per edition ────────────────────────────────────────────

def build_edition_sets(df: pd.DataFrame) -> dict[str, set[str]]:
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
        # Fallback: first word(s) of stripped title
        return f"{head[:20]}\n{year}"
    # Standalone
    short = STANDALONE_SHORT.get(key, key[:20])
    return f"{short}\n{year}"


# ── 5. Overlap matrix ─────────────────────────────────────────────────────────

def overlap_matrix(keys: list[str], sets: dict[str, set[str]]) -> np.ndarray:
    """Cell (i, j) = |sets[i] ∩ sets[j]| / |sets[i]| × 100."""
    n = len(keys)
    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        si = sets[ki]
        if not si:
            continue
        for j, kj in enumerate(keys):
            mat[i, j] = 100.0 * len(si & sets[kj]) / len(si)
    return mat


# ── 6. Plot ───────────────────────────────────────────────────────────────────

def plot(mat: np.ndarray, labels: list[str], out: Path) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(18, 16))

    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("% of row edition's authors also in column edition", fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)

    # Annotate cells with integer percentages
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            text_color = "white" if v < 55 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=5, color=text_color)

    ax.set_title(
        "Author overlap between African American literature anthology editions\n"
        "(cell = % of row edition's authors also present in column edition)",
        fontsize=12, pad=14,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")


# ── 7. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    df = load()
    df = assign_edition_key(df)
    edition_sets = build_edition_sets(df)

    keys = sorted(edition_sets, key=lambda k: sort_year_for(k, df))
    labels = [make_label(k, df) for k in keys]

    mat = overlap_matrix(keys, edition_sets)
    OUT_FILE.parent.mkdir(exist_ok=True)
    plot(mat, labels, OUT_FILE)


if __name__ == "__main__":
    main()
