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


DATA_FILE = (
    Path(__file__).parent.parent
    / "data"
    / "2026-03-13 author ids in afam anthologies.csv"
)
BIRTH_YEAR_FILE = Path(__file__).parent.parent / "data" / "2026-03-27 author birth.csv"
OUT_FILE = Path(__file__).parent / "author_overlap_heatmap.png"
BIRTH_YEAR_WINDOW = 5

# Abbreviated display names for the heatmap axes
SERIES_ABBREV: dict[str, str] = {
    "The Norton Anthology of African American Literature": "NAAAL",
    "Afro-American Writing: An Anthology of Prose and Poetry": "Afro-Am. Writing",
    "African American Literature": "AAL Anthology",
    "The Wiley Blackwell Anthology of African American Literature": "Wiley Blackwell AAL",
}

STANDALONE_SHORT: dict[str, str] = {
    "67": "Amer. Negro Lit.",
    "66": "Readings fr. Negro Authors",
    "62": "Negro Caravan",
    "63": "Amer. Lit. by Negro Authors",
    "56": "Intro to Black Lit.",
    "43": "Black Voices",
    "42": "Cavalcade",
    "46": "Black Insights",
    "48": "Black Lit. in America",
    "47": "Black Writers of America",
    "109": "Black Culture",
    "64": "Cornerstones",
    "44": "AAL Brief Intro",
    "49": "Call & Response",
    "39": "Prentice Hall AAL",
    "86": "Afr. Am. Lit.",
}

# Multi-volume standalones: stripped title|edition → short label
MULTIVOL_SHORT: dict[str, str] = {
    "Blackamerican Literature, 1760-Present|1": "Blackamerican Lit.",
    "The New Cavalcade: African American Writing from 1760 to the Present|1": "New Cavalcade",
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
        df[
            [
                "anthology_id",
                "anthology_title",
                "series",
                "edition_number",
                "volume",
                "publication_year",
            ]
        ]
        .drop_duplicates()
        .iterrows()
    ):
        series = r["series"].strip()
        edition = r["edition_number"].strip()
        volume = r["volume"].strip()
        title = r["anthology_title"].strip()
        year = int(r["publication_year"])
        aid = r["anthology_id"]

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


# ── 3b. Birth-year lookup ─────────────────────────────────────────────────────


def load_birth_years() -> dict[str, int]:
    """Return {author_id (str): birth_year (int)} from the author birth CSV."""
    df = pd.read_csv(BIRTH_YEAR_FILE, dtype=str, na_filter=False)
    result: dict[str, int] = {}
    for _, row in df.iterrows():
        aid = row["id"].strip()
        yr_str = row["birth_year"].strip()
        if not aid or not yr_str:
            continue
        try:
            result[aid] = int(yr_str)
        except ValueError:
            pass
    return result


def _max_known_birth_year(
    author_ids: set[str], birth_years: dict[str, int]
) -> int | None:
    years = [birth_years[a] for a in author_ids if a in birth_years]
    return max(years) if years else None


def _filter_pair(
    si: set[str],
    sj: set[str],
    birth_years: dict[str, int],
    window: int,
) -> tuple[set[str], set[str]]:
    """Return (si_filtered, sj_filtered) restricted to the contemporaneous author pool."""
    max_i = _max_known_birth_year(si, birth_years)
    max_j = _max_known_birth_year(sj, birth_years)
    known = [m for m in (max_i, max_j) if m is not None]
    if not known:
        return si, sj
    cutoff = min(known) + window
    si_filt = {a for a in si if birth_years.get(a, cutoff) <= cutoff}
    sj_filt = {a for a in sj if birth_years.get(a, cutoff) <= cutoff}
    return si_filt, sj_filt


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


# ── 5. Overlap matrices ───────────────────────────────────────────────────────


def overlap_matrix(
    keys: list[str],
    sets: dict[str, set[str]],
    birth_years: dict[str, int],
    window: int = BIRTH_YEAR_WINDOW,
) -> np.ndarray:
    """Cell (i, j) = |si_filt ∩ sj_filt| / |si_filt| × 100 (birth-year adjusted)."""
    n = len(keys)
    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            si_filt, sj_filt = _filter_pair(sets[ki], sets[kj], birth_years, window)
            if not si_filt:
                continue
            mat[i, j] = 100.0 * len(si_filt & sj_filt) / len(si_filt)
    return mat


def jaccard_matrix(
    keys: list[str],
    sets: dict[str, set[str]],
    birth_years: dict[str, int],
    window: int = BIRTH_YEAR_WINDOW,
) -> np.ndarray:
    """Cell (i, j) = |si_filt ∩ sj_filt| / |si_filt ∪ sj_filt| × 100 (birth-year adjusted)."""
    n = len(keys)
    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            si_filt, sj_filt = _filter_pair(sets[ki], sets[kj], birth_years, window)
            union = si_filt | sj_filt
            if not union:
                continue
            mat[i, j] = 100.0 * len(si_filt & sj_filt) / len(union)
    return mat


def counts_matrix(
    keys: list[str],
    sets: dict[str, set[str]],
    birth_years: dict[str, int],
    window: int = BIRTH_YEAR_WINDOW,
) -> np.ndarray:
    """Cell (i, j) = |si_filt ∩ sj_filt| raw count (birth-year adjusted)."""
    n = len(keys)
    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            si_filt, sj_filt = _filter_pair(sets[ki], sets[kj], birth_years, window)
            mat[i, j] = len(si_filt & sj_filt)
    return mat


# ── 6. Plot ───────────────────────────────────────────────────────────────────


def plot(
    mat: np.ndarray,
    labels: list[str],
    out: Path,
    cmap: str = "viridis",
    vmax: float = 100,
    cbar_label: str = "% of row edition's authors also in column edition",
    title: str = (
        "Author overlap between African American literature anthology editions\n"
        "(cell = % of row edition's authors also present in column edition)"
    ),
) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(18, 16))

    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)

    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            text_color = "white" if v < vmax * 0.55 else "black"
            ax.text(
                j, i, f"{v:.0f}", ha="center", va="center", fontsize=5, color=text_color
            )

    ax.set_title(title, fontsize=12, pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")


# ── 7. Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    df = load()
    df = assign_edition_key(df)
    edition_sets = build_edition_sets(df)
    birth_years = load_birth_years()

    keys = sorted(edition_sets, key=lambda k: sort_year_for(k, df))
    labels = [make_label(k, df) for k in keys]

    OUT_FILE.parent.mkdir(exist_ok=True)

    # ── Percentage overlap ────────────────────────────────────────────────────
    pct_mat = overlap_matrix(keys, edition_sets, birth_years)
    pct_kwargs = dict(
        cbar_label="% of row edition's authors also in column edition (birth-year adjusted)",
        title=(
            "Author overlap between African American literature anthology editions\n"
            "(cell = % of row edition's authors also present in column edition,"
            " birth-year adjusted)"
        ),
    )
    plot(pct_mat, labels, OUT_FILE, **pct_kwargs)
    plot(
        pct_mat,
        labels,
        OUT_FILE.with_stem(OUT_FILE.stem + "_bw"),
        cmap="Greys_r",
        **pct_kwargs,
    )

    # ── Jaccard similarity ────────────────────────────────────────────────────
    jac_mat = jaccard_matrix(keys, edition_sets, birth_years)
    jac_kwargs = dict(
        cbar_label="Jaccard similarity × 100 (birth-year adjusted)",
        title=(
            "Author overlap between African American literature anthology editions\n"
            "(cell = Jaccard similarity × 100, birth-year adjusted)"
        ),
    )
    plot(jac_mat, labels, OUT_FILE.with_stem(OUT_FILE.stem + "_jaccard"), **jac_kwargs)
    plot(
        jac_mat,
        labels,
        OUT_FILE.with_stem(OUT_FILE.stem + "_jaccard_bw"),
        cmap="Greys_r",
        **jac_kwargs,
    )

    # ── Raw counts ────────────────────────────────────────────────────────────
    cnt_mat = counts_matrix(keys, edition_sets, birth_years)
    cnt_vmax = int(cnt_mat.max())
    cnt_kwargs = dict(
        vmax=cnt_vmax,
        cbar_label="Authors shared between editions (birth-year adjusted, raw count)",
        title=(
            "Author overlap between African American literature anthology editions\n"
            "(cell = number of authors shared between row and column edition,"
            " birth-year adjusted)"
        ),
    )
    plot(cnt_mat, labels, OUT_FILE.with_stem(OUT_FILE.stem + "_counts"), **cnt_kwargs)
    plot(
        cnt_mat,
        labels,
        OUT_FILE.with_stem(OUT_FILE.stem + "_counts_bw"),
        cmap="Greys_r",
        **cnt_kwargs,
    )


if __name__ == "__main__":
    main()
