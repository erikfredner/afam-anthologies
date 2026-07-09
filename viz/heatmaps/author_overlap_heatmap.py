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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from afam.db import query as query_db
from afam.editions import EDITION_LABELS
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

OUT_FILE = OUTPUT_DIR / "author_overlap_heatmap.png"
BIRTH_YEAR_WINDOW = 5


# ── 1. Load ───────────────────────────────────────────────────────────────────


def load() -> pd.DataFrame:
    """Wide (work × author × edition) frame from the DB; edition_id is the key."""
    return query_db(query_path("works-authors-per-afam-edition"))


# ── 2. Per-edition author sets, years, birth years ────────────────────────────


def build_edition_sets(df: pd.DataFrame) -> dict[int, set[int]]:
    d = df.dropna(subset=["author_id"])
    return {
        int(key): set(grp["author_id"].astype(int))
        for key, grp in d.groupby("edition_id")
    }


def edition_years(df: pd.DataFrame) -> dict[int, int]:
    """{edition_id: first publication year}."""
    years = df.groupby("edition_id")["anthology_publication_year"].min()
    return {int(k): int(v) for k, v in years.items()}


def load_birth_years(df: pd.DataFrame) -> dict[int, int]:
    """{author_id: birth_year} from the wide frame (birth year travels with it)."""
    d = df.dropna(subset=["author_id", "author_birth_year"])
    return {
        int(a): int(y)
        for a, y in d[["author_id", "author_birth_year"]].drop_duplicates().values
    }


def _max_known_birth_year(
    author_ids: set[int], birth_years: dict[int, int]
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


def make_label(key: int, years: dict[int, int]) -> str:
    short = EDITION_LABELS.get(key, str(key))
    return f"{short}\n{years[key]}"


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
    edition_sets = build_edition_sets(df)
    birth_years = load_birth_years(df)
    years = edition_years(df)

    keys = sorted(edition_sets, key=lambda k: years[k])
    labels = [make_label(k, years) for k in keys]

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
