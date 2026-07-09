"""
anthology_overlap_heatmap.py
----------------------------
For every pair of anthology editions, compute the percentage of the row
edition's work IDs that also appear in the column edition.  Editions that
span multiple volumes are merged before comparison.  Renders a viridis
heatmap ordered chronologically by publication year.

Cell (row, col) = |works_in_row ∩ works_in_col| / |works_in_row| × 100
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

OUT_FILE = OUTPUT_DIR / "anthology_overlap_heatmap.png"
BIRTH_YEAR_WINDOW = 5


# ── 1. Load ───────────────────────────────────────────────────────────────────


def load() -> pd.DataFrame:
    """Wide (work × author × edition) frame from the DB; edition_id is the key."""
    return query_db(query_path("works-authors-per-afam-edition"))


# ── 2. Per-edition work sets, years, work birth years ─────────────────────────


def build_edition_sets(df: pd.DataFrame) -> dict[int, set[int]]:
    return {
        int(key): set(grp["work_id"].astype(int))
        for key, grp in df.groupby("edition_id")
    }


def edition_years(df: pd.DataFrame) -> dict[int, int]:
    """{edition_id: first publication year}."""
    years = df.groupby("edition_id")["anthology_publication_year"].min()
    return {int(k): int(v) for k, v in years.items()}


def load_work_birth_years(df: pd.DataFrame) -> dict[int, int]:
    """Return {work_id: max_author_birth_year}.

    Birth year proxy for a work = max birth year across its author(s). Works
    with no known author birth year are omitted (always-include in filtering).
    """
    d = df.dropna(subset=["author_birth_year"])
    by_work = d.groupby("work_id")["author_birth_year"].max()
    return {int(w): int(y) for w, y in by_work.items()}


def _max_known_birth_year(
    work_ids: set[int], work_birth_years: dict[int, int]
) -> int | None:
    years = [work_birth_years[w] for w in work_ids if w in work_birth_years]
    return max(years) if years else None


def _filter_pair(
    si: set[str],
    sj: set[str],
    work_birth_years: dict[str, int],
    window: int,
) -> tuple[set[str], set[str]]:
    """Return (si_filtered, sj_filtered) restricted to the contemporaneous work pool."""
    max_i = _max_known_birth_year(si, work_birth_years)
    max_j = _max_known_birth_year(sj, work_birth_years)
    known = [m for m in (max_i, max_j) if m is not None]
    if not known:
        return si, sj
    cutoff = min(known) + window
    si_filt = {w for w in si if work_birth_years.get(w, cutoff) <= cutoff}
    sj_filt = {w for w in sj if work_birth_years.get(w, cutoff) <= cutoff}
    return si_filt, sj_filt


# ── 4. Ordering and labels ────────────────────────────────────────────────────


def make_label(key: int, years: dict[int, int]) -> str:
    short = EDITION_LABELS.get(key, str(key))
    return f"{short}\n{years[key]}"


# ── 5. Overlap matrices ───────────────────────────────────────────────────────


def overlap_matrix(
    keys: list[str],
    sets: dict[str, set[str]],
    work_birth_years: dict[str, int],
    window: int = BIRTH_YEAR_WINDOW,
) -> np.ndarray:
    """Cell (i, j) = |si_filt ∩ sj_filt| / |si_filt| × 100 (birth-year adjusted)."""
    n = len(keys)
    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            si_filt, sj_filt = _filter_pair(
                sets[ki], sets[kj], work_birth_years, window
            )
            if not si_filt:
                continue
            mat[i, j] = 100.0 * len(si_filt & sj_filt) / len(si_filt)
    return mat


def jaccard_matrix(
    keys: list[str],
    sets: dict[str, set[str]],
    work_birth_years: dict[str, int],
    window: int = BIRTH_YEAR_WINDOW,
) -> np.ndarray:
    """Cell (i, j) = |si_filt ∩ sj_filt| / |si_filt ∪ sj_filt| × 100 (birth-year adjusted)."""
    n = len(keys)
    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            si_filt, sj_filt = _filter_pair(
                sets[ki], sets[kj], work_birth_years, window
            )
            union = si_filt | sj_filt
            if not union:
                continue
            mat[i, j] = 100.0 * len(si_filt & sj_filt) / len(union)
    return mat


def counts_matrix(
    keys: list[str],
    sets: dict[str, set[str]],
    work_birth_years: dict[str, int],
    window: int = BIRTH_YEAR_WINDOW,
) -> np.ndarray:
    """Cell (i, j) = |si_filt ∩ sj_filt| raw count (birth-year adjusted)."""
    n = len(keys)
    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            si_filt, sj_filt = _filter_pair(
                sets[ki], sets[kj], work_birth_years, window
            )
            mat[i, j] = len(si_filt & sj_filt)
    return mat


# ── 6. Plot ───────────────────────────────────────────────────────────────────


def plot(
    mat: np.ndarray,
    labels: list[str],
    out: Path,
    cmap: str = "viridis",
    vmax: float = 100,
    cbar_label: str = "% of row edition's works also in column edition",
    title: str = (
        "Work overlap between African American literature anthology editions\n"
        "(cell = % of row edition's works also present in column edition)"
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
    work_birth_years = load_work_birth_years(df)
    years = edition_years(df)

    keys = sorted(edition_sets, key=lambda k: years[k])
    labels = [make_label(k, years) for k in keys]

    OUT_FILE.parent.mkdir(exist_ok=True)

    # ── Percentage overlap ────────────────────────────────────────────────────
    pct_mat = overlap_matrix(keys, edition_sets, work_birth_years)
    pct_kwargs = dict(
        cbar_label="% of row edition's works also in column edition (birth-year adjusted)",
        title=(
            "Work overlap between African American literature anthology editions\n"
            "(cell = % of row edition's works also present in column edition,"
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
    jac_mat = jaccard_matrix(keys, edition_sets, work_birth_years)
    jac_kwargs = dict(
        cbar_label="Jaccard similarity × 100 (birth-year adjusted)",
        title=(
            "Work overlap between African American literature anthology editions\n"
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
    cnt_mat = counts_matrix(keys, edition_sets, work_birth_years)
    cnt_vmax = int(cnt_mat.max())
    cnt_kwargs = dict(
        vmax=cnt_vmax,
        cbar_label="Works shared between editions (birth-year adjusted, raw count)",
        title=(
            "Work overlap between African American literature anthology editions\n"
            "(cell = number of works shared between row and column edition,"
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
