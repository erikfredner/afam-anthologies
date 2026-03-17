"""retention_from_1941.py
-----------------------
Bar chart showing how much of the 1941 "Negro Caravan" (anthology_id=62)
was retained — by author and by work — in each anthology published after 1941.

Figure: viz/retention_from_1941.png

Usage: uv run python viz/retention_from_1941.py
"""

from __future__ import annotations

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

REF_EDITION_KEY = "62"  # 1941 Negro Caravan (series_id empty → edition_key = anthology_id)
REF_YEAR = 1941


# ── Style constants (match work_selection_divergence.py) ───────────────────────

C_BLUE = "#1f77b4"
C_RED = "#d62728"
GRID_KW = dict(alpha=0.25, linestyle=":")


# ── Label dicts (from work_selection_divergence.py) ───────────────────────────

SERIES_ID_ABBREV: dict[str, str] = {
    "3": "NAAAL",
    "8": "Afro-Am. Writing",
    "12": "AAL Anthology",
    "17": "Wiley Blackwell AAL",
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
    "50": "Blackamerican Lit.",
    "40": "New Cavalcade v.1",
    "41": "New Cavalcade v.2",
}


def make_label(edition_key: str, year: int) -> str:
    if "|" in edition_key:
        series_id, edition = edition_key.rsplit("|", 1)
        abbrev = SERIES_ID_ABBREV.get(series_id, series_id)
        return f"{abbrev} ed.{edition}\n{year}"
    short = STANDALONE_SHORT.get(edition_key, edition_key[:20])
    return f"{short}\n{year}"


# ── Title normalization (private copy) ─────────────────────────────────────────


def _normalize_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"^(from |excerpt from |selections? from )", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip()


# ── Load and prepare (private copy from cumulative_pairwise_agreement.py) ──────


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


# ── Core computation ───────────────────────────────────────────────────────────


def compute_retention(
    long: pd.DataFrame, ref_key: str, after_year: int | None = None
) -> pd.DataFrame:
    """Compute per-edition retention of a reference anthology's authors and works.

    Parameters
    ----------
    long:
        DataFrame with columns: edition_key, ek_year, author_id, norm_title.
    ref_key:
        Edition key of the reference anthology (excluded from output).
    after_year:
        If provided, only include editions with year strictly greater than this value.

    Returns
    -------
    DataFrame with columns: edition_key, year, author_pct, work_pct.
    Sorted by year ascending. Reference edition is excluded.
    """
    authors_ref = frozenset(
        long.loc[long["edition_key"] == ref_key, "author_id"].unique()
    )
    works_ref = frozenset(
        long.loc[long["edition_key"] == ref_key, "norm_title"].unique()
    )

    records = []
    for ek, grp in long.groupby("edition_key"):
        if ek == ref_key:
            continue
        year = int(grp["ek_year"].iloc[0])
        if after_year is not None and year <= after_year:
            continue
        ek_authors = frozenset(grp["author_id"].unique())
        ek_works = frozenset(grp["norm_title"].unique())
        author_pct = (
            100 * len(ek_authors & authors_ref) / len(authors_ref)
            if authors_ref
            else 0.0
        )
        work_pct = (
            100 * len(ek_works & works_ref) / len(works_ref) if works_ref else 0.0
        )
        records.append(
            {
                "edition_key": ek,
                "year": year,
                "author_pct": author_pct,
                "work_pct": work_pct,
            }
        )

    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


# ── Figure ─────────────────────────────────────────────────────────────────────


def plot_retention(df: pd.DataFrame, out_path: Path) -> None:
    """Render the retention grouped bar chart."""
    labels = [make_label(row["edition_key"], row["year"]) for _, row in df.iterrows()]

    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - width / 2, df["author_pct"], width=width, color=C_BLUE, label="Authors")
    ax.bar(x + width / 2, df["work_pct"], width=width, color=C_RED, label="Works")

    ax.axhline(50, color="gray", linestyle="--", lw=1.0, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of 1941 selections retained", fontsize=11)
    ax.set_title(
        "Retention of 1941 Negro Caravan selections in subsequent anthologies",
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis="y", **GRID_KW)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    long = _load_and_prepare(DATA_FILE)
    df = compute_retention(long, REF_EDITION_KEY, after_year=REF_YEAR)

    print(f"\nRetention rows: {len(df)} subsequent editions")
    if not df.empty:
        print(df[["edition_key", "year", "author_pct", "work_pct"]].to_string(index=False))

    out_path = OUT_DIR / "retention_from_1941.png"
    plot_retention(df, out_path)


if __name__ == "__main__":
    main()
