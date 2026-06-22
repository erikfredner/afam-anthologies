"""
vernacular_works.py
-------------------
Resolve the editor-designated vernacular page ranges into the actual works they
contain, and measure how much of each anthology edition is vernacular.

The vernacular page ranges live in ``data/vernacular_pages.csv`` (parsed by
``afam.vernacular``). Pagination lives in ``data_workinanthology.toc_page`` (a
work's start page) and ``toc_next`` (the next work's start page); a work belongs
to a vernacular range when its ``toc_page`` falls within ``[start, end]``.

Two tables:
  * works in range — work title, author(s), volume, and pagination for every
    work the editors marked as vernacular;
  * edition shares — per edition, the share of selections and the share of pages
    occupied by vernacular material (the latter computed straight from the page
    ranges over full book extent).

Usage:
    uv run python analysis/vernacular/vernacular_works.py
    uv run python analysis/vernacular/vernacular_works.py --include-excerpts
    uv run python analysis/vernacular/vernacular_works.py --save-csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from afam import DATA_DIR
from afam.cli import add_root_works_flag, add_save_csv_flag
from afam.db import query
from afam.editions import EDITION_LABELS
from afam.sql import query_path
from afam.vernacular import load_vernacular_ranges

OUT_WORKS = DATA_DIR / "vernacular_works.csv"
OUT_SHARES = DATA_DIR / "vernacular_edition_shares.csv"
OUT_RESELECT = DATA_DIR / "vernacular_work_reselection.csv"


def load_data(only_root_works: bool = True) -> pd.DataFrame:
    """Load every work×volume appearance in AFAM editions with page columns.

    Reuses ``queries/author-page-share-reselection.sql`` (one row per
    work×volume×author×form) and joins ``volume_number`` for labeling. With
    ``only_root_works``, drops excerpt works (``parent_id`` not null).
    """
    df = query(query_path("author-page-share-reselection"))
    volumes = query("SELECT id AS volume_id, volume_number FROM data_volume")
    df = df.merge(volumes, on="volume_id", how="left")

    for col in ["toc_page", "toc_next", "first_toc_page", "last_toc_page"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["edition_year"] = df["edition_year"].astype(int)

    if only_root_works:
        df = df[df["parent_id"].isna()]
    return df


def match_vernacular_works(df: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """Return the works whose start page falls inside a vernacular range.

    One row per matched work×volume, with multi-author works' names joined.
    """
    frames: list[pd.DataFrame] = []
    for rng in ranges.itertuples(index=False):
        sel = df[
            (df["volume_id"] == rng.volume_id)
            & (df["toc_page"] >= rng.page_start)
            & (df["toc_page"] <= rng.page_end)
        ]
        frames.append(sel)

    if not frames:
        return pd.DataFrame()
    matched = pd.concat(frames, ignore_index=True)

    grouped = (
        matched.groupby(
            ["edition_id", "edition_year", "volume_id", "work_id", "work_title"],
            dropna=False,
        )
        .agg(
            volume_number=("volume_number", "first"),
            toc_page=("toc_page", "first"),
            toc_next=("toc_next", "first"),
            authors=(
                "author_name",
                lambda s: ", ".join(sorted({a for a in s if pd.notna(a)})),
            ),
        )
        .reset_index()
    )

    grouped["edition"] = grouped["edition_id"].map(EDITION_LABELS)
    grouped["pages"] = grouped.apply(
        lambda r: (
            f"{int(r['toc_page'])}–{int(r['toc_next'])}"
            if pd.notna(r["toc_next"])
            else f"{int(r['toc_page'])}"
        ),
        axis=1,
    )
    return grouped.sort_values(
        ["edition_year", "volume_id", "toc_page"], kind="mergesort"
    ).reset_index(drop=True)


def compute_edition_shares(df: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """Per-edition share of selections and pages occupied by vernacular material.

    ``pct_selections`` = distinct vernacular works ÷ distinct works in edition.
    ``pct_pages`` = Σ vernacular range lengths ÷ Σ volume extents (full book
    extent, ``last_toc_page − first_toc_page + 1`` summed across volumes).
    """
    vol_to_edition = (
        df[["volume_id", "edition_id"]].drop_duplicates().set_index("volume_id")
    )

    ranges = ranges.copy()
    ranges["length"] = ranges["page_end"] - ranges["page_start"] + 1
    ranges["edition_id"] = ranges["volume_id"].map(vol_to_edition["edition_id"])
    if ranges["edition_id"].isna().any():
        missing = sorted(ranges.loc[ranges["edition_id"].isna(), "volume_id"].unique())
        print(f"WARNING: vernacular volumes not found in AFAM corpus: {missing}")
        ranges = ranges.dropna(subset=["edition_id"])
    vernacular_pages = ranges.groupby("edition_id")["length"].sum()

    matched = match_vernacular_works(df, ranges)
    vernacular_works = (
        matched.groupby("edition_id")["work_id"].nunique()
        if not matched.empty
        else pd.Series(dtype=int)
    )

    volume_extent = (
        df[["volume_id", "edition_id", "first_toc_page", "last_toc_page"]]
        .drop_duplicates("volume_id")
        .assign(extent=lambda d: d["last_toc_page"] - d["first_toc_page"] + 1)
        .groupby("edition_id")["extent"]
        .sum()
    )
    total_works = df.groupby("edition_id")["work_id"].nunique()
    year = df.groupby("edition_id")["edition_year"].first()

    out = pd.DataFrame({"edition_id": total_works.index})
    out["edition"] = out["edition_id"].map(EDITION_LABELS)
    out["year"] = out["edition_id"].map(year).astype(int)
    out["total_works"] = out["edition_id"].map(total_works).fillna(0).astype(int)
    out["vernacular_works"] = (
        out["edition_id"].map(vernacular_works).fillna(0).astype(int)
    )
    out["total_pages"] = out["edition_id"].map(volume_extent).fillna(0).astype(int)
    out["vernacular_pages"] = (
        out["edition_id"].map(vernacular_pages).fillna(0).astype(int)
    )
    out["pct_selections"] = 100 * out["vernacular_works"] / out["total_works"]
    out["pct_pages"] = 100 * out["vernacular_pages"] / out["total_pages"]
    return out.sort_values("year", kind="mergesort").reset_index(drop=True)


def compute_work_reselection(df: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """Per vernacular work: its corpus-wide selection and reselection counts.

    ``selections`` = number of distinct AFAM editions the work appears in (across
    the whole corpus, not only its vernacular-marked appearances). Following the
    repo's reselection convention, a work's ``reselections`` (``selections - 1``,
    the appearances after its debut) are reported over ``opportunities`` = the
    number of AFAM editions published after the work's debut edition.

    One row per unique vernacular work, sorted high→low by selection count.
    """
    if matched.empty:
        return pd.DataFrame()

    editions = (
        df[["edition_id", "edition_year"]]
        .drop_duplicates()
        .sort_values(["edition_year", "edition_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    edition_pos = {eid: i for i, eid in enumerate(editions["edition_id"])}
    n_editions = len(editions)

    vern_ids = matched["work_id"].unique()
    meta = (
        matched.sort_values(["edition_year", "edition_id"], kind="mergesort")
        .groupby("work_id")
        .agg(work_title=("work_title", "first"), authors=("authors", "first"))
    )

    appears = (
        df[df["work_id"].isin(vern_ids)]
        .groupby("work_id")["edition_id"]
        .apply(lambda s: sorted({edition_pos[e] for e in s}))
    )

    rows = []
    for wid in vern_ids:
        positions = appears[wid]
        selections = len(positions)
        debut_pos = positions[0]
        opportunities = n_editions - 1 - debut_pos
        rows.append(
            {
                "work_title": meta.loc[wid, "work_title"],
                "authors": meta.loc[wid, "authors"],
                "selections": selections,
                "reselections": selections - 1,
                "opportunities": opportunities,
            }
        )

    out = pd.DataFrame(rows)
    out["work"] = out.apply(
        lambda r: (
            f'"{r["work_title"]}" by {r["authors"]}'
            if isinstance(r["authors"], str) and r["authors"].strip()
            else f'"{r["work_title"]}"'
        ),
        axis=1,
    )
    return out.sort_values(
        ["selections", "reselections", "work_title"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def print_work_reselection(reselect: pd.DataFrame) -> None:
    print(f"\nVernacular works by selection count ({len(reselect)} works)")
    print("Work, Selections, Reselections")
    for r in reselect.itertuples(index=False):
        print(f"{r.work}, {r.selections}, {r.reselections} / {r.opportunities}")


def print_works(matched: pd.DataFrame) -> None:
    cols = ["edition", "volume_number", "work_title", "authors", "pages"]
    print(f"\nWorks in vernacular ranges ({len(matched)} works)")
    print(
        matched[cols]
        .rename(columns={"volume_number": "vol", "work_title": "work"})
        .to_string(index=False)
    )


def print_shares(shares: pd.DataFrame) -> None:
    cols = [
        "edition",
        "year",
        "vernacular_works",
        "total_works",
        "pct_selections",
        "vernacular_pages",
        "total_pages",
        "pct_pages",
    ]
    print("\nVernacular share by edition")
    print(
        shares[cols].to_string(
            index=False,
            formatters={
                "pct_selections": lambda x: f"{x:.1f}%",
                "pct_pages": lambda x: f"{x:.1f}%",
            },
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_works_flag(parser)
    add_save_csv_flag(parser)
    args = parser.parse_args()

    df = load_data(only_root_works=args.only_root_works)
    ranges = load_vernacular_ranges()
    matched = match_vernacular_works(df, ranges)
    shares = compute_edition_shares(df, ranges)
    reselect = compute_work_reselection(df, matched)

    print_works(matched)
    print_shares(shares)
    print_work_reselection(reselect)

    if args.save_csv:
        OUT_WORKS.parent.mkdir(parents=True, exist_ok=True)
        matched[
            ["edition", "volume_number", "work_title", "authors", "pages", "toc_page"]
        ].to_csv(OUT_WORKS, index=False)
        shares.to_csv(OUT_SHARES, index=False)
        reselect[
            [
                "work",
                "work_title",
                "authors",
                "selections",
                "reselections",
                "opportunities",
            ]
        ].to_csv(OUT_RESELECT, index=False)
        print(f"\nSaved → {OUT_WORKS}")
        print(f"Saved → {OUT_SHARES}")
        print(f"Saved → {OUT_RESELECT}")


if __name__ == "__main__":
    main()
