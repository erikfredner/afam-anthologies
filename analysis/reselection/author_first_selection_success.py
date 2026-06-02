"""author_first_selection_success.py
------------------------------------
For each author, find the first anthology (chronologically) to select them.
Measure how often that debut was validated by subsequent anthologies.
Aggregate to the debut-anthology level to compare across eras.

Usage:
    uv run python analysis/author_first_selection_success.py
    uv run python analysis/author_first_selection_success.py --save-csv
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from afam import DATA_DIR

DATA_FILE = DATA_DIR / "2026-03-13 author ids in afam anthologies.csv"

DETAIL_CSV = "author_first_selection_success_detail.csv"
EDITION_CSV = "author_first_selection_success_edition.csv"


# ── Edition-key logic (verbatim from viz/selection_frequency_decay.py) ─────────


def _strip_volume(title: str) -> str:
    return re.sub(r",?\s+[Vv]ol\.?\s+\d+\s*$", "", title).strip()


def assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    meta_rows: list[dict] = []
    for _, r in (
        df[["anthology_id", "anthology_title", "series", "edition_number", "volume"]]
        .drop_duplicates()
        .iterrows()
    ):
        series = r["series"].strip()
        edition = r["edition_number"].strip()
        volume = r["volume"].strip()
        title = r["anthology_title"].strip()
        aid = r["anthology_id"]

        if series:
            key = f"{series}|{edition}"
        elif volume:
            key = f"{_strip_volume(title)}|{edition}"
        else:
            key = aid

        meta_rows.append({"anthology_id": aid, "edition_key": key})

    return df.merge(pd.DataFrame(meta_rows), on="anthology_id")


# ── Label ───────────────────────────────────────────────────────────────────────

SERIES_ABBREV: dict[str, str] = {
    "The Norton Anthology of African American Literature": "NAAAL",
    "Afro-American Writing: An Anthology of Prose and Poetry": "Afro-Am. Writing",
    "African American Literature": "AAL Anthology",
    "The Wiley Blackwell Anthology of African American Literature": "Wiley Blackwell AAL",
}


def _edition_label(edition_key: str, year: int, ek_to_title: dict[str, str]) -> str:
    if "|" in edition_key:
        series_or_title, edition = edition_key.rsplit("|", 1)
        abbrev = SERIES_ABBREV.get(series_or_title, series_or_title[:25])
        return f"{abbrev} ed.{edition} ({year})"
    title = ek_to_title.get(edition_key, edition_key)
    return f"{title[:30]} ({year})"


# ── Core computation ────────────────────────────────────────────────────────────


def compute(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (detail_df, edition_df) from a raw anthology-author DataFrame."""
    df = df_raw.copy()
    df["publication_year"] = df["publication_year"].astype(int)

    df = assign_edition_key(df)

    # Title lookup: edition_key → first anthology_title seen
    ek_to_title: dict[str, str] = (
        df.groupby("edition_key")["anthology_title"].first().to_dict()
    )

    # One row per (edition_key, author_id)
    pairs = df[["edition_key", "author_id", "publication_year"]].drop_duplicates(
        subset=["edition_key", "author_id"]
    ).copy()

    # Edition year = min publication_year per edition_key
    edition_year = (
        df.groupby("edition_key")["publication_year"]
        .min()
        .rename("edition_year")
        .reset_index()
    )
    pairs = pairs.merge(edition_year, on="edition_key")

    # All unique editions with their years
    all_editions = edition_year.copy()

    # For each author: debut = edition with min year; tie-break by smallest edition_key
    sorted_pairs = pairs.sort_values(["author_id", "edition_year", "edition_key"])
    debut = (
        sorted_pairs.groupby("author_id", sort=False)
        .first()
        .reset_index()
        [["author_id", "edition_key", "edition_year"]]
        .rename(columns={"edition_key": "debut_edition_key", "edition_year": "debut_year"})
    )

    # Author → set of edition_keys they appeared in
    author_ek_set: dict[str, set] = (
        pairs.groupby("author_id")["edition_key"].apply(set).to_dict()
    )

    # Per-author metrics
    records = []
    for _, row in debut.iterrows():
        aid = row["author_id"]
        debut_ek = row["debut_edition_key"]
        debut_yr = int(row["debut_year"])

        subsequent = all_editions[all_editions["edition_year"] > debut_yr]
        subsequent_count = len(subsequent)

        author_eks = author_ek_set.get(aid, set())
        reselection_count = int(subsequent["edition_key"].isin(author_eks).sum())

        if subsequent_count == 0:
            reselection_prob = float("nan")
            ever_reselected = float("nan")
        else:
            reselection_prob = reselection_count / subsequent_count
            ever_reselected = 1.0 if reselection_count > 0 else 0.0

        records.append({
            "debut_edition_key": debut_ek,
            "debut_year": debut_yr,
            "author_id": aid,
            "subsequent_count": subsequent_count,
            "reselection_count": reselection_count,
            "reselection_prob": reselection_prob,
            "ever_reselected": ever_reselected,
        })

    detail_df = pd.DataFrame(records).sort_values(
        ["debut_year", "debut_edition_key", "author_id"]
    ).reset_index(drop=True)

    # Anthology-level aggregate
    agg_records = []
    for (debut_ek, debut_yr), grp in detail_df.groupby(
        ["debut_edition_key", "debut_year"], sort=False
    ):
        n_debut_authors = len(grp)
        n_with_subsequent = int((grp["subsequent_count"] > 0).sum())

        valid_prob = grp["reselection_prob"].dropna()
        mean_reselection_prob = (
            float(valid_prob.mean()) if len(valid_prob) > 0 else float("nan")
        )

        valid_ever = grp["ever_reselected"].dropna()
        ever_reselected_rate = (
            float(valid_ever.mean()) if len(valid_ever) > 0 else float("nan")
        )

        anthology_label = _edition_label(debut_ek, int(debut_yr), ek_to_title)

        agg_records.append({
            "debut_edition_key": debut_ek,
            "debut_year": int(debut_yr),
            "n_debut_authors": n_debut_authors,
            "n_with_subsequent": n_with_subsequent,
            "mean_reselection_prob": mean_reselection_prob,
            "ever_reselected_rate": ever_reselected_rate,
            "anthology_label": anthology_label,
        })

    edition_df = (
        pd.DataFrame(agg_records)
        .sort_values(["debut_year", "debut_edition_key"])
        .reset_index(drop=True)
    )

    return detail_df, edition_df


# ── Stdout table ────────────────────────────────────────────────────────────────


def print_table(edition_df: pd.DataFrame) -> None:
    with_sub = edition_df[edition_df["n_with_subsequent"] > 0].copy()
    without_sub = edition_df[edition_df["n_with_subsequent"] == 0].copy()

    cols = ["Debut anthology", "Year", "N debut", "N w/oppty", "Mean p(resel)", "Ever resel."]
    widths = [35, 4, 7, 9, 12, 11]

    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-" * len(header)
    print(header)
    print(sep)

    for _, row in with_sub.iterrows():
        label = row["anthology_label"][:35]
        year_str = str(row["debut_year"])
        n_auth = str(row["n_debut_authors"])
        n_sub = str(row["n_with_subsequent"])
        prob = (
            f"{row['mean_reselection_prob']:.2f}"
            if not pd.isna(row["mean_reselection_prob"])
            else "N/A"
        )
        ever = (
            f"{row['ever_reselected_rate']:.2f}"
            if not pd.isna(row["ever_reselected_rate"])
            else "N/A"
        )
        print(
            f"{label.ljust(widths[0])} | "
            f"{year_str.ljust(widths[1])} | "
            f"{n_auth.rjust(widths[2])} | "
            f"{n_sub.rjust(widths[3])} | "
            f"{prob.rjust(widths[4])} | "
            f"{ever.rjust(widths[5])}"
        )

    if len(without_sub):
        print()
        print("Most recent debut anthologies (no subsequent opportunities to measure):")
        for _, row in without_sub.iterrows():
            print(
                f"  {row['anthology_label']}: "
                f"{row['n_debut_authors']} debut author(s)"
            )


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Author first-selection success by debut anthology."
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Write output CSVs to data/ instead of printing to stdout.",
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_FILE, dtype=str, na_filter=False)
    detail_df, edition_df = compute(df)

    if args.save_csv:
        detail_path = DATA_DIR / DETAIL_CSV
        edition_path = DATA_DIR / EDITION_CSV
        detail_df.to_csv(detail_path, index=False)
        edition_df.to_csv(edition_path, index=False)
        print(f"Saved → {detail_path}")
        print(f"Saved → {edition_path}")
        return

    print_table(edition_df)


if __name__ == "__main__":
    main()
