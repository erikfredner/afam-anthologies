"""
most_anthologized.py
--------------------
Produces two ranked CSV tables:

  data/most_anthologized_works.csv   — works by number of unique anthology
                                       editions that selected them
  data/most_anthologized_authors.csv — authors by number of unique anthology
                                       editions that selected them

Multi-volume editions (e.g. NAAAL 3rd/4th, Wiley Blackwell vol.1-2) are
merged into a single edition using the same logic as the heatmap scripts.
"""

from __future__ import annotations

import re

import pandas as pd


from afam import DATA_DIR

DATA_FILE = DATA_DIR / "2026-03-13 works per afam anthology.csv"
OUT_WORKS = DATA_DIR / "most_anthologized_works.csv"
OUT_AUTHORS = DATA_DIR / "most_anthologized_authors.csv"


# ── Edition-key logic (mirrors heatmap scripts) ───────────────────────────────


def _strip_volume(title: str) -> str:
    return re.sub(r",?\s+[Vv]ol\.?\s+\d+\s*$", "", title).strip()


def assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    meta_rows: list[dict] = []
    for _, r in (
        df[
            [
                "anthology_id",
                "anthology_title",
                "series_id",
                "anthology_edition",
                "anthology_volume",
            ]
        ]
        .drop_duplicates()
        .iterrows()
    ):
        series_id = r["series_id"].strip()
        edition = r["anthology_edition"].strip()
        volume = r["anthology_volume"].strip()
        title = r["anthology_title"].strip()
        aid = r["anthology_id"]

        if series_id:
            key = f"{series_id}|{edition}"
        elif volume:
            key = f"{_strip_volume(title)}|{edition}"
        else:
            key = aid

        meta_rows.append({"anthology_id": aid, "edition_key": key})

    return df.merge(pd.DataFrame(meta_rows), on="anthology_id")


# ── Works table ───────────────────────────────────────────────────────────────


def build_works_table(df: pd.DataFrame) -> pd.DataFrame:
    """Count unique editions per work_id, keep representative metadata."""
    counts = df.groupby("work_id")["edition_key"].nunique().rename("edition_count")
    # One representative row per work for metadata
    meta = (
        df[["work_id", "work_title", "parent_work_title", "author_names"]]
        .drop_duplicates("work_id")
        .set_index("work_id")
    )
    result = (
        meta.join(counts)
        .reset_index()
        .sort_values("edition_count", ascending=False)[
            [
                "work_id",
                "work_title",
                "author_names",
                "parent_work_title",
                "edition_count",
            ]
        ]
    )
    return result


# ── Authors table ─────────────────────────────────────────────────────────────


def build_authors_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand multi-author rows, then count unique editions per author_id.
    Rows with no author_id are skipped.
    """
    records: list[dict] = []
    for _, row in df[df["author_ids"] != ""].iterrows():
        ids = [i.strip() for i in row["author_ids"].split(",")]
        names = [n.strip() for n in row["author_names"].split(";")]
        # Pad names if counts differ (shouldn't normally happen)
        names += [""] * (len(ids) - len(names))
        for aid, name in zip(ids, names):
            records.append(
                {
                    "author_id": aid,
                    "author_name": name,
                    "edition_key": row["edition_key"],
                }
            )

    expanded = pd.DataFrame(records)

    counts = (
        expanded.groupby("author_id")["edition_key"].nunique().rename("edition_count")
    )
    # Canonical name per author_id (first seen)
    names = expanded.drop_duplicates("author_id").set_index("author_id")["author_name"]
    result = (
        names.to_frame()
        .join(counts)
        .reset_index()
        .sort_values("edition_count", ascending=False)[
            ["author_id", "author_name", "edition_count"]
        ]
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    df = pd.read_csv(DATA_FILE, dtype=str, na_filter=False)
    df = assign_edition_key(df)

    works_table = build_works_table(df)
    works_table.to_csv(OUT_WORKS, index=False)
    print(f"Saved → {OUT_WORKS}  ({len(works_table):,} works)")

    authors_table = build_authors_table(df)
    authors_table.to_csv(OUT_AUTHORS, index=False)
    print(f"Saved → {OUT_AUTHORS}  ({len(authors_table):,} authors)")

    # Preview top 10
    print("\nTop 10 works:")
    print(works_table.head(10).to_string(index=False))
    print("\nTop 10 authors:")
    print(authors_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
