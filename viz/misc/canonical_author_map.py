from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import pandas as pd


from afam import DATA_DIR
from afam.db import query as query_db
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

DEFAULT_OUT_PNG = OUTPUT_DIR / "canonical_author_map.png"
DEFAULT_OUT_CSV = DATA_DIR / "canonical_author_metrics.csv"
TARGET_LABEL_AUTHORS = [
    "Olaudah Equiano",
    "Alain Locke",
    "Langston Hughes",
    "Phillis Wheatley",
]
TARGET_SIGNATURE_TITLES = {
    "Alain Locke": "The New Negro <essay>",
}
CUSTOM_LABEL_TEXT = {
    "Langston Hughes": 'Hughes, "Mother to Son"',
    "Alain Locke": 'Locke, "The New Negro"',
    "Olaudah Equiano": r"Equiano, $\it{The\ Interesting\ Narrative}$",
    "Phillis Wheatley": r"Wheatley, $\it{Poems\ on\ Various\ Subjects}$",
}
LABEL_OFFSETS = {
    "Langston Hughes": (-10, 10),
    "Alain Locke": (10, 14),
    "Olaudah Equiano": (10, -12),
    "Phillis Wheatley": (-10, 8),
}
LABEL_ALIGNMENTS = {
    "Langston Hughes": "right",
    "Alain Locke": "left",
    "Olaudah Equiano": "left",
    "Phillis Wheatley": "right",
}


def _stringify_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure string-like columns are safe for .strip() and CSV output."""
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str)
    return result


def assign_edition_key(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows to anthology editions using series_id when available."""
    result = _stringify_columns(df, ["series_id", "anthology_edition", "anthology_id"])
    result["edition_key"] = result.apply(
        lambda row: (
            f"{row['series_id']}|{row['anthology_edition']}"
            if row["series_id"].strip()
            else row["anthology_id"]
        ),
        axis=1,
    )
    return result


def assign_canonical_work(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve each row to its top-level parent work when parent data exists."""
    result = _stringify_columns(
        df,
        ["work_id", "work_title", "parent_work_id", "parent_work_title"],
    )
    work_to_parent_id = (
        result.drop_duplicates(subset=["work_id"])
        .set_index("work_id")["parent_work_id"]
        .to_dict()
    )
    work_to_title = (
        result.drop_duplicates(subset=["work_id"])
        .set_index("work_id")["work_title"]
        .to_dict()
    )

    def resolve_root_id(work_id: str, parent_work_id: str) -> str:
        current = parent_work_id.strip()
        visited = {work_id}
        while current and current not in visited:
            visited.add(current)
            next_parent = work_to_parent_id.get(current, "").strip()
            if not next_parent:
                return current
            current = next_parent
        return current or work_id

    canonical_ids: list[str] = []
    canonical_titles: list[str] = []
    for row in result.to_dict("records"):
        work_id = row["work_id"].strip()
        work_title = row["work_title"].strip()
        parent_work_id = row["parent_work_id"].strip()
        parent_work_title = row["parent_work_title"].strip()

        if parent_work_id:
            canonical_work_id = resolve_root_id(work_id, parent_work_id)
            canonical_work_title = work_to_title.get(canonical_work_id, "").strip()
            if not canonical_work_title:
                canonical_work_title = parent_work_title or work_title
        elif parent_work_title:
            canonical_work_id = f"title::{parent_work_title}"
            canonical_work_title = parent_work_title
        else:
            canonical_work_id = work_id
            canonical_work_title = work_title

        canonical_ids.append(canonical_work_id)
        canonical_titles.append(canonical_work_title)

    result["canonical_work_id"] = canonical_ids
    result["canonical_work_title"] = canonical_titles
    return result


def explode_authors(df: pd.DataFrame) -> pd.DataFrame:
    """Expand one row per listed author and drop rows with no usable author."""
    df = _stringify_columns(df, ["author_ids", "author_names"])
    records: list[dict] = []
    for row in df.to_dict("records"):
        author_ids = [
            part.strip() for part in row["author_ids"].split(",") if part.strip()
        ]
        author_names = [
            part.strip() for part in row["author_names"].split(";") if part.strip()
        ]
        pairs: list[tuple[str, str]] = []

        if author_ids and author_names and len(author_ids) == len(author_names):
            pairs = list(zip(author_ids, author_names))
        elif len(author_ids) == 1 or len(author_names) == 1:
            pairs = [
                (
                    author_ids[0] if author_ids else "",
                    author_names[0] if author_names else "",
                )
            ]

        for author_id, author_name in pairs:
            if not author_id and not author_name:
                continue
            record = dict(row)
            record["author_id"] = author_id
            record["author_name"] = author_name
            records.append(record)

    return pd.DataFrame.from_records(records)


def build_selection_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create unique author-edition-work selection events."""
    expanded = explode_authors(assign_canonical_work(assign_edition_key(df)))
    selection_cols = [
        "author_id",
        "author_name",
        "edition_key",
        "canonical_work_id",
        "canonical_work_title",
    ]
    selections = (
        expanded.loc[:, selection_cols]
        .drop_duplicates()
        .rename(
            columns={
                "canonical_work_id": "work_id",
                "canonical_work_title": "work_title",
            }
        )
        .sort_values(["author_name", "author_id", "edition_key", "work_title"])
        .reset_index(drop=True)
    )
    return selections


def _shannon_entropy(counts: pd.Series) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probabilities = counts / total
    return float(-(probabilities * probabilities.map(math.log2)).sum())


def compute_author_metrics(selection_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate work-selection concentration metrics within authors."""
    records: list[dict] = []
    for (author_id, author_name), group in selection_df.groupby(
        ["author_id", "author_name"], sort=True, dropna=False
    ):
        work_counts = (
            group.groupby(["work_id", "work_title"], sort=True)
            .size()
            .rename("selection_count")
            .reset_index()
            .sort_values(
                ["selection_count", "work_title", "work_id"],
                ascending=[False, True, True],
            )
            .reset_index(drop=True)
        )
        total_work_selections = int(work_counts["selection_count"].sum())
        signature_row = work_counts.iloc[0]
        records.append(
            {
                "author_id": author_id,
                "author_name": author_name,
                "anthology_appearances": int(group["edition_key"].nunique()),
                "total_work_selections": total_work_selections,
                "distinct_works_selected": int(work_counts["work_id"].nunique()),
                "work_selection_entropy": _shannon_entropy(
                    work_counts["selection_count"]
                ),
                "signature_work_share": float(
                    signature_row["selection_count"] / total_work_selections
                ),
                "signature_work_id": signature_row["work_id"],
                "signature_work_title": signature_row["work_title"],
            }
        )

    metrics = pd.DataFrame.from_records(records)
    return metrics.sort_values(
        ["anthology_appearances", "signature_work_share", "author_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def choose_exemplars(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Pick one preferred row for each requested exemplar author."""
    order = {name: idx for idx, name in enumerate(TARGET_LABEL_AUTHORS)}
    selected_rows: list[pd.Series] = []

    for author_name in TARGET_LABEL_AUTHORS:
        candidates = metrics_df.loc[metrics_df["author_name"] == author_name].copy()
        if candidates.empty:
            continue

        preferred_title = TARGET_SIGNATURE_TITLES.get(author_name)
        if preferred_title is not None:
            preferred = candidates.loc[
                candidates["signature_work_title"] == preferred_title
            ]
            if not preferred.empty:
                candidates = preferred

        chosen = candidates.sort_values(
            ["anthology_appearances", "total_work_selections", "signature_work_share"],
            ascending=[False, False, False],
        ).iloc[0]
        selected_rows.append(chosen)

    if not selected_rows:
        return pd.DataFrame(columns=metrics_df.columns)

    exemplars = pd.DataFrame(selected_rows)
    return exemplars.sort_values(
        by="author_name",
        key=lambda col: col.map(order),
    ).reset_index(drop=True)


def build_label_text(row: pd.Series) -> str:
    """Format annotation text with author and current signature work."""
    custom_text = CUSTOM_LABEL_TEXT.get(row["author_name"])
    if custom_text is not None:
        return custom_text
    return f'{row["author_name"]}, "{row["signature_work_title"]}"'


def make_plot(
    metrics_df: pd.DataFrame, exemplars_df: pd.DataFrame, out_path: Path
) -> None:
    """Render the canonical-author map."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    exemplar_keys = set(zip(exemplars_df["author_id"], exemplars_df["author_name"]))
    exemplar_mask = metrics_df.apply(
        lambda row: (row["author_id"], row["author_name"]) in exemplar_keys,
        axis=1,
    )
    background_df = metrics_df.loc[~exemplar_mask]
    highlighted_df = metrics_df.loc[exemplar_mask]

    ax.scatter(
        background_df["anthology_appearances"],
        background_df["signature_work_share"],
        s=36,
        alpha=0.12,
        color="#2a6f97",
        edgecolors="white",
        linewidths=0.3,
        zorder=1,
    )
    ax.scatter(
        highlighted_df["anthology_appearances"],
        highlighted_df["signature_work_share"],
        s=96,
        alpha=0.95,
        color="#c44536",
        edgecolors="#111111",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_xlabel("Author frequency across African American anthology editions")
    ax.set_ylabel("Signature work share")
    ax.set_title(
        "Canonical-author map\nCollapsed editions; work unit = exact selected text"
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.25)

    for row in exemplars_df.to_dict("records"):
        dx, dy = LABEL_OFFSETS.get(row["author_name"], (8, 8))
        ax.annotate(
            build_label_text(pd.Series(row)),
            xy=(row["anthology_appearances"], row["signature_work_share"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=LABEL_ALIGNMENTS.get(row["author_name"], "left"),
            va="center",
            fontsize=7,
            color="#1b1b1b",
            zorder=4,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def load_data() -> pd.DataFrame:
    """Wide (work × author × edition) frame from the DB, shaped to the legacy
    CSV columns the helpers consume (edition collapses to edition_id; each row
    is already a single author, so author_ids/author_names hold one value)."""
    raw = query_db(query_path("works-authors-per-afam-edition"))
    raw = raw.dropna(subset=["author_id"]).copy()
    return pd.DataFrame(
        {
            "anthology_id": raw["edition_id"].astype(int).astype(str),
            "series_id": "",
            "anthology_edition": "",
            "work_id": raw["work_id"].astype(int).astype(str),
            "work_title": raw["work_title"].fillna(""),
            "parent_work_id": raw["parent_id"].apply(
                lambda v: "" if pd.isna(v) else str(int(v))
            ),
            "parent_work_title": raw["parent_work_title"].fillna(""),
            "author_ids": raw["author_id"].astype(int).astype(str),
            "author_names": raw["author_name"].fillna(""),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot signature-work concentration within authors."
    )
    parser.add_argument(
        "--out-png",
        default=str(DEFAULT_OUT_PNG),
        help="Path to output PNG figure.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(DEFAULT_OUT_CSV),
        help="Path to output metrics CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data()
    selection_df = build_selection_frame(df)
    metrics_df = compute_author_metrics(selection_df)
    exemplars_df = choose_exemplars(metrics_df)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_csv, index=False)
    make_plot(metrics_df, exemplars_df, Path(args.out_png))

    print(f"Saved metrics CSV -> {out_csv}")
    print(f"Saved figure -> {Path(args.out_png)}")


if __name__ == "__main__":
    main()
