"""
predictability_new_focus_per_edition.py
---------------------------------------
Per-edition complement to viz/predictability/predictability_new_focus.py.

The chart shows the trend across editions of the share appearing in the AFAM
tradition for the first time (authors × works × slot-weighted × page-weighted),
with a single OLS line and one residual outlier called out. This script writes
the underlying per-edition numbers — both the proportions plotted in the chart
and the raw slot/page counts behind them — so each edition can be inspected
directly.

Reads:
    data/predictability_over_time_authors.csv
    data/predictability_over_time_works.csv

Writes:
    data/predictability_new_focus_per_edition_authors.csv
    data/predictability_new_focus_per_edition_works.csv

Both inputs are produced by
analysis/predictability/predictability_over_time.py --save-csv. They already
exclude the earliest AFAM edition (1931, Calverton) because it has no prior
pool; this script inherits that exclusion to stay consistent with the chart.

Usage:
    uv run python analysis/predictability/predictability_new_focus_per_edition.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "predictability"))

from predictability_over_time import compute_per_edition  # noqa: E402

from afam import DATA_DIR  # noqa: E402
from afam.editions import EDITION_LABELS  # noqa: E402

OUTPUT_BASENAMES = {
    "authors": "predictability_new_focus_per_edition_authors.csv",
    "works": "predictability_new_focus_per_edition_works.csv",
}


def build_table(per_edition: pd.DataFrame) -> pd.DataFrame:
    overall = per_edition[per_edition["subgroup_dim"] == "all"].copy()
    overall = overall.sort_values("year").reset_index(drop=True)

    total_slots = overall["slots"].astype(int)
    total_pages = overall["pages"].astype(float)
    share_new_slots = 1.0 - overall["fraction_repeat_slot"].astype(float)
    share_new_pages = 1.0 - overall["fraction_repeat_page"].astype(float)

    new_slots = (total_slots * share_new_slots).round().astype(int)
    new_pages = (total_pages * share_new_pages).round().astype(int)

    return pd.DataFrame(
        {
            "edition_id": overall["edition_id"].astype(int),
            "year": overall["year"].astype(int),
            "edition_label": overall["edition_id"]
            .astype(int)
            .map(EDITION_LABELS)
            .fillna("(unlabeled)"),
            "total_slots": total_slots,
            "new_slots": new_slots,
            "share_new_slots": share_new_slots.round(4),
            "total_pages": total_pages.round().astype(int),
            "new_pages": new_pages,
            "share_new_pages": share_new_pages.round(4),
        }
    )


def print_table(entity: str, table: pd.DataFrame) -> None:
    header = (
        f"{'edition':<24}{'year':>6}"
        f"{'slots':>8}{'new':>8}{'new %':>8}"
        f"{'pages':>8}{'new':>8}{'new %':>8}"
    )
    print(f"\n===== {entity.upper()}: share never previously anthologized =====")
    print(header)
    print("-" * len(header))
    for _, r in table.iterrows():
        print(
            f"{r['edition_label']:<24}{int(r['year']):>6}"
            f"{int(r['total_slots']):>8}{int(r['new_slots']):>8}"
            f"{r['share_new_slots'] * 100:>7.1f}%"
            f"{int(r['total_pages']):>8}{int(r['new_pages']):>8}"
            f"{r['share_new_pages'] * 100:>7.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Where to write the per-edition CSVs (default: {DATA_DIR})",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for entity in ("authors", "works"):
        per_ed = compute_per_edition(entity)
        table = build_table(per_ed)
        print_table(entity, table)
        out_path = args.out_dir / OUTPUT_BASENAMES[entity]
        table.to_csv(out_path, index=False)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
