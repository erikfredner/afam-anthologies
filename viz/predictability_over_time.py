"""
predictability_over_time.py (viz)
---------------------------------
Reads the CSVs written by analysis/predictability_over_time.py and produces
faceted line plots of each predictability metric vs. anthology year, with one
subplot per metric and one line per subgroup level.

Two PNGs are written:
  viz/predictability_over_time_authors.png
  viz/predictability_over_time_works.png

Usage:
    uv run python viz/predictability_over_time.py
    uv run python viz/predictability_over_time.py --csv-dir data/ --out-dir viz/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "viz"

PLOT_METRICS = [
    ("logit_auc", "Logit AUC (in_E ~ log(1+prior_count))"),
    ("recall_at_k", "Recall@K  (K = edition size)"),
    ("fraction_repeat_slot", "Fraction repeat (slot-weighted)"),
    ("fraction_repeat_page", "Fraction repeat (page-weighted)"),
    ("fraction_frequent_canon_slot", "Fraction from ≥3-prior canon (slot)"),
    ("lift_over_chance", "Lift over chance (fraction_repeat / base_rate)"),
    ("gini_prior_count_in_e", "Gini of prior_count among E"),
    ("base_rate", "Base rate (slots / prior pool)"),
]

SUBGROUP_PALETTES = {
    "all": {"all": ("#1f1f1f", "-", 2.6, 1.0)},
    "gender": {
        "female": ("#d62728", "-", 2.0, 0.9),
        "male": ("#1f77b4", "-", 2.0, 0.9),
        "unknown": ("#7f7f7f", ":", 1.4, 0.5),
    },
    "birth_cohort": {
        "≤1900": ("#4c72b0", "-", 1.8, 0.9),
        "1901-1950": ("#dd8452", "-", 1.8, 0.9),
        ">1950": ("#55a467", "-", 1.8, 0.9),
        "unknown": ("#a0a0a0", ":", 1.2, 0.4),
    },
    "debut_era": {
        "pre-1970": ("#8172b3", "-", 1.8, 0.9),
        "1970-1995": ("#c44e52", "-", 1.8, 0.9),
        "post-1995": ("#ccb974", "-", 1.8, 0.9),
    },
    "root_excerpt": {
        "root": ("#1f77b4", "-", 1.8, 0.9),
        "excerpt": ("#ff7f0e", "--", 1.5, 0.8),
    },
}


def render(df: pd.DataFrame, out_path: Path, title_suffix: str) -> None:
    available = [(m, lbl) for m, lbl in PLOT_METRICS if m in df.columns]
    n = len(available)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), sharex=True)
    axes = axes.flatten()

    legend_handles: dict[str, plt.Line2D] = {}

    for ax, (metric, label) in zip(axes, available):
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.2, linestyle=":")
        ax.set_xlabel("Anthology year")
        for dim in df["subgroup_dim"].unique():
            sub = df[df["subgroup_dim"] == dim]
            palette = SUBGROUP_PALETTES.get(dim, {})
            for level in sub["subgroup_level"].unique():
                rows = sub[sub["subgroup_level"] == level].sort_values("year")
                rows = rows.dropna(subset=[metric])
                if rows.empty:
                    continue
                color, ls, lw, alpha = palette.get(level, ("#888888", "-", 1.4, 0.6))
                label_key = f"{dim}: {level}" if dim != "all" else "overall"
                (handle,) = ax.plot(
                    rows["year"],
                    rows[metric],
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=alpha,
                    marker="o",
                    markersize=3.5,
                    label=label_key,
                )
                if label_key not in legend_handles:
                    legend_handles[label_key] = handle

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"AFAM anthology selection predictability over time — {title_suffix}",
        fontsize=13,
        y=1.00,
    )
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="lower center",
        ncol=min(len(legend_handles), 6),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help=f"Directory containing the predictability CSVs (default: {DEFAULT_CSV_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Where to write the PNGs (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    for entity in ["authors", "works"]:
        csv_path = args.csv_dir / f"predictability_over_time_{entity}.csv"
        if not csv_path.exists():
            print(
                f"Skipping {entity}: {csv_path} not found. "
                f"Run analysis/predictability_over_time.py --save-csv first."
            )
            continue
        df = pd.read_csv(csv_path)
        out_path = args.out_dir / f"predictability_over_time_{entity}.png"
        render(df, out_path, title_suffix=entity)


if __name__ == "__main__":
    main()
