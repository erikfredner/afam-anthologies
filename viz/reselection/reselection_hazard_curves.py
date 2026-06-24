"""
reselection_hazard_curves.py
----------------------------
Figures for the discrete-time reselection hazard model
(analysis/reselection/reselection_hazard.py):

  output/reselection_survival.png   author vs work survival S(t) with 95% CI
                                     bands -- P(not yet reselected after t
                                     editions since debut)
  output/reselection_hazard.png     discrete hazard h(t) by editions-since-debut
  output/reselection_hr_forest.png  hazard ratios (exp coef, 95% CI) from the
                                     combined complementary-log-log model, with
                                     the editions-at-risk baseline dummies
                                     suppressed

Usage:
    uv run python viz/reselection/reselection_hazard_curves.py
    uv run python viz/reselection/reselection_hazard_curves.py --event-model recurrent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from afam.cli import add_root_works_flag
from afam.viz_style import AUTHOR_COLOR, DPI, FIGSIZE, OUTPUT_DIR, WORK_COLOR

# Reuse the analysis logic rather than recomputing it here.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "analysis" / "reselection")
)
from reselection_hazard import (  # noqa: E402
    build_all_person_periods,
    fit_hazard_models,
    lifetable_survival,
    load_data,
)

TYPE_COLOR = {"author": AUTHOR_COLOR, "work": WORK_COLOR}

OUT_SURVIVAL = OUTPUT_DIR / "reselection_survival.png"
OUT_HAZARD = OUTPUT_DIR / "reselection_hazard.png"
OUT_FOREST = OUTPUT_DIR / "reselection_hr_forest.png"


def plot_survival(lifetable) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for etype, grp in lifetable.groupby("entity_type"):
        grp = grp.sort_values("t_since_debut")
        color = TYPE_COLOR.get(etype, None)
        ax.step(
            grp["t_since_debut"],
            grp["survival"],
            where="post",
            label=etype,
            color=color,
            lw=2,
        )
        ax.fill_between(
            grp["t_since_debut"],
            grp["survival_lo"],
            grp["survival_hi"],
            step="post",
            color=color,
            alpha=0.15,
        )
    ax.set_xlabel("Editions since debut (t)")
    ax.set_ylabel("S(t): P(not yet reselected)")
    ax.set_title("Reselection survival by entity type")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_SURVIVAL, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {OUT_SURVIVAL}")


def plot_hazard(lifetable) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    types = list(lifetable["entity_type"].unique())
    width = 0.8 / max(len(types), 1)
    for i, etype in enumerate(types):
        grp = lifetable[lifetable["entity_type"] == etype].sort_values("t_since_debut")
        x = (
            grp["t_since_debut"].to_numpy(dtype=float)
            + (i - (len(types) - 1) / 2) * width
        )
        ax.bar(
            x,
            grp["hazard"],
            width=width,
            label=etype,
            color=TYPE_COLOR.get(etype, None),
            alpha=0.85,
        )
    ax.set_xlabel("Editions since debut (t)")
    ax.set_ylabel("Discrete hazard h(t)")
    ax.set_title("Per-edition reselection hazard by entity type")
    ax.grid(alpha=0.25, linestyle=":", axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_HAZARD, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {OUT_HAZARD}")


def plot_hr_forest(coef_df) -> None:
    combined = coef_df[
        (coef_df["model"] == "combined") & (coef_df["link"] == "cloglog")
    ]
    # Suppress the intercept and the editions-at-risk baseline dummies; those are
    # nuisance terms, not effects of interest.
    keep = combined[
        ~combined["term"].str.startswith("Intercept")
        & ~combined["term"].str.contains("C(t_since_debut)", regex=False)
    ].copy()
    if keep.empty:
        print("No non-baseline terms to plot; skipping forest.")
        return

    keep = keep.iloc[::-1]
    y = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], max(FIGSIZE[1], 0.4 * len(keep) + 1)))
    ax.errorbar(
        keep["hazard_ratio"],
        y,
        xerr=[
            keep["hazard_ratio"] - keep["ci_lo"],
            keep["ci_hi"] - keep["hazard_ratio"],
        ],
        fmt="o",
        color="#333333",
        ecolor="#888888",
        capsize=3,
    )
    ax.axvline(1.0, color="red", lw=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(keep["term"])
    ax.set_xlabel("Hazard ratio (exp coef), 95% CI")
    ax.set_title("Reselection hazard ratios (cloglog model)")
    ax.grid(alpha=0.25, linestyle=":", axis="x")
    fig.tight_layout()
    fig.savefig(OUT_FOREST, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {OUT_FOREST}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_works_flag(parser)
    parser.add_argument(
        "--event-model", choices=("first", "recurrent"), default="first"
    )
    parser.add_argument("--scope", choices=("all", "cross-series"), default="all")
    args = parser.parse_args()

    raw, all_editions, form_map = load_data(args.only_root_works)

    # Survival/hazard curves are a single-event construct.
    pp_first = build_all_person_periods(
        raw, all_editions, form_map, event_model="first", scope=args.scope
    )
    lifetable = lifetable_survival(pp_first)
    plot_survival(lifetable)
    plot_hazard(lifetable)

    pp_model = build_all_person_periods(
        raw, all_editions, form_map, event_model=args.event_model, scope=args.scope
    )
    results = fit_hazard_models(pp_model)
    plot_hr_forest(results["coefficients"])


if __name__ == "__main__":
    main()
