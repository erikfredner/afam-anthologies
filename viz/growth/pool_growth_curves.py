"""
pool_growth_curves.py
---------------------
Plot the growth of the ever-anthologized author pool against the ever-
anthologized work pool across AFAM anthologies (1929-2025).

Two figures:

  output/pool_growth_curves.png       essay-ready. Top panel: cumulative pool
                                      size on a log axis with the fitted Heaps
                                      curves overlaid. Bottom panel: the share
                                      of each anthology's selections that are
                                      new to the pool -- the growth rate in the
                                      most literal sense, and where the 2025
                                      convergence is visible.

  output/pool_growth_diagnostics.png  the statistics. Left: the Heaps log-log
                                      fit with both exponents. Right: the
                                      loyalty-matched null distribution of
                                      Delta beta against the observed value and
                                      the bootstrap 95% CI.

All computation lives in analysis/growth/pool_growth_rates.py; this script only
draws it.

Usage:
    uv run python viz/growth/pool_growth_curves.py
    uv run python viz/growth/pool_growth_curves.py --time-axis year
    uv run python viz/growth/pool_growth_curves.py --only-root-works
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "growth"))

from pool_growth_rates import (
    apply_scope,
    bootstrap_delta_beta,
    build_steps,
    compute_curves,
    fit_both,
    load_data,
    loyalty_matched_null,
)

from afam.viz_style import OUTPUT_DIR

# Authors blue, works red, matching viz/reselection/cumulative_pairwise_agreement.py
# (note this is the inverse of afam.viz_style's AUTHOR_COLOR / WORK_COLOR).
C_AUTHOR = "#1f77b4"
C_WORK = "#d62728"
GRID_KW = {"alpha": 0.25, "linestyle": ":"}


def make_labels(curves) -> list[str]:
    """Two-line anthology labels, as in viz/reselection/retention_from_1929.py."""
    return [f"{row.label}\n{int(row.year)}" for row in curves.itertuples()]


def plot_curves(curves, fit_a, fit_w, out_path: Path) -> None:
    fig, (ax_pool, ax_nov) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]}
    )
    t = curves["t"].to_numpy()

    # ── Top: cumulative pool size, log scale, with fitted Heaps curves ────────
    for entity, fit, color, name in (
        ("authors", fit_a, C_AUTHOR, "Authors"),
        ("works", fit_w, C_WORK, "Works"),
    ):
        pool = curves[f"pool_{entity}"].to_numpy()
        ax_pool.plot(
            t,
            pool,
            color=color,
            lw=2,
            marker="o",
            markersize=4,
            label=f"{name} ever anthologized (n={pool[-1]:,})",
        )
        fitted = np.exp(fit.log_k) * curves[f"slots_{entity}"].to_numpy() ** fit.beta
        ax_pool.plot(
            t,
            fitted,
            color=color,
            lw=1.2,
            linestyle="--",
            alpha=0.75,
            label=rf"  Heaps fit, $\beta$ = {fit.beta:.3f} ($R^2$ = {fit.beta_r2:.3f})",
        )

    ax_pool.set_yscale("log")
    ax_pool.set_ylabel("Cumulative pool size (log scale)")
    ax_pool.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_pool.grid(True, which="both", axis="y", **GRID_KW)
    ax_pool.legend(frameon=False, fontsize=9, loc="lower right")
    ax_pool.set_title(
        "The pool of anthologized works grows faster than the pool of anthologized authors",
        fontsize=13,
        pad=12,
    )
    ratio = fit_w.beta / fit_a.beta
    ax_pool.text(
        0.015,
        0.95,
        f"Work pool grows {(ratio - 1) * 100:.0f}% faster per unit of selection effort\n"
        f"({fit_w.rate * 100:.1f}% vs {fit_a.rate * 100:.1f}% per anthology)",
        transform=ax_pool.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.75"},
    )

    # ── Bottom: per-anthology novelty rate ───────────────────────────────────
    ax_nov.plot(
        t,
        curves["novelty_authors"] * 100,
        color=C_AUTHOR,
        lw=2,
        marker="o",
        markersize=4,
        label="Authors new to the pool",
    )
    ax_nov.plot(
        t,
        curves["novelty_works"] * 100,
        color=C_WORK,
        lw=2,
        marker="s",
        markersize=4,
        label="Works new to the pool",
    )
    ax_nov.fill_between(
        t,
        curves["novelty_authors"] * 100,
        curves["novelty_works"] * 100,
        where=curves["novelty_works"] >= curves["novelty_authors"],
        color=C_WORK,
        alpha=0.10,
        interpolate=True,
    )
    ax_nov.set_ylabel("Share of the anthology's\nselections new to the pool")
    ax_nov.set_ylim(0, 105)
    ax_nov.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax_nov.grid(True, axis="y", **GRID_KW)
    ax_nov.legend(frameon=False, fontsize=9)

    ax_nov.set_xticks(t)
    ax_nov.set_xticklabels(make_labels(curves), rotation=45, ha="right", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_diagnostics(curves, fit_a, fit_w, boot, null, out_path: Path) -> None:
    fig, (ax_fit, ax_null) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: the Heaps log-log fit ──────────────────────────────────────────
    for entity, fit, color, name in (
        ("authors", fit_a, C_AUTHOR, "Authors"),
        ("works", fit_w, C_WORK, "Works"),
    ):
        slots = curves[f"slots_{entity}"].to_numpy(dtype=float)
        pool = curves[f"pool_{entity}"].to_numpy(dtype=float)
        ax_fit.scatter(slots, pool, color=color, s=26, zorder=3, label=name)
        grid = np.linspace(slots.min(), slots.max(), 100)
        ax_fit.plot(
            grid,
            np.exp(fit.log_k) * grid**fit.beta,
            color=color,
            lw=1.4,
            linestyle="--",
            label=rf"  $\beta$ = {fit.beta:.3f} $\pm$ {fit.beta_se:.3f}",
        )
    ax_fit.set_xscale("log")
    ax_fit.set_yscale("log")
    ax_fit.set_xlabel("Cumulative selection slots (log scale)")
    ax_fit.set_ylabel("Distinct entities ever selected (log scale)")
    ax_fit.set_title("Heaps' law fit: pool $\\propto$ effort$^{\\beta}$", fontsize=11)
    ax_fit.grid(True, which="both", **GRID_KW)
    ax_fit.legend(frameon=False, fontsize=9, loc="upper left")

    # ── Right: the loyalty-matched null ──────────────────────────────────────
    draws = null["_draws"]
    observed = null["observed_delta_beta"]
    ax_null.hist(
        draws,
        bins=40,
        color="0.6",
        edgecolor="white",
        linewidth=0.4,
        label=f"Null $\\Delta\\beta$ ({null['n_sims']:,} sims)",
    )
    ax_null.axvline(
        observed,
        color=C_WORK,
        lw=2.2,
        label=f"Observed $\\Delta\\beta$ = {observed:.3f}",
    )
    ax_null.axvspan(
        boot["delta_beta_lo"],
        boot["delta_beta_hi"],
        color=C_WORK,
        alpha=0.16,
        label="Bootstrap 95% CI",
    )
    ax_null.set_xlabel(r"$\Delta\beta$  ($\beta_{works} - \beta_{authors}$)")
    ax_null.set_ylabel(f"Null simulations (of {null['n_sims']:,})")
    ax_null.set_title(
        "Loyalty-matched null: work-slot counts held fixed,\n"
        "repeats drawn at the observed author retention rate",
        fontsize=11,
    )
    ax_null.grid(True, axis="y", **GRID_KW)
    ax_null.legend(frameon=False, fontsize=9, loc="upper center")
    ax_null.text(
        0.98,
        0.60,
        f"observed sits {null['z_vs_null']:.0f} sd\nabove the null\n"
        f"(p $\\leq$ {null['p_null_ge_observed']:.3f})",
        transform=ax_null.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.75"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--include-excerpts",
        dest="only_root_works",
        action="store_false",
        default=False,
        help="include excerpt works alongside root works (default here)",
    )
    group.add_argument(
        "--only-root-works",
        dest="only_root_works",
        action="store_true",
        help="exclude excerpts",
    )
    parser.add_argument(
        "--authored-only", action="store_true", help="drop works with no author"
    )
    parser.add_argument("--time-axis", choices=["edition", "year"], default="edition")
    parser.add_argument("--n", type=int, default=1000, help="null-model simulations")
    parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    raw = load_data()
    scoped = apply_scope(raw, args.only_root_works, args.authored_only)
    steps = build_steps(raw, args.time_axis)
    curves = compute_curves(scoped, steps)
    fit_a, fit_w = fit_both(curves)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(curves, fit_a, fit_w, args.out_dir / "pool_growth_curves.png")

    boot = bootstrap_delta_beta(scoped, steps, n_boot=args.n_boot, seed=args.seed)
    null = loyalty_matched_null(curves, n_sims=args.n, seed=args.seed)
    plot_diagnostics(
        curves, fit_a, fit_w, boot, null, args.out_dir / "pool_growth_diagnostics.png"
    )


if __name__ == "__main__":
    main()
