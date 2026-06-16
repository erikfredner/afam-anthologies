"""
author_work_shared_scatter.py
-----------------------------
For every cross-series pair of anthology editions, plot:

    x = |authors in common|
    y = |works in common|

The y = x reference line makes the dominant pattern visually obvious:
most pairs share more authors than works (points cluster below y = x).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from afam.db import query as query_db
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

OUT_FILE = OUTPUT_DIR / "author_work_shared_scatter.png"
OUT_FILE_LOG = OUTPUT_DIR / "author_work_shared_scatter_log.png"
DEFAULT_BANDS = [5, 10, 20]
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_SEED = 42

SERIES_ID_ABBREV: dict[str, str] = {
    "3": "NAAAL",
    "8": "Afro-Am. Writing",
    "12": "AAL Anthology",
    "17": "Wiley Blackwell AAL",
    "19": "Cavalcade",
}

STANDALONE_SHORT: dict[str, str] = {
    "67": "Amer. Negro Lit.",
    "66": "Readings fr. Negro Authors",
    "62": "Negro Caravan",
    "63": "Amer. Lit. by Negro Authors",
    "56": "Intro to Black Lit.",
    "43": "Black Voices",
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
}


# ── Load and prepare ──────────────────────────────────────────────────────────


def load_and_prepare_db() -> tuple[
    dict[str, set], dict[str, set], dict[str, int], dict[str, str]
]:
    df = query_db(query_path("work-selection-divergence"))

    df["edition_key"] = df.apply(
        lambda r: (
            f"{int(r['series_id'])}|{r['edition_number']}"
            if pd.notna(r["series_id"])
            else str(r["edition_id"])
        ),
        axis=1,
    )

    work_sets: dict[str, set] = {
        key: set(grp["work_id"].dropna().astype(str))
        for key, grp in df.groupby("edition_key")
    }
    author_sets: dict[str, set] = {
        key: set(grp["author_id"].dropna().astype(str))
        for key, grp in df.groupby("edition_key")
    }
    year_map: dict[str, int] = (
        df.groupby("edition_key")["anthology_publication_year"]
        .min()
        .astype(int)
        .to_dict()
    )

    name_map: dict[str, str] = {}
    for key, grp in df.groupby("edition_key"):
        row = grp.iloc[0]
        year = int(grp["anthology_publication_year"].min())
        if pd.notna(row.get("series_name")) and row["series_name"]:
            name_map[key] = f"{row['series_name']} ({year})"
        else:
            title = row.get("edition_title") or key
            name_map[key] = f"{title} ({year})"

    return work_sets, author_sets, year_map, name_map


# ── Series helpers ────────────────────────────────────────────────────────────


def _same_series(ki: str, kj: str) -> bool:
    if "|" not in ki or "|" not in kj:
        return False
    return ki.rsplit("|", 1)[0] == kj.rsplit("|", 1)[0]


# ── Pair data ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairRecord:
    shared_authors: int
    shared_works: int
    key_i: str
    key_j: str
    year_i: int
    year_j: int

    @property
    def year_gap(self) -> int:
        return abs(self.year_i - self.year_j)

    @property
    def authors_gt_works(self) -> bool:
        return self.shared_authors > self.shared_works

    @property
    def works_gt_authors(self) -> bool:
        return self.shared_works > self.shared_authors

    @property
    def equal(self) -> bool:
        return self.shared_authors == self.shared_works

    @property
    def nonzero_overlap(self) -> bool:
        return self.shared_authors > 0 or self.shared_works > 0


@dataclass(frozen=True)
class OutcomeSummary:
    n: int
    authors_gt_works: int
    works_gt_authors: int
    equal: int

    @property
    def author_gt_rate(self) -> float:
        return self.authors_gt_works / self.n if self.n else float("nan")

    @property
    def works_gt_rate(self) -> float:
        return self.works_gt_authors / self.n if self.n else float("nan")

    @property
    def equal_rate(self) -> float:
        return self.equal / self.n if self.n else float("nan")


@dataclass(frozen=True)
class BandSummary:
    band: int
    close_n: int
    close_k: int
    close_rate: float
    close_ci_low: float
    close_ci_high: float
    far_n: int
    far_k: int
    far_rate: float
    far_ci_low: float
    far_ci_high: float
    rate_diff: float
    z_p: float
    permutation_p: float


def build_pair_records(
    keys: list[str],
    work_sets: dict[str, set],
    author_sets: dict[str, set],
    year_map: dict[str, int],
    cross_series_only: bool = True,
) -> list[PairRecord]:
    """Return every unique anthology-edition pair in the requested pair universe."""
    pairs: list[PairRecord] = []
    for i, ki in enumerate(keys):
        for j in range(i + 1, len(keys)):
            kj = keys[j]
            if cross_series_only and _same_series(ki, kj):
                continue
            shared_a = len(author_sets.get(ki, set()) & author_sets.get(kj, set()))
            shared_w = len(work_sets.get(ki, set()) & work_sets.get(kj, set()))
            pairs.append(
                PairRecord(
                    shared_authors=shared_a,
                    shared_works=shared_w,
                    key_i=ki,
                    key_j=kj,
                    year_i=int(year_map[ki]),
                    year_j=int(year_map[kj]),
                )
            )
    return pairs


def build_pairs(
    keys: list[str],
    work_sets: dict[str, set],
    author_sets: dict[str, set],
    cross_series_only: bool = True,
) -> list[tuple[int, int, str, str]]:
    """Backward-compatible nonzero-overlap pair tuples used by older callers."""
    pairs = []
    for i, ki in enumerate(keys):
        for j in range(i + 1, len(keys)):
            kj = keys[j]
            if cross_series_only and _same_series(ki, kj):
                continue
            shared_a = len(author_sets.get(ki, set()) & author_sets.get(kj, set()))
            shared_w = len(work_sets.get(ki, set()) & work_sets.get(kj, set()))
            if shared_a > 0 or shared_w > 0:
                pairs.append((shared_a, shared_w, ki, kj))
    return pairs


def summarize_outcomes(pairs: list[PairRecord]) -> OutcomeSummary:
    return OutcomeSummary(
        n=len(pairs),
        authors_gt_works=sum(p.authors_gt_works for p in pairs),
        works_gt_authors=sum(p.works_gt_authors for p in pairs),
        equal=sum(p.equal for p in pairs),
    )


def _rate_ci(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    low, high = proportion_confint(k, n, alpha=0.05, method="wilson")
    return k / n, float(low), float(high)


def _two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    try:
        _, p_value = proportions_ztest([k1, k2], [n1, n2])
    except (ValueError, ZeroDivisionError):
        return float("nan")
    return float(p_value)


def _rate_difference(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    return k1 / n1 - k2 / n2


def permutation_p_value(
    pairs: list[PairRecord],
    year_map: dict[str, int],
    band: int,
    permutations: int,
    seed: int,
) -> float:
    """Shuffle publication years across editions and test close-vs-far rate gap."""
    if permutations <= 0:
        return float("nan")

    observed_close = [p for p in pairs if p.year_gap <= band]
    observed_far = [p for p in pairs if p.year_gap > band]
    observed = _rate_difference(
        sum(p.authors_gt_works for p in observed_close),
        len(observed_close),
        sum(p.authors_gt_works for p in observed_far),
        len(observed_far),
    )
    if pd.isna(observed):
        return float("nan")

    keys = sorted({p.key_i for p in pairs} | {p.key_j for p in pairs})
    years = np.array([int(year_map[k]) for k in keys])
    key_index = {k: idx for idx, k in enumerate(keys)}
    pair_indices = np.array(
        [(key_index[p.key_i], key_index[p.key_j]) for p in pairs], dtype=int
    )
    outcomes = np.array([p.authors_gt_works for p in pairs], dtype=int)

    rng = np.random.default_rng(seed)
    exceeded = 0
    valid = 0
    for _ in range(permutations):
        perm_years = rng.permutation(years)
        gaps = np.abs(perm_years[pair_indices[:, 0]] - perm_years[pair_indices[:, 1]])
        close_mask = gaps <= band
        far_mask = ~close_mask
        close_n = int(close_mask.sum())
        far_n = int(far_mask.sum())
        if close_n == 0 or far_n == 0:
            continue
        diff = outcomes[close_mask].mean() - outcomes[far_mask].mean()
        valid += 1
        if abs(diff) >= abs(observed):
            exceeded += 1

    return (exceeded + 1) / (valid + 1) if valid else float("nan")


def summarize_proximity_bands(
    pairs: list[PairRecord],
    year_map: dict[str, int],
    bands: list[int],
    permutations: int,
    seed: int,
) -> list[BandSummary]:
    summaries: list[BandSummary] = []
    for band in bands:
        close = [p for p in pairs if p.year_gap <= band]
        far = [p for p in pairs if p.year_gap > band]
        close_k = sum(p.authors_gt_works for p in close)
        far_k = sum(p.authors_gt_works for p in far)
        close_rate, close_low, close_high = _rate_ci(close_k, len(close))
        far_rate, far_low, far_high = _rate_ci(far_k, len(far))
        summaries.append(
            BandSummary(
                band=band,
                close_n=len(close),
                close_k=close_k,
                close_rate=close_rate,
                close_ci_low=close_low,
                close_ci_high=close_high,
                far_n=len(far),
                far_k=far_k,
                far_rate=far_rate,
                far_ci_low=far_low,
                far_ci_high=far_high,
                rate_diff=_rate_difference(close_k, len(close), far_k, len(far)),
                z_p=_two_proportion_p(close_k, len(close), far_k, len(far)),
                permutation_p=permutation_p_value(
                    pairs, year_map, band, permutations, seed
                ),
            )
        )
    return summaries


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{100 * x:.1f}%"


def fmt_p(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    if x < 0.001:
        return f"{x:.2e}"
    return f"{x:.4f}"


def print_outcome_summary(label: str, summary: OutcomeSummary) -> None:
    print(f"\n{label}")
    print(
        f"  {summary.authors_gt_works} of {summary.n} pairs "
        f"({fmt_pct(summary.author_gt_rate)}) have more shared authors than works."
    )
    print(
        f"  {summary.works_gt_authors} of {summary.n} pairs "
        f"({fmt_pct(summary.works_gt_rate)}) have more shared works than authors."
    )
    print(
        f"  {summary.equal} of {summary.n} pairs "
        f"({fmt_pct(summary.equal_rate)}) are equal."
    )


def print_band_summaries(summaries: list[BandSummary]) -> None:
    print("\nPublication-year proximity")
    for row in summaries:
        print(f"\n  Within {row.band} years vs. farther apart")
        print(
            f"    Close: n={row.close_n:3d}  k={row.close_k:3d}  "
            f"rate={fmt_pct(row.close_rate)}  "
            f"95% CI [{fmt_pct(row.close_ci_low)}, {fmt_pct(row.close_ci_high)}]"
        )
        print(
            f"    Far:   n={row.far_n:3d}  k={row.far_k:3d}  "
            f"rate={fmt_pct(row.far_rate)}  "
            f"95% CI [{fmt_pct(row.far_ci_low)}, {fmt_pct(row.far_ci_high)}]"
        )
        print(
            f"    Diff={fmt_pct(row.rate_diff)}  "
            f"z-test p={fmt_p(row.z_p)}  permutation p={fmt_p(row.permutation_p)}"
        )


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot(
    pairs: list[tuple[int, int]],
    out: Path,
    log: bool = False,
    annotate_pairs: list[tuple[int, int, str]] | None = None,
) -> None:
    more_a = [(a, w) for a, w in pairs if a > w]
    more_w = [(a, w) for a, w in pairs if w > a]
    equal = [(a, w) for a, w in pairs if a == w]

    lo = 0.5 if log else 0.0

    def xy(pts: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        arr = np.array(pts, dtype=float)
        return arr[:, 0], arr[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8))

    all_xs = [p[0] for p in pairs]
    all_ys = [p[1] for p in pairs]
    data_max = max(max(all_xs, default=1), max(all_ys, default=1))
    lim = max(data_max * 1.05, 1.0)
    ax.plot([lo, lim], [lo, lim], color="grey", linestyle="--", linewidth=1, zorder=0)

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")

    groups = [
        (more_a, "o", f"More shared authors ({len(more_a)})"),
        (more_w, "^", f"More shared works ({len(more_w)})"),
        (equal, "s", f"Equal ({len(equal)})"),
    ]
    for pts, marker, label in groups:
        if not pts:
            continue
        px, py = xy(pts)
        if log:
            px = np.maximum(px, lo)
            py = np.maximum(py, lo)
        ax.scatter(
            px,
            py,
            alpha=0.6,
            s=40,
            color="steelblue",
            marker=marker,
            linewidths=0,
            label=label,
        )

    ax.set_xlim(lo if log else 0, lim)
    ax.set_ylim(lo if log else 0, lim)
    ax.legend(fontsize=9, loc="upper left")

    ax.set_xlabel("Shared authors", fontsize=13)
    ax.set_ylabel("Shared works", fontsize=13)
    ax.set_title(
        "Shared authors vs. shared works between anthology pairs\n"
        "(each point = one pair; dashed line = equal shared authors and works)",
        fontsize=11,
    )

    ax.text(
        0.72,
        0.22,
        "More shared\nauthors",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="grey",
        style="italic",
    )
    ax.text(
        0.22,
        0.72,
        "More shared\nworks",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="grey",
        style="italic",
    )

    text_positions = [(0.58, 0.10), (0.30, 0.90)]
    for (ann_a, ann_w, label), (tx, ty) in zip(annotate_pairs or [], text_positions):
        ann_x = max(ann_a, lo)
        ann_y = max(ann_w, lo)
        ax.annotate(
            label,
            xy=(ann_x, ann_y),
            xycoords="data",
            xytext=(tx, ty),
            textcoords="axes fraction",
            fontsize=7.5,
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8),
        )

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Argparse ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scatterplot of shared authors vs. shared works for AFAM anthology pairs"
    )
    parser.add_argument(
        "--include-within-series",
        action="store_true",
        help=(
            "Include within-series pairs (e.g. NAAAL ed.1 vs NAAAL ed.2). "
            "Excluded by default."
        ),
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        type=int,
        default=DEFAULT_BANDS,
        metavar="YEARS",
        help="Publication-year proximity bands to test (default: 5 10 20).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        metavar="N",
        help=f"Publication-year permutation count (default: {DEFAULT_PERMUTATIONS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="INT",
        help=f"Random seed for permutation tests (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    cross_series_only = not args.include_within_series

    work_sets, author_sets, year_map, name_map = load_and_prepare_db()

    all_keys = sorted(
        set(work_sets) | set(author_sets),
        key=lambda k: year_map.get(k, 0),
    )

    pair_records = build_pair_records(
        all_keys, work_sets, author_sets, year_map, cross_series_only
    )
    nonzero_pair_records = [p for p in pair_records if p.nonzero_overlap]
    pairs = [(p.shared_authors, p.shared_works) for p in pair_records]

    pair_scope = "cross-series pairs" if cross_series_only else "all anthology pairs"
    print_outcome_summary(
        f"All {pair_scope} (including zero-overlap pairs)",
        summarize_outcomes(pair_records),
    )
    print_outcome_summary(
        f"Nonzero-overlap {pair_scope} (legacy denominator)",
        summarize_outcomes(nonzero_pair_records),
    )
    print_band_summaries(
        summarize_proximity_bands(
            pair_records, year_map, args.bands, args.permutations, args.seed
        )
    )

    def make_annotation(p: PairRecord) -> tuple[int, int, str]:
        ann_a, ann_w, ki, kj = p.shared_authors, p.shared_works, p.key_i, p.key_j
        label = f"{name_map.get(ki, ki)}\nvs. {name_map.get(kj, kj)}"
        return (ann_a, ann_w, label)

    annotate_pairs = []
    if nonzero_pair_records:
        min_pair = min(
            nonzero_pair_records, key=lambda p: (p.shared_works, p.shared_authors)
        )
        max_pair = max(
            nonzero_pair_records, key=lambda p: (p.shared_works, p.shared_authors)
        )

        ann_min = make_annotation(min_pair)
        ann_max = make_annotation(max_pair)

        for desc, ann in [("Fewest", ann_min), ("Most", ann_max)]:
            print(
                f"{desc} nonzero shared works: {ann[2].replace(chr(10), ' ')} "
                f"({ann[1]} works, {ann[0]} authors)"
            )

        annotate_pairs = [ann_min, ann_max]

    OUT_FILE.parent.mkdir(exist_ok=True)
    plot(pairs, OUT_FILE, annotate_pairs=annotate_pairs)
    plot(pairs, OUT_FILE_LOG, log=True, annotate_pairs=annotate_pairs)


if __name__ == "__main__":
    main()
