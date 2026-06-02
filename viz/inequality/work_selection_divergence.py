"""work_selection_divergence.py
------------------------------
Six figures examining work selection divergence across African American
literature anthologies.

Usage: uv run python viz/work_selection_divergence.py
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


# ── Paths ──────────────────────────────────────────────────────────────────────

from afam.db import query as query_db
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

OUT_DIR = OUTPUT_DIR


# ── Style constants ────────────────────────────────────────────────────────────

PALETTE_SEQ = "viridis"
C_BLUE = "#1f77b4"
C_RED = "#d62728"
GRID_KW = dict(alpha=0.25, linestyle=":")


# ── Label dicts (verbatim from anthology_overlap_heatmap.py, adapted) ─────────

SERIES_ABBREV: dict[str, str] = {
    "The Norton Anthology of African American Literature": "NAAAL",
    "Afro-American Writing: An Anthology of Prose and Poetry": "Afro-Am. Writing",
    "African American Literature": "AAL Anthology",
    "The Wiley Blackwell Anthology of African American Literature": "Wiley Blackwell AAL",
}

# Numeric series_id → abbreviated name (new data format uses numeric series_id)
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

# Multi-volume standalones (kept for reference; new data uses anthology_id keys)
MULTIVOL_SHORT: dict[str, str] = {
    "Blackamerican Literature, 1760-Present|1": "Blackamerican Lit.",
}


def make_label(edition_key: str, year: int) -> str:
    if "|" in edition_key:
        series_id, edition = edition_key.rsplit("|", 1)
        abbrev = SERIES_ID_ABBREV.get(series_id, series_id)
        return f"{abbrev} ed.{edition}\n{year}"
    short = STANDALONE_SHORT.get(edition_key, edition_key[:20])
    return f"{short}\n{year}"


# ── Title normalization ────────────────────────────────────────────────────────


def normalize_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"^(from |excerpt from |selections? from )", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip()


# ── Name utilities ─────────────────────────────────────────────────────────────


def last_name(full_name: str) -> str:
    parts = full_name.strip().split()
    prefixes = {"du", "de", "van", "von", "la", "le"}
    if len(parts) >= 2 and parts[-2].lower() in prefixes:
        return " ".join(parts[-2:])
    return parts[-1] if parts else full_name


# ── DB helpers ────────────────────────────────────────────────────────────────


# ── Load and prepare ───────────────────────────────────────────────────────────


def load_and_prepare_db() -> pd.DataFrame:
    df = query_db(query_path("work-selection-divergence"))
    print(f"Loaded {len(df)} rows, {df['edition_id'].nunique()} raw anthologies")

    # Drop authorless works
    df = df[df["author_id"].notna()].copy()

    # Build edition_key: series_id|edition_number when series, else str(edition_id)
    df["edition_key"] = df.apply(
        lambda r: (
            f"{int(r['series_id'])}|{r['edition_number']}"
            if pd.notna(r["series_id"])
            else str(r["edition_id"])
        ),
        axis=1,
    )

    # Year per edition_key (minimum year across volumes of same edition)
    year_by_key = (
        df.groupby("edition_key")["anthology_publication_year"].min().astype(int)
    )
    df["ek_year"] = df["edition_key"].map(year_by_key)

    # Short label for display
    df["short_label"] = df.apply(
        lambda r: make_label(r["edition_key"], int(r["ek_year"])), axis=1
    )

    # Normalize work titles
    df["norm_title"] = df["work_title"].apply(normalize_title)

    print(f"  Unique work_ids: {df['work_id'].nunique()}")
    print(f"  Unique norm_titles: {df['norm_title'].nunique()}")
    print(f"  Unique edition_keys: {df['edition_key'].nunique()}")
    print(f"  Unique authors: {df['author_id'].nunique()}")
    print(f"  Total rows: {len(df)}")
    return df


# ── Save figure ────────────────────────────────────────────────────────────────


def save_fig(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png / .pdf")


# ── Shared intermediate tables ─────────────────────────────────────────────────


def build_shared_tables(
    long: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    author_ant_count = long.groupby("author_id")["edition_key"].nunique()
    author_work_count = long.groupby("author_id")["norm_title"].nunique()
    work_pop = long.groupby(["author_id", "norm_title"])["edition_key"].nunique()
    author_name_lookup = (
        long[["author_id", "author_name"]]
        .drop_duplicates("author_id")
        .set_index("author_id")["author_name"]
    )
    return author_ant_count, author_work_count, work_pop, author_name_lookup


# ── Figure 1 – Author Frequency vs. Work Consensus ────────────────────────────


def fig1_author_consensus(
    author_ant_count: pd.Series,
    work_pop: pd.Series,
    author_name_lookup: pd.Series,
    out_dir: Path,
) -> None:
    qualifying = author_ant_count[author_ant_count >= 2].index
    level0 = work_pop.index.get_level_values(0)

    records = []
    for aid in qualifying:
        if aid not in level0:
            continue
        n_ant = author_ant_count[aid]
        counts = work_pop.loc[aid]
        score = counts.max() / n_ant
        records.append({"author_id": aid, "n_ant": n_ant, "consensus": score})

    if not records:
        print("Figure 1: no qualifying authors")
        return

    df_plot = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(
        df_plot["n_ant"],
        df_plot["consensus"],
        alpha=0.4,
        color=C_BLUE,
        s=40,
        label="Authors (≥2 anthologies)",
    )

    # Label the 15 most-anthologized
    top15 = df_plot.nlargest(15, "n_ant")
    texts = []
    for _, row in top15.iterrows():
        name = author_name_lookup.get(row["author_id"], row["author_id"])
        t = ax.text(row["n_ant"], row["consensus"], last_name(name), fontsize=8)
        texts.append(t)
    if adjust_text:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.4, color="gray"))

    ax.axhline(
        0.5,
        color="gray",
        linestyle="--",
        lw=1.2,
        label="Majority agreement threshold",
    )
    ax.grid(True, **GRID_KW)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("Number of anthologies including author", fontsize=11)
    ax.set_ylabel(
        "Work consensus score\n(max fraction agreeing on any single work)", fontsize=11
    )
    ax.set_title(
        "Author frequency vs. work consensus\nacross African American literature anthologies",
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, "fig1_author_frequency_consensus", out_dir)


# ── Figure 2 – Pairwise Ratio Distribution ────────────────────────────────────


def fig2_pairwise_ratio(long: pd.DataFrame, out_dir: Path) -> float:
    edition_keys = long["edition_key"].unique().tolist()

    # Author sets per edition_key
    ek_authors: dict[str, set] = {
        ek: set(grp["author_id"].unique()) for ek, grp in long.groupby("edition_key")
    }

    # Work sets per (edition_key, author_id)
    ek_author_works: dict[str, dict[str, set]] = {}
    for (ek, aid), grp in long.groupby(["edition_key", "author_id"]):
        ek_author_works.setdefault(ek, {})[aid] = set(grp["norm_title"].unique())

    ratios: list[float] = []
    for ek_a, ek_b in combinations(edition_keys, 2):
        shared_authors = ek_authors.get(ek_a, set()) & ek_authors.get(ek_b, set())
        if not shared_authors:
            continue
        with_shared = sum(
            1
            for aid in shared_authors
            if ek_author_works.get(ek_a, {}).get(aid, set())
            & ek_author_works.get(ek_b, {}).get(aid, set())
        )
        ratios.append(with_shared / len(shared_authors))

    if not ratios:
        print("Figure 2: no anthology pairs with shared authors")
        return 0.0

    arr = np.array(ratios)
    mean_r = float(arr.mean())
    med_r = float(np.median(arr))
    print(f"Figure 2: {len(arr)} pairs, median ratio={med_r:.3f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(arr, bins=20, color=C_BLUE, alpha=0.8, edgecolor="white")
    ax.axvline(mean_r, color="black", lw=1.5, linestyle="-", label=f"Mean {mean_r:.2f}")
    ax.axvline(med_r, color=C_RED, lw=1.5, linestyle="--", label=f"Median {med_r:.2f}")
    ax.set_xlabel("Fraction of shared authors with ≥1 shared work", fontsize=11)
    ax.set_ylabel("Number of anthology pairs", fontsize=11)
    ax.set_title(
        f"Pairwise work overlap across anthology pairs\n"
        f"(conditional on shared authors; {len(arr)} pairs total)",
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, **GRID_KW)
    fig.tight_layout()
    save_fig(fig, "fig2_pairwise_ratio_histogram", out_dir)
    return med_r


# ── Figure 3 – Work Proliferation by Author (lollipop) ────────────────────────


def fig3_work_proliferation(
    author_ant_count: pd.Series,
    author_work_count: pd.Series,
    author_name_lookup: pd.Series,
    out_dir: Path,
) -> None:
    top20_ids = author_ant_count.nlargest(20).index
    rows = [
        {
            "author_id": aid,
            "n_ant": int(author_ant_count[aid]),
            "n_works": int(author_work_count.get(aid, 0)),
            "name": author_name_lookup.get(aid, aid),
        }
        for aid in top20_ids
    ]
    # Sort descending so i=0 is most-anthologized; invert_yaxis puts it at top
    df3 = (
        pd.DataFrame(rows).sort_values("n_ant", ascending=False).reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 9))
    added_ant = False
    added_wk = False
    for i, row in df3.iterrows():
        n_ant = row["n_ant"]
        n_works = row["n_works"]
        ax.plot(
            [min(n_works, n_ant), max(n_works, n_ant)],
            [i, i],
            color="gray",
            lw=0.8,
            zorder=1,
        )
        ant_label = "Anthologies" if not added_ant else None
        wk_label = "Unique works" if not added_wk else None
        ax.scatter(n_ant, i, color=C_BLUE, s=60, zorder=2, label=ant_label)
        ax.scatter(n_works, i, color=C_RED, s=40, zorder=2, marker="D", label=wk_label)
        added_ant = added_wk = True
        ratio_str = f"{n_works / n_ant:.2f}" if n_ant else "—"
        ax.text(
            max(n_works, n_ant) + 0.3,
            i,
            ratio_str,
            va="center",
            fontsize=7,
            color="gray",
        )

    ax.set_yticks(range(len(df3)))
    ax.set_yticklabels([last_name(r["name"]) for _, r in df3.iterrows()], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title(
        "Work proliferation by author\n(top 20 most-anthologized authors)", fontsize=12
    )
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis="x", **GRID_KW)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "fig3_work_proliferation", out_dir)


# ── Figure 4 – Long Tail of Work Selections ───────────────────────────────────


def fig4_work_frequency(long: pd.DataFrame, out_dir: Path) -> float:
    per_work = long.groupby("norm_title")["edition_key"].nunique()
    freq_dist = per_work.value_counts().sort_index()
    total_works = len(per_work)
    singletons = int(freq_dist.get(1, 0))
    singleton_pct = 100.0 * singletons / total_works

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(freq_dist.index, freq_dist.values, color=C_BLUE, width=0.6)

    # Log scale if singletons dominate
    if len(freq_dist) > 1 and singletons > 5 * freq_dist.iloc[1:].max():
        ax.set_yscale("log")
        y_label = "Number of works (log scale)"
    else:
        y_label = "Number of works"

    # Annotate each bar with its count
    for bar, val in zip(bars, freq_dist.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Call out the singleton bar
    ax.annotate(
        f"{singleton_pct:.0f}% of all works",
        xy=(1, singletons),
        xytext=(freq_dist.index.max() * 0.35, singletons * 0.5),
        arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
        fontsize=9,
        color="gray",
    )

    # Name a few top works
    max_freq = int(per_work.max())
    top_works = per_work[per_work == max_freq].index[:5].tolist()
    top_str = "\n".join(t[:40] for t in top_works)
    ax.text(
        0.98,
        0.97,
        f"Most anthologized ({max_freq}×):\n{top_str}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="gray",
        style="italic",
    )

    ax.set_xlabel("Number of anthologies including a work", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title("Long tail of work selections across anthologies", fontsize=12)
    ax.grid(True, axis="y", **GRID_KW)
    fig.tight_layout()
    save_fig(fig, "fig4_work_frequency_distribution", out_dir)
    return singleton_pct


# ── Figure 5 – Within-Author Jaccard Similarity ───────────────────────────────


def _parse_birth_year(raw_vals: pd.Series) -> int | None:
    """Return the first clean 4-digit birth year found in the series."""
    for val in raw_vals:
        if pd.notna(val):
            try:
                yr = int(val)
                if 1600 <= yr <= 2000:
                    return yr
            except (ValueError, TypeError):
                pass
    return None


def fig5_jaccard(
    long: pd.DataFrame,
    author_ant_count: pd.Series,
    author_name_lookup: pd.Series,
    out_dir: Path,
) -> float:
    # Compute pairwise Jaccard for all authors in ≥3 anthologies.
    # Jaccard = |A∩B| / |A∪B| is always in [0, 1] by definition.
    qualifying = author_ant_count[author_ant_count >= 3].index

    jaccard_records: list[dict] = []
    for aid in qualifying:
        sub = long[long["author_id"] == aid]
        ant_works = sub.groupby("edition_key")["norm_title"].apply(set)
        editions = ant_works.index.tolist()
        if len(editions) < 2:
            continue
        for e1, e2 in combinations(editions, 2):
            w1, w2 = ant_works[e1], ant_works[e2]
            union = w1 | w2
            j = len(w1 & w2) / len(union) if union else 0.0
            jaccard_records.append({"author_id": aid, "jaccard": j})

    if not jaccard_records:
        print("Figure 5: no qualifying Jaccard pairs")
        return 0.0

    df_j = pd.DataFrame(jaccard_records)
    overall_median = float(df_j["jaccard"].median())

    # Restrict to top 10 authors by anthology count
    top10_ids = author_ant_count.nlargest(10).index.tolist()
    df_top = df_j[df_j["author_id"].isin(top10_ids)].copy()
    df_top["author_name"] = df_top["author_id"].map(author_name_lookup)
    df_top["last_nm"] = df_top["author_name"].apply(
        lambda n: last_name(n) if n else "Unknown"
    )

    # Build birth-year lookup for sort order
    birth_years: dict[str, int] = {}
    for aid in top10_ids:
        yr = _parse_birth_year(long.loc[long["author_id"] == aid, "author_birth_year"])
        if yr is not None:
            birth_years[aid] = yr

    # Order rows birth-year ascending (oldest author at top of horizontal plot)
    top10_sorted = sorted(top10_ids, key=lambda a: birth_years.get(a, 9999))
    aid_to_last = {
        aid: last_name(author_name_lookup.get(aid, aid)) for aid in top10_ids
    }
    present = set(df_top["last_nm"].unique())
    order = [aid_to_last[a] for a in top10_sorted if aid_to_last[a] in present]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Jittered points first so box outline sits on top
    sns.stripplot(
        data=df_top,
        x="jaccard",
        y="last_nm",
        order=order,
        color=C_BLUE,
        alpha=0.4,
        size=3.5,
        jitter=True,
        ax=ax,
    )
    # Clear (unfilled) box with visible median line drawn on top of points
    sns.boxplot(
        data=df_top,
        x="jaccard",
        y="last_nm",
        order=order,
        fill=False,
        color="black",
        width=0.45,
        showfliers=False,
        linewidth=1.2,
        ax=ax,
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Jaccard similarity of work selections", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(
        "Within-author work selection agreement across anthology pairs\n"
        "(top 10 authors by anthology count, ordered by birth year)",
        fontsize=12,
    )
    ax.grid(True, axis="x", **GRID_KW)
    fig.tight_layout()
    save_fig(fig, "fig5_jaccard_similarity", out_dir)
    return overall_median


# ── Figure 6 – Canonical Works Tile Map ───────────────────────────────────────


def fig6_tilemap(
    long: pd.DataFrame,
    author_ant_count: pd.Series,
    author_name_lookup: pd.Series,
    out_dir: Path,
) -> None:
    top_authors = author_ant_count.nlargest(12).index.tolist()

    # Edition_keys sorted chronologically
    ek_year_min = long.groupby("edition_key")["ek_year"].min()
    sorted_eks = ek_year_min.sort_values().index.tolist()
    ek_labels = {
        ek: long.loc[long["edition_key"] == ek, "short_label"].iloc[0]
        for ek in sorted_eks
    }
    n_ek = len(sorted_eks)

    # Build panels: works per author, filtered to ≥2 anthologies if >100 rows
    panels: list[tuple[str, list[str]]] = []
    omitted_notes: list[str] = []
    for aid in top_authors:
        sub = long[long["author_id"] == aid]
        work_freq = sub.groupby("norm_title")["edition_key"].nunique()
        total_rows = len(work_freq)
        min_freq = 2 if total_rows > 100 else 1
        filtered = work_freq[work_freq >= min_freq]
        if total_rows > 100:
            n_omit = total_rows - len(filtered)
            name = author_name_lookup.get(aid, aid)
            omitted_notes.append(f"{last_name(name)}: {n_omit} singleton works omitted")
        works_sorted = filtered.sort_values(ascending=False).index.tolist()
        panels.append((aid, works_sorted))

    # Panel heights proportional to work count (min 1)
    heights = [max(len(ws), 1) for _, ws in panels]
    total_height = max(14, sum(heights) * 0.35 + 3)
    n_pan = len(panels)

    fig = plt.figure(figsize=(14, total_height))
    gs = gridspec.GridSpec(n_pan, 1, hspace=0.08, height_ratios=heights)

    for i, (aid, works_sorted) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        sub = long[long["author_id"] == aid]

        # Binary presence matrix
        mat = np.zeros((max(len(works_sorted), 1), n_ek), dtype=int)
        for wi, wt in enumerate(works_sorted):
            ek_set = set(sub.loc[sub["norm_title"] == wt, "edition_key"].unique())
            for ei, ek in enumerate(sorted_eks):
                mat[wi, ei] = 1 if ek in ek_set else 0

        ax.imshow(
            mat, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="none"
        )

        # Work row labels (left y-axis)
        short_works = [w[:35] for w in works_sorted]
        ax.set_yticks(range(len(works_sorted)))
        ax.set_yticklabels(short_works, fontsize=6)

        # Author label on y-axis
        name = author_name_lookup.get(aid, aid)
        ax.set_ylabel(
            last_name(name),
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
            labelpad=55,
        )

        # X-axis ticks only on the bottom panel
        if i == n_pan - 1:
            ax.set_xticks(range(n_ek))
            ax.set_xticklabels(
                [ek_labels[ek] for ek in sorted_eks],
                rotation=45,
                ha="right",
                fontsize=6,
            )
        else:
            ax.set_xticks([])

    fig.suptitle("Work selections by anthology for top authors", fontsize=13, y=1.005)
    if omitted_notes:
        fig.text(
            0.01,
            -0.01,
            "Omitted (singleton works): " + "; ".join(omitted_notes),
            fontsize=6,
            color="gray",
            ha="left",
            va="top",
        )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.97, bottom=0.08)
    save_fig(fig, "fig6_canonical_works_tilemap", out_dir)


# ── Narrative summary ──────────────────────────────────────────────────────────


def print_summary(
    long: pd.DataFrame,
    author_ant_count: pd.Series,
    author_work_count: pd.Series,
    work_pop: pd.Series,
    author_name_lookup: pd.Series,
    median_pairwise: float,
    overall_jaccard: float,
) -> None:
    per_work = long.groupby("norm_title")["edition_key"].nunique()
    total_works = len(per_work)
    singletons = int((per_work == 1).sum())
    singleton_pct = 100.0 * singletons / total_works

    # Authors with consensus > 0.5 among those in 5+ anthologies
    freq_authors = author_ant_count[author_ant_count >= 5].index
    level0 = work_pop.index.get_level_values(0)
    high_consensus = sum(
        1
        for aid in freq_authors
        if aid in level0 and work_pop.loc[aid].max() / author_ant_count[aid] > 0.5
    )

    # Most proliferative: highest n_works / n_anthologies ratio
    common = author_ant_count.index.intersection(author_work_count.index)
    ratios = author_work_count[common] / author_ant_count[common]
    most_prolif_id = ratios.idxmax()
    most_prolif_name = author_name_lookup.get(most_prolif_id, most_prolif_id)
    most_prolif_ratio = float(ratios.max())

    print()
    print("=" * 70)
    print("NARRATIVE SUMMARY")
    print("=" * 70)
    print(
        f"Dataset: {long['edition_key'].nunique()} anthology editions, "
        f"{long['author_id'].nunique()} authors, {total_works} unique works "
        f"(after title normalization)."
    )
    print(
        f"{singletons} of {total_works} works ({singleton_pct:.0f}%) appear in only "
        f"one anthology — the long tail dominates the corpus."
    )
    print(
        f"Among authors in ≥5 anthologies, {high_consensus} have a consensus "
        f"score >0.5 (majority of their anthologies agree on a single work)."
    )
    print(
        f"Median pairwise fraction of shared authors also sharing ≥1 work: "
        f"{median_pairwise:.2f} — most anthology pairs diverge on specific works."
    )
    print(
        f"Most proliferative author (works-per-anthology ratio): "
        f"{most_prolif_name} ({most_prolif_ratio:.2f})."
    )
    print(
        f"Overall median within-author Jaccard similarity: {overall_jaccard:.2f} "
        f"(low values confirm wide divergence in which works represent each author)."
    )
    print("=" * 70)


# ── Argparse ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Six figures on work selection divergence in African American lit anthologies"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    long = load_and_prepare_db()

    author_ant_count, author_work_count, work_pop, author_name_lookup = (
        build_shared_tables(long)
    )

    fig1_author_consensus(author_ant_count, work_pop, author_name_lookup, args.out_dir)
    median_pairwise = fig2_pairwise_ratio(long, args.out_dir)
    fig3_work_proliferation(
        author_ant_count, author_work_count, author_name_lookup, args.out_dir
    )
    fig4_work_frequency(long, args.out_dir)
    overall_jaccard = fig5_jaccard(
        long, author_ant_count, author_name_lookup, args.out_dir
    )
    fig6_tilemap(long, author_ant_count, author_name_lookup, args.out_dir)

    print_summary(
        long,
        author_ant_count,
        author_work_count,
        work_pop,
        author_name_lookup,
        median_pairwise,
        overall_jaccard,
    )


if __name__ == "__main__":
    main()
