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
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import psycopg
from dotenv import dotenv_values


ENV_FILE = Path(__file__).parent.parent / ".env"
QUERIES = Path(__file__).parent.parent / "queries"
OUT_FILE = Path(__file__).parent / "author_work_shared_scatter.png"

SERIES_ID_ABBREV: dict[str, str] = {
    "3": "NAAAL",
    "8": "Afro-Am. Writing",
    "12": "AAL Anthology",
    "17": "Wiley Blackwell AAL",
}

STANDALONE_SHORT: dict[str, str] = {
    "67": "Amer. Negro Lit.",
    "66": "Readings fr. Negro Authors",
    "62": "Negro Caravan",
    "63": "Amer. Lit. by Negro Authors",
    "56": "Intro to Black Lit.",
    "43": "Black Voices",
    "42": "Cavalcade",
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
    "40": "New Cavalcade v.1",
    "41": "New Cavalcade v.2",
}


# ── DB helpers ────────────────────────────────────────────────────────────────


def parse_db_params(env_file: Path) -> dict[str, str]:
    env = dotenv_values(env_file)
    raw = env["DATABASE_URL"]
    return {
        "host": re.search(r"-h\s+(\S+)", raw).group(1),
        "user": re.search(r"-U\s+(\S+)", raw).group(1),
        "password": re.search(r"PGPASSWORD=(\S+)", raw).group(1),
        "dbname": raw.split()[-1],
    }


def query_db(params: dict[str, str], sql_file: Path) -> pd.DataFrame:
    sql = sql_file.read_text()
    with psycopg.connect(**params) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc.name for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


# ── Load and prepare ──────────────────────────────────────────────────────────


def load_and_prepare_db(
    params: dict[str, str],
) -> tuple[dict[str, set], dict[str, set], dict[str, int]]:
    df = query_db(params, QUERIES / "work-selection-divergence.sql")

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
    return work_sets, author_sets, year_map


# ── Series helpers ────────────────────────────────────────────────────────────


def _same_series(ki: str, kj: str) -> bool:
    if "|" not in ki or "|" not in kj:
        return False
    return ki.rsplit("|", 1)[0] == kj.rsplit("|", 1)[0]


# ── Pair data ─────────────────────────────────────────────────────────────────


def build_pairs(
    keys: list[str],
    work_sets: dict[str, set],
    author_sets: dict[str, set],
    cross_series_only: bool = True,
) -> list[tuple[int, int]]:
    """Return (shared_authors, shared_works) for each unique pair with at least one shared item."""
    pairs = []
    for i, ki in enumerate(keys):
        for j in range(i + 1, len(keys)):
            kj = keys[j]
            if cross_series_only and _same_series(ki, kj):
                continue
            shared_a = len(author_sets.get(ki, set()) & author_sets.get(kj, set()))
            shared_w = len(work_sets.get(ki, set()) & work_sets.get(kj, set()))
            if shared_a > 0 or shared_w > 0:
                pairs.append((shared_a, shared_w))
    return pairs


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot(pairs: list[tuple[int, int]], out: Path) -> None:
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(xs, ys, alpha=0.6, s=40, color="steelblue", linewidths=0)

    lim = max(max(xs), max(ys)) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=1, zorder=0)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

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
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    cross_series_only = not args.include_within_series

    params = parse_db_params(ENV_FILE)
    work_sets, author_sets, year_map = load_and_prepare_db(params)

    all_keys = sorted(
        set(work_sets) | set(author_sets),
        key=lambda k: year_map.get(k, 0),
    )

    pairs = build_pairs(all_keys, work_sets, author_sets, cross_series_only)

    total = len(pairs)
    n_below = sum(1 for a, w in pairs if a > w)
    n_above = sum(1 for a, w in pairs if w > a)
    n_equal = sum(1 for a, w in pairs if a == w)
    print(
        f"{n_below} of {total} pairs ({100 * n_below / total:.1f}%) have more shared authors than works."
    )
    print(
        f"{n_above} of {total} pairs ({100 * n_above / total:.1f}%) have more shared works than authors."
    )
    print(f"{n_equal} pairs ({100 * n_equal / total:.1f}%) are equal.")

    OUT_FILE.parent.mkdir(exist_ok=True)
    plot(pairs, OUT_FILE)


if __name__ == "__main__":
    main()
