"""
selection_frequency_decay_db.py
---------------------------------
Recreates the selection frequency decay figure using live database data.

For every work and author selected in an African-American Literature anthology
edition, counts how many distinct editions include them, then plots:
  - Main panel: survival curve (% of items in >= k editions) for works and authors
  - Bottom panel: absolute N counts per k, with annotations when N < 10
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
from dotenv import dotenv_values


ENV_FILE      = Path(__file__).parent.parent / ".env"
WORK_SQL_FILE = Path(__file__).parent.parent / "queries" / "work-edition-counts-afam.sql"
AUTH_SQL_FILE = Path(__file__).parent.parent / "queries" / "author-edition-counts-afam.sql"
OUT_FILE      = Path(__file__).parent / "selection_frequency_decay_db.png"

WORK_COLOR   = "#1f77b4"
AUTHOR_COLOR = "#d62728"


def parse_db_params(env_file: Path) -> dict[str, str]:
    env = dotenv_values(env_file)
    raw = env["DATABASE_URL"]
    return {
        "host":     re.search(r"-h\s+(\S+)", raw).group(1),
        "user":     re.search(r"-U\s+(\S+)", raw).group(1),
        "password": re.search(r"PGPASSWORD=(\S+)", raw).group(1),
        "dbname":   raw.split()[-1],
    }


def query(sql: str, params: dict[str, str]) -> pd.DataFrame:
    with psycopg.connect(**params) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc.name for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def survival_stats(
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ks, ns (absolute counts >= k), pcts for k = 1 … max."""
    max_k = int(counts.max())
    ks    = np.arange(1, max_k + 1)
    ns    = np.array([int(np.sum(counts >= k)) for k in ks])
    pcts  = 100.0 * ns / len(counts)
    return ks, ns, pcts


def plot(
    work_ks: np.ndarray, work_ns: np.ndarray, work_pcts: np.ndarray, n_works: int,
    auth_ks: np.ndarray, auth_ns: np.ndarray, auth_pcts: np.ndarray, n_authors: int,
    out: Path,
) -> None:
    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ── Main panel: survival curves ───────────────────────────────────────────
    work_label   = f"Works (N={n_works:,})"
    author_label = f"Authors (N={n_authors:,})"

    ax.plot(work_ks, work_pcts, color=WORK_COLOR, linewidth=0.8, alpha=0.4, zorder=2)
    ax.scatter(work_ks, work_pcts,
               color=WORK_COLOR, marker="o", zorder=3, label=work_label)

    ax.plot(auth_ks, auth_pcts, color=AUTHOR_COLOR, linewidth=0.8, alpha=0.4, zorder=2)
    ax.scatter(auth_ks, auth_pcts,
               color=AUTHOR_COLOR, marker="s", zorder=3, label=author_label)

    max_k = max(int(work_ks.max()), int(auth_ks.max()))

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.text(max_k + 0.45, 51.5, "50%", fontsize=8, color="gray", va="bottom")

    ax.set_ylabel("% of items selected in ≥ k editions", fontsize=11)
    ax.set_title(
        "Author and work selections across anthologies",
        fontsize=12,
    )
    ax.set_xlim(0.5, max_k + 0.5)
    ax.set_ylim(-2, 105)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle=":")

    # ── Bottom panel: absolute N counts ──────────────────────────────────────
    bar_w = 0.4

    ax_n.bar(work_ks - bar_w / 2, work_ns, width=bar_w,
             color=WORK_COLOR, alpha=0.7, label="Works")
    ax_n.bar(auth_ks + bar_w / 2, auth_ns, width=bar_w,
             color=AUTHOR_COLOR, alpha=0.7, label="Authors")

    ax_n.set_yscale("log")
    ax_n.set_xlabel("Number of anthology editions selecting an item (k)", fontsize=11)
    ax_n.set_ylabel("N (≥ k editions)", fontsize=9)
    ax_n.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_n.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    params = parse_db_params(ENV_FILE)

    work_df = query(WORK_SQL_FILE.read_text(), params)
    auth_df = query(AUTH_SQL_FILE.read_text(), params)

    work_counts   = work_df["edition_count"].to_numpy(dtype=int)
    author_counts = auth_df["edition_count"].to_numpy(dtype=int)

    work_ks, work_ns, work_pcts     = survival_stats(work_counts)
    auth_ks, auth_ns, auth_pcts     = survival_stats(author_counts)

    OUT_FILE.parent.mkdir(exist_ok=True)
    plot(
        work_ks, work_ns, work_pcts, len(work_counts),
        auth_ks, auth_ns, auth_pcts, len(author_counts),
        OUT_FILE,
    )


if __name__ == "__main__":
    main()
