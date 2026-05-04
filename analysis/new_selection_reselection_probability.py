"""
new_selection_reselection_probability.py
-----------------------------------------
For any work or author making its debut in an AFAM-tagged anthology, what is
the probability that it will be selected again by at least one subsequent
AFAM anthology?

Definitions
-----------
debut    : first appearance across all AFAM editions (min publication year)
reselected: appears in at least one AFAM edition with year > debut_year
excluded : items debuting in the most recent AFAM year have no subsequent
           opportunity and are excluded from the denominator

Outputs overall probability + 95% Wilson score confidence interval for both
works and authors, then the same statistics broken down by debut decade.

Usage
-----
    uv run python analysis/new_selection_reselection_probability.py
    uv run python analysis/new_selection_reselection_probability.py --only-root-works
    uv run python analysis/new_selection_reselection_probability.py --save-csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import dotenv_values
from statsmodels.stats.proportion import proportion_confint


ENV_FILE = Path(__file__).parent.parent / ".env"
QUERIES = Path(__file__).parent.parent / "queries"
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_CSV = DATA_DIR / "new_selection_reselection_probability.csv"


# ── DB helpers ────────────────────────────────────────────────────────────────


def parse_db_params(env_file: Path) -> dict[str, str]:
    env = dotenv_values(env_file)
    raw = env["DATABASE_URL"]
    return {
        "host":     re.search(r"-h\s+(\S+)", raw).group(1),
        "user":     re.search(r"-U\s+(\S+)", raw).group(1),
        "password": re.search(r"PGPASSWORD=(\S+)", raw).group(1),
        "dbname":   raw.split()[-1],
    }


def query_db(params: dict[str, str], sql_file: Path) -> pd.DataFrame:
    sql = sql_file.read_text()
    with psycopg.connect(**params) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc.name for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


# ── Statistics ────────────────────────────────────────────────────────────────


def reselection_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    k = int(df["is_reselected"].sum())
    p = k / n if n else float("nan")
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson") if n else (float("nan"), float("nan"))
    return {"n": n, "k": k, "p": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}


def decade_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decade"] = (df["debut_year"] // 10 * 10).astype(int)
    rows = []
    for decade, grp in df.groupby("decade"):
        s = reselection_stats(grp)
        rows.append({"decade": decade, **s})
    return pd.DataFrame(rows).set_index("decade")


# ── Reporting ─────────────────────────────────────────────────────────────────


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_summary(label: str, stats: dict) -> None:
    print(
        f"  {label:<10}  n={stats['n']:>5}  k={stats['k']:>5}  "
        f"p={_fmt_pct(stats['p'])}  "
        f"95% CI [{_fmt_pct(stats['ci_lo'])}, {_fmt_pct(stats['ci_hi'])}]"
    )


def print_decade_table(label: str, bkdn: pd.DataFrame) -> None:
    print(f"\n  {label} by debut decade:")
    print(f"  {'Decade':<8}  {'n':>5}  {'k':>5}  {'p':>7}  {'CI lo':>7}  {'CI hi':>7}")
    for decade, row in bkdn.iterrows():
        print(
            f"  {decade:<8}  {int(row['n']):>5}  {int(row['k']):>5}  "
            f"{_fmt_pct(row['p']):>7}  {_fmt_pct(row['ci_lo']):>7}  {_fmt_pct(row['ci_hi']):>7}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only-root-works", action="store_true", help="exclude excerpts (works with a parent)")
    parser.add_argument("--save-csv", action="store_true", help="write results to data/")
    args = parser.parse_args()

    params = parse_db_params(ENV_FILE)

    works_df = query_db(params, QUERIES / "new-selection-reselection-afam-works.sql")
    authors_df = query_db(params, QUERIES / "new-selection-reselection-afam-authors.sql")

    if args.only_root_works:
        works_df = works_df[works_df["parent_id"].isna()].copy()

    works_label = "Works (root only)" if args.only_root_works else "Works"

    works_stats = reselection_stats(works_df)
    authors_stats = reselection_stats(authors_df)

    print("\nProbability that a debut selection is reselected by any subsequent AFAM anthology")
    print("=" * 80)
    print_summary(works_label, works_stats)
    print_summary("Authors", authors_stats)

    works_bkdn = decade_breakdown(works_df)
    authors_bkdn = decade_breakdown(authors_df)
    print_decade_table(works_label, works_bkdn)
    print_decade_table("Authors", authors_bkdn)
    print()

    if args.save_csv:
        rows = []
        for entity, stats, bkdn in [
            (works_label, works_stats, works_bkdn),
            ("Authors", authors_stats, authors_bkdn),
        ]:
            rows.append({"entity": entity, "decade": "overall", **stats})
            for decade, row in bkdn.iterrows():
                rows.append({"entity": entity, "decade": decade, **row.to_dict()})
        out = pd.DataFrame(rows)
        out.to_csv(OUT_CSV, index=False)
        print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
