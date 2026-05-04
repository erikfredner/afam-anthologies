"""
works_in_half_or_more_afam_eds.py
----------------------------------
Lists works appearing in at least half of all African American literary
anthology editions, queried live from the database.

Output columns:
  Work        — '"Title" by Author'
  Anthologies — number of distinct editions that selected the work
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import dotenv_values

ENV_FILE = Path(__file__).parent.parent / ".env"
SQL_FILE = Path(__file__).parent.parent / "queries" / "works-in-half-or-more-afam-eds.sql"
OUT_CSV  = Path(__file__).parent.parent / "data" / "works_in_half_or_more_afam_eds.csv"


def parse_db_params(env_file: Path) -> dict[str, str]:
    env = dotenv_values(env_file)
    raw = env["DATABASE_URL"]
    return {
        "host":     re.search(r"-h\s+(\S+)", raw).group(1),
        "user":     re.search(r"-U\s+(\S+)", raw).group(1),
        "password": re.search(r"PGPASSWORD=(\S+)", raw).group(1),
        "dbname":   raw.split()[-1],
    }


def join_authors(names: pd.Series) -> str:
    unique = list(dict.fromkeys(names))
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return ", ".join(unique[:-1]) + f", and {unique[-1]}"


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["work_id", "work_title", "edition_count"], sort=False)
        .agg(author_name=("author_name", join_authors))
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["edition_count", "author_name", "work_title"],
        ascending=[False, True, True],
    )
    result = pd.DataFrame({
        "Work":        grouped.apply(lambda r: f'"{r["work_title"]}" by {r["author_name"]}', axis=1),
        "Anthologies": grouped["edition_count"],
    })
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help=f"Write results to {OUT_CSV}",
    )
    args = parser.parse_args()

    params = parse_db_params(ENV_FILE)
    sql = SQL_FILE.read_text()

    with psycopg.connect(**params) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc.name for desc in cur.description]
        raw = pd.DataFrame(cur.fetchall(), columns=cols)

    table = build_table(raw)

    print(table.to_string(index=False))
    print(f"\n{len(table)} works")

    if args.save_csv:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_CSV, index=False)
        print(f"Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
