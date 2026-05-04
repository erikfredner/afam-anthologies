"""
authors_in_half_or_more_afam_eds.py
------------------------------------
Lists authors appearing in at least half of all African American literary
anthology editions, queried live from the database.

Output columns:
  Author      — "Name (b. YYYY)" or "Name" when birth year is unknown
  Anthologies — number of distinct editions that selected the author
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import dotenv_values

ENV_FILE = Path(__file__).parent.parent / ".env"
SQL_FILE = Path(__file__).parent.parent / "queries" / "authors-in-half-or-more-afam-eds.sql"
OUT_CSV  = Path(__file__).parent.parent / "data" / "authors_in_half_or_more_afam_eds.csv"


def parse_db_params(env_file: Path) -> dict[str, str]:
    env = dotenv_values(env_file)
    raw = env["DATABASE_URL"]
    return {
        "host":     re.search(r"-h\s+(\S+)", raw).group(1),
        "user":     re.search(r"-U\s+(\S+)", raw).group(1),
        "password": re.search(r"PGPASSWORD=(\S+)", raw).group(1),
        "dbname":   raw.split()[-1],
    }


def format_author(name: str, birth_year) -> str:
    if birth_year and str(birth_year).strip() and not (
        isinstance(birth_year, float) and birth_year != birth_year  # NaN check
    ):
        return f"{name} (b. {int(birth_year)})"
    return name


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["edition_count", "birth_year"], ascending=[False, True], na_position="last")
    result = pd.DataFrame({
        "Author":      df.apply(lambda r: format_author(r["author_name"], r["birth_year"]), axis=1),
        "Anthologies": df["edition_count"],
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
    print(f"\n{len(table)} authors")

    if args.save_csv:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_CSV, index=False)
        print(f"Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
