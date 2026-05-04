"""
half_or_more_sentences.py
--------------------------
Prints two summary sentences with accurate counts of authors and works
that appear in at least half of all African American literary anthology editions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import dotenv_values

ENV_FILE = Path(__file__).parent.parent / ".env"
QUERIES  = Path(__file__).parent.parent / "queries"


def parse_db_params(env_file: Path) -> dict[str, str]:
    env = dotenv_values(env_file)
    raw = env["DATABASE_URL"]
    return {
        "host":     re.search(r"-h\s+(\S+)", raw).group(1),
        "user":     re.search(r"-U\s+(\S+)", raw).group(1),
        "password": re.search(r"PGPASSWORD=(\S+)", raw).group(1),
        "dbname":   raw.split()[-1],
    }


def query(params: dict[str, str], sql_file: Path) -> pd.DataFrame:
    sql = sql_file.read_text()
    with psycopg.connect(**params) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc.name for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def main() -> None:
    params = parse_db_params(ENV_FILE)

    authors_all  = query(params, QUERIES / "author-edition-counts-afam.sql")
    authors_half = query(params, QUERIES / "authors-in-half-or-more-afam-eds.sql")
    works_all    = query(params, QUERIES / "work-edition-counts-afam.sql")
    works_half   = query(params, QUERIES / "works-in-half-or-more-afam-eds.sql")

    total_authors = len(authors_all)
    half_authors  = len(authors_half)
    pct_authors   = round(half_authors / total_authors * 100)

    total_works = len(works_all)
    half_works  = works_half["work_id"].nunique()
    pct_works   = round(half_works / total_works * 100)

    print(
        f"Of the {total_authors:,} unique authors who appeared in any anthology, "
        f"{half_authors:,} ({pct_authors}%) appear in half or more of the anthologies."
    )
    print(
        f"Of the {total_works:,} unique works, "
        f"{half_works:,} ({pct_works}%) appear in half or more of the anthologies."
    )


if __name__ == "__main__":
    main()
