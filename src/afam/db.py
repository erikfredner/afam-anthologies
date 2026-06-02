"""PostgreSQL connection helpers.

Parses the `DATABASE_URL=PGPASSWORD=… psql -h … -U … <dbname>` line from
`.env` and exposes thin wrappers for executing SQL against the
`anthologies` database.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import dotenv_values

from . import ENV_FILE


def parse_db_params(env_file: Path = ENV_FILE) -> dict[str, str]:
    env = dotenv_values(env_file)
    raw = env["DATABASE_URL"]
    return {
        "host": re.search(r"-h\s+(\S+)", raw).group(1),
        "user": re.search(r"-U\s+(\S+)", raw).group(1),
        "password": re.search(r"PGPASSWORD=(\S+)", raw).group(1),
        "dbname": raw.split()[-1],
    }


def connect(env_file: Path = ENV_FILE) -> psycopg.Connection:
    return psycopg.connect(**parse_db_params(env_file))


def query(
    sql: str | Path,
    params: dict[str, Any] | tuple | None = None,
    env_file: Path = ENV_FILE,
) -> pd.DataFrame:
    """Execute SQL (string or path to .sql file) and return a DataFrame."""
    if isinstance(sql, Path):
        sql = sql.read_text()
    with connect(env_file) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [desc.name for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
