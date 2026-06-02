"""Reusable argparse flags shared across scripts."""

from __future__ import annotations

import argparse


def add_root_works_flag(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive --only-root-works / --include-excerpts flags.

    DB-backed scripts default to root-only; pass --include-excerpts to add excerpts.
    The dest is `only_root_works` (bool). Default is True.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only-root-works",
        dest="only_root_works",
        action="store_true",
        default=True,
        help="exclude excerpts (default)",
    )
    group.add_argument(
        "--include-excerpts",
        dest="only_root_works",
        action="store_false",
        help="include excerpt works alongside root works",
    )


def add_save_csv_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="write outputs to data/ instead of stdout",
    )
