"""Name formatting and sorting helpers."""

from __future__ import annotations

SURNAME_PARTICLES = {"da", "de", "del", "der", "di", "du", "la", "le", "van", "von"}
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def author_last_name(full_name: object) -> str:
    parts = str(full_name).strip().split()
    while parts and parts[-1].rstrip(".").lower() in NAME_SUFFIXES:
        parts.pop()
    if len(parts) >= 2 and parts[-2].lower() in SURNAME_PARTICLES:
        return " ".join(parts[-2:])
    return parts[-1] if parts else ""


def author_sort_key(full_name: object) -> tuple[str, str]:
    name = str(full_name).strip()
    return (author_last_name(name).casefold(), name.casefold())
