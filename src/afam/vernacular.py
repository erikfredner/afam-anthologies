"""Parser for the editor-designated vernacular page ranges.

The corpus editors mark certain page ranges as *vernacular* material
(spirituals, blues, work songs, folktales, sermons, rap lyrics, ...). That
grouping lives nowhere in the database as a single variable — only in the
hand-maintained, messy ``data/vernacular_pages.csv``.

Each CSV row maps an anthology to one or two **volume IDs**
(``data_volume.id``; two-volume editions list both, first = v1, second = v2)
and a free-text ``Vernacular Pages`` field. The field's quirks:

  * ``n/a`` (whole field, or a ``vN n/a`` segment) means no vernacular pages.
  * Ranges are separated by commas and/or newlines: ``455-458, 463-466`` or
    ``28-69\\n235-245\\n...``.
  * Two-volume editions prefix ranges with ``v1 ``/``v2 ``; a prefix-less line
    inherits the most recent prefix (e.g. ``v1 100-109\\n291-304\\nv2 n/a``
    puts both ranges on v1 and leaves v2 empty).

``parse_vernacular_row`` resolves one CSV row into ``(volume_id, slot, start,
end)`` records; ``load_vernacular_ranges`` applies it across the file.
"""

from __future__ import annotations

import re

import pandas as pd

from .data import load_csv

# A leading "v1 " / "v2 " volume prefix on a pages segment.
_SLOT_PREFIX = re.compile(r"^v(\d+)\s*", re.IGNORECASE)
# A "start-end" page range (en dashes are normalized to hyphens first).
_RANGE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


def parse_vernacular_row(db_id: str, pages: str) -> list[dict]:
    """Resolve one CSV row into vernacular page-range records.

    ``db_id`` is the raw ``Db ID`` cell (one volume ID, or two comma-separated
    IDs for a two-volume edition). ``pages`` is the raw ``Vernacular Pages``
    cell. Returns a list of ``{"volume_id", "slot", "page_start", "page_end"}``
    dicts — empty when the row carries no vernacular pages.

    ``slot`` is 1 for ``v1`` ranges, 2 for ``v2`` ranges, and ``None`` for a
    single-volume row with no ``vN`` prefix. Raises ``ValueError`` on any
    segment that is neither ``n/a`` nor a ``start-end`` range.
    """
    volume_ids = [int(tok) for tok in db_id.split(",") if tok.strip()]
    if not volume_ids:
        return []

    field = (pages or "").strip()
    if not field or field.lower() == "n/a":
        return []

    records: list[dict] = []
    current_slot: int | None = None
    # Split on newlines and commas; both appear as range separators.
    for raw in re.split(r"[\n,]", field):
        segment = raw.strip().replace("–", "-")  # normalize en dash
        if not segment:
            continue

        prefix = _SLOT_PREFIX.match(segment)
        if prefix:
            current_slot = int(prefix.group(1))
            segment = segment[prefix.end() :].strip()

        if not segment or segment.lower() == "n/a":
            continue

        match = _RANGE.match(segment)
        if not match:
            raise ValueError(
                f"Unparseable vernacular segment {segment!r} "
                f"(Db ID {db_id!r}, pages {pages!r})"
            )

        if current_slot is None:
            volume_id = volume_ids[0]
        elif current_slot <= len(volume_ids):
            volume_id = volume_ids[current_slot - 1]
        else:
            raise ValueError(
                f"Volume slot v{current_slot} has no matching Db ID in {db_id!r}"
            )

        records.append(
            {
                "volume_id": volume_id,
                "slot": current_slot,
                "page_start": int(match.group(1)),
                "page_end": int(match.group(2)),
            }
        )

    return records


def load_vernacular_ranges(csv_name: str = "vernacular_pages.csv") -> pd.DataFrame:
    """Load the vernacular page ranges as a tidy DataFrame.

    Columns: ``edition_title``, ``volume_id``, ``slot`` (1/2/None),
    ``page_start``, ``page_end`` — one row per range. Anthologies marked
    ``n/a`` contribute no rows.
    """
    raw = load_csv(csv_name)
    raw.columns = [c.strip() for c in raw.columns]

    rows: list[dict] = []
    for _, r in raw.iterrows():
        for rec in parse_vernacular_row(r["Db ID"], r["Vernacular Pages"]):
            rows.append({"edition_title": r["Title"].strip(), **rec})

    return pd.DataFrame(
        rows, columns=["edition_title", "volume_id", "slot", "page_start", "page_end"]
    )
