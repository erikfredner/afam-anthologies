from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "concentration"))

from poet_signature_concentration import build_signature_concentration  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────────────


def make_raw(rows):
    """Minimal author-page-share-reselection-shaped frame.

    Columns mirror queries/author-page-share-reselection.sql; page columns are
    filled so author_form_concentration.compute can build spans without error.
    """
    cols = [
        "work_id",
        "work_title",
        "parent_id",
        "volume_id",
        "edition_id",
        "edition_year",
        "author_id",
        "author_name",
        "form_id",
        "form_name",
        "toc_page",
        "toc_next",
        "first_toc_page",
        "last_toc_page",
    ]
    return pd.DataFrame(rows, columns=cols)


def _poem_row(work_id, title, author_id, author_name, edition_id):
    """One single-page poem selection in its own single-work volume."""
    return [
        work_id,
        title,
        None,
        9000 + work_id * 10 + edition_id,
        edition_id,
        1990 + edition_id,
        author_id,
        author_name,
        7,
        "poetry",
        10.0,
        11.0,
        10.0,
        11.0,
    ]


def poetry_rows():
    """Two poets with contrasting concentration.

    Walker (concentrated): "For My People" in editions 1-4, plus a second poem in
    edition 1 only -- 5 total selections, 2 distinct poems, signature poem covers
    4/4 editions.
    Hughes (dispersed): a different poem in each of 4 editions -- 4 selections, 4
    distinct poems, top poem covers 1/4 editions.
    """
    rows = []
    for ed in range(1, 5):
        rows.append(_poem_row(100, "For My People", 1, "Margaret Walker", ed))
    rows.append(_poem_row(101, "Lineage", 1, "Margaret Walker", 1))
    for ed in range(1, 5):
        rows.append(_poem_row(200 + ed, f"Hughes Poem {ed}", 2, "Langston Hughes", ed))
    return rows


@pytest.fixture
def built():
    import author_form_concentration as afc

    raw = make_raw(poetry_rows())
    outputs = afc.compute(raw)
    table = build_signature_concentration(outputs, raw, min_editions=2)
    return table.set_index("author_name")


# ── total_selections is merged back from the concentration table ─────────────────


def test_total_selections_counts_every_work_edition(built):
    walker = built.loc["Margaret Walker"]
    # 4 editions of "For My People" + 1 of "Lineage" = 5 selections across 4 editions.
    assert walker["total_selections"] == 5
    assert walker["poetry_editions"] == 4
    assert walker["distinct_poems"] == 2


def test_total_selections_at_least_poetry_editions(built):
    assert (built["total_selections"] >= built["poetry_editions"]).all()


# ── signature concentration metrics ──────────────────────────────────────────────


def test_walker_signature_dominates(built):
    walker = built.loc["Margaret Walker"]
    assert walker["top_poem"] == "For My People"
    assert walker["top_poem_editions"] == 4
    assert walker["top_poem_coverage"] == pytest.approx(1.0)
    # 4 of the poet's 5 poetry selections are the one signature poem.
    assert walker["signature_work_share_count"] == pytest.approx(0.8)


def test_hughes_is_fully_dispersed(built):
    hughes = built.loc["Langston Hughes"]
    assert hughes["distinct_poems"] == 4
    assert hughes["top_poem_coverage"] == pytest.approx(0.25)


# ── ranking: concentrated poet sorts above the dispersed one ─────────────────────


def test_concentrated_poet_ranks_first(built):
    assert built.index[0] == "Margaret Walker"


def test_min_editions_filter_drops_low_volume_poets():
    import author_form_concentration as afc

    raw = make_raw(poetry_rows())
    outputs = afc.compute(raw)
    table = build_signature_concentration(outputs, raw, min_editions=5)
    assert table.empty
