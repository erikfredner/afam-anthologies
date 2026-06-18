"""Unit tests for the messy vernacular-page-range parser."""

from __future__ import annotations

import pytest

from afam.vernacular import load_vernacular_ranges, parse_vernacular_row


# ── parse_vernacular_row ────────────────────────────────────────────────────


def test_single_volume_comma_ranges():
    """Comma-separated ranges on a single volume → one record each, slot None."""
    recs = parse_vernacular_row("86", "455-458, 463-466")
    assert recs == [
        {"volume_id": 86, "slot": None, "page_start": 455, "page_end": 458},
        {"volume_id": 86, "slot": None, "page_start": 463, "page_end": 466},
    ]


def test_single_volume_newline_ranges():
    """Newlines separate ranges just like commas."""
    recs = parse_vernacular_row("64", "11-28\n81-91")
    assert [(r["page_start"], r["page_end"]) for r in recs] == [(11, 28), (81, 91)]
    assert all(r["volume_id"] == 64 and r["slot"] is None for r in recs)


def test_two_volume_v1_v2_prefixes():
    """v1/v2 prefixes map ranges to the first/second listed volume ID."""
    recs = parse_vernacular_row("138, 139", "v1 9-71\nv2 11-91")
    assert recs == [
        {"volume_id": 138, "slot": 1, "page_start": 9, "page_end": 71},
        {"volume_id": 139, "slot": 2, "page_start": 11, "page_end": 91},
    ]


def test_prefix_inheritance_and_vn_na():
    """A prefix-less line inherits the prior slot; `v2 n/a` yields no range."""
    recs = parse_vernacular_row("52, 53", "v1 100-109\n291-304\nv2 n/a")
    assert recs == [
        {"volume_id": 52, "slot": 1, "page_start": 100, "page_end": 109},
        {"volume_id": 52, "slot": 1, "page_start": 291, "page_end": 304},
    ]


def test_whole_field_na():
    assert parse_vernacular_row("42", "n/a") == []
    assert parse_vernacular_row("42", "") == []


def test_deconcatenated_ranges():
    """The Prentice Hall fix `112-127, 154-155` parses as two ranges."""
    recs = parse_vernacular_row("39", "84-85, 112-127, 154-155")
    assert [(r["page_start"], r["page_end"]) for r in recs] == [
        (84, 85),
        (112, 127),
        (154, 155),
    ]


def test_corrected_single_ranges():
    """The de-corrupted vol 63 / vol 85 cells parse as plain ranges."""
    assert parse_vernacular_row("63", "1-17") == [
        {"volume_id": 63, "slot": None, "page_start": 1, "page_end": 17}
    ]
    assert parse_vernacular_row("85", "2-31") == [
        {"volume_id": 85, "slot": None, "page_start": 2, "page_end": 31}
    ]


def test_en_dash_normalized():
    recs = parse_vernacular_row("36", "10–151")  # en dash
    assert recs == [{"volume_id": 36, "slot": None, "page_start": 10, "page_end": 151}]


def test_unparseable_segment_raises():
    """A bare 5-digit serial (reintroduced corruption) is rejected."""
    with pytest.raises(ValueError):
        parse_vernacular_row("63", "47880")


def test_slot_without_matching_id_raises():
    with pytest.raises(ValueError):
        parse_vernacular_row("36", "v2 10-20")


# ── load_vernacular_ranges ──────────────────────────────────────────────────


def test_load_vernacular_ranges_shape():
    df = load_vernacular_ranges()
    assert list(df.columns) == [
        "edition_title",
        "volume_id",
        "slot",
        "page_start",
        "page_end",
    ]
    # v1-only two-volume edition: volume 52 present, its empty v2 (53) absent.
    assert 52 in set(df["volume_id"])
    assert 53 not in set(df["volume_id"])
    # No range may start above its end.
    assert (df["page_end"] >= df["page_start"]).all()
