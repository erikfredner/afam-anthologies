"""Unit tests for the docs/ bibliography generators in scripts/.

The pandoc subprocess is never invoked here: what is worth testing is the key
bookkeeping around it — which entries end up in which section, and whether a
malformed or missing record is caught rather than published.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from bibliography import (
    assert_all_rendered,
    bib_keys,
    nocite_metadata,
    retag_bib_div,
)
from build_comprehensive_bibliography import split_keys

BIB = """\
@book{alpha1929,
  title = {A},
  year = 1929
}

@incollection{beta1941,
  title = {B},
  year = 1941
}

@book{gamma1968,
  title = {C},
  year = 1968
}
"""


@pytest.fixture
def all_bib(tmp_path: Path) -> Path:
    path = tmp_path / "all.bib"
    path.write_text(BIB, encoding="utf-8")
    return path


# ── key parsing ─────────────────────────────────────────────────────────


def test_bib_keys_are_returned_in_file_order(all_bib: Path):
    """Order matters: it is what keeps the two sections' A–Z runs comparable."""
    assert bib_keys(all_bib) == ["alpha1929", "beta1941", "gamma1968"]


def test_bib_keys_reads_every_entry_type(all_bib: Path):
    """@incollection counts too — the reader keys off @\\w+, not @book."""
    assert "beta1941" in bib_keys(all_bib)


def test_bib_keys_ignores_at_signs_inside_fields(tmp_path: Path):
    """An "@" in a field value is not a record header."""
    path = tmp_path / "x.bib"
    path.write_text(
        "@book{k1,\n  title = {Mail to @nobody},\n  note = {see @k2}\n}\n",
        encoding="utf-8",
    )
    assert bib_keys(path) == ["k1"]


# ── nocite metadata ─────────────────────────────────────────────────────


def test_nocite_defaults_to_every_entry():
    assert nocite_metadata() == '---\nnocite: "@*"\n---\n'


def test_nocite_lists_requested_keys():
    """An explicit list is how one .bib yields two separate bibliographies."""
    assert nocite_metadata(["a1", "b2"]) == '---\nnocite: "@a1, @b2"\n---\n'


def test_nocite_of_an_empty_list_selects_nothing():
    """Not "@*": an empty section must stay empty rather than render the lot."""
    assert nocite_metadata([]) == '---\nnocite: ""\n---\n'


# ── section split ───────────────────────────────────────────────────────


def test_split_keys_partitions_the_full_list(tmp_path: Path, all_bib: Path):
    selected = tmp_path / "selected.bib"
    selected.write_text(BIB.split("@incollection")[0], encoding="utf-8")
    in_analysis, considered = split_keys(all_bib, selected)
    assert in_analysis == ["alpha1929"]
    assert considered == ["beta1941", "gamma1968"]


def test_split_keys_preserves_file_order_within_each_section(
    tmp_path: Path, all_bib: Path
):
    selected = tmp_path / "selected.bib"
    selected.write_text(
        "@book{gamma1968,\n  year = 1968\n}\n\n@book{alpha1929,\n  year = 1929\n}\n",
        encoding="utf-8",
    )
    in_analysis, considered = split_keys(all_bib, selected)
    # Ordered by the full bibliography, not by the selected file.
    assert in_analysis == ["alpha1929", "gamma1968"]
    assert considered == ["beta1941"]


def test_split_keys_rejects_a_selected_key_missing_from_the_full_list(
    tmp_path: Path, all_bib: Path
):
    """Such a key would be silently dropped: every entry renders from all_bib."""
    selected = tmp_path / "selected.bib"
    selected.write_text("@book{delta1993,\n  year = 1993\n}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        split_keys(all_bib, selected)
    assert "delta1993" in str(excinfo.value)


# ── rendered-output checks ──────────────────────────────────────────────


def test_assert_all_rendered_counts_entries():
    fragment = '<div class="csl-entry">A</div><div class="csl-entry">B</div>'
    assert assert_all_rendered(fragment, 2, "test") == 2


def test_assert_all_rendered_exits_when_citeproc_drops_one():
    with pytest.raises(SystemExit):
        assert_all_rendered('<div class="csl-entry">A</div>', 2, "test")


def test_retag_bib_div_renames_only_the_wrapper():
    """Two bibliographies on one page cannot both be id="refs"."""
    fragment = '<div id="refs" class="csl-bib-body"><div id="ref-a1">x</div></div>'
    retagged = retag_bib_div(fragment, "refs-selected")
    assert 'id="refs-selected"' in retagged
    assert 'id="refs"' not in retagged
    assert 'id="ref-a1"' in retagged
