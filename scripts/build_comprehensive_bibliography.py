"""Render the comprehensive-anthology .bib files as a static page for GitHub Pages.

`data/comprehensive_all.bib` lists every anthology of African American literature
that was eligible for consideration in this project; `data/comprehensive_selected.bib`
lists the subset whose contents were indexed and analyzed. The selected keys are a
strict subset of the full list, and the shared records are identical between the two
files, so the page is rendered entirely from `comprehensive_all.bib` — the selected
file is read only for its citation keys, which split the bibliography into two
sections: the anthologies in the analysis, and those considered but not included.

Requires a `pandoc` binary on PATH (citeproc is built into pandoc 3.x).

    uv run python scripts/build_comprehensive_bibliography.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bibliography import (
    PAGE_CSS,
    assert_all_rendered,
    bib_keys,
    render_bibliography,
    require_files,
    retag_bib_div,
)

from afam import DATA_DIR, DOCS_DIR, REPO_ROOT

DEFAULT_ALL_BIB = DATA_DIR / "comprehensive_all.bib"
DEFAULT_SELECTED_BIB = DATA_DIR / "comprehensive_selected.bib"
DEFAULT_CSL = REPO_ROOT / "chicago-notes-bibliography.csl"
DEFAULT_OUT = DOCS_DIR / "comprehensive_bibliography.html"

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anthologies considered</title>
<style>
{css}
</style>
</head>
<body>

<h1>Anthologies considered</h1>

<p class="lede">
  The {total} anthologies of African American literature that were eligible for
  this project, and the {n_selected} of them whose contents were indexed and
  analyzed. Chicago style.
</p>

<h2>In the analysis ({n_selected})</h2>

<p>
  The anthologies behind every
  <a href="index.html">figure and table</a> on this site.
</p>

{selected_bibliography}

<h2>Considered, not included ({n_other})</h2>

<p>
  These titles were eligible for consideration but are not part of the analysis.
</p>

{other_bibliography}

<p class="note">
  <a href="index.html">All figures and tables</a>
</p>

</body>
</html>
"""


def split_keys(all_bib: Path, selected_bib: Path) -> tuple[list[str], list[str]]:
    """Partition the full key list into selected and set-aside, in file order.

    The selected file must be a subset of the full one: a key that appears only
    in the selected file would silently vanish from the page, since every entry
    is rendered from `all_bib`.
    """
    all_keys = bib_keys(all_bib)
    selected = set(bib_keys(selected_bib))
    stray = sorted(selected - set(all_keys))
    if stray:
        sys.exit(
            f"{selected_bib.name} holds {len(stray)} key(s) missing from "
            f"{all_bib.name}: {', '.join(stray)}"
        )
    return (
        [key for key in all_keys if key in selected],
        [key for key in all_keys if key not in selected],
    )


def build(all_bib: Path, selected_bib: Path, csl: Path, out: Path) -> tuple[int, int]:
    """Write the page and return the (selected, set-aside) entry counts."""
    require_files(
        (all_bib, "bibliography"),
        (selected_bib, "selected bibliography"),
        (csl, "CSL style"),
    )

    selected_keys, other_keys = split_keys(all_bib, selected_bib)
    sections = []
    for keys, what in ((selected_keys, "selected"), (other_keys, "considered")):
        fragment = render_bibliography(all_bib, csl, keys)
        assert_all_rendered(fragment, len(keys), f"the {what} entry list")
        sections.append(retag_bib_div(fragment, f"refs-{what}"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        PAGE_TEMPLATE.format(
            css=PAGE_CSS,
            total=len(selected_keys) + len(other_keys),
            n_selected=len(selected_keys),
            n_other=len(other_keys),
            selected_bibliography=sections[0],
            other_bibliography=sections[1],
        ),
        encoding="utf-8",
    )
    return len(selected_keys), len(other_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all-bib",
        type=Path,
        default=DEFAULT_ALL_BIB,
        help="BibTeX input holding every eligible anthology (default: %(default)s)",
    )
    parser.add_argument(
        "--selected-bib",
        type=Path,
        default=DEFAULT_SELECTED_BIB,
        help="BibTeX input naming the analyzed subset (default: %(default)s)",
    )
    parser.add_argument(
        "--csl", type=Path, default=DEFAULT_CSL, help="CSL style (default: %(default)s)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="HTML output (default: %(default)s)",
    )
    args = parser.parse_args()

    n_selected, n_other = build(args.all_bib, args.selected_bib, args.csl, args.out)
    print(f"wrote {args.out} ({n_selected} analyzed, {n_other} considered)")


if __name__ == "__main__":
    main()
