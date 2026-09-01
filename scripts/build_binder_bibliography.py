"""Render the 1986 NAAAL binder .bib as a static page for GitHub Pages.

`data/1986_naaal_binder_tocs.bib` lists the books whose tables of contents Henry
Louis Gates Jr. collected into binders for the editors of *The Norton Anthology of
African American Literature* at their 1986 meeting. This script formats every entry
as a Chicago notes-bibliography list and writes it to docs/, which is tracked and
served by GitHub Pages.

Requires a `pandoc` binary on PATH (citeproc is built into pandoc 3.x).

    uv run python scripts/build_binder_bibliography.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bibliography import (
    PAGE_CSS,
    assert_all_rendered,
    count_bib_entries,
    render_bibliography,
    require_files,
)

from afam import DATA_DIR, DOCS_DIR, REPO_ROOT

DEFAULT_BIB = DATA_DIR / "1986_naaal_binder_tocs.bib"
DEFAULT_CSL = REPO_ROOT / "chicago-notes-bibliography.csl"
DEFAULT_OUT = DOCS_DIR / "1986_binder_bibliography.html"

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The 1986 NAAAL binders — works consulted</title>
<style>
{css}
</style>
</head>
<body>

<h1>The 1986 NAAAL binders</h1>

<p class="lede">
  The {count} anthologies and collections whose tables of contents Henry Louis
  Gates Jr. gave the editors of <i>The Norton Anthology of African American
  Literature</i> at their first meeting, in 1986. Chicago style; each entry links
  to its Library of Congress record.
</p>

<h2>Works in the binders</h2>

{bibliography}

<p class="note">
  <a href="index.html">All figures and tables</a> &middot;
  Source and analysis code:
  <a href="https://github.com/erikfredner/afam-anthologies">github.com/erikfredner/afam-anthologies</a>
</p>

</body>
</html>
"""


def build(bib: Path, csl: Path, out: Path) -> int:
    """Write the page and return the number of entries rendered."""
    require_files((bib, "bibliography"), (csl, "CSL style"))

    fragment = render_bibliography(bib, csl)
    rendered = assert_all_rendered(fragment, count_bib_entries(bib), bib.name)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        PAGE_TEMPLATE.format(css=PAGE_CSS, count=rendered, bibliography=fragment),
        encoding="utf-8",
    )
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bib",
        type=Path,
        default=DEFAULT_BIB,
        help="BibTeX input (default: %(default)s)",
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

    count = build(args.bib, args.csl, args.out)
    print(f"wrote {args.out} ({count} entries)")


if __name__ == "__main__":
    main()
