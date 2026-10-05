"""Render the 1986 NAAAL binder bibliography as a static page for GitHub Pages.

`data/1986_naaal_binders.md` lists the books whose tables of contents Henry Louis
Gates Jr. collected into binders for the editors of *The Norton Anthology of African
American Literature* at their 1986 meeting. The list is hand-corrected Chicago
notes-bibliography Markdown, one entry per paragraph, in display order. It is the
source of record rather than citeproc output: it uses forms citeproc cannot produce
from BibTeX, such as an original date in brackets after a reprint's (`1970 [1849]`)
and an earlier name in brackets after the author's (`Baraka, Amiri [LeRoi Jones]`).
This script wraps every entry in the same markup the citeproc-rendered bibliography
pages use and writes the page to docs/, which is tracked and served by GitHub Pages.

Requires a `pandoc` binary on PATH.

    uv run python scripts/build_binder_bibliography.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from bibliography import PAGE_CSS, assert_all_rendered, require_files

from afam import DATA_DIR, DOCS_DIR

DEFAULT_SOURCE = DATA_DIR / "1986_naaal_binders.md"
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
  Henry Louis Gates Jr. gave the editors of <i>The Norton Anthology of
  African American Literature</i> these {count} anthologies and collections
  at their first meeting, in 1986. Citations follow Chicago style, and entries
  link to their Library of Congress records where one is given.
</p>

<h2>Works in the binders</h2>

{bibliography}

<p class="note">
  <a href="index.html">All figures and tables</a>
</p>

</body>
</html>
"""


def read_entries(source: Path) -> list[str]:
    """The bibliography entries in `source`: one per non-blank line."""
    lines = source.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def entries_markdown(entries: list[str]) -> str:
    """Wrap each entry in a pandoc fenced div matching citeproc's markup.

    The outer div carries citeproc's `refs` id and hanging-indent class, so the
    shared stylesheet formats this page exactly like the citeproc-rendered one.
    """
    body = "\n\n".join(f"::: {{.csl-entry role=listitem}}\n{e}\n:::" for e in entries)
    return (
        ":::: {#refs .references .csl-bib-body .hanging-indent role=list}\n\n"
        f"{body}\n\n::::\n"
    )


def render_entries(entries: list[str]) -> str:
    """Return the entries as an HTML fragment."""
    if shutil.which("pandoc") is None:
        sys.exit(
            "pandoc not found on PATH. Install it (e.g. `brew install pandoc`) and "
            "re-run."
        )
    proc = subprocess.run(
        ["pandoc", "--from=markdown", "--to=html", "--wrap=preserve"],
        input=entries_markdown(entries),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"pandoc failed (exit {proc.returncode}):\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def build(source: Path, out: Path) -> int:
    """Write the page and return the number of entries rendered."""
    require_files((source, "bibliography"))

    entries = read_entries(source)
    fragment = render_entries(entries)
    rendered = assert_all_rendered(fragment, len(entries), source.name)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        PAGE_TEMPLATE.format(css=PAGE_CSS, count=rendered, bibliography=fragment),
        encoding="utf-8",
    )
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Markdown bibliography, one entry per line (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="HTML output (default: %(default)s)",
    )
    args = parser.parse_args()

    count = build(args.source, args.out)
    print(f"wrote {args.out} ({count} entries)")


if __name__ == "__main__":
    main()
