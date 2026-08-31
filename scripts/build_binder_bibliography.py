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
import re
import shutil
import subprocess
import sys
from pathlib import Path

from afam import DATA_DIR, DOCS_DIR, REPO_ROOT

DEFAULT_BIB = DATA_DIR / "1986_naaal_binder_tocs.bib"
DEFAULT_CSL = REPO_ROOT / "chicago-notes-bibliography.csl"
DEFAULT_OUT = DOCS_DIR / "1986_binder_bibliography.html"

# `nocite: "@*"` tells citeproc to put every entry in the bibliography without any
# in-text citation, which is the whole document here.
PANDOC_STDIN = '---\nnocite: "@*"\n---\n'

# Styling mirrors docs/index.html so the two pages read as one site; the
# hanging-indent rule is the one addition, and normally comes from pandoc's
# standalone template, which we do not use.
PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The 1986 NAAAL binders — works consulted</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    max-width: 42rem;
    margin: 0 auto;
    padding: 3rem 1.25rem 4rem;
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
  }}
  h1 {{ font-size: 1.6rem; line-height: 1.25; margin: 0 0 0.75rem; }}
  h2 {{ font-size: 1.1rem; margin: 2.5rem 0 0.5rem; }}
  p {{ margin: 0 0 1rem; }}
  .lede {{ color: #555; }}
  a {{ color: #2a78d6; }}
  .note {{ font-size: 0.9rem; color: #555; }}
  .hanging-indent div.csl-entry {{ margin-left: 2em; text-indent: -2em; }}
  div.csl-entry {{ margin-bottom: 0.9rem; }}
  @media (prefers-color-scheme: dark) {{
    .lede, .note {{ color: #aaa; }}
    a {{ color: #6ea8e8; }}
  }}
</style>
</head>
<body>

<h1>The 1986 NAAAL binders</h1>

<p class="lede">
  In 1986, Henry Louis Gates Jr. convened the editors of <i>The Norton Anthology of
  African American Literature</i> for their first meeting and gave each of them a
  binder of photocopied tables of contents from earlier anthologies and collections
  of African American writing. The {count} books listed below are the works those
  tables of contents came from.
</p>

<p class="lede">
  Entries are alphabetical, in Chicago notes-bibliography style. Each links to its
  record in the Library of Congress catalog.
</p>

<h2>Works in the binders</h2>

{bibliography}

<p class="note">
  Source and analysis code:
  <a href="https://github.com/erikfredner/afam-anthologies">github.com/erikfredner/afam-anthologies</a>
</p>

</body>
</html>
"""


def count_bib_entries(bib: Path) -> int:
    """Count `@type{key,` records in a BibTeX file."""
    return len(re.findall(r"^@\w+\s*\{", bib.read_text(encoding="utf-8"), re.MULTILINE))


def render_bibliography(bib: Path, csl: Path) -> str:
    """Return the citeproc-rendered bibliography as an HTML fragment."""
    if shutil.which("pandoc") is None:
        sys.exit(
            "pandoc not found on PATH. Install it (e.g. `brew install pandoc`) and "
            "re-run; pandoc 3.x includes citeproc."
        )
    proc = subprocess.run(
        [
            "pandoc",
            "--from=markdown",
            "--to=html",
            "--citeproc",
            f"--csl={csl}",
            f"--bibliography={bib}",
        ],
        input=PANDOC_STDIN,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"pandoc failed (exit {proc.returncode}):\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def build(bib: Path, csl: Path, out: Path) -> int:
    """Write the page and return the number of entries rendered."""
    for path, what in ((bib, "bibliography"), (csl, "CSL style")):
        if not path.exists():
            sys.exit(f"{what} file not found: {path}")

    fragment = render_bibliography(bib, csl)
    rendered = fragment.count('class="csl-entry"')
    expected = count_bib_entries(bib)
    if rendered != expected:
        sys.exit(
            f"citeproc rendered {rendered} entries but {bib.name} holds {expected}; "
            "an entry was dropped (check pandoc's warnings for malformed records)."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        PAGE_TEMPLATE.format(count=rendered, bibliography=fragment), encoding="utf-8"
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
