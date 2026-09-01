"""Shared pandoc/citeproc plumbing for the bibliography pages in docs/.

`build_binder_bibliography.py` and `build_comprehensive_bibliography.py` both
turn a BibTeX file into a Chicago notes-bibliography list wrapped in the site's
house style. Everything they have in common lives here: the citation-key
reader, the pandoc invocation, and the stylesheet the two pages share.

Requires a `pandoc` binary on PATH (citeproc is built into pandoc 3.x).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

# `@type{key,` at the start of a line — the record header of a BibTeX entry.
_ENTRY_RE = re.compile(r"^@\w+\s*\{([^,\s]+)\s*,", re.MULTILINE)

# Styling mirrors docs/index.html so the pages read as one site; the
# hanging-indent rule is the one addition, and normally comes from pandoc's
# standalone template, which we do not use.
PAGE_CSS = """  :root { color-scheme: light dark; }
  body {
    max-width: 42rem;
    margin: 0 auto;
    padding: 3rem 1.25rem 4rem;
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
  }
  h1 { font-size: 1.6rem; line-height: 1.25; margin: 0 0 0.75rem; }
  h2 { font-size: 1.1rem; margin: 2.5rem 0 0.5rem; }
  p { margin: 0 0 1rem; }
  .lede { color: #555; }
  a { color: #2a78d6; }
  .note { font-size: 0.9rem; color: #555; }
  .hanging-indent div.csl-entry { margin-left: 2em; text-indent: -2em; }
  div.csl-entry { margin-bottom: 0.9rem; }
  @media (prefers-color-scheme: dark) {
    .lede, .note { color: #aaa; }
    a { color: #6ea8e8; }
  }"""


def bib_keys(bib: Path) -> list[str]:
    """Citation keys in a BibTeX file, in the order they appear."""
    return _ENTRY_RE.findall(bib.read_text(encoding="utf-8"))


def count_bib_entries(bib: Path) -> int:
    """Count `@type{key,` records in a BibTeX file."""
    return len(bib_keys(bib))


def count_entries(fragment: str) -> int:
    """Count rendered entries in a citeproc HTML fragment."""
    return fragment.count('class="csl-entry"')


def nocite_metadata(keys: Iterable[str] | None = None) -> str:
    """The pandoc stdin document: a YAML block and no body.

    `nocite` tells citeproc which entries to put in the bibliography without any
    in-text citation, which is the whole document for these pages. `None` means
    every entry in the bibliography file; an explicit key list is how one .bib
    yields two separate bibliographies without writing temporary files.
    """
    selector = "@*" if keys is None else ", ".join(f"@{key}" for key in keys)
    return f'---\nnocite: "{selector}"\n---\n'


def render_bibliography(bib: Path, csl: Path, keys: Iterable[str] | None = None) -> str:
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
        input=nocite_metadata(keys),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"pandoc failed (exit {proc.returncode}):\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def retag_bib_div(fragment: str, div_id: str) -> str:
    """Rename citeproc's `id="refs"` wrapper, so two fragments can share a page.

    Every citeproc run labels its bibliography `refs`; a page holding more than
    one bibliography would otherwise repeat that id.
    """
    return fragment.replace('id="refs"', f'id="{div_id}"', 1)


def assert_all_rendered(fragment: str, expected: int, source: str) -> int:
    """Exit unless citeproc rendered every entry it was asked for.

    A malformed record is dropped silently, which would quietly shorten a
    published bibliography, so the count is checked rather than trusted.
    """
    rendered = count_entries(fragment)
    if rendered != expected:
        sys.exit(
            f"citeproc rendered {rendered} entries but {source} holds {expected}; "
            "an entry was dropped (check pandoc's warnings for malformed records)."
        )
    return rendered


def require_files(*paths: tuple[Path, str]) -> None:
    """Exit naming the first missing input, so a typo'd flag is legible."""
    for path, what in paths:
        if not path.exists():
            sys.exit(f"{what} file not found: {path}")
