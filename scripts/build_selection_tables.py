"""Render the author and work selection records as static pages.

Two pages, built live from PostgreSQL and written into docs/, which is tracked
and served by GitHub Pages:

    docs/author_selection_records.html  all 575 authors ever selected for an
                                        African American anthology
    docs/work_selection_records.html    all 3,236 works, counting excerpts in
                                        their own right alongside root works

Both records are given as percentages: the share of the 26 anthologies that
selected the entity, and the share of the anthologies published after its debut
that selected it again.

Works with no author in the database keep an empty Author cell. That absence is
evidence -- 298 of the 3,236 works are unattributed -- and filling it with
"Anonymous" would assert an authorship claim the record does not make.

The output is plain hand-editable HTML -- no template engine, no CDN, and the
only script is ~30 lines of inline column sorting -- so prose and individual
rows can be corrected by hand afterwards. Re-running this script overwrites
those edits.

    uv run python scripts/build_selection_tables.py
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pandas as pd

from afam import DOCS_DIR
from afam.db import query
from afam.names import author_sort_key
from afam.sql import query_path

DEFAULT_AUTHORS_OUT = DOCS_DIR / "author_selection_records.html"
DEFAULT_WORKS_OUT = DOCS_DIR / "work_selection_records.html"

EN_DASH = "–"
EM_DASH = "—"

# Some DB titles carry a trailing angle-bracket qualifier that disambiguates two
# records sharing a title -- Locke's essay "The New Negro <essay>" against the
# anthology "The New Negro <edited collection>" it opens. The brackets are an
# editing convention, not part of the title, but the qualifier still has to
# survive to the page or the two rows become indistinguishable.
_QUALIFIER = re.compile(r"\s*<([^<>]+)>\s*$")

# Styling mirrors docs/index.html so the pages read as one site; the table,
# .qual and .wrap rules are the additions, and the body is wider than the 42rem
# set for prose because the works table carries five columns.
PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    max-width: 54rem;
    margin: 0 auto;
    padding: 3rem 1.25rem 4rem;
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
  }}
  h1 {{ font-size: 1.6rem; line-height: 1.25; margin: 0 0 0.75rem; }}
  h2 {{ font-size: 1.1rem; margin: 2.5rem 0 0.5rem; }}
  p {{ margin: 0 0 1rem; }}
  .lede {{ color: #555; max-width: 42rem; }}
  a {{ color: #2a78d6; }}
  .note {{ font-size: 0.9rem; color: #555; }}
  .wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.95rem; }}
  th, td {{
    text-align: left;
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid #e6e6e6;
    vertical-align: top;
  }}
  th {{ cursor: pointer; user-select: none; white-space: nowrap; }}
  th .arrow {{ color: #999; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  th.num {{ text-align: right; }}
  .qual {{ color: #777; font-size: 0.85em; }}
  @media (prefers-color-scheme: dark) {{
    .lede, .note {{ color: #aaa; }}
    a {{ color: #6ea8e8; }}
    th, td {{ border-bottom-color: #333; }}
    .qual {{ color: #999; }}
  }}
</style>
</head>
<body>

<h1>{heading}</h1>

{lede}

<h2>{table_heading}</h2>

<div class="wrap">
{table}
</div>

<p class="note">
  <a href="index.html">All figures and tables</a> &middot;
  Source and analysis code:
  <a href="https://github.com/erikfredner/afam-anthologies">github.com/erikfredner/afam-anthologies</a>
</p>

<script>
// Click a column heading to sort by it; click again to reverse. Cells carry a
// data-sort attribute when their display text does not sort correctly on its
// own ("22/25" should rank by rate, "1901-1967" by birth year).
document.querySelectorAll("table").forEach(function (table) {{
  var body = table.tBodies[0];
  table.querySelectorAll("th").forEach(function (th, column) {{
    th.addEventListener("click", function () {{
      var numeric = th.classList.contains("num");
      // First click sorts the way the column is most useful: counts high-to-low,
      // names A-to-Z. Clicking again reverses.
      var descending = th.dataset.direction
        ? th.dataset.direction !== "desc"
        : numeric;
      var rows = Array.prototype.slice.call(body.rows);
      rows.sort(function (a, b) {{
        var x = key(a.cells[column]);
        var y = key(b.cells[column]);
        if (numeric) {{ return descending ? y - x : x - y; }}
        return descending ? String(y).localeCompare(x) : String(x).localeCompare(y);
      }});
      table.querySelectorAll("th").forEach(function (other) {{
        delete other.dataset.direction;
        other.querySelector(".arrow").textContent = "";
      }});
      th.dataset.direction = descending ? "desc" : "asc";
      th.querySelector(".arrow").textContent = descending ? " ▼" : " ▲";
      rows.forEach(function (row) {{ body.appendChild(row); }});
    }});
  }});
  function key(cell) {{
    var raw = cell.dataset.sort !== undefined ? cell.dataset.sort : cell.textContent.trim();
    var number = parseFloat(raw);
    return isNaN(number) ? raw : number;
  }}
}});
</script>

</body>
</html>
"""


def split_qualifier(title: object) -> tuple[str, str]:
    """Split a trailing ``<qualifier>`` off a title: ("The New Negro", "essay")."""
    text = str(title).strip()
    match = _QUALIFIER.search(text)
    if not match:
        return text, ""
    return text[: match.start()].strip(), match.group(1).strip()


def format_title(title: object) -> str:
    """Render a title as escaped HTML, with any qualifier as a muted parenthetical."""
    text, qualifier = split_qualifier(title)
    rendered = html.escape(text)
    if qualifier:
        rendered += f' <span class="qual">({html.escape(qualifier)})</span>'
    return rendered


def format_life_dates(birth: object, death: object) -> str:
    """Render an author's dates: "1901-1967", "b. 1944", or an em dash."""
    has_birth = pd.notna(birth)
    has_death = pd.notna(death)
    if has_birth and has_death:
        return f"{int(birth)}{EN_DASH}{int(death)}"
    if has_birth:
        return f"b. {int(birth)}"
    if has_death:
        return f"d. {int(death)}"
    return EM_DASH


def format_percent(part: object, whole: object) -> str:
    """Render a share as a whole percent, or an em dash when there is no denominator."""
    if not whole or pd.isna(whole) or pd.isna(part):
        return EM_DASH
    return f"{round(100 * int(part) / int(whole))}%"


def join_author_names(names: pd.Series) -> str:
    """
    Join an work's authors, or return "" when the database records none.

    Deliberately unlike works_in_half_or_more_afam_eds.join_authors, which
    credits an unattributed work to "Anonymous". 298 of the 3,236 selected works
    have no author on record, and that silence is itself evidence about how the
    anthologies handle unattributed material -- naming an author for them would
    assert a claim the record does not make.
    """
    unique = list(dict.fromkeys(n for n in names if pd.notna(n)))
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return ", ".join(unique[:-1]) + f", and {unique[-1]}"


def _rate_sort(row: pd.Series) -> str:
    """
    Sort key for the reselection column: the exact rate, so rows that round to
    the same percent still order correctly.

    An entity that debuted in the most recent anthology has had no chance to be
    reselected -- 20 authors and 70 works. Those sort as -1 rather than as an
    empty string, which the page's numeric comparison would read as NaN and
    leave in arbitrary order; -1 ranks "no opportunity" just below a genuine 0%.
    """
    if not row["opportunities"] or pd.isna(row["reselection_rate"]):
        return "-1"
    return f"{float(row['reselection_rate']):.6f}"


def _cell(text: str, sort: object = None, numeric: bool = False) -> dict:
    """One rendered table cell. `text` is already-escaped HTML."""
    return {"html": text, "sort": sort, "numeric": numeric}


def build_author_rows(df: pd.DataFrame, total_editions: int) -> list[list[dict]]:
    """Rows for the authors page, sorted by selections desc then surname."""
    df = df.assign(_sort=df["author_name"].map(author_sort_key)).sort_values(
        ["edition_count", "_sort"], ascending=[False, True]
    )
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                _cell(html.escape(str(row["author_name"]))),
                _cell(
                    format_life_dates(row["birth_year"], row["death_year"]),
                    sort=row["birth_year"] if pd.notna(row["birth_year"]) else "",
                ),
                _cell(
                    format_percent(row["edition_count"], total_editions),
                    sort=int(row["edition_count"]),
                    numeric=True,
                ),
                _cell(
                    format_percent(row["reselection_count"], row["opportunities"]),
                    sort=_rate_sort(row),
                    numeric=True,
                ),
            ]
        )
    return rows


def build_work_rows(df: pd.DataFrame, total_editions: int) -> list[list[dict]]:
    """Rows for the works page, one per work, sorted by selections desc."""
    grouped = (
        df.groupby(["work_id", "work_title", "edition_count"], sort=False)
        .agg(
            author_name=("author_name", join_author_names),
            parent_work_title=("parent_work_title", "first"),
            reselection_count=("reselection_count", "first"),
            opportunities=("opportunities", "first"),
            reselection_rate=("reselection_rate", "first"),
        )
        .reset_index()
    )
    grouped = grouped.assign(
        _sort=grouped["author_name"].map(author_sort_key)
    ).sort_values(
        ["edition_count", "_sort", "work_title"], ascending=[False, True, True]
    )
    rows = []
    for _, row in grouped.iterrows():
        parent = row["parent_work_title"]
        rows.append(
            [
                _cell(
                    format_title(row["work_title"]),
                    sort=split_qualifier(row["work_title"])[0],
                ),
                _cell(html.escape(str(row["author_name"]))),
                _cell(
                    format_title(parent) if pd.notna(parent) else EM_DASH,
                    sort=split_qualifier(parent)[0] if pd.notna(parent) else "",
                ),
                _cell(
                    format_percent(row["edition_count"], total_editions),
                    sort=int(row["edition_count"]),
                    numeric=True,
                ),
                _cell(
                    format_percent(row["reselection_count"], row["opportunities"]),
                    sort=_rate_sort(row),
                    numeric=True,
                ),
            ]
        )
    return rows


def render_table(headers: list[str], rows: list[list[dict]]) -> str:
    """Render rows as an HTML table. Cell `html` is inserted as-is, so escape first."""
    numeric_columns = {
        i for i in range(len(headers)) if any(row[i]["numeric"] for row in rows)
    }
    head = "".join(
        f"<th{' class="num"' if i in numeric_columns else ''}>"
        f'{html.escape(header)}<span class="arrow"></span></th>'
        for i, header in enumerate(headers)
    )
    lines = ["<table>", f"<thead><tr>{head}</tr></thead>", "<tbody>"]
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            attrs = ' class="num"' if i in numeric_columns else ""
            if cell["sort"] is not None:
                attrs += f' data-sort="{html.escape(str(cell["sort"]), quote=True)}"'
            cells.append(f"<td{attrs}>{cell['html']}</td>")
        lines.append(f"<tr>{''.join(cells)}</tr>")
    lines += ["</tbody>", "</table>"]
    return "\n".join(lines)


def total_editions() -> int:
    """Number of AFAM-tagged editions -- the denominator of a selection record."""
    return int(query(query_path("afam-edition-count"))["n"].iloc[0])


def build_authors_page(out: Path) -> int:
    """Write the authors page and return the number of rows."""
    total = total_editions()
    rows = build_author_rows(query(query_path("authors-anthology-record")), total)
    lede = f"""<p class="lede">
  Every one of the {len(rows)} authors ever selected for an anthology of African
  American literature &mdash; from those in all {total} anthologies published
  between 1929 and 2025 down to those chosen exactly once.
</p>

<p class="lede">
  <b>Selected</b> is the share of the {total} anthologies that chose any work by
  the author. <b>Reselected</b> is the share of the anthologies published after
  the author's first appearance that chose them again; it is blank for the
  authors who debuted in the most recent anthology and have had no opportunity
  yet. Click any column heading to sort by it.
</p>"""
    page = PAGE_TEMPLATE.format(
        title="Author selection records",
        heading="Author selection records",
        lede=lede,
        table_heading=f"All {len(rows)} authors selected in {total} anthologies",
        table=render_table(["Author", "Dates", "Selected", "Reselected"], rows),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return len(rows)


def build_works_page(out: Path) -> int:
    """Write the works page and return the number of rows."""
    total = total_editions()
    rows = build_work_rows(query(query_path("works-anthology-record")), total)
    unattributed = sum(1 for row in rows if not row[1]["html"])
    lede = f"""<p class="lede">
  Every one of the {len(rows):,} works ever selected for an anthology of African
  American literature, across the {total} anthologies published between 1929 and
  2025. Excerpts count in their own right rather than being folded into the book
  they come from, so a chapter and the book it was taken from each keep their own
  row; <b>In</b> names that larger work &mdash; the volume a poem was collected
  in, or the book an excerpt came from.
</p>

<p class="lede">
  <b>Selected</b> is the share of the {total} anthologies that chose the work.
  <b>Reselected</b> is the share of the anthologies published after the work's
  first appearance that chose it again; it is blank for works that debuted in the
  most recent anthology and have had no opportunity yet.
</p>

<p class="lede">
  {unattributed} of these works have no author in the database, and their
  <b>Author</b> cell is left empty rather than credited to &ldquo;Anonymous.&rdquo;
  The anthologies print much of this material &mdash; spirituals, work songs,
  folktales &mdash; without attribution, and recording that silence as a name
  would assert an authorship claim the sources do not make. Click any column
  heading to sort by it.
</p>"""
    page = PAGE_TEMPLATE.format(
        title="Work selection records",
        heading="Work selection records",
        lede=lede,
        table_heading=f"All {len(rows):,} works selected in {total} anthologies",
        table=render_table(["Work", "Author", "In", "Selected", "Reselected"], rows),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--authors-out",
        type=Path,
        default=DEFAULT_AUTHORS_OUT,
        help="Authors page output (default: %(default)s)",
    )
    parser.add_argument(
        "--works-out",
        type=Path,
        default=DEFAULT_WORKS_OUT,
        help="Works page output (default: %(default)s)",
    )
    parser.add_argument(
        "--only",
        choices=("authors", "works", "both"),
        default="both",
        help="Which page to build (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.only in ("authors", "both"):
        count = build_authors_page(args.authors_out)
        print(f"wrote {args.authors_out} ({count} authors)")
    if args.only in ("works", "both"):
        count = build_works_page(args.works_out)
        print(f"wrote {args.works_out} ({count} works)")


if __name__ == "__main__":
    main()
