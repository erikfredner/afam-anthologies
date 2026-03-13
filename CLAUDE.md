# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python analysis scripts for studying African American literary anthologies — examining author/work overlap, reselection rates, and predictability of inclusion in *The Norton Anthology of African American Literature* (NAAAL).

## Commands

```bash
# Install dependencies
uv sync

# Run a script
uv run python overlap_naaal_1996.py
uv run python freq_bucket_predictability.py --save-csv
uv run python logistic_predictability_naaal1996.py
uv run python author_selection_spread.py

# Lint
uv run ruff check .
uv run ruff format .

# viz/ scripts take positional or named args (see --help)
uv run python viz/anthology_network.py --csv data/myfile.csv --out network.png
uv run python viz/reselection_probability.py data/myfile.csv
```

## Architecture

Scripts are standalone — each is a self-contained analysis with a `main()` and `if __name__ == "__main__"` guard. There is no shared library or import graph between scripts.

**Root-level scripts** do statistical/overlap analysis:
- `overlap_naaal_1996.py` — work and author overlap between NAAAL 1996 and pre-1996 anthologies
- `freq_bucket_predictability.py` — recall analysis by prior appearance frequency buckets
- `logistic_predictability_naaal1996.py` — logistic regression predicting NAAAL 1996 inclusion
- `author_selection_spread.py` — scatter plot of author breadth vs. work repetition

**`viz/` scripts** produce charts and network graphs, often requiring a CSV path argument.

## Data

CSV files live in `data/` (gitignored). Two primary datasets:

| File | Used by |
|------|---------|
| `data/202505121539 authors works.csv` | Most analysis scripts |
| `data/202505131423 author work form.csv` | `author_selection_spread.py` |

Key columns: `work_id`, `work_author`, `author_id`, `anthology_id`, `anthology_series`, `anthology_edition`, `anthology_year`, `series_id`, `parent_work_id` / `parent_work_title`, `work_form`.

**`edition_key`** is a derived column computed consistently across scripts:
- `{series_id}_{anthology_edition}` when `series_id` is non-empty
- `anthology_id` otherwise

PNG outputs are also gitignored and written to `viz/` or the current directory.

## Flags common across scripts

- `--only-root-works` — restrict analysis to works without a parent work (`parent_work_id` or `parent_work_title` is empty), i.e. exclude excerpts/selections
- `--save-csv` — write output tables to `data/` instead of printing to stdout
