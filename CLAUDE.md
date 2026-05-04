# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python analysis scripts for studying African American literary anthologies — examining author/work overlap, reselection rates, and predictability of inclusion in *The Norton Anthology of African American Literature* (NAAAL).

## Commands

```bash
# Install dependencies
uv sync

# Run an analysis script
uv run python analysis/simulate_naaal2025_selection.py
uv run python analysis/reselection_vs_chance.py --only-root-works
uv run python analysis/logistic_predictability_naaal2025.py --save-csv

# Run a viz script
uv run python viz/anthology_overlap_heatmap.py
uv run python viz/anthology_network.py --csv data/myfile.csv --out network.png
uv run python viz/reselection_probability.py data/myfile.csv

# Run tests
uv run pytest

# Lint
uv run ruff check .
uv run ruff format .
```

## Architecture

Scripts are standalone — each is a self-contained analysis with a `main()` and `if __name__ == "__main__"` guard. There is no shared library or import graph between scripts.

**`analysis/`** — statistical/overlap analysis scripts that print results to stdout or write CSVs:

| Script | Description |
|--------|-------------|
| `overlap_naaal_1996.py` | Work and author overlap between NAAAL 1996 and pre-1996 anthologies |
| `freq_bucket_predictability.py` | Recall analysis by prior appearance frequency buckets |
| `logistic_predictability_naaal1996.py` | Logistic regression predicting NAAAL 1996 inclusion |
| `logistic_predictability_naaal2025.py` | Logistic regression predicting NAAAL 2025 author inclusion |
| `logistic_predictability_naaal2025_works.py` | Logistic regression predicting NAAAL 2025 work inclusion |
| `simulate_naaal1996_selection.py` | Logit + Bayesian models trained on pre-1996 data to predict NAAAL 1996 |
| `simulate_naaal2025_selection.py` | Logit + Bayesian models trained on pre-2025 data to predict NAAAL 2025 |
| `simulate_author_work_overlap.py` | Monte Carlo simulation of author vs. work pairwise overlap |
| `reselection_vs_chance.py` | Reselection rates vs. chance with statistical significance tests |
| `author_first_selection_success.py` | Author debut selection and subsequent retention rates |
| `author_vs_work_debut_reselection.py` | Author vs. work debut reselection comparison |
| `post_debut_performance.py` | Post-debut selection performance analysis |
| `most_anthologized.py` | Most frequently anthologized authors and works |
| `authors_without_frequent_works.py` | Authors who appear often but whose individual works are rarely reselected |
| `women_author_gaps.py` | Gender gap analysis using author gender data |

**`viz/`** — scripts that produce charts, network graphs, and formatted tables. Most hardcode a data path at the top of the file; some require a `--csv` argument. PNG outputs are gitignored and written to `viz/` or the current directory.

**`tests/`** — pytest unit tests for analysis functions. Tests import internal `compute()` and helper functions directly from `analysis/` scripts via `sys.path.insert`, so they cover business logic without invoking `main()`.

## Data

CSV files live in `data/` (gitignored). Scripts hardcode data paths as module-level `DATA_FILE` constants (overridable via `--data-file` in some scripts). Primary datasets:

| File | Used by |
|------|---------|
| `2026-03-13 works per afam anthology.csv` | Most 2026-era analysis scripts |
| `2026-03-13 work ids in afam anthologies.csv` | `logistic_predictability_naaal2025_works.py` |
| `2026-03-13 author ids in afam anthologies.csv` | `author_first_selection_success.py`, `logistic_predictability_naaal2025.py` |
| `2026-03-17 af am anthology authors with genders.csv` | `women_author_gaps.py` |
| `202505121539 authors works.csv` | `freq_bucket_predictability.py`, `logistic_predictability_naaal1996.py`, `overlap_naaal_1996.py` |

Key columns in `202505` data: `work_id`, `work_author`, `author_id`, `anthology_id`, `anthology_series`, `anthology_edition`, `anthology_year`, `series_id`, `parent_work_id` / `parent_work_title`, `work_form`.

Key columns in `2026-03-13` data: `series_id`, `anthology_edition`, `anthology_volume`, `anthology_title`, `anthology_id`, `publication_year`, `work_id` / `author_id`.

**`edition_key`** is a derived column computed consistently across scripts:
- In `202505` data: `{series_id}_{anthology_edition}` when `series_id` is non-empty, else `anthology_id`
- In `2026` data: `{series_id}|{anthology_edition}` (pipe-separated) when `series_id` is non-empty, else `anthology_id`

## Flags common across scripts

- `--only-root-works` — restrict analysis to works without a parent work (`parent_work_id` or `parent_work_title` is empty), i.e. exclude excerpts/selections
- `--save-csv` — write output tables to `data/` instead of printing to stdout
- `--data-file` — override the default `DATA_FILE` path (supported in newer logistic scripts)
