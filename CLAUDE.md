# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python analysis scripts for studying African American literary anthologies — examining author/work overlap, reselection rates, and predictability of inclusion in *The Norton Anthology of African American Literature* (NAAAL).

## Commands

```bash
# Requires Python 3.13+ (3.14 in .python-version)
# Install dependencies
uv sync

# Run an analysis script
uv run python analysis/logistic_reselection.py --year 2025 --mode both
uv run python analysis/reselection_vs_chance.py --only-root-works
uv run python analysis/new_selection_reselection_probability.py --save-csv

# Run a viz script
uv run python viz/anthology_overlap_heatmap.py
uv run python viz/anthology_network.py --csv data/myfile.csv --out network.png

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_gender_reselection.py

# Lint
uv run ruff check .
uv run ruff format .
```

## Architecture

Scripts are standalone — each is a self-contained analysis with a `main()` and `if __name__ == "__main__"` guard. There is no shared library or import graph between scripts.

**`analysis/`** — statistical/overlap analysis scripts that print results to stdout or write CSVs:

| Script | Description |
|--------|-------------|
| `logistic_reselection.py` | Unified logistic regression predicting author or work reselection (DB-backed; replaces the old per-edition logistic scripts) |
| `new_selection_reselection_probability.py` | Reselection probability for debut authors/works, with Wilson CI by decade (DB-backed) |
| `authors_in_half_or_more_afam_eds.py` | Authors appearing in ≥50% of AFAM editions (DB-backed) |
| `works_in_half_or_more_afam_eds.py` | Works appearing in ≥50% of AFAM editions (DB-backed) |
| `half_or_more_sentences.py` | Summary sentences for authors/works in ≥50% of editions (DB-backed) |
| `overlap_naaal_1996.py` | Work and author overlap between NAAAL 1996 and pre-1996 anthologies |
| `freq_bucket_predictability.py` | Recall analysis by prior appearance frequency buckets |
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

**`viz/`** — 34 scripts producing charts, network graphs, and formatted tables. PNG outputs are gitignored and written to `viz/` or the current directory. Categories:
- **Heatmaps**: `anthology_overlap_heatmap.py`, `author_work_ratio_heatmap.py` — overlap matrices between editions
- **Networks**: `anthology_network.py`, `work_network.py` — bipartite co-occurrence graphs (require `--csv`)
- **Scatter/line**: `author_selection_spread.py`, `reselection_probability.py` — per-author or per-work trends
- **Tables/summaries**: formatted text tables, gender summaries, retention summaries

Most scripts use hardcoded DB or CSV paths; `anthology_network.py` and a few others require `--csv`.

**`tests/`** — pytest unit tests for analysis functions. Tests import internal `compute()` and helper functions directly from `analysis/` scripts via `sys.path.insert`, so they cover business logic without invoking `main()`.

**`queries/`** — SQL files read and executed by DB-backed scripts against the PostgreSQL database.

## Database schema

Key tables (PostgreSQL, database `anthologies`):

| Table | Key columns |
|-------|-------------|
| `data_series` | `id`, `name` — anthology series (e.g. id=3 = NAAAL, id=17 = Wiley Blackwell) |
| `data_edition` | `id`, `year`, `edition_number`, `series_id`, `title` — one row per anthology edition |
| `data_volume` | `id`, `edition_id`, `volume_number` — physical volumes within an edition |
| `data_work` | `id`, `title`, `parent_id` — `parent_id` links excerpts to their root work |
| `data_workinanthology` | `work_id`, `volume_id` — many-to-many works ↔ volumes |
| `data_author` | `id`, `name`, `birth_year`, `death_year` |
| `data_work_authors` | `work_id`, `author_id` — many-to-many works ↔ authors |
| `data_literarytradition` | `id`, `name` — e.g. `'African-American Literature'` |
| `data_edition_literary_traditions` | `edition_id`, `literarytradition_id` — tags editions by tradition |

**Join path for works in an edition:** `data_edition` → `data_volume` (via `edition_id`) → `data_workinanthology` (via `volume_id`) → `data_work` (via `work_id`)

**Filtering to AFAM tradition:** join `data_edition_literary_traditions` + `data_literarytradition` and filter `lt.name = 'African-American Literature'`. There are 26 such editions (1929–2025).

**Key edition IDs** (NAAAL series_id=3): 1997 ed.1 → id=16, 2004 ed.2 → id=17, 2014 ed.3 → id=13, 2025 ed.4 → id=43. Call and Response 1998 → id=60 (no series).

## Data access: two patterns

**CSV-backed scripts** (older): hardcode a `DATA_FILE` constant pointing to `data/*.csv`. Overridable via `--data-file` in some scripts.

**DB-backed scripts** (newer): connect to PostgreSQL via `psycopg`, reading connection params from `.env`. SQL is stored in `queries/*.sql` and loaded at runtime. These scripts do not use local CSV files as input.

### Database setup

`.env` must contain a `DATABASE_URL` in this format:
```
DATABASE_URL=PGPASSWORD=<password> psql -h <host> -U <user> <dbname>
```

The DB helper in each script parses this string to extract `host`, `user`, `password`, and `dbname`.

## Data (CSV)

CSV files live in `data/` (gitignored). Primary source datasets:

| File | Used by |
|------|---------|
| `2026-03-13 works per afam anthology.csv` | Most 2026-era CSV-backed analysis scripts |
| `2026-03-13 work ids in afam anthologies.csv` | Some logistic/work scripts |
| `2026-03-13 author ids in afam anthologies.csv` | `author_first_selection_success.py` |
| `2026-03-17 af am anthology authors with genders.csv` | `women_author_gaps.py` |
| `202505121539 authors works.csv` | `freq_bucket_predictability.py`, `overlap_naaal_1996.py` |

Key columns in `202505` data: `work_id`, `work_author`, `author_id`, `anthology_id`, `anthology_series`, `anthology_edition`, `anthology_year`, `series_id`, `parent_work_id` / `parent_work_title`, `work_form`.

Key columns in `2026-03-13` data: `series_id`, `anthology_edition`, `anthology_volume`, `anthology_title`, `anthology_id`, `publication_year`, `work_id` / `author_id`.

**`edition_key`** is a derived column computed consistently across scripts:
- In `202505` data: `{series_id}_{anthology_edition}` when `series_id` is non-empty, else `anthology_id`
- In `2026` data: `{series_id}|{anthology_edition}` (pipe-separated) when `series_id` is non-empty, else `anthology_id`

## Flags common across scripts

- `--only-root-works` / `--include-excerpts` — DB-backed scripts default to root-only; pass `--include-excerpts` to add excerpt works. CSV-backed scripts include excerpts by default; pass `--only-root-works` to filter them out. "Root works" means `parent_work_id` / `parent_work_title` is empty.
- `--save-csv` — write output tables to `data/` instead of printing to stdout
- `--data-file` — override the default `DATA_FILE` path (supported in some CSV-backed scripts)
- `--year` / `--mode` — in `logistic_reselection.py`: target edition year and whether to model `authors`, `works`, or `both`
