# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python analysis scripts for studying African American literary anthologies — examining author/work overlap, reselection rates, and predictability of inclusion in *The Norton Anthology of African American Literature* (NAAAL).

## Git convention

Commit directly to `main` when asked to commit. Do **not** create a feature branch first.

## Commands

```bash
# Requires Python 3.13+ (3.14 in .python-version)
# Install dependencies (also installs the local `afam` package in editable mode)
uv sync

# Run an analysis script
uv run python analysis/predictability/logistic_reselection.py --year 2025 --mode both
uv run python analysis/reselection/reselection_vs_chance.py --only-root-works
uv run python analysis/reselection/new_selection_reselection_probability.py --save-csv

# Run a viz script (figures land in output/)
uv run python viz/heatmaps/anthology_overlap_heatmap.py
uv run python viz/networks/anthology_network.py --csv data/myfile.csv --out output/network.png

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_gender_reselection.py

# Lint
uv run ruff check .
uv run ruff format .
```

## Repository layout

```
afam-anthologies/
├── src/afam/         # shared utility package — imported by every script
├── analysis/         # statistical/stdout/CSV analysis scripts (subfolders by theme)
├── viz/              # figure-producing scripts (subfolders by theme)
├── queries/          # SQL files loaded by DB-backed scripts
├── data/             # CSV inputs and CSV outputs (gitignored)
├── output/           # generated PNG/PDF figures (gitignored)
└── tests/            # pytest suites that import script helpers via sys.path
```

## Shared utilities (`src/afam/`)

Every script imports its DB connection, SQL loading, CSV path, and figure-output helpers from `afam.*` instead of redefining them. The package is installed in editable mode by `uv sync` via the `src/` layout declared in `pyproject.toml`.

| Module | What it exposes |
|---|---|
| `afam` (package root) | `REPO_ROOT`, `DATA_DIR`, `QUERIES_DIR`, `OUTPUT_DIR`, `ENV_FILE` |
| `afam.db` | `parse_db_params(env_file)`, `connect(env_file)`, `query(sql_or_path, params)` |
| `afam.sql` | `query_path(name)`, `load_query(name)` — resolves `queries/<name>.sql` |
| `afam.data` | `strip_volume`, `assign_edition_key_pipe`, `assign_edition_key_underscore`, `load_csv(name)` |
| `afam.viz_style` | `OUTPUT_DIR` (auto-mkdir), `DPI`, `FIGSIZE`, `WORK_COLOR`, `AUTHOR_COLOR` |
| `afam.cli` | `add_root_works_flag(parser)`, `add_save_csv_flag(parser)` |
| `afam.editions` | `EDITION_LABELS` (edition_id → short human-readable name) |
| `afam.names` | `author_last_name`, `author_sort_key` — surname-aware name parsing/sorting (handles particles like "van"/"de" and suffixes like "Jr.") |
| `afam.vernacular` | `parse_vernacular_row`, `load_vernacular_ranges` — parse the messy editor-designated vernacular page ranges in `data/vernacular_pages.csv` (handles `v1`/`v2` volume prefixes, prefix inheritance, comma/newline splits, `n/a`) |

A typical DB-backed script:

```python
from afam.db import query
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

df = query(query_path("works-per-afam-edition"))
fig.savefig(OUTPUT_DIR / "my_figure.png", dpi=300)
```

## `analysis/` — non-figure outputs

Scripts that print to stdout or write CSVs to `data/`.

| Subfolder | Scripts | Purpose |
|---|---|---|
| `overlap/` | `overlap_naaal_1996`, `simulate_author_work_overlap`, `half_or_more_sentences`, `authors_in_half_or_more_afam_eds`, `works_in_half_or_more_afam_eds`, `half_or_more_summary`, `per_author_work_overlap`, `author_disagreement` | Author/work overlap counts, Monte Carlo simulations, and multi-metric author-disagreement verdicts |
| `reselection/` | `reselection_vs_chance`, `new_selection_reselection_probability`, `author_first_selection_success`, `author_vs_work_debut_reselection`, `post_debut_performance`, `authors_without_frequent_works`, `early_selection_dropouts`, `work_pool_dilution` | Debut reselection rates, post-debut retention, early-dropout and work-pool-dilution analyses |
| `concentration/` | `author_form_concentration` | Page-weighted and count-weighted per-author-per-form concentration across selected works |
| `vernacular/` | `vernacular_works` | Resolve editor-designated vernacular page ranges into the works they contain; per-edition share of selections and pages that are vernacular |
| `predictability/` | `logistic_reselection`, `freq_bucket_predictability`, `predictability_over_time`, `predictability_new_focus_per_edition`, `simulate_naaal1996_selection`, `simulate_naaal2025_selection`, `work_selection_probability_model` | Logistic regression and predictability metrics for NAAAL inclusion |
| `influence/` | `anthology_influence` | Per-edition influence on subsequent editions: forward pickup rate of each edition's selections (all and debuts-only) vs. corpus baseline |
| `gender/` | `women_author_gaps`, `author_gender_summary` | Gender-gap analyses |
| `summaries/` | `most_anthologized`, `works_without_authors_naal_naaal`, `count_anthologies`, `count_works`, `edition_stats`, `edition_summary`, `format_author_anthology_counts`, `format_work_anthology_counts`, `author_never_cut`, `compare_series_authors`, `naal_exclusive_authors`, `naal_exclusive_works`, `list_reselected_works` | Counts, formatted tables, exclusivity lists |

## `viz/` — figure-producing scripts

Each script writes its PNG/PDF to `output/`.

| Subfolder | Scripts | Purpose |
|---|---|---|
| `heatmaps/` | `anthology_overlap_heatmap`, `author_overlap_heatmap`, `author_work_ratio_heatmap`, `author_work_shared_scatter` | Pairwise overlap matrices between editions |
| `networks/` | `anthology_network`, `work_network` | Bipartite co-occurrence graphs (require `--csv`) |
| `reselection/` | `reselection_probability`, `work_reselection_probability`, `gender_reselection`, `cumulative_pairwise_agreement`, `series_pair_reselection`, `author_selection_spread`, `author_page_share_reselection`, `retention_from_1929`, `retention_from_1941`, `first_selection_success`, `continuation_probability`, `edition_pair_retention_scatter`, `debut_reselection_forest`, `frequent_author_work_selection_scatter` | Per-author/work reselection and retention trends |
| `predictability/` | `predictability_over_time`, `predictability_repeat_focus`, `predictability_new_focus`, `naaal1996_prior_selection` | Visualizations of predictability metrics and logistic-model curves |
| `inequality/` | `inclusion_inequality`, `work_selection_divergence`, `selection_frequency_decay`, `selection_frequency_distribution` | Frequency distributions, divergence, survival curves |
| `influence/` | `anthology_influence_lift` | Dumbbell chart ranking editions by influence lift on subsequent editions |
| `misc/` | `canonical_author_map`, `form_by_edition` | One-off canonical-author and literary-form visualizations |
| `vernacular/` | `vernacular_share` | Bar chart of each edition's vernacular share, by selection count and by page extent |

## `tests/`

Pytest unit tests for helpers inside analysis/viz scripts. They `sys.path.insert` the relevant subfolder and import `compute()`/helper functions directly, so tests cover business logic without invoking `main()`. Whenever a script moves between subfolders, update the matching `sys.path.insert` line.

## `queries/`

SQL files executed by DB-backed scripts via `afam.sql.query_path(name)` (name is the file's stem; the `.sql` suffix is optional).

## Database schema

Key tables (PostgreSQL, database `anthologies`):

| Table | Key columns |
|---|---|
| `data_series` | `id`, `name` — anthology series (e.g. id=3 = NAAAL, id=17 = Wiley Blackwell) |
| `data_edition` | `id`, `year`, `edition_number`, `series_id`, `title` — one row per edition |
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

## Database setup

`.env` at the repo root must contain a `DATABASE_URL` in this format:

```
DATABASE_URL=PGPASSWORD=<password> psql -h <host> -U <user> <dbname>
```

`afam.db.parse_db_params()` parses this string into psycopg connection kwargs.

## Data inputs (CSV)

CSV files live in `data/` (gitignored). DB-backed scripts read live from PostgreSQL; only a few legacy scripts still read CSV. Primary source datasets:

| File | Used by |
|---|---|
| `2026-03-13 works per afam anthology.csv` | Most 2026-era CSV-backed scripts |
| `2026-03-13 work ids in afam anthologies.csv` | Some heatmap/inequality scripts |
| `2026-03-13 author ids in afam anthologies.csv` | `analysis/reselection/author_first_selection_success.py`, heatmaps |
| `2026-03-17 af am anthology authors with genders.csv` | Gender analyses |
| `202505121539 authors works.csv` | `analysis/predictability/freq_bucket_predictability.py`, `analysis/overlap/overlap_naaal_1996.py` |
| `vernacular_pages.csv` | `afam.vernacular` (vernacular analysis/viz) — hand-maintained `volume_id → page ranges` map of editor-designated vernacular material |

Two `edition_key` formats are derived consistently across scripts via `afam.data`:

- **Pipe format** (`{series}|{edition}`, 2026 datasets) — `assign_edition_key_pipe(df)`
- **Underscore format** (`{series_id}_{anthology_edition}`, 202505 dataset) — `assign_edition_key_underscore(df)`

## Outputs

Generated PNG/PDF figures are written to `output/` (auto-created by `afam.viz_style`, gitignored). Generated CSVs are written to `data/`.

## Flags common across scripts

- `--only-root-works` / `--include-excerpts` — root-only is the default for DB-backed scripts; pass `--include-excerpts` to include excerpt works. "Root works" means `parent_id` is null. Shared parser scaffolding is available via `afam.cli.add_root_works_flag(parser)`.
- `--save-csv` — write output tables to `data/` instead of stdout. Available via `afam.cli.add_save_csv_flag(parser)`.
- `--year` / `--mode` — in `analysis/predictability/logistic_reselection.py`: target edition year and whether to model `authors`, `works`, or `both`.
