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
| `afam.data` | `load_csv(name)` — reads the one remaining hand-maintained CSV (`data/vernacular_pages.csv`) |
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
| `overlap/` | `simulate_author_work_overlap`, `author_work_agreement_models`, `author_work_agreement_stress_test`, `half_or_more_sentences`, `authors_in_half_or_more_afam_eds`, `works_in_half_or_more_afam_eds`, `half_or_more_summary`, `per_author_work_overlap`, `author_disagreement`, `sterling_brown_poem_overlap` | Author/work overlap counts, Monte Carlo simulations, multi-metric author-disagreement verdicts, and a single-author (Sterling Brown) pairwise poem-overlap case study; `author_work_agreement_models` tests whether "authors agree more than works" is a pool-size artifact — author-first vs work-first generative models (uniform and Zipf-weighted "fractal canonicity" variants with per-trial fixed work rankings) run against real edition shapes (`--mode real`) and a synthetic parameter sweep over concentration × works-per-author × corpus size (`--mode synthetic`); `--mode fit` grid-searches the author-canon model jointly over concentration γ × corpus inflation λ (each author's effective corpus gains round(λ×observed) unobserved works whose picks are unique never-shared selections) against all five observed statistics (RMS z), and `--decouple-allocation` randomly re-matches per-author work counts to selected authors as a rank-vs-allocation robustness check; primary metric is mean Jaccard(authors) − Jaccard(works), with the shared-count A>W pair rate for continuity; `author_work_agreement_stress_test` is the chronological validation runner (prequential grid fitting before a temporal cutoff, frozen-rank held-out identity scoring, split-simulation posterior-predictive checks) whose model families {uniform, author-only, work-only, mixed} × {± same-series carryover} include δ_author/δ_work log-boosts for candidates in the focal edition's most recent same-series predecessor (works are reselected within a series at ~74% vs ~18% across series) and count rank scores by distinct series (`--rank-basis`), so cross-editor canonicity is not conflated with series inertia |
| `reselection/` | `reselection_vs_chance`, `new_selection_reselection_probability`, `author_first_selection_success`, `author_vs_work_debut_reselection`, `post_debut_performance`, `authors_without_frequent_works`, `early_selection_dropouts`, `work_pool_dilution`, `reselection_hazard` | Debut reselection rates, post-debut retention, early-dropout and work-pool-dilution analyses; `reselection_hazard` is the edition-indexed discrete-time survival/hazard model (entity-edition person-period table, first-event & recurrent, cloglog/logit GLM) comparing author vs. work reselection net of exposure |
| `concentration/` | `author_form_concentration`, `poet_work_dispersal`, `poet_signature_concentration` | Page-weighted and count-weighted per-author-per-form concentration across selected works; per-poet work-level dispersal (top-poem edition coverage vs. effective number of poems, plus an all-forms `distinct_works_all_forms` count — e.g. Hughes 141 vs McKay 58); `poet_signature_concentration` ranks the inverse — poets whose selection collapses onto one signature poem (high selection volume + few distinct poems + dominant top-poem coverage) |
| `vernacular/` | `vernacular_works`, `vernacular_reselection_test` | Resolve editor-designated vernacular page ranges into the works they contain; per-edition share of selections and pages that are vernacular; `vernacular_reselection_test` tests whether vernacular works are reselected at different rates than non-vernacular works appearing in vernacular-containing editions — Test 1 pools V vs NV on ever-reselected and per-opportunity metrics across three scopes (all, cross-series, and cross-series restricted to only opportunities/reselections in later editions that themselves contain a non-zero number of vernacular works — `*_vern_eds` metrics, with a Test 1b delta report against the plain cross-series estimate), chi²/Fisher/z; Test 2 compares the vernacular rate to \|V\|-sized Monte Carlo NV samples (`--n`/`--seed`) for every metric/scope including `*_vern_eds`, drawn uniformly and debut-edition-stratified (the stratified null equalizes reselection opportunities, since vernacular material clusters in early editions) |
| `predictability/` | `logistic_reselection`, `predictability_over_time`, `predictability_new_focus_per_edition`, `simulate_naaal1996_selection`, `simulate_naaal2025_selection`, `work_selection_probability_model` | Logistic regression and predictability metrics for NAAAL inclusion; `logistic_reselection` also prints a discrete "exactly k prior anthologies" selection-rate table (per-exact-count, not pooled ">= k") and a summary sentence naming the first prior-count that crosses a >50% selection rate for authors/works, plus the size/share of the pool selected for that many or more prior anthologies; `predictability_over_time` exposes `compute_per_edition()` for the new-focus/repeat-focus consumers |
| `influence/` | `anthology_influence` | Per-edition influence on subsequent editions: forward pickup rate of each edition's selections (all and debuts-only) vs. corpus baseline |
| `growth/` | `pool_growth_rates` | How fast the pool of *ever-anthologized* works grows relative to the pool of *ever-anthologized* authors. Two scale-free estimators fit on one chronological sweep: the Heaps exponent β (OLS of log pool on log cumulative selection slots — primary, since it compares the two pools at equal editorial effort) and a per-anthology exponential rate. Headline: β 0.811 vs 0.640 (works accumulate ~27% faster per unit of effort), 11.3%/anthology vs 8.0% (~40% faster). Three tests: (1) Wilcoxon + exact sign test on per-edition novelty rates paired by anthology, plus Kendall τ on the works-minus-authors gap to check whether it narrows; (2) a cluster bootstrap resampling **authors** (works are nested inside them) for a 95% CI on Δβ — the repo's first bootstrap, vectorized via per-author contribution matrices so each replicate is one matrix-vector product; (3) the **loyalty-matched null**, which holds each edition's work-slot count at its observed value and draws repeats at that edition's observed *author* retention rate, so scale is preserved and only loyalty is swapped — this is what shows the gap is editorial behavior rather than the mechanical consequence of works outnumbering authors 5.6-to-1. `--time-axis {edition,year}`; note the primary scope **inverts** the repo-wide root-works default (it keeps excerpts and unauthored works so the pools match the project's 575 / 3,236 counts), with all four scopes printed as a robustness table |
| `gender/` | `women_author_gaps`, `author_gender_summary`, `gender_work_consistency` | Gender-gap analyses; `gender_work_consistency` tests whether work-selection consistency (Jaccard of an author's work sets across editions, within literary form) differs by gender |
| `summaries/` | `works_without_authors_naal_naaal`, `edition_stats`, `edition_summary`, `format_author_anthology_counts`, `format_work_anthology_counts`, `author_never_cut`, `compare_series_authors`, `naal_exclusive_authors`, `naal_exclusive_works`, `list_reselected_works` | Counts, formatted tables, exclusivity lists; `format_author_anthology_counts` also ranks author selection records with `--percentiles` (percentile rank + "top X%" share, always computed over all 575 ever-selected authors regardless of `--min-anthologies`) and `--author NAME` for a one-sentence standing report |
| `robustness/` | `edge_case_sensitivity` | Sensitivity check for the three anthologies excluded from the AFAM corpus as non-comparable edge cases (*Black Culture* 1972 id=79, *Crossing the Danger Water* 1993 id=62, *African American Literature* series ed.1 1993 id=22 — none has a `data_edition_literary_traditions` row); widens the AFAM tag filter in-memory via a `UNION` (no DB writes) to re-run the four headline author-vs-work claims (never-repeated debut share, ≥half-of-anthologies selection counts, debut reselection-rate comparison, paired edition retention) with vs. without these editions |

## `viz/` — figure-producing scripts

Each script writes its PNG/PDF to `output/`.

| Subfolder | Scripts | Purpose |
|---|---|---|
| `heatmaps/` | `anthology_overlap_heatmap`, `author_overlap_heatmap`, `author_work_ratio_heatmap`, `author_work_shared_scatter` | Pairwise overlap matrices between editions |
| `networks/` | `anthology_network`, `work_network` | Bipartite co-occurrence graphs (require `--csv`) |
| `reselection/` | `reselection_probability`, `work_reselection_probability`, `gender_reselection`, `cumulative_pairwise_agreement`, `series_pair_reselection`, `author_selection_spread`, `author_page_share_reselection`, `retention_from_1929`, `retention_from_1941`, `first_selection_success`, `continuation_probability`, `edition_pair_retention_scatter`, `debut_reselection_forest`, `frequent_author_work_selection_scatter`, `reselection_hazard_curves` | Per-author/work reselection and retention trends; `reselection_hazard_curves` plots survival S(t), discrete hazard h(t), and the hazard-ratio forest from `analysis/reselection/reselection_hazard`; `edition_pair_retention_scatter` takes `--authored-works-only`, which drops works with no author on record (298 of 2,673) and writes the variant to `edition_pair_retention_scatter_authored_only.*` — unauthored works can never contribute to author retention, so this restricts both axes to the same authored material (author retention exceeds work retention in 311/311 chronological pairs vs. 310/311 in the default scope) |
| `predictability/` | `predictability_over_time`, `predictability_repeat_focus`, `predictability_new_focus`, `naaal1996_prior_selection` | Visualizations of predictability metrics and logistic-model curves |
| `inequality/` | `inclusion_inequality`, `work_selection_divergence`, `selection_frequency_decay`, `selection_frequency_distribution` | Frequency distributions, divergence, survival curves |
| `influence/` | `anthology_influence_lift` | Dumbbell chart ranking editions by influence lift on subsequent editions |
| `growth/` | `pool_growth_curves` | Two figures for `analysis/growth/pool_growth_rates` (all computation is imported, none duplicated). `pool_growth_curves.png` is essay-ready: cumulative author vs. work pool on a log axis with the fitted Heaps curves overlaid, over a per-anthology novelty panel showing the share of each anthology's selections new to the pool (works sit above authors in 24 of 25 steps; the 2025 *NAAAL* is the lone crossover). `pool_growth_diagnostics.png` shows the Heaps log-log fit and the loyalty-matched null distribution of Δβ against the observed value and bootstrap CI. Authors blue / works red, matching `cumulative_pairwise_agreement` (the inverse of `afam.viz_style`'s `AUTHOR_COLOR` / `WORK_COLOR`) |
| `misc/` | `canonical_author_map`, `form_by_edition` | One-off canonical-author and literary-form visualizations |
| `vernacular/` | `vernacular_share`, `vernacular_reselection_forest`, `vernacular_debut_reselection` | Bar chart of each edition's vernacular share, by selection count and by page extent; forest plot of the twelve `vernacular_reselection_test` estimates (six metrics, including the `*_vern_eds` cross-series-restricted scope) — observed vernacular rate (95% Wilson CI) over each Monte Carlo null's 95% interval, per metric × sampling mode; `vernacular_debut_reselection` is a per-debut-edition jittered boxplot of later-edition reselection counts (V vs. NV points over one box per edition, `--cross-series` to switch scope), visualizing the same debut-clustering confound the stratified null corrects for |

## `tests/`

Pytest unit tests for helpers inside analysis/viz scripts. They `sys.path.insert` the relevant subfolder and import `compute()`/helper functions directly, so tests cover business logic without invoking `main()`. Whenever a script moves between subfolders, update the matching `sys.path.insert` line.

## `queries/`

SQL files executed by DB-backed scripts via `afam.sql.query_path(name)` (name is the file's stem; the `.sql` suffix is optional).

The workhorse replacement for the legacy "works per afam anthology" CSV dump is `works-authors-per-afam-edition.sql` — one row per (work, author) over AFAM-tagged editions, carrying work/parent titles, edition year/series/number, and author name + birth year. Another shared query added for the CSV→DB migration: `naal-american-authors-works.sql` (Norton American series_id 1 + 3, not AFAM-restricted, for the NAAL-exclusivity / series-comparison scripts).

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
| `data_genders` | `id`, `name` — `Male`, `Female`, `Other` |
| `data_author_genders` | `author_id`, `genders_id` — many-to-many authors ↔ genders (collapse to one label, e.g. `MIN(genders_id)`) |
| `data_form` | `id`, `name` — literary form (`poetry`, `fiction`, `nonfiction`, `drama`, `song`, `folk`, …) |
| `data_work_form` | `work_id`, `form_id` — many-to-many works ↔ forms (multi-form works collapse to lowest `form_id`; excerpts inherit parent form) |

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

Every script under `analysis/` and `viz/` reads live from PostgreSQL. The only CSV input that remains is one hand-maintained file:

| File | Used by |
|---|---|
| `vernacular_pages.csv` | `afam.vernacular` (vernacular analysis/viz) — hand-maintained `volume_id → page ranges` map of editor-designated vernacular material |

Editions are volume-collapsed in the DB (one `data_edition.id` per edition), so scripts group on `edition_id` directly — there is no longer any CSV-derived `edition_key` reconstruction. The DB-dump CSVs that scripts used to read (the 2026-era works/authors exports, the genders export, the 202505 datasets, the author-birth export) are obsolete; their values now come from the queries below, chiefly `queries/works-authors-per-afam-edition.sql`.

## Outputs

Generated PNG/PDF figures are written to `output/` (auto-created by `afam.viz_style`, gitignored). Generated CSVs are written to `data/`.

## Flags common across scripts

- `--only-root-works` / `--include-excerpts` — root-only is the default for DB-backed scripts; pass `--include-excerpts` to include excerpt works. "Root works" means `parent_id` is null. Shared parser scaffolding is available via `afam.cli.add_root_works_flag(parser)`.
- `--save-csv` — write output tables to `data/` instead of stdout. Available via `afam.cli.add_save_csv_flag(parser)`.
- `--year` / `--mode` — in `analysis/predictability/logistic_reselection.py`: target edition year and whether to model `authors`, `works`, or `both`.
