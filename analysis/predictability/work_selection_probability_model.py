"""
work_selection_probability_model.py
-----------------------------------
Per-edition probability model that scores every selection appearing in an AFAM
anthology and flags the top-N most-probable and top-N least-probable per edition.

Four factors compose the probability score:

    1. length_z_by_form           — appearance's share of edition pages, z-scored
                                    within the work's literary form (pooled across
                                    all 26 AFAM editions). Page share — not raw
                                    pages — is used so that "long for form" stays
                                    comparable across editions of 500–2500+ pages.
                                    Higher → more improbable.
    2. prior_editions_count       — distinct prior AFAM editions (year < current)
                                    that included this exact work_id. Higher → more
                                    probable.
    3. pct_pages_of_edition       — appearance's page span / total edition pages.
                                    Total edition pages is summed from each volume's
                                    last_toc_page − first_toc_page + 1 (volume
                                    metadata), not from per-work spans. Higher →
                                    more improbable.
    4. author_prior_selection_rate — opportunities-normalized rate of the author's
                                    prior selections (mirrors the convention in
                                    analysis/reselection/post_debut_performance.py).
                                    Max across multi-author works. Higher → more
                                    probable.

Two scores are emitted side-by-side:

    prob_heuristic = z(prior_editions_count) + z(author_prior_selection_rate)
                     − z(length_z_by_form)   − z(pct_pages_of_edition)

    prob_logit     = predicted P(selected_in_E | log1p(prior_count),
                                                avg_length_z_by_form,
                                                author_prior_selection_rate)
                     from a per-edition statsmodels Logit fit on the counterfactual
                     pool of works that appeared in any earlier AFAM edition.

The unit of analysis is every appearance in data_workinanthology — one row per
(work_id × volume_id × edition_id) — including excerpts (parent_id IS NOT NULL).

Outputs (with --save-csv):
    data/selection_probability_top_per_edition.csv
    data/selection_probability_bottom_per_edition.csv

Usage:
    uv run python analysis/predictability/work_selection_probability_model.py
    uv run python analysis/predictability/work_selection_probability_model.py --save-csv
    uv run python analysis/predictability/work_selection_probability_model.py --top-n 5
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError

from afam import DATA_DIR
from afam.cli import add_save_csv_flag
from afam.db import query
from afam.sql import query_path

TOP_CSV = DATA_DIR / "selection_probability_top_per_edition.csv"
BOTTOM_CSV = DATA_DIR / "selection_probability_bottom_per_edition.csv"

MIN_FORM_N = 5
DEFAULT_TOP_N = 10

OUTPUT_COLUMNS = [
    "year",
    "edition_id",
    "edition_label",
    "work_id",
    "work_title",
    "author_names",
    "form",
    "pages",
    "length_z_by_form",
    "prior_editions_count",
    "pct_pages_of_edition",
    "author_prior_selection_rate",
    "author_is_debut",
    "prob_heuristic",
    "prob_logit",
    "rank_in_edition",
]


# ── Page span (mirrors predictability_over_time.compute_work_volume_spans) ─────


def compute_work_volume_spans(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (work_id, volume_id) with the work's page span in that volume.

    Span = toc_next - toc_page when both present; falls back to
    last_toc_page - toc_page + 1 when toc_next is NULL.
    """
    spans = raw[
        ["work_id", "volume_id", "toc_page", "toc_next", "last_toc_page"]
    ].drop_duplicates(["work_id", "volume_id"])
    span = spans["toc_next"] - spans["toc_page"]
    fallback = spans["last_toc_page"] - spans["toc_page"] + 1
    span = span.where(spans["toc_next"].notna(), fallback)
    span = span.clip(lower=0).fillna(0).astype(int)
    return pd.DataFrame(
        {
            "work_id": spans["work_id"].values,
            "volume_id": spans["volume_id"].values,
            "span": span.values,
        }
    )


# ── Form attachment (per work, with parent fallback) ──────────────────────────


def build_form_map(form_rows: pd.DataFrame, parent_map: pd.DataFrame) -> pd.Series:
    """Return Series indexed by work_id with the canonical form name.

    Multi-form works collapse to the lowest form_id (stable preference).
    Works without a direct form row inherit their parent's form (one hop).
    """
    direct = (
        form_rows.sort_values("form_id")
        .drop_duplicates("work_id", keep="first")
        .set_index("work_id")["form_name"]
    )
    parent_of = parent_map.set_index("work_id")["parent_id"]
    out = direct.copy()
    missing = parent_of.index.difference(out.index)
    if len(missing):
        inherited = parent_of.loc[missing].map(direct)
        inherited = inherited.dropna()
        out = pd.concat([out, inherited])
    out.name = "form"
    return out


# ── Selection-level frame ─────────────────────────────────────────────────────


def build_selection_frame(
    raw: pd.DataFrame,
    spans: pd.DataFrame,
    form_by_work: pd.Series,
) -> pd.DataFrame:
    """One row per (work_id, volume_id, edition_id) with pages, year, form."""
    base = raw[
        [
            "work_id",
            "volume_id",
            "edition_id",
            "anthology_publication_year",
            "parent_id",
        ]
    ].drop_duplicates(["work_id", "volume_id", "edition_id"])
    base = base.merge(spans, on=["work_id", "volume_id"], how="left")
    base = base.rename(columns={"anthology_publication_year": "year", "span": "pages"})
    base["pages"] = base["pages"].fillna(0).astype(int)
    base["form"] = base["work_id"].map(form_by_work)
    return base


# ── Edition page totals from volume metadata ──────────────────────────────────


def compute_edition_total_pages(raw: pd.DataFrame) -> pd.Series:
    """Total physical pages per edition, summed from each volume's
    `last_toc_page - first_toc_page + 1` (the actual paginated extent of the book),
    not from summing per-work spans. Anthology editions vary from ~500 to 2500+ pp,
    so a 10-page excerpt is a very different share in each — this denominator
    captures that. Volumes missing pagination metadata are dropped.
    """
    vols = raw[
        ["edition_id", "volume_id", "first_toc_page", "last_toc_page"]
    ].drop_duplicates(["edition_id", "volume_id"])
    vols = vols.dropna(subset=["first_toc_page", "last_toc_page"])
    vols["volume_pages"] = (
        vols["last_toc_page"].astype(int) - vols["first_toc_page"].astype(int) + 1
    ).clip(lower=0)
    return (
        vols.groupby("edition_id")["volume_pages"].sum().rename("edition_total_pages")
    )


# ── Length z-score within form (operates on page share) ───────────────────────


def length_zscore_by_form(
    frame: pd.DataFrame,
    value_col: str = "pct_pages_of_edition",
    min_n: int = MIN_FORM_N,
) -> pd.Series:
    """Per-row z-score of `value_col` within the row's form, pooled across editions.

    `value_col` should be a proportional length (e.g. pct_pages_of_edition) so
    that "long for form" accounts for varying anthology sizes. Forms with fewer
    than `min_n` rows or zero variance yield NaN.
    """
    stats = frame.groupby("form")[value_col].agg(["mean", "std", "count"])
    means = frame["form"].map(stats["mean"])
    stds = frame["form"].map(stats["std"])
    counts = frame["form"].map(stats["count"])
    z = (frame[value_col] - means) / stds.replace(0, np.nan)
    z = z.where(counts >= min_n)
    return z


# ── Prior reselection count per appearance ────────────────────────────────────


def prior_editions_count(frame: pd.DataFrame) -> pd.Series:
    """For each row, count distinct prior editions (year < current) including this work.

    Uses edition-level dedup so multi-volume editions count once per edition.
    """
    work_edition_year = frame[["work_id", "edition_id", "year"]].drop_duplicates(
        ["work_id", "edition_id"]
    )
    work_edition_year = work_edition_year.sort_values(["work_id", "year", "edition_id"])
    work_edition_year["prior_count"] = work_edition_year.groupby("work_id").cumcount()
    lookup = work_edition_year.set_index(["work_id", "edition_id"])["prior_count"]
    keys = list(zip(frame["work_id"], frame["edition_id"]))
    return pd.Series(lookup.reindex(keys).values, index=frame.index, dtype=int)


# ── Author opportunities-normalized prior rate ────────────────────────────────


def author_prior_rate_table(
    author_rows: pd.DataFrame,
    edition_years: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per (author_id, edition_id) with the opportunities-normalized
    prior selection rate at the time of that edition.

    opportunities = AFAM editions with year(e) < target_year and
                    year(e) >= author_first_afam_year
    selections   = opportunities that included any work by this author
    rate         = selections / opportunities  (NaN when denom = 0 → marked as debut)

    Parameters
    ----------
    author_rows : DataFrame
        Columns: author_id, edition_id, year — one row per (author, edition).
    edition_years : DataFrame
        Columns: edition_id, year — distinct AFAM editions.
    """
    author_first = author_rows.groupby("author_id")["year"].min().rename("first_year")
    author_eds = author_rows[["author_id", "edition_id"]].drop_duplicates()
    author_ed_set = author_eds.groupby("author_id")["edition_id"].apply(set)

    eds = edition_years.drop_duplicates("edition_id").sort_values("year")
    ed_year_arr = eds["year"].to_numpy()
    ed_id_arr = eds["edition_id"].to_numpy()

    records: list[dict[str, Any]] = []
    for author_id, first_year in author_first.items():
        sel_eds = author_ed_set.get(author_id, set())
        for ed_id, target_year in zip(ed_id_arr, ed_year_arr):
            opp_mask = (ed_year_arr < target_year) & (ed_year_arr >= first_year)
            opps = ed_id_arr[opp_mask]
            n_opps = len(opps)
            if n_opps == 0:
                rate = np.nan
                is_debut = True
            else:
                n_sel = sum(1 for e in opps if e in sel_eds)
                rate = n_sel / n_opps
                is_debut = False
            records.append(
                {
                    "author_id": author_id,
                    "edition_id": ed_id,
                    "author_prior_selection_rate": rate,
                    "author_is_debut": is_debut,
                }
            )
    return pd.DataFrame(records)


def attach_author_rate_to_frame(
    frame: pd.DataFrame,
    raw: pd.DataFrame,
    author_rate: pd.DataFrame,
) -> pd.DataFrame:
    """Attach max author rate (across multi-author works) to each selection row."""
    work_authors = raw[["work_id", "author_id"]].dropna(subset=["author_id"])
    work_authors = work_authors.drop_duplicates()
    work_authors["author_id"] = work_authors["author_id"].astype(int)

    rate_lookup = author_rate.set_index(["author_id", "edition_id"])
    pairs = frame[["work_id", "edition_id"]].merge(
        work_authors, on="work_id", how="left"
    )
    pairs = pairs.dropna(subset=["author_id"]).copy()
    pairs["author_id"] = pairs["author_id"].astype(int)
    keys = list(zip(pairs["author_id"], pairs["edition_id"]))
    looked = rate_lookup.reindex(keys)
    pairs["author_prior_selection_rate"] = looked["author_prior_selection_rate"].values
    pairs["author_is_debut"] = looked["author_is_debut"].values

    grouped = (
        pairs.groupby(["work_id", "edition_id"])
        .agg(
            author_prior_selection_rate=("author_prior_selection_rate", "max"),
            author_is_debut=("author_is_debut", "all"),
        )
        .reset_index()
    )
    out = frame.merge(grouped, on=["work_id", "edition_id"], how="left")
    out["author_prior_selection_rate"] = out["author_prior_selection_rate"].fillna(0.0)
    out["author_is_debut"] = out["author_is_debut"].fillna(True).astype(bool)
    return out


# ── Composite heuristic score ─────────────────────────────────────────────────


def zscore_column(s: pd.Series) -> pd.Series:
    mean = s.mean(skipna=True)
    std = s.std(skipna=True)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def heuristic_score(frame: pd.DataFrame) -> pd.Series:
    """+z(prior) +z(author_rate) − z(length_by_form) − z(pct_pages). NaN-safe."""
    prior_z = zscore_column(frame["prior_editions_count"].astype(float))
    author_z = zscore_column(frame["author_prior_selection_rate"].astype(float))
    length_z = zscore_column(frame["length_z_by_form"].astype(float)).fillna(0.0)
    pct_z = zscore_column(frame["pct_pages_of_edition"].astype(float))
    return prior_z + author_z - length_z - pct_z


# ── Logistic regression score (per edition) ───────────────────────────────────


def fit_logit_safe(y: pd.Series, X: pd.DataFrame) -> Any:
    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X)
    try:
        return model.fit(disp=False)
    except (LinAlgError, ValueError, np.linalg.LinAlgError):
        return model.fit_regularized(alpha=1e-6, L1_wt=0.0, disp=False)


def compute_logit_scores(
    frame: pd.DataFrame,
    edition_years: pd.DataFrame,
    raw: pd.DataFrame,
    author_rate: pd.DataFrame,
) -> pd.Series:
    """For each selected row, attach the predicted P(selected_in_E) from a logit
    fit on the per-edition counterfactual pool.

    Counterfactual pool for edition E = every work_id that appeared in at least
    one prior AFAM edition (year < year(E)). Features per candidate:
        log1p(prior_count_before_E)
        avg_length_z_by_form_before_E    (mean over prior appearances)
        author_prior_selection_rate_before_E (max across authors, computed earlier)
    Outcome: 1 if work_id appears in E, else 0.
    """
    length_by_we = (
        frame.groupby(["work_id", "edition_id"], dropna=False)["length_z_by_form"]
        .mean()
        .rename("length_z")
    )
    work_edition = (
        frame[["work_id", "edition_id", "year"]]
        .drop_duplicates(["work_id", "edition_id"])
        .merge(length_by_we, on=["work_id", "edition_id"], how="left")
        .sort_values(["work_id", "year", "edition_id"])
        .reset_index(drop=True)
    )

    work_authors = (
        raw[["work_id", "author_id"]].dropna(subset=["author_id"]).drop_duplicates()
    )
    work_authors["author_id"] = work_authors["author_id"].astype(int)
    author_rate_lookup = author_rate.set_index(["author_id", "edition_id"])[
        "author_prior_selection_rate"
    ]

    eds_sorted = edition_years.drop_duplicates("edition_id").sort_values("year")

    scores = pd.Series(np.nan, index=frame.index)

    for _, ed_row in eds_sorted.iterrows():
        target_ed = int(ed_row["edition_id"])
        target_year = int(ed_row["year"])

        prior_appearances = work_edition[work_edition["year"] < target_year]
        if prior_appearances.empty:
            continue

        pool_stats = (
            prior_appearances.groupby("work_id")
            .agg(
                prior_count=("edition_id", "nunique"),
                avg_length_z=("length_z", "mean"),
            )
            .reset_index()
        )

        target_works = set(
            frame.loc[frame["edition_id"] == target_ed, "work_id"].unique()
        )
        pool_stats["selected_in_E"] = (
            pool_stats["work_id"].isin(target_works).astype(int)
        )

        pool_with_authors = pool_stats.merge(work_authors, on="work_id", how="left")
        pool_with_authors["author_rate"] = pool_with_authors.apply(
            lambda r: (
                author_rate_lookup.get((int(r["author_id"]), target_ed), np.nan)
                if pd.notna(r["author_id"])
                else np.nan
            ),
            axis=1,
        )
        pool_grouped = (
            pool_with_authors.groupby("work_id")
            .agg(
                prior_count=("prior_count", "first"),
                avg_length_z=("avg_length_z", "first"),
                selected_in_E=("selected_in_E", "first"),
                author_rate=("author_rate", "max"),
            )
            .reset_index()
        )
        pool_grouped["author_rate"] = pool_grouped["author_rate"].fillna(0.0)
        pool_grouped["avg_length_z"] = pool_grouped["avg_length_z"].fillna(0.0)
        pool_grouped["log1p_prior"] = np.log1p(pool_grouped["prior_count"])

        X = pool_grouped[["log1p_prior", "avg_length_z", "author_rate"]]
        y = pool_grouped["selected_in_E"]
        if y.nunique() < 2:
            continue

        try:
            result = fit_logit_safe(y, X)
            preds = result.predict(sm.add_constant(X, has_constant="add"))
        except Exception:
            continue

        pred_by_work = dict(zip(pool_grouped["work_id"], preds))

        mask = frame["edition_id"] == target_ed
        scores.loc[mask] = frame.loc[mask, "work_id"].map(pred_by_work)

    return scores


# ── Edition / work / author metadata ──────────────────────────────────────────


def load_metadata() -> dict[str, pd.DataFrame]:
    work_titles = query("SELECT id AS work_id, title AS work_title FROM data_work")
    authors = query("SELECT id AS author_id, name AS author_name FROM data_author")
    work_authors = query(
        """
        SELECT wa.work_id, wa.author_id, a.name AS author_name
        FROM data_work_authors wa
        JOIN data_author a ON a.id = wa.author_id
        """
    )
    editions = query(
        """
        SELECT e.id AS edition_id, e.year, e.edition_number, e.title AS edition_title,
               s.name AS series_name
        FROM data_edition e
        LEFT JOIN data_series s ON s.id = e.series_id
        """
    )
    form_rows = query(
        """
        SELECT wf.work_id, f.id AS form_id, f.name AS form_name
        FROM data_work_form wf
        JOIN data_form f ON f.id = wf.form_id
        """
    )
    parent_map = query(
        "SELECT id AS work_id, parent_id FROM data_work WHERE parent_id IS NOT NULL"
    )
    return {
        "work_titles": work_titles,
        "authors": authors,
        "work_authors": work_authors,
        "editions": editions,
        "form_rows": form_rows,
        "parent_map": parent_map,
    }


SERIES_ABBREV = {
    "The Norton Anthology of African American Literature": "NAAAL",
    "The Wiley Blackwell Anthology of African American Literature": "Wiley Blackwell AAL",
    "Afro-American Writing: An Anthology of Prose and Poetry": "Afro-Am. Writing",
    "African American Literature": "AAL Anthology",
    "Cavalcade: Negro American writing from 1760 to the present": "Cavalcade",
}


def _clean_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip()


def edition_label(row: pd.Series) -> str:
    series = _clean_str(row.get("series_name"))
    edition_num = _clean_str(row.get("edition_number"))
    title = _clean_str(row.get("edition_title"))
    year = int(row["year"])
    if series:
        abbrev = SERIES_ABBREV.get(series, series[:30])
        suffix = f" ed.{edition_num}" if edition_num else ""
        return f"{abbrev}{suffix} ({year})"
    return f"{title[:60]} ({year})"


def author_names_per_work_edition(
    raw: pd.DataFrame, authors: pd.DataFrame
) -> pd.DataFrame:
    wa = raw[["work_id", "edition_id", "author_id"]].dropna(subset=["author_id"])
    wa = wa.drop_duplicates()
    wa["author_id"] = wa["author_id"].astype(int)
    wa = wa.merge(authors, on="author_id", how="left")
    grouped = (
        wa.sort_values(["work_id", "edition_id", "author_id"])
        .groupby(["work_id", "edition_id"])["author_name"]
        .apply(lambda s: ", ".join(s.dropna().astype(str)))
        .reset_index()
        .rename(columns={"author_name": "author_names"})
    )
    return grouped


# ── End-to-end ────────────────────────────────────────────────────────────────


def build_model_frame(
    raw: pd.DataFrame, meta: dict[str, pd.DataFrame], exclude_na_form: bool = True
) -> pd.DataFrame:
    spans = compute_work_volume_spans(raw)
    form_by_work = build_form_map(meta["form_rows"], meta["parent_map"])
    selection = build_selection_frame(raw, spans, form_by_work)

    n_before = len(selection)
    selection = selection[selection["pages"] > 0].copy()
    n_after_pages = len(selection)
    selection = selection[selection["form"].notna()].copy()
    n_after_form = len(selection)
    if exclude_na_form:
        selection = selection[selection["form"] != "na"].copy()
    n_final = len(selection)

    print(
        f"[filter] {n_before} appearances → {n_after_pages} after pages>0 → "
        f"{n_after_form} after form attached → {n_final} after na-form filter."
    )

    edition_totals = compute_edition_total_pages(raw)
    span_totals = (
        selection.groupby("edition_id")["pages"].sum().rename("span_total_pages")
    )
    selection = selection.merge(edition_totals, on="edition_id", how="left")
    selection = selection.merge(span_totals, on="edition_id", how="left")
    # Fall back to summed spans when volume metadata is missing for an edition.
    selection["edition_total_pages"] = selection["edition_total_pages"].fillna(
        selection["span_total_pages"]
    )
    selection["pct_pages_of_edition"] = (
        selection["pages"] / selection["edition_total_pages"]
    )

    selection["length_z_by_form"] = length_zscore_by_form(
        selection, value_col="pct_pages_of_edition"
    )

    selection["prior_editions_count"] = prior_editions_count(selection)

    author_rows = (
        raw[["author_id", "edition_id", "anthology_publication_year"]]
        .dropna(subset=["author_id"])
        .drop_duplicates()
        .rename(columns={"anthology_publication_year": "year"})
    )
    author_rows["author_id"] = author_rows["author_id"].astype(int)
    edition_years = (
        raw[["edition_id", "anthology_publication_year"]]
        .drop_duplicates()
        .rename(columns={"anthology_publication_year": "year"})
    )
    author_rate = author_prior_rate_table(author_rows, edition_years)
    n_debut_rows = int(author_rate["author_is_debut"].sum())
    print(f"[author-rate] {n_debut_rows} (author, edition) pairs are debut.")
    selection = attach_author_rate_to_frame(selection, raw, author_rate)

    selection["prob_heuristic"] = heuristic_score(selection)
    selection["prob_logit"] = compute_logit_scores(
        selection, edition_years, raw, author_rate
    )

    selection = enrich_metadata(selection, raw, meta)
    return selection


def enrich_metadata(
    frame: pd.DataFrame, raw: pd.DataFrame, meta: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    titles = meta["work_titles"].set_index("work_id")["work_title"]
    frame["work_title"] = frame["work_id"].map(titles)

    authors_per = author_names_per_work_edition(raw, meta["authors"])
    frame = frame.merge(authors_per, on=["work_id", "edition_id"], how="left")
    frame["author_names"] = frame["author_names"].fillna("")

    ed_meta = meta["editions"].copy()
    ed_meta["edition_label"] = ed_meta.apply(edition_label, axis=1)
    frame = frame.merge(
        ed_meta[["edition_id", "edition_label"]], on="edition_id", how="left"
    )
    return frame


def rank_per_edition(
    frame: pd.DataFrame, top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top_table, bottom_table) — top-N most/least probable per edition.

    Both are sorted by year ascending, then within-edition by composite score.
    Aggregates duplicate (work_id, edition_id) rows from multi-volume editions
    by summing pages and recomputing per-edition share before ranking.
    """
    collapsed = (
        frame.groupby(["work_id", "edition_id"], dropna=False)
        .agg(
            year=("year", "first"),
            edition_label=("edition_label", "first"),
            work_title=("work_title", "first"),
            author_names=("author_names", "first"),
            form=("form", "first"),
            pages=("pages", "sum"),
            length_z_by_form=("length_z_by_form", "mean"),
            prior_editions_count=("prior_editions_count", "first"),
            pct_pages_of_edition=("pct_pages_of_edition", "sum"),
            author_prior_selection_rate=("author_prior_selection_rate", "first"),
            author_is_debut=("author_is_debut", "first"),
            prob_heuristic=("prob_heuristic", "mean"),
            prob_logit=("prob_logit", "mean"),
        )
        .reset_index()
    )

    top_chunks = []
    bottom_chunks = []
    for ed_id, sub in collapsed.groupby("edition_id"):
        sub_sorted_desc = sub.sort_values("prob_heuristic", ascending=False)
        head = sub_sorted_desc.head(top_n).copy()
        head["rank_in_edition"] = range(1, len(head) + 1)
        top_chunks.append(head)

        sub_sorted_asc = sub.sort_values("prob_heuristic", ascending=True)
        tail = sub_sorted_asc.head(top_n).copy()
        tail["rank_in_edition"] = range(1, len(tail) + 1)
        bottom_chunks.append(tail)

    top_table = pd.concat(top_chunks, ignore_index=True).sort_values(
        ["year", "edition_id", "rank_in_edition"]
    )
    bottom_table = pd.concat(bottom_chunks, ignore_index=True).sort_values(
        ["year", "edition_id", "rank_in_edition"]
    )

    return top_table[OUTPUT_COLUMNS].reset_index(drop=True), bottom_table[
        OUTPUT_COLUMNS
    ].reset_index(drop=True)


# ── CLI / output ──────────────────────────────────────────────────────────────


def format_stdout_table(df: pd.DataFrame, header: str) -> str:
    lines = [header, "=" * len(header)]
    for ed_id, sub in df.groupby("edition_id", sort=False):
        label = sub["edition_label"].iloc[0]
        lines.append(f"\n{label}")
        lines.append("-" * len(label))
        for _, r in sub.iterrows():
            author = r["author_names"][:30] if r["author_names"] else "(no author)"
            title = (r["work_title"] or "")[:42]
            length_z = (
                f"{r['length_z_by_form']:+5.2f}"
                if pd.notna(r["length_z_by_form"])
                else " n/a "
            )
            logit = f"{r['prob_logit']:.3f}" if pd.notna(r["prob_logit"]) else " n/a "
            lines.append(
                f"  {int(r['rank_in_edition']):2d}. "
                f"{title:<44} {author:<32} {r['form']:<10} "
                f"pgs={int(r['pages']):3d}  lenZ={length_z}  "
                f"prior={int(r['prior_editions_count']):2d}  "
                f"pct={r['pct_pages_of_edition'] * 100:4.1f}%  "
                f"authR={r['author_prior_selection_rate']:.2f}"
                f"{' (debut)' if bool(r['author_is_debut']) else ''}  "
                f"hScore={r['prob_heuristic']:+5.2f}  pLogit={logit}"
            )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_save_csv_flag(p)
    p.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"selections per edition per table (default {DEFAULT_TOP_N})",
    )
    p.add_argument(
        "--include-na-form",
        dest="exclude_na_form",
        action="store_false",
        default=True,
        help="include rows whose form is 'na' (excluded by default)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw = query(query_path("predictability_over_time"))
    meta = load_metadata()
    frame = build_model_frame(raw, meta, exclude_na_form=args.exclude_na_form)
    top_table, bottom_table = rank_per_edition(frame, top_n=args.top_n)

    if args.save_csv:
        top_table.to_csv(TOP_CSV, index=False)
        bottom_table.to_csv(BOTTOM_CSV, index=False)
        print(f"wrote {TOP_CSV.relative_to(DATA_DIR.parent)} ({len(top_table)} rows)")
        print(
            f"wrote {BOTTOM_CSV.relative_to(DATA_DIR.parent)} "
            f"({len(bottom_table)} rows)"
        )
    else:
        print(
            format_stdout_table(
                top_table, "TABLE 1 — Most probable selections per edition"
            )
        )
        print()
        print(
            format_stdout_table(
                bottom_table, "TABLE 2 — Least probable selections per edition"
            )
        )


if __name__ == "__main__":
    main()
