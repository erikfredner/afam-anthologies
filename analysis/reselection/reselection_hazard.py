"""
reselection_hazard.py
---------------------
Discrete-time (edition-indexed) hazard model for author and work reselection
across the African-American Literature anthology editions (1929-2025).

Reselection is reframed as a time-to-event process on the *edition clock* rather
than calendar years: the 26 AFAM editions are the discrete, irregularly spaced
opportunities at which an already-debuted author or work can be reselected. The
unit of analysis is an entity-edition "person-period" row -- one row per
(entity, post-debut edition) -- with:

  * left-truncation  : an entity enters the risk set only at its debut edition,
                       so rows begin at debut + 1;
  * right-censoring   : the last observed edition (2025 NAAAL) has no successor,
                       so a 2025 debut generates zero post-debut rows and is
                       censored at entry -- it contributes exposure context but
                       no event likelihood.

This makes the author-vs-work question a direct comparison of *hazards* net of
exposure (the number of editions at risk = the number of person-period rows),
which is exactly the reviewer worry that older entities have had more chances.

Two event models are supported via --event-model:

  first      single-event survival -- each entity's rows stop at (and include)
             its first reselection; censored if it never fires. The classic
             survival framing; extends years_to_first_reselection.
  recurrent  every post-debut edition stays at risk and reselection may recur;
             the time-varying prior_selections covariate carries the
             state-dependence / Matthew-effect signal.

Models (statsmodels only -- no lifelines dependency):

  * Combined discrete-time hazard GLM pooling authors and works:
        event ~ C(entity_type) + C(t_since_debut) + C(debut_decade)
                + prior_selections
    fit with a complementary-log-log link (Prentice-Gloeckler grouped
    proportional hazards; exp(coef) are hazard ratios) and, for robustness, a
    logit link (pooled-logistic discrete hazard). Cluster-robust standard errors
    by entity_id. The C(entity_type) hazard ratio is the headline,
    exposure-adjusted author-vs-work verdict.
  * A works-only model that additionally adjusts for literary form C(form_name).
  * A nonparametric life-table (discrete Kaplan-Meier) of survival S(t) and
    hazard h(t) by editions-since-debut, separately for authors and works, with
    Greenwood-formula confidence intervals -- the empirical curves the figures
    draw.

Outputs:
  data/reselection_hazard_person_period.csv   (with --save-csv)
  data/reselection_hazard_coefficients.csv    (with --save-csv)
  data/reselection_hazard_lifetable.csv        (with --save-csv)

Usage:
    uv run python analysis/reselection/reselection_hazard.py
    uv run python analysis/reselection/reselection_hazard.py --event-model recurrent
    uv run python analysis/reselection/reselection_hazard.py --scope cross-series
    uv run python analysis/reselection/reselection_hazard.py --save-csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from afam import DATA_DIR
from afam.cli import add_root_works_flag, add_save_csv_flag
from afam.db import query
from afam.sql import query_path

# The edition-ordering and entry-group helpers are shared with the debut script
# so debut/exposure definitions stay identical across the reselection analyses.
from author_vs_work_debut_reselection import (
    add_entry_group,
    build_author_rows,
    build_edition_table,
    build_work_rows,
)

OUT_PP_CSV = DATA_DIR / "reselection_hazard_person_period.csv"
OUT_COEF_CSV = DATA_DIR / "reselection_hazard_coefficients.csv"
OUT_LIFETABLE_CSV = DATA_DIR / "reselection_hazard_lifetable.csv"

EVENT_MODELS = ("first", "recurrent")
SCOPES = ("all", "cross-series")


# -- Data loading ---------------------------------------------------------------


def load_data(only_root_works: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load AFAM appearances, the full edition table, and a work->form map.

    The edition table is built from the full data *before* the optional root-work
    filter so that opportunity denominators reflect the whole AFAM sequence.
    """
    raw = query(query_path("works-per-afam-edition"))
    raw["anthology_publication_year"] = raw["anthology_publication_year"].astype(int)
    raw = add_entry_group(raw)
    all_editions = build_edition_table(raw)
    form_map = load_form_map(raw)

    if only_root_works:
        raw = raw[raw["parent_id"].isna()].copy()

    return raw, all_editions, form_map


def load_form_map(raw: pd.DataFrame) -> pd.Series:
    """Map work_id -> form_name, inheriting a parent work's form for excerpts."""
    forms = query(query_path("work-forms-afam"))
    base = (
        forms.dropna(subset=["work_id"])
        .assign(work_id=lambda d: d["work_id"].astype("int64"))
        .set_index("work_id")["form_name"]
    )

    # Excerpts (parent_id set) without their own form inherit the parent's form.
    parents = (
        raw.dropna(subset=["work_id", "parent_id"])[["work_id", "parent_id"]]
        .astype({"work_id": "int64", "parent_id": "int64"})
        .drop_duplicates("work_id")
    )
    resolved = dict(base)
    for work_id, parent_id in parents.itertuples(index=False):
        if work_id not in resolved and parent_id in resolved:
            resolved[work_id] = resolved[parent_id]
    return pd.Series(resolved, name="form_name")


# -- Person-period construction -------------------------------------------------


def build_person_period(
    entity_rows: pd.DataFrame,
    id_col: str,
    entity_type: str,
    all_editions: pd.DataFrame,
    *,
    event_model: str = "first",
    scope: str = "all",
    form_map: pd.Series | None = None,
) -> pd.DataFrame:
    """Return one row per (entity, post-debut edition) -- the risk set table.

    Rows begin at the edition after debut (left-truncation). ``event`` flags
    whether the entity was selected in that edition. For ``event_model="first"``
    each entity's rows are truncated at and including its first event. For
    ``scope="cross-series"`` only editions outside the debut's entry_group are at
    risk, mirroring the cross-series sensitivity of the debut analysis.
    """
    cols = [
        "entity_id",
        "entity_type",
        "edition_id",
        "edition_order",
        "year",
        "t_since_debut",
        "event",
        "debut_year",
        "debut_decade",
        "prior_selections",
        "entry_group",
        "form_name",
    ]
    if entity_rows.empty:
        return pd.DataFrame(columns=cols)

    ed_lookup = all_editions[
        ["edition_id", "anthology_publication_year", "edition_order", "entry_group"]
    ].rename(columns={"anthology_publication_year": "year"})

    appearances = (
        entity_rows[[id_col, "edition_id"]]
        .dropna(subset=[id_col, "edition_id"])
        .drop_duplicates([id_col, "edition_id"])
        .merge(ed_lookup, on="edition_id", how="inner")
        .sort_values([id_col, "edition_order"], kind="mergesort")
    )
    if appearances.empty:
        return pd.DataFrame(columns=cols)

    all_orders = all_editions["edition_order"].to_numpy(dtype=int)
    order_to_group = all_editions.set_index("edition_order")["entry_group"].to_dict()
    order_to_year = all_editions.set_index("edition_order")[
        "anthology_publication_year"
    ].to_dict()
    order_to_edition = all_editions.set_index("edition_order")["edition_id"].to_dict()

    selected_orders = appearances.groupby(id_col)["edition_order"].apply(set).to_dict()
    debut = appearances.groupby(id_col).first()

    records: list[dict] = []
    for entity_id, drow in debut.iterrows():
        debut_order = int(drow["edition_order"])
        debut_year = int(drow["year"])
        debut_group = drow["entry_group"]
        entity_selected = selected_orders.get(entity_id, set())

        later_orders = [int(o) for o in all_orders if int(o) > debut_order]
        if scope == "cross-series":
            later_orders = [
                o for o in later_orders if order_to_group.get(o) != debut_group
            ]

        prior_selections = 0
        for order in later_orders:
            event = 1 if order in entity_selected else 0
            records.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "edition_id": order_to_edition[order],
                    "edition_order": order,
                    "year": int(order_to_year[order]),
                    "t_since_debut": order - debut_order,
                    "event": event,
                    "debut_year": debut_year,
                    "debut_decade": (debut_year // 10) * 10,
                    "prior_selections": prior_selections,
                    "entry_group": debut_group,
                    "form_name": (
                        form_map.get(entity_id) if form_map is not None else None
                    ),
                }
            )
            prior_selections += event
            if event_model == "first" and event == 1:
                break

    return pd.DataFrame(records, columns=cols)


def build_all_person_periods(
    raw: pd.DataFrame,
    all_editions: pd.DataFrame,
    form_map: pd.Series,
    *,
    event_model: str = "first",
    scope: str = "all",
) -> pd.DataFrame:
    """Pooled author + work person-period table for the combined hazard model."""
    authors = build_person_period(
        build_author_rows(raw),
        "author_id",
        "author",
        all_editions,
        event_model=event_model,
        scope=scope,
        form_map=None,
    )
    works = build_person_period(
        build_work_rows(raw),
        "work_id",
        "work",
        all_editions,
        event_model=event_model,
        scope=scope,
        form_map=form_map,
    )
    return pd.concat([authors, works], ignore_index=True)


# -- Nonparametric life table ---------------------------------------------------


def lifetable_survival(pp_df: pd.DataFrame) -> pd.DataFrame:
    """Discrete Kaplan-Meier life table by t_since_debut, per entity_type.

    Expects a *first-event* person-period table (each entity contributes a row
    for every edition it was at risk, ending at its first event). Returns one row
    per (entity_type, t) with the risk set, events, hazard h(t), survival S(t),
    and a Greenwood-based 95% CI for S(t).
    """
    cols = [
        "entity_type",
        "t_since_debut",
        "n_at_risk",
        "n_events",
        "hazard",
        "survival",
        "survival_lo",
        "survival_hi",
    ]
    if pp_df.empty:
        return pd.DataFrame(columns=cols)

    out: list[dict] = []
    for entity_type, grp in pp_df.groupby("entity_type"):
        agg = (
            grp.groupby("t_since_debut")["event"]
            .agg(n_at_risk="count", n_events="sum")
            .reset_index()
            .sort_values("t_since_debut")
        )
        survival = 1.0
        greenwood_sum = 0.0
        for _, row in agg.iterrows():
            n = int(row["n_at_risk"])
            d = int(row["n_events"])
            hazard = d / n if n else float("nan")
            survival *= 1.0 - hazard
            if n > d:
                greenwood_sum += d / (n * (n - d))
            se = survival * np.sqrt(greenwood_sum)
            out.append(
                {
                    "entity_type": entity_type,
                    "t_since_debut": int(row["t_since_debut"]),
                    "n_at_risk": n,
                    "n_events": d,
                    "hazard": hazard,
                    "survival": survival,
                    "survival_lo": max(0.0, survival - 1.96 * se),
                    "survival_hi": min(1.0, survival + 1.96 * se),
                }
            )
    return pd.DataFrame(out, columns=cols)


# -- Hazard regression ----------------------------------------------------------


def _fit_glm(formula: str, data: pd.DataFrame, link: str):
    """Fit a discrete-time hazard GLM with cluster-robust SEs by entity.

    Returns the fitted results, falling back to a non-robust fit if a given
    statsmodels build rejects clustered covariance for this family.
    """
    family = sm.families.Binomial(
        link=sm.families.links.CLogLog()
        if link == "cloglog"
        else sm.families.links.Logit()
    )
    model = sm.GLM.from_formula(formula, data=data, family=family)
    try:
        return model.fit(
            cov_type="cluster", cov_kwds={"groups": data["entity_id"].astype(str)}
        )
    except (ValueError, TypeError):
        return model.fit()


def fit_hazard_models(pp_df: pd.DataFrame) -> dict:
    """Fit the combined and works-only discrete-time hazard models.

    Returns a dict with fitted result objects and a tidy coefficient/HR table.
    """
    results: dict = {}
    if pp_df.empty:
        results["coefficients"] = pd.DataFrame(
            columns=[
                "model",
                "link",
                "term",
                "coef",
                "hazard_ratio",
                "ci_lo",
                "ci_hi",
                "p",
            ]
        )
        return results

    data = pp_df.copy()
    # Time-since-debut levels and decades with no variation are handled by patsy.
    # prior_selections is a time-varying covariate, but in the first-event model
    # it is identically zero (rows stop at the first event before any prior
    # selection accrues), which would be a degenerate all-zero regressor. Include
    # it only when it actually varies (the recurrent model).
    terms = ["C(t_since_debut)", "C(debut_decade)"]
    if data["prior_selections"].nunique() > 1:
        terms.append("prior_selections")
    base_terms = " + ".join(terms)
    combined_formula = f"event ~ C(entity_type) + {base_terms}"

    coef_rows: list[dict] = []
    for link in ("cloglog", "logit"):
        res = _fit_glm(combined_formula, data, link)
        results[f"combined_{link}"] = res
        coef_rows.extend(_tidy(res, model="combined", link=link))

    works = data[data["entity_type"] == "work"].copy()
    if not works.empty and works["form_name"].notna().any():
        works_formula = f"event ~ {base_terms} + C(form_name)"
        try:
            res_w = _fit_glm(works_formula, works, "cloglog")
            results["works_cloglog"] = res_w
            coef_rows.extend(_tidy(res_w, model="works", link="cloglog"))
        except Exception:  # noqa: BLE001 - degenerate form strata are non-fatal
            pass

    results["coefficients"] = pd.DataFrame(coef_rows)
    return results


def _tidy(res, *, model: str, link: str) -> list[dict]:
    """Flatten a fitted result into coefficient rows with hazard ratios."""
    params = res.params
    conf = res.conf_int()
    pvals = res.pvalues
    rows: list[dict] = []
    for term in params.index:
        coef = float(params[term])
        lo, hi = float(conf.loc[term, 0]), float(conf.loc[term, 1])
        # A separated baseline dummy (a sparse t_since_debut or debut_decade
        # level) can send a CI bound to +/-inf; exp() then overflows. That is a
        # legitimate "unbounded" interval, so silence the warning rather than
        # clip away the signal.
        with np.errstate(over="ignore"):
            rows.append(
                {
                    "model": model,
                    "link": link,
                    "term": term,
                    "coef": coef,
                    "hazard_ratio": float(np.exp(coef)),
                    "ci_lo": float(np.exp(lo)),
                    "ci_hi": float(np.exp(hi)),
                    "p": float(pvals[term]),
                }
            )
    return rows


# -- Reporting ------------------------------------------------------------------


def _entity_type_term(coef_df: pd.DataFrame) -> pd.DataFrame:
    return coef_df[coef_df["term"].str.contains("entity_type", regex=False)]


def print_summary(pp_df: pd.DataFrame, results: dict, lifetable: pd.DataFrame) -> None:
    n_entities = pp_df.groupby("entity_type")["entity_id"].nunique()
    n_events = pp_df.groupby("entity_type")["event"].sum()
    print("Person-period risk set")
    print("-" * 60)
    for etype in ("author", "work"):
        ents = int(n_entities.get(etype, 0))
        rows = int((pp_df["entity_type"] == etype).sum())
        evs = int(n_events.get(etype, 0))
        print(
            f"  {etype:7s}: {ents:5d} entities at risk, "
            f"{rows:6d} entity-editions, {evs:5d} events"
        )

    coef_df = results.get("coefficients", pd.DataFrame())
    et = _entity_type_term(coef_df)
    if not et.empty:
        print("\nAuthor-vs-work hazard ratio (reference = author)")
        print("-" * 60)
        for _, row in et.iterrows():
            print(
                f"  [{row['link']:7s}] HR = {row['hazard_ratio']:.3f} "
                f"(95% CI {row['ci_lo']:.3f}-{row['ci_hi']:.3f}), p = {row['p']:.4g}"
            )
        print(
            "  HR > 1: works reselected at a higher per-edition hazard than\n"
            "  authors, net of editions-at-risk, debut era, and prior selections."
        )

    if not lifetable.empty:
        print("\nLife-table survival S(t) = P(not yet reselected after t editions)")
        print("-" * 60)
        for etype, grp in lifetable.groupby("entity_type"):
            head = grp.head(6)
            curve = ", ".join(
                f"t{int(r.t_since_debut)}={r.survival:.2f}" for r in head.itertuples()
            )
            print(f"  {etype:7s}: {curve}")


# -- CLI ------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_works_flag(parser)
    add_save_csv_flag(parser)
    parser.add_argument(
        "--event-model",
        choices=EVENT_MODELS,
        default="first",
        help="first: time-to-first-reselection; recurrent: repeated reselection.",
    )
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="all",
        help="all: any later edition is at risk; cross-series: only later "
        "editions outside the debut's series.",
    )
    args = parser.parse_args()

    raw, all_editions, form_map = load_data(args.only_root_works)
    pp_df = build_all_person_periods(
        raw,
        all_editions,
        form_map,
        event_model=args.event_model,
        scope=args.scope,
    )
    results = fit_hazard_models(pp_df)

    # The life table is a single-event (survival) construct; build a first-event
    # table for it regardless of the regression's event model.
    if args.event_model == "first":
        pp_first = pp_df
    else:
        pp_first = build_all_person_periods(
            raw, all_editions, form_map, event_model="first", scope=args.scope
        )
    lifetable = lifetable_survival(pp_first)

    if args.save_csv:
        pp_df.to_csv(OUT_PP_CSV, index=False)
        results["coefficients"].to_csv(OUT_COEF_CSV, index=False)
        lifetable.to_csv(OUT_LIFETABLE_CSV, index=False)
        print(f"Wrote {OUT_PP_CSV}")
        print(f"Wrote {OUT_COEF_CSV}")
        print(f"Wrote {OUT_LIFETABLE_CSV}")
        return

    print(
        f"Event model: {args.event_model}   Scope: {args.scope}   "
        f"Root works only: {args.only_root_works}\n"
    )
    print_summary(pp_df, results, lifetable)


if __name__ == "__main__":
    main()
