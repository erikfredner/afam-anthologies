"""
gender_work_consistency.py
--------------------------
Do editors disagree more about *which works* represent women writers than men?

The overarching finding of this project is that editors agree more about which
*authors* to anthologize than which *works* represent them. This script tests
whether that work-level inconsistency differs by author gender, holding literary
form constant (poets vs. poets, fiction writers vs. fiction writers, etc.).

Null hypothesis: there is no difference between men and women in the consistency
of the works chosen to represent them across editions. The substantive argument
for a difference is the "test of time": women were canonized more recently, so
their canonical set of works has had less time to stabilize.

Consistency metric (per author-form unit, between two editions):
    Jaccard(W_i, W_j) = |W_i ∩ W_j| / |W_i ∪ W_j|
averaged over edition pairs. We report Jaccard as the primary statistic plus raw
overlap counts / works-per-edition as descriptive context.

Two edition-pairing schemes are reported separately:
  - cross_series : pairs of editions from *different* series (or standalones).
                   Measures independent editorial agreement; the headline.
  - consecutive  : pairs of editions adjacent in the global AFAM chronology
                   (includes same-series transitions, e.g. NAAAL 1997→2004).
                   The literal "from one anthology to the next".

Each author is split into one *unit per literary form* (Langston Hughes → a
poetry unit and a nonfiction unit) and compared only against same-form units.

Results are reported two ways:
  1. As a whole  — per author-form mean Jaccard; per-form Female-vs-Male tests
                   plus a pooled, form- and test-of-time-controlled regression.
  2. One-to-next — transition-level Jaccard with an author-clustered regression.

Gender is sourced from the database (data_author_genders → data_genders),
not from the legacy gender CSV.

Usage:
    uv run python analysis/gender/gender_work_consistency.py
    uv run python analysis/gender/gender_work_consistency.py --include-excerpts
    uv run python analysis/gender/gender_work_consistency.py --save-csv --min-n 8
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from afam import DATA_DIR
from afam.cli import add_root_works_flag, add_save_csv_flag
from afam.db import query as query_db
from afam.sql import query_path

OUT_UNITS = DATA_DIR / "gender_work_consistency_units.csv"
OUT_TRANS = DATA_DIR / "gender_work_consistency_transitions.csv"

UNKNOWN_FORM = "Unknown"
UNKNOWN_GENDER = "Unknown"
PAIRINGS = ("cross_series", "consecutive")


# ── Preparation helpers (no DB access; unit-testable) ─────────────────────────


def assign_canonical_form(df: pd.DataFrame) -> pd.Series:
    """Map each work_id to one canonical form name.

    Multi-form works collapse to the lowest form_id (stable preference). Works
    without a direct form row inherit their parent's form (one hop). Anything
    still unmapped becomes ``"Unknown"``. Mirrors ``build_form_map`` in
    analysis/predictability/work_selection_probability_model.py.

    Returns a Series indexed by work_id.
    """
    forms = df[["work_id", "form_id", "form_name"]].dropna(subset=["form_id"])
    direct = (
        forms.sort_values("form_id")
        .drop_duplicates("work_id", keep="first")
        .set_index("work_id")["form_name"]
    )
    parent_of = (
        df[["work_id", "parent_id"]].drop_duplicates().set_index("work_id")["parent_id"]
    )
    out = direct.copy()
    missing = parent_of.index.difference(out.index)
    if len(missing):
        inherited = parent_of.loc[missing].map(direct).dropna()
        out = pd.concat([out, inherited])

    all_works = pd.Index(df["work_id"].unique())
    out = out.reindex(all_works).fillna(UNKNOWN_FORM)
    out.name = "form"
    out.index.name = "work_id"
    return out


def collapse_gender(df: pd.DataFrame) -> pd.Series:
    """Map each author_id to one gender label, filling gaps with ``"Unknown"``.

    The SQL query already collapses the many-to-many gender table to one row per
    author, but this defends against an author carrying multiple gender values in
    the frame by taking the first non-null label in sorted order.

    Returns a Series indexed by author_id.
    """
    g = df[["author_id", "gender"]].copy()
    g = g[g["author_id"].notna()]
    g = g.sort_values("gender")
    label = (
        g.dropna(subset=["gender"])
        .drop_duplicates("author_id", keep="first")
        .set_index("author_id")["gender"]
    )
    all_authors = pd.Index(df.loc[df["author_id"].notna(), "author_id"].unique())
    label = label.reindex(all_authors).fillna(UNKNOWN_GENDER)
    label.name = "gender"
    label.index.name = "author_id"
    return label


def _series_identity(series_id, edition_id) -> str:
    """Standalone editions (null series) are their own singleton series."""
    if pd.notna(series_id):
        return f"series_{int(series_id)}"
    return f"standalone_{int(edition_id)}"


def prepare_selections(df: pd.DataFrame, only_root_works: bool) -> pd.DataFrame:
    """Reduce the raw query to one row per (author_id, form, edition_id, work_id).

    Drops rows without an author, optionally drops excerpts, assigns a canonical
    form per work and a single gender per author, and attaches a series identity
    per edition.
    """
    df = df[df["author_id"].notna()].copy()
    if only_root_works:
        df = df[df["parent_id"].isna()].copy()

    form_by_work = assign_canonical_form(df)
    gender_by_author = collapse_gender(df)

    df["author_id"] = df["author_id"].astype(int)
    df["form"] = df["work_id"].map(form_by_work)
    df["gender"] = df["author_id"].map(gender_by_author)
    df["series_identity"] = [
        _series_identity(s, e) for s, e in zip(df["series_id"], df["edition_id"])
    ]

    sel = df[
        [
            "author_id",
            "author_name",
            "gender",
            "form",
            "edition_id",
            "edition_year",
            "series_identity",
            "work_id",
        ]
    ].drop_duplicates(subset=["author_id", "form", "edition_id", "work_id"])
    return sel.reset_index(drop=True)


def global_edition_order(sel: pd.DataFrame) -> dict[int, int]:
    """Map edition_id → chronological rank over all AFAM editions in the frame."""
    eds = (
        sel[["edition_id", "edition_year"]]
        .drop_duplicates()
        .sort_values(["edition_year", "edition_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return {int(eid): i for i, eid in enumerate(eds["edition_id"])}


# ── Pairing + Jaccard ─────────────────────────────────────────────────────────


def _jaccard(s1: set, s2: set) -> float:
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def _unit_pairs(
    edition_ids: list[int],
    edition_order: dict[int, int],
    edition_series: dict[int, str],
    pairing: str,
) -> list[tuple[int, int]]:
    """Edition pairs for one unit under the requested pairing scheme.

    Pairs are returned (earlier, later) by global chronological order.
    """
    eds = sorted(edition_ids, key=lambda e: edition_order[e])
    pairs: list[tuple[int, int]] = []
    for ei, ej in combinations(eds, 2):
        if pairing == "cross_series":
            if edition_series[ei] != edition_series[ej]:
                pairs.append((ei, ej))
        elif pairing == "consecutive":
            if edition_order[ej] == edition_order[ei] + 1:
                pairs.append((ei, ej))
        else:  # pragma: no cover - guarded by caller
            raise ValueError(f"unknown pairing: {pairing}")
    return pairs


def _unit_index(sel: pd.DataFrame):
    """Build per-unit lookup structures keyed by (author_id, form).

    Returns (worksets, info, edition_year, edition_series) where worksets maps a
    unit to {edition_id: set(work_id)} and info carries author_name + gender.
    """
    worksets: dict[tuple[int, str], dict[int, set]] = {}
    info: dict[tuple[int, str], dict] = {}
    for (aid, form), grp in sel.groupby(["author_id", "form"]):
        ed_works: dict[int, set] = {}
        for eid, ework in grp.groupby("edition_id"):
            ed_works[int(eid)] = set(ework["work_id"].astype(int))
        worksets[(int(aid), form)] = ed_works
        first = grp.iloc[0]
        info[(int(aid), form)] = {
            "author_name": first["author_name"],
            "gender": first["gender"],
        }
    edition_year = dict(
        zip(sel["edition_id"].astype(int), sel["edition_year"].astype(int))
    )
    edition_series = dict(zip(sel["edition_id"].astype(int), sel["series_identity"]))
    return worksets, info, edition_year, edition_series


# ── Metric 1: per author-form consistency ─────────────────────────────────────


def compute_unit_consistency(sel: pd.DataFrame, pairing: str) -> pd.DataFrame:
    """One row per (author_id, form): mean Jaccard + descriptive stats.

    Units with no qualifying edition pair under the given pairing are dropped.
    """
    edition_order = global_edition_order(sel)
    worksets, info, edition_year, edition_series = _unit_index(sel)

    rows: list[dict] = []
    for unit, ed_works in worksets.items():
        aid, form = unit
        pairs = _unit_pairs(
            list(ed_works.keys()), edition_order, edition_series, pairing
        )
        if not pairs:
            continue

        jaccs = [_jaccard(ed_works[i], ed_works[j]) for i, j in pairs]
        overlaps = [len(ed_works[i] & ed_works[j]) for i, j in pairs]
        per_ed_counts = [len(s) for s in ed_works.values()]
        years = [edition_year[e] for e in ed_works]

        rows.append(
            {
                "author_id": aid,
                "author_name": info[unit]["author_name"],
                "gender": info[unit]["gender"],
                "form": form,
                "pairing": pairing,
                "n_editions_in_form": len(ed_works),
                "n_pairs": len(pairs),
                "mean_jaccard": float(np.mean(jaccs)),
                "median_jaccard": float(np.median(jaccs)),
                "mean_overlap_count": float(np.mean(overlaps)),
                "mean_works_per_edition": float(np.mean(per_ed_counts)),
                "debut_year": min(years),
                "span_years": max(years) - min(years),
            }
        )

    return pd.DataFrame(rows)


# ── Metric 2: transition-level (one anthology to the next) ────────────────────


def compute_transitions(sel: pd.DataFrame) -> pd.DataFrame:
    """One row per (author_id, form, consecutive-edition-pair) the unit spans.

    Long form behind the author-clustered transition regression.
    """
    edition_order = global_edition_order(sel)
    worksets, info, edition_year, edition_series = _unit_index(sel)

    rows: list[dict] = []
    for unit, ed_works in worksets.items():
        aid, form = unit
        pairs = _unit_pairs(
            list(ed_works.keys()), edition_order, edition_series, "consecutive"
        )
        for ei, ej in pairs:
            ya, yb = edition_year[ei], edition_year[ej]
            rows.append(
                {
                    "author_id": aid,
                    "author_name": info[unit]["author_name"],
                    "gender": info[unit]["gender"],
                    "form": form,
                    "edition_earlier": ei,
                    "edition_later": ej,
                    "year_earlier": ya,
                    "year_later": yb,
                    "midpoint_year": (ya + yb) / 2,
                    "jaccard": _jaccard(ed_works[ei], ed_works[ej]),
                    "overlap_count": len(ed_works[ei] & ed_works[ej]),
                    "n_works_earlier": len(ed_works[ei]),
                }
            )

    return pd.DataFrame(rows)


# ── Statistics ────────────────────────────────────────────────────────────────


def cliffs_delta(female: np.ndarray, male: np.ndarray) -> float:
    """Cliff's delta in [-1, 1]; positive ⇒ women more consistent than men."""
    f = np.asarray(female)
    m = np.asarray(male)
    if len(f) == 0 or len(m) == 0:
        return float("nan")
    diff = f[:, None] - m[None, :]
    return float((np.sign(diff)).sum() / (len(f) * len(m)))


def per_form_tests(units: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Female-vs-Male Mann–Whitney U on mean Jaccard, per form."""
    from scipy.stats import mannwhitneyu

    rows: list[dict] = []
    for form, grp in units.groupby("form"):
        f = grp.loc[grp["gender"] == "Female", "mean_jaccard"].to_numpy()
        m = grp.loc[grp["gender"] == "Male", "mean_jaccard"].to_numpy()
        row = {
            "form": form,
            "n_female": len(f),
            "n_male": len(m),
            "median_female": float(np.median(f)) if len(f) else float("nan"),
            "median_male": float(np.median(m)) if len(m) else float("nan"),
            "cliffs_delta": cliffs_delta(f, m),
            "powered": len(f) >= min_n and len(m) >= min_n,
        }
        if len(f) >= 1 and len(m) >= 1 and (len(f) + len(m)) >= 3:
            u, p = mannwhitneyu(f, m, alternative="two-sided")
            row["U"] = float(u)
            row["p_value"] = float(p)
        else:
            row["U"] = float("nan")
            row["p_value"] = float("nan")
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(
        ["powered", "n_female"], ascending=[False, False]
    )
    return out.reset_index(drop=True)


def van_elteren(units: pd.DataFrame) -> dict:
    """Stratified (van Elteren) Female-vs-Male rank test, strata = form.

    Distribution-free pooled cross-check on the form-controlled gender effect.
    Uses design-free weights w_s = 1/(N_s + 1). Returns the z statistic and a
    two-sided p-value, plus the strata that contributed.
    """
    from scipy.stats import norm

    num = 0.0
    var = 0.0
    used: list[str] = []
    for form, grp in units.groupby("form"):
        sub = grp[grp["gender"].isin(["Female", "Male"])]
        f = sub.loc[sub["gender"] == "Female", "mean_jaccard"].to_numpy()
        m = sub.loc[sub["gender"] == "Male", "mean_jaccard"].to_numpy()
        n_f, n_m, n = len(f), len(m), len(f) + len(m)
        if n_f == 0 or n_m == 0 or n < 2:
            continue
        values = np.concatenate([f, m])
        ranks = pd.Series(values).rank(method="average").to_numpy()
        r_f = ranks[:n_f].sum()
        w = 1.0 / (n + 1)
        exp_f = n_f * (n + 1) / 2
        tie_term = (ranks**2).sum() - n * ((n + 1) / 2) ** 2
        var_s = (n_f * n_m) / (n * (n - 1)) * tie_term
        num += w * (r_f - exp_f)
        var += w**2 * var_s
        used.append(form)

    if var <= 0:
        return {"z": float("nan"), "p_value": float("nan"), "strata": used}
    z = num / np.sqrt(var)
    return {"z": float(z), "p_value": float(2 * norm.sf(abs(z))), "strata": used}


def pooled_regression(units: pd.DataFrame):
    """OLS mean_jaccard ~ gender + form + debut_year + n_editions_in_form.

    Restricted to Male/Female units. The gender[Female] coefficient is the gap
    in work-selection consistency net of form and "test of time" controls.
    Returns the fitted results object, or None if too few rows.
    """
    import statsmodels.formula.api as smf

    sub = units[units["gender"].isin(["Female", "Male"])].copy()
    if len(sub) < 10 or sub["form"].nunique() < 1:
        return None
    sub["is_female"] = (sub["gender"] == "Female").astype(int)
    formula = "mean_jaccard ~ is_female + C(form) + debut_year + n_editions_in_form"
    return smf.ols(formula, data=sub).fit()


def transition_regression(trans: pd.DataFrame):
    """OLS jaccard ~ gender + form + midpoint_year with author-clustered SEs."""
    import statsmodels.formula.api as smf

    sub = trans[trans["gender"].isin(["Female", "Male"])].copy()
    if len(sub) < 10:
        return None
    sub["is_female"] = (sub["gender"] == "Female").astype(int)
    formula = "jaccard ~ is_female + C(form) + midpoint_year"
    return smf.ols(formula, data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["author_id"]}
    )


# ── Reporting ─────────────────────────────────────────────────────────────────


def _print_per_form(units: pd.DataFrame, pairing: str, min_n: int) -> None:
    print(f"\n=== Per-form Female-vs-Male consistency — pairing: {pairing} ===")
    print(
        "Mean Jaccard per author-form unit; Mann–Whitney U (two-sided). "
        f"'powered' = both groups ≥ {min_n}."
    )
    table = per_form_tests(units, min_n)
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(table.round(4).to_string(index=False))


def _print_regression(res, title: str, key: str = "is_female") -> None:
    print(f"\n=== {title} ===")
    if res is None:
        print("(insufficient data to fit)")
        return
    coef = res.params.get(key, float("nan"))
    p = res.pvalues.get(key, float("nan"))
    ci = res.conf_int().loc[key] if key in res.params.index else [float("nan")] * 2
    print(f"n = {int(res.nobs)}")
    print(
        f"{key}: coef = {coef:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  p = {p:.4g}"
    )
    print("(negative ⇒ women's work selections are less consistent than men's)")


def run_report(sel: pd.DataFrame, min_n: int, save_csv: bool) -> None:
    unit_frames = {p: compute_unit_consistency(sel, p) for p in PAIRINGS}
    trans = compute_transitions(sel)

    print("\n################ AS A WHOLE (per author-form) ################")
    for pairing in PAIRINGS:
        units = unit_frames[pairing]
        _print_per_form(units, pairing, min_n)
        _print_regression(
            pooled_regression(units),
            f"Pooled form-controlled regression — pairing: {pairing}",
        )
        ve = van_elteren(units)
        print(
            f"Van Elteren stratified rank test (strata={len(ve['strata'])} forms): "
            f"z = {ve['z']:.3f}, p = {ve['p_value']:.4g}"
        )

    print("\n################ ONE ANTHOLOGY TO THE NEXT (transitions) ################")
    if trans.empty:
        print("(no consecutive-edition transitions found)")
    else:
        summary = (
            trans[trans["gender"].isin(["Female", "Male"])]
            .groupby(["form", "gender"])
            .agg(
                n_transitions=("jaccard", "size"),
                median_jaccard=("jaccard", "median"),
                mean_jaccard=("jaccard", "mean"),
            )
            .reset_index()
        )
        with pd.option_context("display.width", 240):
            print("\nTransition Jaccard by form × gender:")
            print(summary.round(4).to_string(index=False))
        _print_regression(
            transition_regression(trans),
            "Transition regression (author-clustered SEs)",
        )

    if save_csv:
        all_units = pd.concat(unit_frames.values(), ignore_index=True)
        OUT_UNITS.parent.mkdir(parents=True, exist_ok=True)
        all_units.to_csv(OUT_UNITS, index=False)
        trans.to_csv(OUT_TRANS, index=False)
        print(f"\nSaved → {OUT_UNITS}")
        print(f"Saved → {OUT_TRANS}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_root_works_flag(parser)
    add_save_csv_flag(parser)
    parser.add_argument(
        "--min-n",
        type=int,
        default=5,
        help="minimum units per gender in a form to call a per-form test 'powered'",
    )
    args = parser.parse_args()

    raw = query_db(query_path("gender-work-consistency"))
    sel = prepare_selections(raw, only_root_works=args.only_root_works)

    scope = "root works only" if args.only_root_works else "root works + excerpts"
    n_authors = sel["author_id"].nunique()
    print(f"Scope: {scope}")
    print(
        f"{len(sel):,} selections · {n_authors:,} authors · "
        f"{sel['edition_id'].nunique()} editions · {sel['form'].nunique()} forms"
    )

    run_report(sel, min_n=args.min_n, save_csv=args.save_csv)


if __name__ == "__main__":
    main()
