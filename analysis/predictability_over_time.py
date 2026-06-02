"""
predictability_over_time.py
---------------------------
Tests whether author and work selections in AFAM-tagged anthologies
become more *predictable* from prior editions as we approach the present
(1929 → 2025), under several complementary metrics and subgroups.

For each AFAM edition E (sorted by year, excluding the earliest), the script
treats every earlier AFAM edition as the prior pool and computes:

  • base_rate                 — chance prob. of any prior entity landing in E
  • fraction_repeat           — share of E drawn from any prior entity
  • fraction_frequent_canon   — share of E from entities with ≥3 prior eds
  • recall_at_K               — top-K prior canon's recall in E (K = slots)
  • lift_over_chance          — fraction_repeat / base_rate
  • logit_auc / brier / R²    — ROC-AUC, Brier, McFadden R² of
                                  in_E ~ log(1 + prior_count) over prior pool
  • gini_prior_count_in_E     — concentration of E on the canonical tail

Slot-weighted variants count each work / author equally.  Page-weighted
variants weight each work by its TOC span and roll those spans up to authors.

Subgroups: gender (CSV-backed, authors only), birth cohort, debut-era, and
root works vs excerpts (works only).

Usage:
    uv run python analysis/predictability_over_time.py
    uv run python analysis/predictability_over_time.py --mode authors --save-csv
    uv run python analysis/predictability_over_time.py --include-excerpts
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import statsmodels.api as sm
from dotenv import dotenv_values
from numpy.linalg import LinAlgError
from scipy.stats import linregress, norm, spearmanr
from sklearn.metrics import roc_auc_score
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError

# Logit fits routinely hit perfect/near-perfect separation on small pools; we
# fall back to L2-regularised estimates via fit_logit_safe, so silence the noise.
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
QUERIES = ROOT / "queries"
DATA_DIR = ROOT / "data"
GENDER_CSV = DATA_DIR / "2026-03-17 af am anthology authors with genders.csv"

FREQUENT_CANON_THRESHOLD = 3
MIN_SUBGROUP_SIZE = 5

SubgroupTag = tuple[str, str]  # (dimension, level)


# ── DB helpers (parse_db_params / query_db copied from logistic_reselection.py) ─


def parse_db_params(env_file: Path = ENV_FILE) -> dict[str, str]:
    env = dotenv_values(env_file)
    raw = env["DATABASE_URL"]
    return {
        "host": re.search(r"-h\s+(\S+)", raw).group(1),
        "user": re.search(r"-U\s+(\S+)", raw).group(1),
        "password": re.search(r"PGPASSWORD=(\S+)", raw).group(1),
        "dbname": raw.split()[-1],
    }


def query_db(params: dict[str, str], sql_file: Path) -> pd.DataFrame:
    sql = sql_file.read_text()
    with psycopg.connect(**params) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc.name for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


# ── Page spans ────────────────────────────────────────────────────────────────


def compute_work_volume_spans(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (work_id, volume_id) with the work's page span in that volume.

    Span = toc_next - toc_page when both present (and positive).  When toc_next
    is NULL, falls back to last_toc_page - toc_page + 1 (treating the work as
    the last entry in its volume).
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


# ── Long entity-per-edition tables ────────────────────────────────────────────


def build_work_per_edition(
    raw: pd.DataFrame, work_volume_spans: pd.DataFrame
) -> pd.DataFrame:
    """One row per (work_id, edition_id) with year, parent_id, span (pages)."""
    base = raw[
        [
            "work_id",
            "volume_id",
            "edition_id",
            "anthology_publication_year",
            "parent_id",
        ]
    ].drop_duplicates(["work_id", "volume_id", "edition_id"])
    base = base.merge(work_volume_spans, on=["work_id", "volume_id"], how="left")
    grouped = (
        base.groupby(
            ["work_id", "edition_id", "anthology_publication_year"], dropna=False
        )
        .agg(span=("span", "sum"), parent_id=("parent_id", "first"))
        .reset_index()
        .rename(columns={"anthology_publication_year": "year"})
    )
    grouped["slot_w"] = 1
    grouped["page_w"] = grouped["span"].astype(int)
    return grouped


def build_author_per_edition(
    raw: pd.DataFrame, work_volume_spans: pd.DataFrame
) -> pd.DataFrame:
    """One row per (author_id, edition_id) with year, summed page span, birth_year.

    Pages credited to an author are the sum over unique (work_id, volume_id)
    pairs in that edition where the author appears.  Multi-author works
    therefore credit each author for the full span.
    """
    base = raw.dropna(subset=["author_id"]).copy()
    base["author_id"] = base["author_id"].astype(int)
    base = base[
        [
            "author_id",
            "work_id",
            "volume_id",
            "edition_id",
            "anthology_publication_year",
            "author_birth_year",
        ]
    ].drop_duplicates(["author_id", "work_id", "volume_id", "edition_id"])
    base = base.merge(work_volume_spans, on=["work_id", "volume_id"], how="left")
    grouped = (
        base.groupby(
            ["author_id", "edition_id", "anthology_publication_year"], dropna=False
        )
        .agg(span=("span", "sum"), birth_year=("author_birth_year", "first"))
        .reset_index()
        .rename(columns={"anthology_publication_year": "year"})
    )
    grouped["slot_w"] = 1
    grouped["page_w"] = grouped["span"].astype(int)
    return grouped


# ── Subgroup tagging ──────────────────────────────────────────────────────────


def load_gender_map(gender_csv: Path) -> dict[int, str]:
    if not gender_csv.exists():
        return {}
    df = pd.read_csv(gender_csv, dtype=str, na_filter=False)
    df = df[df["gender"].str.strip() != ""]
    df["gender"] = df["gender"].str.strip().str.lower()
    df["author_id"] = pd.to_numeric(df["author_id"], errors="coerce")
    df = df.dropna(subset=["author_id"])
    df["author_id"] = df["author_id"].astype(int)
    return dict(df.drop_duplicates("author_id")[["author_id", "gender"]].values)


def birth_cohort(year: float | None) -> str:
    if year is None or (isinstance(year, float) and math.isnan(year)):
        return "unknown"
    y = int(year)
    if y <= 1900:
        return "≤1900"
    if y <= 1950:
        return "1901-1950"
    return ">1950"


def debut_era(year: int) -> str:
    if year < 1970:
        return "pre-1970"
    if year < 1996:
        return "1970-1995"
    return "post-1995"


def attach_subgroups_authors(
    author_per_ed: pd.DataFrame, gender_map: dict[int, str]
) -> pd.DataFrame:
    df = author_per_ed.copy()
    df["sg_gender"] = df["author_id"].map(gender_map).fillna("unknown")
    df["sg_birth_cohort"] = df["birth_year"].apply(birth_cohort)
    debut = df.groupby("author_id")["year"].min()
    df["sg_debut_era"] = df["author_id"].map(debut).map(debut_era)
    return df


def attach_subgroups_works(work_per_ed: pd.DataFrame) -> pd.DataFrame:
    df = work_per_ed.copy()
    debut = df.groupby("work_id")["year"].min()
    df["sg_debut_era"] = df["work_id"].map(debut).map(debut_era)
    df["sg_root_excerpt"] = df["parent_id"].apply(
        lambda x: (
            "root"
            if x is None or (isinstance(x, float) and math.isnan(x))
            else "excerpt"
        )
    )
    return df


# ── Stats helpers ─────────────────────────────────────────────────────────────


def gini(values: np.ndarray) -> float:
    """Gini coefficient on a non-negative array.  Returns 0 if all equal/empty."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0 or v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * v) / (n * v.sum())) - (n + 1) / n)


def fit_logit_safe(y: pd.Series, x: pd.Series):
    X = sm.add_constant(x.rename("x"), has_constant="add")
    model = sm.Logit(y, X)
    try:
        return model.fit(disp=False)
    except (LinAlgError, ValueError, PerfectSeparationError):
        return model.fit_regularized(alpha=1e-6, L1_wt=0.0, disp=False)


def mann_kendall(x: np.ndarray) -> tuple[float, float]:
    """Return (S, two-sided p) for the Mann-Kendall trend test (no ties correction)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 4:
        return float("nan"), float("nan")
    s = 0.0
    for i in range(n - 1):
        s += np.sign(x[i + 1 :] - x[i]).sum()
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s <= 0:
        return float(s), float("nan")
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = float(2 * (1 - norm.cdf(abs(z))))
    return float(s), p


# ── Per-edition metric computation ────────────────────────────────────────────


def compute_edition_metrics(
    ent_per_ed: pd.DataFrame,
    entity_col: str,
    year: int,
    edition_id: int,
) -> dict | None:
    """Compute predictability metrics for one edition under one subgroup slice.

    `ent_per_ed` already restricted to the subgroup level if applicable.
    Returns None if E has fewer than MIN_SUBGROUP_SIZE entities or no prior pool.
    """
    e_rows = ent_per_ed[ent_per_ed["edition_id"] == edition_id]
    if e_rows.empty:
        return None
    e_entities = set(e_rows[entity_col].astype(int))
    if len(e_entities) < MIN_SUBGROUP_SIZE:
        return None

    e_per_entity_pages = e_rows.groupby(entity_col)["page_w"].sum()
    e_slot_total = len(e_entities)
    e_page_total = float(e_per_entity_pages.sum())

    prior = ent_per_ed[ent_per_ed["year"] < year]
    if prior.empty:
        return None
    prior_counts = prior.groupby(entity_col)["edition_id"].nunique()
    prior_pool = set(prior_counts.index.astype(int))

    in_e_and_prior = e_entities & prior_pool
    fraction_repeat_slot = len(in_e_and_prior) / e_slot_total
    page_repeat = float(
        e_per_entity_pages[e_per_entity_pages.index.isin(in_e_and_prior)].sum()
    )
    fraction_repeat_page = (
        (page_repeat / e_page_total) if e_page_total > 0 else float("nan")
    )

    frequent = set(
        prior_counts[prior_counts >= FREQUENT_CANON_THRESHOLD].index.astype(int)
    )
    in_e_and_freq = e_entities & frequent
    fraction_freq_slot = len(in_e_and_freq) / e_slot_total
    page_freq = float(
        e_per_entity_pages[e_per_entity_pages.index.isin(in_e_and_freq)].sum()
    )
    fraction_freq_page = (
        (page_freq / e_page_total) if e_page_total > 0 else float("nan")
    )

    k = e_slot_total
    top_k = set(prior_counts.sort_values(ascending=False).head(k).index.astype(int))
    recall_at_k = len(top_k & e_entities) / k

    base_rate = e_slot_total / len(prior_pool) if prior_pool else float("nan")
    lift = (
        (fraction_repeat_slot / base_rate)
        if base_rate and base_rate > 0
        else float("nan")
    )

    pool_df = pd.DataFrame({"entity": sorted(prior_pool)})
    pool_df["prior_count"] = pool_df["entity"].map(prior_counts).astype(int)
    pool_df["in_e"] = pool_df["entity"].isin(e_entities).astype(int)
    pool_df = pool_df[pool_df["prior_count"] >= 1]

    auc = brier = mcfadden = float("nan")
    if len(pool_df) >= 10 and pool_df["in_e"].nunique() == 2:
        x = np.log1p(pool_df["prior_count"].to_numpy(dtype=float))
        y_arr = pool_df["in_e"].to_numpy(dtype=int)
        try:
            result = fit_logit_safe(pd.Series(y_arr), pd.Series(x))
            probs = np.asarray(result.predict(), dtype=float)
            auc = float(roc_auc_score(y_arr, probs))
            brier = float(np.mean((probs - y_arr) ** 2))
            if (
                hasattr(result, "llf")
                and hasattr(result, "llnull")
                and result.llnull != 0
            ):
                mcfadden = 1.0 - float(result.llf / result.llnull)
        except Exception:
            pass

    e_prior_counts = (
        pd.Series(list(e_entities)).map(prior_counts).fillna(0).astype(int).to_numpy()
    )
    gini_e = gini(e_prior_counts.astype(float))

    return {
        "edition_id": edition_id,
        "year": year,
        "slots": e_slot_total,
        "pages": e_page_total,
        "prior_pool_size": len(prior_pool),
        "base_rate": base_rate,
        "fraction_repeat_slot": fraction_repeat_slot,
        "fraction_repeat_page": fraction_repeat_page,
        "fraction_frequent_canon_slot": fraction_freq_slot,
        "fraction_frequent_canon_page": fraction_freq_page,
        "recall_at_k": recall_at_k,
        "lift_over_chance": lift,
        "logit_auc": auc,
        "logit_brier": brier,
        "mcfadden_r2": mcfadden,
        "gini_prior_count_in_e": gini_e,
    }


# ── Subgroup iteration ────────────────────────────────────────────────────────


SUBGROUP_DIMS_AUTHORS = [
    ("all", lambda df: [("all", "all", df)]),
    (
        "gender",
        lambda df: [
            (dim, lvl, df[df["sg_gender"] == lvl])
            for lvl in sorted(df["sg_gender"].unique())
            for dim in ["gender"]
        ],
    ),
    (
        "birth_cohort",
        lambda df: [
            (dim, lvl, df[df["sg_birth_cohort"] == lvl])
            for lvl in ["≤1900", "1901-1950", ">1950", "unknown"]
            for dim in ["birth_cohort"]
            if (df["sg_birth_cohort"] == lvl).any()
        ],
    ),
    (
        "debut_era",
        lambda df: [
            (dim, lvl, df[df["sg_debut_era"] == lvl])
            for lvl in ["pre-1970", "1970-1995", "post-1995"]
            for dim in ["debut_era"]
            if (df["sg_debut_era"] == lvl).any()
        ],
    ),
]

SUBGROUP_DIMS_WORKS = [
    ("all", lambda df: [("all", "all", df)]),
    (
        "debut_era",
        lambda df: [
            (dim, lvl, df[df["sg_debut_era"] == lvl])
            for lvl in ["pre-1970", "1970-1995", "post-1995"]
            for dim in ["debut_era"]
            if (df["sg_debut_era"] == lvl).any()
        ],
    ),
    (
        "root_excerpt",
        lambda df: [
            (dim, lvl, df[df["sg_root_excerpt"] == lvl])
            for lvl in ["root", "excerpt"]
            for dim in ["root_excerpt"]
            if (df["sg_root_excerpt"] == lvl).any()
        ],
    ),
]


def iter_subgroup_slices(ent_per_ed: pd.DataFrame, entity_type: str):
    dims = SUBGROUP_DIMS_AUTHORS if entity_type == "authors" else SUBGROUP_DIMS_WORKS
    for _, slicer in dims:
        for dim, level, sub_df in slicer(ent_per_ed):
            if len(sub_df) == 0:
                continue
            yield dim, level, sub_df


def compute_all_metrics(ent_per_ed: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    """One row per (entity_type, subgroup_dim, subgroup_level, edition).

    Iterates editions ordered by year (skipping the earliest, which has no prior).
    """
    entity_col = "author_id" if entity_type == "authors" else "work_id"
    edition_meta = (
        ent_per_ed[["edition_id", "year"]]
        .drop_duplicates()
        .sort_values(["year", "edition_id"])
        .reset_index(drop=True)
    )
    earliest_year = int(edition_meta["year"].min())

    rows: list[dict] = []
    for dim, level, sub_df in iter_subgroup_slices(ent_per_ed, entity_type):
        for _, ed in edition_meta.iterrows():
            year = int(ed["year"])
            if year == earliest_year:
                continue
            metrics = compute_edition_metrics(
                sub_df, entity_col, year, int(ed["edition_id"])
            )
            if metrics is None:
                continue
            metrics.update(
                {
                    "entity_type": entity_type,
                    "subgroup_dim": dim,
                    "subgroup_level": level,
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


# ── Trend tests ───────────────────────────────────────────────────────────────


TREND_METRICS = [
    "base_rate",
    "fraction_repeat_slot",
    "fraction_repeat_page",
    "fraction_frequent_canon_slot",
    "fraction_frequent_canon_page",
    "recall_at_k",
    "lift_over_chance",
    "logit_auc",
    "logit_brier",
    "mcfadden_r2",
    "gini_prior_count_in_e",
]


def compute_trends(per_edition: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = per_edition.groupby(["entity_type", "subgroup_dim", "subgroup_level"])
    for (etype, dim, level), grp in grouped:
        grp_sorted = grp.sort_values("year")
        for metric in TREND_METRICS:
            series = grp_sorted[["year", metric]].dropna()
            if len(series) < 4:
                rows.append(
                    {
                        "entity_type": etype,
                        "subgroup_dim": dim,
                        "subgroup_level": level,
                        "metric": metric,
                        "n_editions": len(series),
                        "ols_slope": float("nan"),
                        "ols_p": float("nan"),
                        "spearman_rho": float("nan"),
                        "spearman_p": float("nan"),
                        "mk_s": float("nan"),
                        "mk_p": float("nan"),
                    }
                )
                continue
            try:
                lr = linregress(series["year"], series[metric])
                slope, lr_p = float(lr.slope), float(lr.pvalue)
            except Exception:
                slope, lr_p = float("nan"), float("nan")
            try:
                rho, rho_p = spearmanr(series["year"], series[metric])
                rho, rho_p = float(rho), float(rho_p)
            except Exception:
                rho, rho_p = float("nan"), float("nan")
            mk_s, mk_p = mann_kendall(series[metric].to_numpy())
            rows.append(
                {
                    "entity_type": etype,
                    "subgroup_dim": dim,
                    "subgroup_level": level,
                    "metric": metric,
                    "n_editions": len(series),
                    "ols_slope": slope,
                    "ols_p": lr_p,
                    "spearman_rho": rho,
                    "spearman_p": rho_p,
                    "mk_s": mk_s,
                    "mk_p": mk_p,
                }
            )
    return pd.DataFrame(rows)


# ── Output ────────────────────────────────────────────────────────────────────


SUMMARY_METRICS = [
    "logit_auc",
    "recall_at_k",
    "fraction_repeat_slot",
    "fraction_frequent_canon_slot",
    "lift_over_chance",
    "gini_prior_count_in_e",
]


def print_trend_summary(trends: pd.DataFrame) -> None:
    print("\n===== TREND SUMMARY (overall subgroup only) =====")
    print("Positive ρ / S / OLS slope = metric rising over 1929 → 2025.")
    print("Spearman p and MK p are two-sided.\n")
    overall = trends[trends["subgroup_dim"] == "all"].copy()
    overall = overall[overall["metric"].isin(SUMMARY_METRICS)]
    for etype in ["authors", "works"]:
        sub = overall[overall["entity_type"] == etype]
        if sub.empty:
            continue
        print(f"-- {etype.upper()} --")
        print(
            f"{'metric':<32}{'n':>4}{'OLS slope':>14}{'OLS p':>10}"
            f"{'Spearman ρ':>14}{'ρ p':>10}{'MK S':>10}{'MK p':>10}"
        )
        for _, r in sub.iterrows():
            print(
                f"{r['metric']:<32}{int(r['n_editions']):>4}"
                f"{r['ols_slope']:>14.5f}{r['ols_p']:>10.3f}"
                f"{r['spearman_rho']:>14.3f}{r['spearman_p']:>10.3f}"
                f"{r['mk_s']:>10.1f}{r['mk_p']:>10.3f}"
            )
        print()


def print_subgroup_summary(trends: pd.DataFrame, metric: str = "logit_auc") -> None:
    print(f"\n===== SUBGROUP TREND for {metric} =====")
    print("Spearman ρ and two-sided p, grouped by subgroup level.\n")
    sub = trends[(trends["metric"] == metric) & (trends["subgroup_dim"] != "all")]
    if sub.empty:
        print("(no subgroup data)\n")
        return
    for etype in ["authors", "works"]:
        e_sub = sub[sub["entity_type"] == etype]
        if e_sub.empty:
            continue
        print(f"-- {etype.upper()} --")
        print(
            f"{'dim':<16}{'level':<14}{'n':>4}{'Spearman ρ':>14}{'ρ p':>10}{'MK S':>10}"
        )
        for _, r in e_sub.iterrows():
            print(
                f"{r['subgroup_dim']:<16}{r['subgroup_level']:<14}"
                f"{int(r['n_editions']):>4}"
                f"{r['spearman_rho']:>14.3f}{r['spearman_p']:>10.3f}"
                f"{r['mk_s']:>10.1f}"
            )
        print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["authors", "works", "both"],
        default="both",
        help="Compute metrics for authors, works, or both (default: both)",
    )
    parser.add_argument(
        "--include-excerpts",
        action="store_true",
        help="Include excerpt works (parent_id NOT NULL). Default: root works only.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Write per-edition + trend CSVs to --out-dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory for CSVs (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--gender-csv",
        type=Path,
        default=GENDER_CSV,
        help=f"Path to gender annotations (default: {GENDER_CSV.name})",
    )
    args = parser.parse_args()

    params = parse_db_params(ENV_FILE)
    raw = query_db(params, QUERIES / "predictability_over_time.sql")

    if not args.include_excerpts:
        raw_works = raw[raw["parent_id"].isna()].copy()
    else:
        raw_works = raw.copy()

    work_volume_spans_all = compute_work_volume_spans(raw)
    work_volume_spans_works = compute_work_volume_spans(raw_works)

    print(
        f"\n===== DATA OVERVIEW =====\n"
        f"Rows from SQL                : {len(raw):,}\n"
        f"AFAM editions                : {raw['edition_id'].nunique()}\n"
        f"Years covered                : "
        f"{int(raw['anthology_publication_year'].min())}"
        f"–{int(raw['anthology_publication_year'].max())}\n"
        f"Root works only              : {not args.include_excerpts}\n"
        f"Frequent-canon threshold     : ≥ {FREQUENT_CANON_THRESHOLD} prior editions\n"
        f"Min subgroup × edition size  : {MIN_SUBGROUP_SIZE}\n"
        f"Note: earliest edition (no prior) is excluded from metrics.\n"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["authors", "works"] if args.mode == "both" else [args.mode]
    all_per_ed = []
    all_trends = []

    for mode in modes:
        if mode == "authors":
            gender_map = load_gender_map(args.gender_csv)
            author_per_ed = build_author_per_edition(raw, work_volume_spans_all)
            author_per_ed = attach_subgroups_authors(author_per_ed, gender_map)
            per_ed = compute_all_metrics(author_per_ed, "authors")
        else:
            work_per_ed = build_work_per_edition(raw_works, work_volume_spans_works)
            work_per_ed = attach_subgroups_works(work_per_ed)
            per_ed = compute_all_metrics(work_per_ed, "works")

        trends = compute_trends(per_ed)
        all_per_ed.append(per_ed)
        all_trends.append(trends)

        if args.save_csv:
            out_path = args.out_dir / f"predictability_over_time_{mode}.csv"
            per_ed.to_csv(out_path, index=False)
            print(f"Saved → {out_path}")

    combined_trends = (
        pd.concat(all_trends, ignore_index=True) if all_trends else pd.DataFrame()
    )
    if args.save_csv and not combined_trends.empty:
        out_path = args.out_dir / "predictability_over_time_trends.csv"
        combined_trends.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

    if not combined_trends.empty:
        print_trend_summary(combined_trends)
        print_subgroup_summary(combined_trends, "logit_auc")
        print_subgroup_summary(combined_trends, "recall_at_k")


if __name__ == "__main__":
    main()
