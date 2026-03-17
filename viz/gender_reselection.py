"""gender_reselection.py
-----------------------
Analyse whether male and female anthology authors are reselected at different
rates across subsequent anthologies.

Figure: viz/gender_reselection.png
Usage: uv run python viz/gender_reselection.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import StratifiedTable

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_FILE = (
    Path(__file__).parent.parent
    / "data"
    / "2026-03-17 af am anthology authors with genders.csv"
)
OUT_DIR = Path(__file__).parent

# ── Style constants (from retention_from_1929.py) ─────────────────────────────

C_BLUE = "#1f77b4"
C_RED = "#d62728"
C_GRAY = "#888888"
GRID_KW = dict(alpha=0.25, linestyle=":")

GENDER_COLORS = {"Male": C_BLUE, "Female": C_RED, "Unmarked": C_GRAY}


# ── Core functions (exported for tests) ───────────────────────────────────────


def build_gender_group(gender: str) -> str:
    """Map raw gender string to Male / Female / Unmarked."""
    if gender == "Male":
        return "Male"
    if gender == "Female":
        return "Female"
    return "Unmarked"


def compute_opportunities(selections: pd.DataFrame) -> pd.DataFrame:
    """Build opportunity table: every (author, edition) where edition year > author debut year.

    Parameters
    ----------
    selections:
        DataFrame with columns: author_id, gender_group, edition_key, ek_year.
        One row per (author, edition) — already deduplicated.

    Returns
    -------
    DataFrame with columns: author_id, gender_group, edition_key, ek_year, selected.
    """
    author_meta = (
        selections.groupby("author_id")
        .agg(first_year=("ek_year", "min"), gender_group=("gender_group", "first"))
        .reset_index()
    )

    all_eks = selections[["edition_key", "ek_year"]].drop_duplicates()
    actual = set(zip(selections["author_id"], selections["edition_key"]))

    opps = []
    for _, auth in author_meta.iterrows():
        for _, ek_row in all_eks.iterrows():
            if ek_row["ek_year"] > auth["first_year"]:
                opps.append(
                    {
                        "author_id": auth["author_id"],
                        "gender_group": auth["gender_group"],
                        "edition_key": ek_row["edition_key"],
                        "ek_year": ek_row["ek_year"],
                        "selected": int(
                            (auth["author_id"], ek_row["edition_key"]) in actual
                        ),
                    }
                )

    if not opps:
        return pd.DataFrame(
            columns=["author_id", "gender_group", "edition_key", "ek_year", "selected"]
        )
    return pd.DataFrame(opps)


def aggregate_by_year(opps: pd.DataFrame) -> pd.DataFrame:
    """Per (ek_year, gender_group): opportunities, reselections, rate."""
    agg = (
        opps.groupby(["ek_year", "gender_group"])
        .agg(
            opportunities=("selected", "count"),
            reselections=("selected", "sum"),
        )
        .reset_index()
    )
    agg["rate"] = agg["reselections"] / agg["opportunities"]
    return agg


def compute_cumulative(agg: pd.DataFrame) -> pd.DataFrame:
    """Add cum_opportunities, cum_reselections, cum_rate per gender, sorted by year."""
    chunks = []
    for gender, grp in agg.groupby("gender_group"):
        grp = grp.sort_values("ek_year").copy()
        grp["cum_opportunities"] = grp["opportunities"].cumsum()
        grp["cum_reselections"] = grp["reselections"].cumsum()
        grp["cum_rate"] = grp["cum_reselections"] / grp["cum_opportunities"]
        chunks.append(grp)
    return pd.concat(chunks, ignore_index=True).sort_values(["gender_group", "ek_year"])


# ── Load and prepare ───────────────────────────────────────────────────────────


def _load_and_prepare(data_file: Path) -> pd.DataFrame:
    df = pd.read_csv(data_file, dtype=str, na_filter=False)
    print(f"Loaded {len(df)} rows, {df['anthology_id'].nunique()} raw anthologies")

    # Build edition_key: series_id|edition when series, else anthology_id
    df["edition_key"] = df.apply(
        lambda r: (
            f"{r['series_id']}|{r['edition']}"
            if r["series_id"].strip()
            else r["anthology_id"]
        ),
        axis=1,
    )

    # Year per edition_key (minimum publication_year across volumes of same edition)
    df["_pub_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    year_by_key = df.groupby("edition_key")["_pub_year"].min().astype(int)
    df["ek_year"] = df["edition_key"].map(year_by_key)
    df.drop(columns=["_pub_year"], inplace=True)

    # Gender group
    df["gender_group"] = df["gender"].apply(build_gender_group)

    print(f"  Unique edition_keys: {df['edition_key'].nunique()}")
    print(f"  Unique authors: {df['author_id'].nunique()}")
    gender_counts = df.groupby("gender_group").size()
    print(f"  Gender distribution:\n{gender_counts.to_string()}")

    return df


# ── Statistical tests ──────────────────────────────────────────────────────────


def _cmh_male_female(opps_df: pd.DataFrame) -> None:
    """Cochran-Mantel-Haenszel test for Male vs Female, stratified by edition_key."""
    mf = opps_df[opps_df["gender_group"].isin(["Male", "Female"])]
    tables = []
    strata_used = []
    for ek, grp in mf.groupby("edition_key"):
        m = grp[grp["gender_group"] == "Male"]["selected"]
        f = grp[grp["gender_group"] == "Female"]["selected"]
        if len(m) == 0 or len(f) == 0:
            continue
        tbl = np.array(
            [[m.sum(), (m == 0).sum()], [f.sum(), (f == 0).sum()]],
            dtype=float,
        )
        # Skip degenerate strata (all zeros in a row/col)
        if tbl.min() == 0 and (tbl[0].sum() == 0 or tbl[1].sum() == 0):
            continue
        tables.append(tbl)
        strata_used.append(ek)

    print(
        "\n========== Cochran-Mantel-Haenszel test "
        "(Male vs Female, stratified by edition) =========="
    )
    print(f"Strata used: {len(strata_used)} of {mf['edition_key'].nunique()} editions")

    if len(tables) < 2:
        print("  Too few valid strata — CMH test not computed.")
        return

    st = StratifiedTable(tables)
    cmh = st.test_null_odds()
    or_mh = st.oddsratio_pooled
    print(f"CMH statistic: {cmh.statistic:.3f}, p={cmh.pvalue:.4g}")
    print(f"Mantel-Haenszel common OR (Male vs Female): {or_mh:.3f}")
    print(
        "  (OR < 1 means male authors reselected at lower rate after controlling "
        "for which anthology)"
    )


def print_statistics(opps_df: pd.DataFrame) -> None:
    """Print chi-squared tests and logistic regression to stdout."""
    totals = (
        opps_df.groupby("gender_group")
        .agg(
            selected=("selected", "sum"),
            not_selected=("selected", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    print("\n========== 3×2 Contingency Table (gender × selected) — unadjusted ==========")
    print(f"{'Group':<12} {'Selected':>10} {'Not selected':>14} {'Total':>8}")
    for _, row in totals.iterrows():
        total = row["selected"] + row["not_selected"]
        print(
            f"{row['gender_group']:<12} {row['selected']:>10} {row['not_selected']:>14} {total:>8}"
        )

    table_3x2 = totals[["selected", "not_selected"]].values
    chi2, p, dof, _ = chi2_contingency(table_3x2)
    print(f"\nChi-squared (3-way): χ²={chi2:.3f}, df={dof}, p={p:.4g}")

    # Pairwise Male vs Female (unadjusted)
    male_row = totals[totals["gender_group"] == "Male"]
    female_row = totals[totals["gender_group"] == "Female"]
    if not male_row.empty and not female_row.empty:
        m_sel = int(male_row["selected"].values[0])
        m_not = int(male_row["not_selected"].values[0])
        f_sel = int(female_row["selected"].values[0])
        f_not = int(female_row["not_selected"].values[0])
        tbl_2x2 = np.array([[m_sel, m_not], [f_sel, f_not]])
        chi2_mf, p_mf, _, _ = chi2_contingency(tbl_2x2)
        or_mf = (m_sel * f_not) / (m_not * f_sel) if (m_not * f_sel) > 0 else float("nan")
        print(f"\nPairwise Male vs Female (unadjusted): χ²={chi2_mf:.3f}, p={p_mf:.4g}, OR={or_mf:.3f}")

    # CMH test — controls for within-anthology pool composition
    _cmh_male_female(opps_df)

    # Logistic regression Model 1: year only (unadjusted for anthology)
    print(
        "\n========== Logistic Regression Model 1 (unadjusted): "
        "selected ~ C(gender_group) + ek_year =========="
    )
    try:
        m1 = smf.logit("selected ~ C(gender_group) + ek_year", data=opps_df).fit(disp=False)
        _print_gender_coefs(m1)
    except Exception as exc:
        print(f"Model 1 failed: {exc}")

    # Logistic regression Model 2: edition fixed effects (controls for anthology selectivity)
    print(
        "\n========== Logistic Regression Model 2 (edition fixed effects): "
        "selected ~ C(gender_group) + C(edition_key) =========="
    )
    try:
        m2 = smf.logit(
            "selected ~ C(gender_group) + C(edition_key)", data=opps_df
        ).fit(disp=False)
        _print_gender_coefs(m2)
    except Exception as exc:
        print(f"Model 2 failed: {exc}")


def _print_gender_coefs(model) -> None:
    """Print only the gender_group and intercept rows from a fitted model."""
    coef_df = pd.DataFrame(
        {"coef": model.params, "OR": np.exp(model.params), "p": model.pvalues}
    )
    gender_rows = coef_df[
        coef_df.index.str.contains("gender_group|Intercept", regex=True)
    ]
    print(gender_rows.to_string(float_format=lambda x: f"{x:.4f}"))


# ── Figure ─────────────────────────────────────────────────────────────────────


def plot_gender_reselection(
    cum_df: pd.DataFrame, agg_df: pd.DataFrame, out_path: Path
) -> None:
    """Render two-panel gender reselection figure."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 10))

    # Top panel: cumulative reselection rate over time
    for gender, grp in cum_df.groupby("gender_group"):
        color = GENDER_COLORS.get(gender, C_GRAY)
        grp_sorted = grp.sort_values("ek_year")
        ax_top.plot(
            grp_sorted["ek_year"],
            grp_sorted["cum_rate"],
            color=color,
            label=gender,
            linewidth=2.0,
        )

    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("Cumulative reselection rate")
    ax_top.set_title("Cumulative author reselection rate by gender")
    ax_top.legend(frameon=False)
    ax_top.grid(True, **GRID_KW)

    # Bottom panel: per-anthology rate vs. opportunities
    overall_rate = agg_df["reselections"].sum() / agg_df["opportunities"].sum()
    for gender, grp in agg_df.groupby("gender_group"):
        color = GENDER_COLORS.get(gender, C_GRAY)
        ax_bot.scatter(
            grp["opportunities"],
            grp["rate"],
            color=color,
            alpha=0.7,
            label=gender,
        )

    ax_bot.axhline(
        overall_rate,
        color="black",
        linestyle="--",
        lw=1.0,
        label=f"Overall avg ({overall_rate:.2f})",
    )
    ax_bot.set_xlabel("Opportunities")
    ax_bot.set_ylabel("Reselection rate")
    ax_bot.set_title("Reselection rate vs. opportunities by gender per anthology")
    ax_bot.legend(frameon=False)
    ax_bot.grid(True, **GRID_KW)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    df = _load_and_prepare(DATA_FILE)

    # One row per (author, edition)
    selections = df.drop_duplicates(["author_id", "edition_key"])[
        ["author_id", "gender_group", "edition_key", "ek_year"]
    ].copy()

    opps_df = compute_opportunities(selections)
    print(f"\nTotal opportunities : {len(opps_df)}")
    print(f"Total reselections  : {opps_df['selected'].sum()}")

    agg_df = aggregate_by_year(opps_df)
    cum_df = compute_cumulative(agg_df)

    print_statistics(opps_df)

    out_path = OUT_DIR / "gender_reselection.png"
    plot_gender_reselection(cum_df, agg_df, out_path)


if __name__ == "__main__":
    main()
