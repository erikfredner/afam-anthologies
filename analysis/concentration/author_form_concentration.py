"""
author_form_concentration.py
----------------------------
Measure how concentrated each author × form anthology record is across that
author-form's selected works.

The headline metric is effective_works_page = 1 / HHI, where work weights are
page-share weighted within each author × form × edition. A count-weighted
variant is included alongside it for comparison.

Usage:
    uv run python analysis/concentration/author_form_concentration.py
    uv run python analysis/concentration/author_form_concentration.py --include-excerpts
    uv run python analysis/concentration/author_form_concentration.py --save-csv
    uv run python analysis/concentration/author_form_concentration.py --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "viz" / "reselection"))
sys.path.insert(0, str(REPO_ROOT / "viz" / "inequality"))

from author_page_share_reselection import (  # noqa: E402
    build_author_form_counts,
    build_work_per_edition,
    compute_work_volume_spans,
)
from inclusion_inequality import gini  # noqa: E402

from afam import DATA_DIR  # noqa: E402
from afam.cli import add_root_works_flag, add_save_csv_flag  # noqa: E402
from afam.db import query as query_db  # noqa: E402
from afam.names import author_last_name  # noqa: E402
from afam.sql import query_path  # noqa: E402
from afam.viz_style import OUTPUT_DIR  # noqa: E402

OUT_PER_WORK = DATA_DIR / "author_form_concentration_per_work.csv"
OUT_CONCENTRATION = DATA_DIR / "author_form_concentration.csv"
PLOT_FORMS = ("poetry", "nonfiction", "fiction")
OUT_FORM_HTML = {
    form: OUTPUT_DIR / f"author_form_concentration_{form}.html" for form in PLOT_FORMS
}

UNIT_COLS = ["author_id", "form_id", "form_name"]
FOCAL_AUTHORS = {"Langston Hughes", "Claude McKay"}


def _unknown_form_rows(raw: pd.DataFrame) -> pd.DataFrame:
    forms = raw[["work_id", "form_id", "form_name"]].drop_duplicates(
        ["work_id", "form_id", "form_name"]
    )
    works_without_form = (
        forms.groupby("work_id")["form_name"]
        .apply(lambda s: s.notna().any())
        .rename("has_form")
        .reset_index()
    )
    unknown_rows = works_without_form.loc[~works_without_form["has_form"], ["work_id"]]
    unknown_rows = unknown_rows.assign(form_id=pd.NA, form_name="Unknown")
    return pd.concat(
        [forms.dropna(subset=["form_name"]), unknown_rows], ignore_index=True
    )


def build_work_author_form_per_edition(
    raw: pd.DataFrame,
    work_per_edition: pd.DataFrame,
    counts: pd.DataFrame,
) -> pd.DataFrame:
    """One row per author × form × work × edition with split page spans.

    This is the work-grain sibling of
    author_page_share_reselection.build_author_form_per_edition: each work's
    span is split equally among co-authors and co-forms, but work_id/title are
    retained instead of being summed away.
    """
    author_links = raw.dropna(subset=["author_id"])[
        ["work_id", "author_id", "author_name"]
    ].drop_duplicates(["work_id", "author_id"])
    forms = _unknown_form_rows(raw)

    title_cols = ["work_id"]
    if "work_title" in raw.columns:
        title_cols.append("work_title")
    work_titles = raw[title_cols].drop_duplicates("work_id")
    if "work_title" not in work_titles.columns:
        work_titles = work_titles.assign(work_title=work_titles["work_id"].map(str))

    cartesian = author_links.merge(forms, on="work_id")
    cartesian = cartesian.merge(work_titles, on="work_id", how="left")
    cartesian = cartesian.merge(
        counts[["work_id", "author_count", "form_count"]], on="work_id"
    )
    merged = cartesian.merge(work_per_edition, on="work_id")
    merged["author_form_work_pages"] = merged["span"] / (
        merged["author_count"].clip(lower=1) * merged["form_count"].clip(lower=1)
    )

    return (
        merged.groupby(
            [
                "author_id",
                "author_name",
                "form_id",
                "form_name",
                "work_id",
                "work_title",
                "edition_id",
                "edition_year",
            ],
            dropna=False,
        )
        .agg(author_form_work_pages=("author_form_work_pages", "sum"))
        .reset_index()
    )


def shannon_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return float("nan")
    return float(-(p * np.log(p)).sum())


def concentration_stats(weights: pd.Series | np.ndarray) -> dict[str, float]:
    values = np.asarray(weights, dtype=float)
    n = len(values)
    total = float(np.nansum(values))
    if n == 0 or not np.isfinite(total) or total <= 0:
        return {
            "hhi": float("nan"),
            "effective_works": float("nan"),
            "entropy": float("nan"),
            "normalized_entropy": float("nan"),
            "gini": float("nan"),
            "signature_work_share": float("nan"),
        }

    p = values / total
    hhi = float(np.sum(p * p))
    entropy = shannon_entropy(p)
    return {
        "hhi": hhi,
        "effective_works": float(1 / hhi) if hhi > 0 else float("nan"),
        "entropy": entropy,
        "normalized_entropy": float(entropy / np.log(n)) if n > 1 else float("nan"),
        "gini": float(gini(values)),
        "signature_work_share": float(np.max(p)),
    }


def _weighted_per_work(work_author_form: pd.DataFrame) -> pd.DataFrame:
    denominators = (
        work_author_form.groupby(UNIT_COLS + ["edition_id"], dropna=False)
        .agg(author_form_edition_pages=("author_form_work_pages", "sum"))
        .reset_index()
    )
    shares = work_author_form.merge(
        denominators, on=UNIT_COLS + ["edition_id"], how="left"
    )
    shares["page_share_work_edition"] = shares["author_form_work_pages"] / shares[
        "author_form_edition_pages"
    ].replace(0, np.nan)

    per_work = (
        shares.groupby(
            UNIT_COLS + ["work_id", "work_title", "author_name"],
            dropna=False,
        )
        .agg(
            n_editions_selecting_work=("edition_id", "nunique"),
            page_weight=("page_share_work_edition", "sum"),
            count_weight=("edition_id", "nunique"),
        )
        .reset_index()
    )

    page_totals = (
        per_work.groupby(UNIT_COLS, dropna=False)["page_weight"]
        .sum()
        .rename("page_weight_total")
        .reset_index()
    )
    count_totals = (
        per_work.groupby(UNIT_COLS, dropna=False)["count_weight"]
        .sum()
        .rename("count_weight_total")
        .reset_index()
    )
    per_work = per_work.merge(page_totals, on=UNIT_COLS, how="left")
    per_work = per_work.merge(count_totals, on=UNIT_COLS, how="left")
    per_work["page_p"] = per_work["page_weight"] / per_work[
        "page_weight_total"
    ].replace(0, np.nan)
    per_work["count_p"] = per_work["count_weight"] / per_work[
        "count_weight_total"
    ].replace(0, np.nan)

    per_work["author_last_name"] = per_work["author_name"].map(author_last_name)
    return per_work[
        [
            "author_id",
            "author_name",
            "author_last_name",
            "form_id",
            "form_name",
            "work_id",
            "work_title",
            "n_editions_selecting_work",
            "page_weight",
            "page_p",
            "count_weight",
            "count_p",
        ]
    ].sort_values(
        ["author_last_name", "author_name", "form_name", "page_p", "work_title"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )


def _signature(
    per_work: pd.DataFrame, unit: tuple, weight_col: str
) -> tuple[object, object]:
    sub = per_work
    for col, value in zip(UNIT_COLS, unit, strict=True):
        sub = sub[sub[col].isna()] if pd.isna(value) else sub[sub[col] == value]
    if sub.empty or float(sub[weight_col].sum()) <= 0:
        return (pd.NA, pd.NA)
    row = sub.sort_values(
        [weight_col, "work_title", "work_id"], ascending=[False, True, True]
    ).iloc[0]
    return (row["work_id"], row["work_title"])


def _build_concentration(
    per_work: pd.DataFrame, work_author_form: pd.DataFrame
) -> pd.DataFrame:
    diagnostics = (
        work_author_form.groupby(UNIT_COLS, dropna=False)
        .agg(
            distinct_editions=("edition_id", "nunique"),
            total_selections=("edition_id", "size"),
            n_zero_span_workedition=(
                "author_form_work_pages",
                lambda s: int((s == 0).sum()),
            ),
        )
        .reset_index()
    )
    unit_meta = (
        per_work.groupby(UNIT_COLS, dropna=False)
        .agg(
            author_name=("author_name", "first"),
            author_last_name=("author_last_name", "first"),
            distinct_works=("work_id", "nunique"),
        )
        .reset_index()
    )

    records: list[dict] = []
    for unit, group in per_work.groupby(UNIT_COLS, dropna=False, sort=False):
        page = concentration_stats(group["page_weight"])
        count = concentration_stats(group["count_weight"])
        page_sig_id, page_sig_title = _signature(per_work, unit, "page_weight")
        count_sig_id, count_sig_title = _signature(per_work, unit, "count_weight")
        records.append(
            {
                **dict(zip(UNIT_COLS, unit, strict=True)),
                "page_weights_degenerate": bool(
                    pd.isna(page["hhi"]) and float(group["page_weight"].sum()) == 0
                ),
                "hhi_page": page["hhi"],
                "effective_works_page": page["effective_works"],
                "entropy_page": page["entropy"],
                "normalized_entropy_page": page["normalized_entropy"],
                "gini_page": page["gini"],
                "signature_work_share_page": page["signature_work_share"],
                "signature_work_id": page_sig_id,
                "signature_work_title": page_sig_title,
                "hhi_count": count["hhi"],
                "effective_works_count": count["effective_works"],
                "entropy_count": count["entropy"],
                "normalized_entropy_count": count["normalized_entropy"],
                "gini_count": count["gini"],
                "signature_work_share_count": count["signature_work_share"],
                "signature_work_id_count": count_sig_id,
                "signature_work_title_count": count_sig_title,
            }
        )

    concentration = pd.DataFrame(records)
    concentration = concentration.merge(unit_meta, on=UNIT_COLS, how="left")
    concentration = concentration.merge(diagnostics, on=UNIT_COLS, how="left")
    concentration = concentration[
        [
            "author_id",
            "author_name",
            "author_last_name",
            "form_id",
            "form_name",
            "distinct_works",
            "distinct_editions",
            "total_selections",
            "page_weights_degenerate",
            "n_zero_span_workedition",
            "hhi_page",
            "effective_works_page",
            "entropy_page",
            "normalized_entropy_page",
            "gini_page",
            "signature_work_share_page",
            "signature_work_id",
            "signature_work_title",
            "hhi_count",
            "effective_works_count",
            "entropy_count",
            "normalized_entropy_count",
            "gini_count",
            "signature_work_share_count",
            "signature_work_id_count",
            "signature_work_title_count",
        ]
    ]
    return concentration.sort_values(
        ["total_selections", "signature_work_share_page"],
        ascending=[False, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def compute(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute per-work weights and per-author-form concentration tables."""
    work_spans = compute_work_volume_spans(raw)
    work_per_edition = build_work_per_edition(raw, work_spans)
    counts = build_author_form_counts(raw)
    work_author_form = build_work_author_form_per_edition(raw, work_per_edition, counts)
    per_work = _weighted_per_work(work_author_form).reset_index(drop=True)
    concentration = _build_concentration(per_work, work_author_form)
    return {"per_work": per_work, "concentration": concentration}


def _fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def print_concentration(
    concentration: pd.DataFrame, *, min_selections: int = 5, limit: int = 25
) -> None:
    table = concentration[concentration["total_selections"] >= min_selections].copy()
    if table.empty:
        print(f"No author-form units with total_selections >= {min_selections}.")
        return
    table = table.head(limit)
    cols = [
        "author_name",
        "form_name",
        "distinct_works",
        "total_selections",
        "effective_works_page",
        "hhi_page",
        "signature_work_share_page",
        "signature_work_title",
        "effective_works_count",
    ]
    print(f"\nAuthor-form concentration (min selections: {min_selections})")
    print(
        table[cols]
        .assign(
            effective_works_page=lambda d: d["effective_works_page"].map(_fmt),
            hhi_page=lambda d: d["hhi_page"].map(_fmt),
            signature_work_share_page=lambda d: d["signature_work_share_page"].map(
                _fmt
            ),
            effective_works_count=lambda d: d["effective_works_count"].map(_fmt),
        )
        .to_string(index=False)
    )


def print_focal(concentration: pd.DataFrame) -> None:
    focal = concentration[
        concentration["author_name"].isin(["Langston Hughes", "Claude McKay"])
        & (concentration["form_name"].str.casefold() == "poetry")
    ].copy()
    print("\nHughes vs. McKay poetry")
    if focal.empty:
        print("No focal poetry rows found.")
        return
    cols = [
        "author_name",
        "distinct_works",
        "total_selections",
        "effective_works_page",
        "hhi_page",
        "signature_work_share_page",
        "signature_work_title",
    ]
    print(
        focal[cols]
        .assign(
            effective_works_page=lambda d: d["effective_works_page"].map(_fmt),
            hhi_page=lambda d: d["hhi_page"].map(_fmt),
            signature_work_share_page=lambda d: d["signature_work_share_page"].map(
                _fmt
            ),
        )
        .to_string(index=False)
    )


def prepare_plot_data(
    concentration: pd.DataFrame,
    *,
    min_selections: int = 5,
    min_distinct_works: int = 2,
    max_forms: int | None = 8,
) -> pd.DataFrame:
    df = concentration[
        (concentration["total_selections"] >= min_selections)
        & (concentration["distinct_works"] >= min_distinct_works)
        & (~concentration["page_weights_degenerate"])
    ].copy()
    finite_cols = [
        "hhi_page",
        "effective_works_page",
        "signature_work_share_page",
        "distinct_works",
    ]
    for col in finite_cols:
        df = df[np.isfinite(df[col].to_numpy(dtype=float))]
    if df.empty:
        return df

    form_order = (
        df.groupby("form_name", dropna=False)["total_selections"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if max_forms is not None:
        form_order = form_order[:max_forms]
        df = df[df["form_name"].isin(form_order)].copy()

    df["form_name"] = pd.Categorical(
        df["form_name"], categories=form_order, ordered=True
    )
    df["concentration_percentile_page"] = df.groupby("form_name", observed=True)[
        "hhi_page"
    ].rank(pct=True)
    return df.sort_values(["form_name", "total_selections"], ascending=[True, False])


def _hover_text(df: pd.DataFrame) -> list[str]:
    return [
        (
            f"<b>{row.author_name}</b><br>"
            f"Form: {row.form_name}<br>"
            f"Distinct works: {row.distinct_works}<br>"
            f"Selections: {row.total_selections}<br>"
            f"Effective works, page-weighted: {row.effective_works_page:.3f}<br>"
            f"HHI, page-weighted: {row.hhi_page:.3f}<br>"
            f"Within-form HHI percentile: {row.concentration_percentile_page:.1%}<br>"
            f"Signature share, page-weighted: {row.signature_work_share_page:.1%}<br>"
            f"Signature work: {row.signature_work_title}"
        )
        for row in df.itertuples(index=False)
    ]


def _label_text(df: pd.DataFrame) -> list[str]:
    return [
        str(row.author_last_name)
        if row.author_name in FOCAL_AUTHORS or row.concentration_percentile_page >= 0.9
        else ""
        for row in df.itertuples(index=False)
    ]


def plot_form_interactive(
    concentration: pd.DataFrame,
    form: str,
    *,
    min_selections: int = 5,
    min_distinct_works: int = 2,
    top_n: int = 12,
    out_path: Path | None = None,
) -> None:
    """Write one interactive HTML file for a single literary form."""
    df = prepare_plot_data(
        concentration,
        min_selections=min_selections,
        min_distinct_works=min_distinct_works,
        max_forms=None,
    )
    df = df[df["form_name"].astype(str).str.casefold() == form.casefold()].copy()
    if df.empty:
        print(
            "No plottable author-form units with "
            f"total_selections >= {min_selections} and "
            f"distinct_works >= {min_distinct_works} for {form!r}."
        )
        return

    out_path = out_path or OUT_FORM_HTML[form]
    df["marker_size"] = 8 + 3 * np.sqrt(df["distinct_works"].to_numpy(dtype=float))

    ranked = (
        df.sort_values(
            ["concentration_percentile_page", "hhi_page", "total_selections"],
            ascending=[False, False, False],
        )
        .head(top_n)
        .copy()
    )
    ranked = ranked.sort_values("hhi_page")
    bar_labels = ranked["author_last_name"].where(
        ranked["author_last_name"].astype(str).str.len() > 0, ranked["author_name"]
    )
    bar_colors = np.where(
        ranked["author_name"].isin(FOCAL_AUTHORS), "#d62728", "#4c78a8"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=(
            "Selection volume vs. effective works",
            f"Most concentrated {form} author-form units",
        ),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=df["total_selections"],
            y=df["effective_works_page"],
            mode="markers+text",
            text=_label_text(df),
            textposition="top center",
            textfont={"size": 10},
            marker={
                "size": df["marker_size"],
                "color": df["concentration_percentile_page"],
                "colorscale": "Magma_r",
                "cmin": 0,
                "cmax": 1,
                "opacity": 0.82,
                "line": {"color": "white", "width": 0.8},
                "colorbar": {
                    "title": "Within-form<br>HHI percentile",
                    "tickformat": ".0%",
                    "x": 0.58,
                },
            },
            hovertext=_hover_text(df),
            hoverinfo="text",
            name="Author-form units",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ranked["hhi_page"],
            y=bar_labels,
            orientation="h",
            marker={"color": bar_colors},
            customdata=np.stack(
                [
                    ranked["author_name"],
                    ranked["effective_works_page"],
                    ranked["total_selections"],
                    ranked["signature_work_title"],
                    ranked["signature_work_share_page"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "HHI, page-weighted: %{x:.3f}<br>"
                "Effective works: %{customdata[1]:.3f}<br>"
                "Selections: %{customdata[2]}<br>"
                "Signature work: %{customdata[3]}<br>"
                "Signature share: %{customdata[4]:.1%}<extra></extra>"
            ),
            showlegend=False,
            name="HHI",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Total work-edition selections", row=1, col=1)
    fig.update_yaxes(title_text="Effective works, page-weighted", row=1, col=1)
    fig.update_xaxes(title_text="HHI, page-weighted", range=[0, 1], row=1, col=2)
    fig.update_yaxes(automargin=True, row=1, col=2)
    fig.update_layout(
        title={
            "text": (
                f"Author-form concentration: {form}<br>"
                "<sup>Higher effective works means a record is spread across more works; "
                "higher HHI percentile means more concentrated relative to same-form peers.</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        width=1280,
        height=720,
        margin={"l": 70, "r": 70, "t": 110, "b": 70},
        showlegend=False,
    )
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    print(f"Wrote {out_path}")


def plot_outputs(
    outputs: dict[str, pd.DataFrame],
    *,
    min_selections: int = 5,
    min_distinct_works: int = 2,
) -> None:
    for form in PLOT_FORMS:
        plot_form_interactive(
            outputs["concentration"],
            form,
            min_selections=min_selections,
            min_distinct_works=min_distinct_works,
        )


def save_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    outputs["per_work"].to_csv(OUT_PER_WORK, index=False)
    outputs["concentration"].to_csv(OUT_CONCENTRATION, index=False)
    print(f"Wrote {OUT_PER_WORK}")
    print(f"Wrote {OUT_CONCENTRATION}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_root_works_flag(parser)
    add_save_csv_flag(parser)
    parser.add_argument(
        "--min-selections",
        type=int,
        default=5,
        help="minimum total work-edition selections for the printed table",
    )
    parser.add_argument(
        "--min-plot-works",
        type=int,
        default=2,
        help="minimum distinct works for inclusion in plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip writing interactive HTML visualizations to output/",
    )
    args = parser.parse_args()

    raw = query_db(query_path("author-page-share-reselection"))
    if args.only_root_works:
        raw = raw[raw["parent_id"].isna()].reset_index(drop=True)

    outputs = compute(raw)
    if not args.no_plots:
        plot_outputs(
            outputs,
            min_selections=args.min_selections,
            min_distinct_works=args.min_plot_works,
        )
    if args.save_csv:
        save_outputs(outputs)
    else:
        print_concentration(
            outputs["concentration"], min_selections=args.min_selections
        )
        print_focal(outputs["concentration"])


if __name__ == "__main__":
    main()
