"""Frequent author work-selection scatter charts.

For authors selected in at least half of African-American Literature anthology
editions, plot each root work by:

  * x = the author's post-debut selection record
  * y = the work's post-debut selection record

Two chart variants are written: raw counts and opportunity-based proportions.
Opportunities are editions with publication year greater than the entity's
debut year.

Usage:
    uv run python viz/reselection/frequent_author_work_selection_scatter.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px

from afam.cli import add_root_works_flag
from afam.db import query
from afam.sql import query_path
from afam.viz_style import DPI, OUTPUT_DIR

OUT_COUNTS_SVG = OUTPUT_DIR / "frequent_author_work_selection_counts.svg"
OUT_COUNTS_HTML = OUTPUT_DIR / "frequent_author_work_selection_counts.html"
OUT_RATES_SVG = OUTPUT_DIR / "frequent_author_work_selection_rates.svg"
OUT_RATES_HTML = OUTPUT_DIR / "frequent_author_work_selection_rates.html"

DEFAULT_SEED = 42
GRID_KW = dict(alpha=0.25, linestyle=":")
CMAP = "viridis"
POINT_COLOR = "#2f6f8f"
HOVER_TEMPLATE = (
    "<b>%{customdata[0]} by %{customdata[1]}</b><br>"
    "Work: %{customdata[2]} of %{customdata[3]} opps.<br>"
    "Author: %{customdata[4]} of %{customdata[5]} opps."
    "<extra></extra>"
)


@dataclass(frozen=True)
class OutputPaths:
    counts_svg: Path = OUT_COUNTS_SVG
    counts_html: Path = OUT_COUNTS_HTML
    rates_svg: Path = OUT_RATES_SVG
    rates_html: Path = OUT_RATES_HTML


def _clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _is_root_parent(value: object) -> bool:
    return _clean_id(value) == ""


def _safe_rate(selections: int, opportunities: int) -> float:
    return selections / opportunities if opportunities else float("nan")


def load_data(only_root_works: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full opportunity rows, plotted appearance rows)."""
    df_full = query(query_path("post-debut-performance"))
    df_full["anthology_publication_year"] = df_full[
        "anthology_publication_year"
    ].astype(int)
    df = filter_root_works(df_full) if only_root_works else df_full.copy()
    return df_full, df


def filter_root_works(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["parent_id"].map(_is_root_parent)].copy()


def build_edition_table(df_full: pd.DataFrame) -> pd.DataFrame:
    """One row per anthology edition in the AFAM opportunity universe."""
    editions = (
        df_full[["edition_id", "anthology_publication_year"]]
        .dropna(subset=["edition_id", "anthology_publication_year"])
        .drop_duplicates("edition_id")
        .rename(columns={"anthology_publication_year": "year"})
        .sort_values(["year", "edition_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    editions["year"] = editions["year"].astype(int)
    return editions


def build_entity_pairs(
    df: pd.DataFrame, id_col: str, editions: pd.DataFrame
) -> pd.DataFrame:
    """Return one row per selected (entity, edition), with edition year."""
    pairs = df[[id_col, "edition_id"]].dropna(subset=[id_col, "edition_id"]).copy()
    pairs[id_col] = pairs[id_col].map(_clean_id)
    pairs = pairs[pairs[id_col] != ""]
    pairs = pairs.drop_duplicates([id_col, "edition_id"])
    return pairs.merge(editions, on="edition_id", how="inner")


def compute_year_based_records(
    pairs: pd.DataFrame, id_col: str, editions: pd.DataFrame
) -> pd.DataFrame:
    """Compute selection counts/rates after debut using year > debut year."""
    cols = [
        id_col,
        "debut_year",
        "opportunities",
        "selections",
        "selection_rate",
        "total_selection_count",
    ]
    if pairs.empty:
        return pd.DataFrame(columns=cols)

    entity_editions = pairs.groupby(id_col)["edition_id"].apply(set).to_dict()
    debut_years = pairs.groupby(id_col)["year"].min().to_dict()
    rows: list[dict] = []
    all_editions = editions[["edition_id", "year"]].drop_duplicates()

    for entity_id in sorted(entity_editions, key=lambda x: str(x)):
        debut_year = int(debut_years[entity_id])
        selected_editions = entity_editions[entity_id]
        later_editions = all_editions[all_editions["year"] > debut_year]
        selections = int(later_editions["edition_id"].isin(selected_editions).sum())
        opportunities = int(len(later_editions))
        rows.append(
            {
                id_col: entity_id,
                "debut_year": debut_year,
                "opportunities": opportunities,
                "selections": selections,
                "selection_rate": _safe_rate(selections, opportunities),
                "total_selection_count": len(selected_editions),
            }
        )

    return pd.DataFrame(rows, columns=cols)


def qualifying_author_ids(author_pairs: pd.DataFrame, total_editions: int) -> set[str]:
    """Authors selected in at least half of AFAM anthology editions."""
    if author_pairs.empty:
        return set()
    counts = author_pairs.groupby("author_id")["edition_id"].nunique()
    return set(counts[counts * 2 >= total_editions].index)


def _metadata(df: pd.DataFrame, columns: list[str], id_col: str) -> pd.DataFrame:
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.DataFrame(columns=[id_col])
    out = df[present].dropna(subset=[id_col]).copy()
    out[id_col] = out[id_col].map(_clean_id)
    out = out[out[id_col] != ""]
    return out.drop_duplicates(id_col)


def build_author_work_rows(
    df_full: pd.DataFrame,
    df_works: pd.DataFrame,
    editions: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per qualifying author/work pair with chart metrics."""
    author_pairs = build_entity_pairs(df_full, "author_id", editions)
    work_pairs = build_entity_pairs(df_works, "work_id", editions)
    author_records = compute_year_based_records(
        author_pairs, "author_id", editions
    ).rename(
        columns={
            "debut_year": "author_debut_year",
            "opportunities": "author_opportunities",
            "selections": "author_selections",
            "selection_rate": "author_selection_rate",
            "total_selection_count": "author_total_selection_count",
        }
    )
    work_records = compute_year_based_records(work_pairs, "work_id", editions).rename(
        columns={
            "debut_year": "work_debut_year",
            "opportunities": "work_opportunities",
            "selections": "work_selections",
            "selection_rate": "work_selection_rate",
            "total_selection_count": "work_total_selection_count",
        }
    )

    frequent_authors = qualifying_author_ids(author_pairs, len(editions))
    author_work = (
        df_works[["author_id", "work_id"]]
        .dropna(subset=["author_id", "work_id"])
        .copy()
    )
    author_work["author_id"] = author_work["author_id"].map(_clean_id)
    author_work["work_id"] = author_work["work_id"].map(_clean_id)
    author_work = author_work[
        (author_work["author_id"] != "")
        & (author_work["work_id"] != "")
        & author_work["author_id"].isin(frequent_authors)
    ].drop_duplicates(["author_id", "work_id"])

    author_meta = _metadata(
        df_full,
        ["author_id", "author_name", "author_birth_year"],
        "author_id",
    )
    work_meta = _metadata(
        df_works,
        ["work_id", "work_title", "parent_id", "parent_work_title"],
        "work_id",
    )

    out = (
        author_work.merge(author_records, on="author_id", how="left")
        .merge(work_records, on="work_id", how="left")
        .merge(author_meta, on="author_id", how="left")
        .merge(work_meta, on="work_id", how="left")
    )

    if out.empty:
        out["work_rate_spread"] = pd.Series(dtype=float)
        return out

    spread = out.groupby("author_id")["work_selection_rate"].transform(
        lambda s: float(s.max() - s.median()) if int(s.notna().sum()) > 1 else 0.0
    )
    out["work_rate_spread"] = spread
    return out.sort_values(
        ["author_selections", "work_selections", "author_name", "work_title"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def add_jitter(
    rows: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    out_x: str,
    out_y: str,
    amount: float,
    seed: int,
    clip: tuple[float, float] | None = None,
) -> pd.DataFrame:
    out = rows.copy()
    rng = np.random.default_rng(seed)
    if out.empty:
        out[out_x] = pd.Series(dtype=float)
        out[out_y] = pd.Series(dtype=float)
        return out

    x = out[x_col].astype(float).to_numpy()
    y = out[y_col].astype(float).to_numpy()
    x_jittered = x + rng.uniform(-amount, amount, size=len(out))
    y_jittered = y + rng.uniform(-amount, amount, size=len(out))
    if clip is not None:
        x_jittered = np.clip(x_jittered, clip[0], clip[1])
        y_jittered = np.clip(y_jittered, clip[0], clip[1])
    out[out_x] = x_jittered
    out[out_y] = y_jittered
    return out


def build_plot_frames(
    rows: pd.DataFrame, seed: int = DEFAULT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows[rows["work_opportunities"] > 1].copy()
    counts = add_jitter(
        rows,
        x_col="author_selections",
        y_col="work_selections",
        out_x="author_selections_jittered",
        out_y="work_selections_jittered",
        amount=0.18,
        seed=seed,
    )
    rates = rows.dropna(subset=["author_selection_rate", "work_selection_rate"]).copy()
    rates = add_jitter(
        rates,
        x_col="author_selection_rate",
        y_col="work_selection_rate",
        out_x="author_selection_rate_jittered",
        out_y="work_selection_rate_jittered",
        amount=0.012,
        seed=seed,
        clip=(0.0, 1.0),
    )
    return counts, rates


def add_hover_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hover_work"] = out["work_title"].fillna(out["work_id"]).astype(str)
    out["hover_author"] = out["author_name"].fillna(out["author_id"]).astype(str)
    return out


def _axis_limit(values: pd.Series) -> float:
    if values.empty or values.max() <= 0:
        return 1.0
    return float(math.ceil(values.max()))


def plot_svg(
    df: pd.DataFrame,
    *,
    out_path: Path,
    chart_type: str,
    title: str,
    x: str,
    y: str,
    x_raw: str,
    y_raw: str,
    x_label: str,
    y_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    if df.empty:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
    else:
        color_values = df["work_rate_spread"].fillna(0.0)
        sc = ax.scatter(
            df[x],
            df[y],
            c=color_values,
            cmap=CMAP,
            s=46,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.45,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Max - median work selection rate")

        if chart_type == "rate":
            ax.plot([0, 1], [0, 1], color="#555555", lw=1.0, linestyle="--")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        else:
            lim = max(_axis_limit(df[x_raw]), _axis_limit(df[y_raw]))
            ax.plot([0, lim], [0, lim], color="#555555", lw=1.0, linestyle="--")
            ax.set_xlim(-0.5, lim + 0.5)
            ax.set_ylim(-0.5, lim + 0.5)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, **GRID_KW)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", format="svg")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_html(
    df: pd.DataFrame,
    *,
    out_path: Path,
    chart_type: str,
    title: str,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = add_hover_columns(df)

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="work_rate_spread",
        color_continuous_scale=CMAP,
        custom_data=[
            "hover_work",
            "hover_author",
            "work_selections",
            "work_opportunities",
            "author_selections",
            "author_opportunities",
        ],
        labels={
            x: x_label,
            y: y_label,
            "work_rate_spread": "Max - median work selection rate",
        },
        title=title,
    )
    if chart_type == "rate":
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color="#555555", width=1, dash="dash"),
        )
        fig.update_xaxes(range=[-0.02, 1.02], tickformat=".0%")
        fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%")
    elif not df.empty:
        lim = max(
            _axis_limit(df["author_selections"]), _axis_limit(df["work_selections"])
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=lim,
            y1=lim,
            line=dict(color="#555555", width=1, dash="dash"),
        )
        fig.update_xaxes(range=[-0.5, lim + 0.5], dtick=1)
        fig.update_yaxes(range=[-0.5, lim + 0.5], dtick=1)

    fig.update_traces(
        hovertemplate=HOVER_TEMPLATE,
        marker=dict(size=8, opacity=0.78, line=dict(width=0.5, color="white")),
    )
    fig.update_layout(
        template="plotly_white",
        width=960,
        height=760,
        margin=dict(l=80, r=30, t=80, b=70),
    )
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved {out_path}")


def write_outputs(rows: pd.DataFrame, paths: OutputPaths, seed: int) -> tuple[int, int]:
    counts, rates = build_plot_frames(rows, seed=seed)

    plot_svg(
        counts,
        out_path=paths.counts_svg,
        chart_type="count",
        title="Frequent authors: author vs. work post-debut selections",
        x="author_selections_jittered",
        y="work_selections_jittered",
        x_raw="author_selections",
        y_raw="work_selections",
        x_label="Author selections after debut",
        y_label="Work selections after debut",
    )
    plot_html(
        counts,
        out_path=paths.counts_html,
        chart_type="count",
        title="Frequent authors: author vs. work post-debut selections",
        x="author_selections_jittered",
        y="work_selections_jittered",
        x_label="Author selections after debut",
        y_label="Work selections after debut",
    )
    plot_svg(
        rates,
        out_path=paths.rates_svg,
        chart_type="rate",
        title="Frequent authors: author vs. work post-debut selection rates",
        x="author_selection_rate_jittered",
        y="work_selection_rate_jittered",
        x_raw="author_selection_rate",
        y_raw="work_selection_rate",
        x_label="Author selection rate after debut",
        y_label="Work selection rate after debut",
    )
    plot_html(
        rates,
        out_path=paths.rates_html,
        chart_type="rate",
        title="Frequent authors: author vs. work post-debut selection rates",
        x="author_selection_rate_jittered",
        y="work_selection_rate_jittered",
        x_label="Author selection rate after debut",
        y_label="Work selection rate after debut",
    )
    return len(counts), len(rates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_works_flag(parser)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for deterministic jitter (default: {DEFAULT_SEED}).",
    )
    parser.add_argument("--counts-svg", type=Path, default=OUT_COUNTS_SVG)
    parser.add_argument("--counts-html", type=Path, default=OUT_COUNTS_HTML)
    parser.add_argument("--rates-svg", type=Path, default=OUT_RATES_SVG)
    parser.add_argument("--rates-html", type=Path, default=OUT_RATES_HTML)
    args = parser.parse_args()

    df_full, df_works = load_data(args.only_root_works)
    editions = build_edition_table(df_full)
    rows = build_author_work_rows(df_full, df_works, editions)
    paths = OutputPaths(
        counts_svg=args.counts_svg,
        counts_html=args.counts_html,
        rates_svg=args.rates_svg,
        rates_html=args.rates_html,
    )
    count_points, rate_points = write_outputs(rows, paths, seed=args.seed)
    excluded = len(rows) - count_points
    print(
        f"Built {len(rows):,} author-work rows; excluded {excluded:,} with "
        f"<=1 work opportunity. Plotted {count_points:,} count points and "
        f"{rate_points:,} rate points."
    )


if __name__ == "__main__":
    main()
