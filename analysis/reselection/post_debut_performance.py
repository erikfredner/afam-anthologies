"""
post_debut_performance.py
--------------------------
Ranks authors and works by how often they are reselected after debut compared
with a chance model based on later AFAM anthology editions.

For every author and work, computes:
  - first_year:        year of initial selection
  - opportunities:     count of later anthology editions
  - selections:        later editions in which the entity appeared
  - selection_rate:    selections / opportunities
  - expected_count:    sum of later edition-level null probabilities
  - expected_rate:     expected_count / opportunities
  - obs_over_expected: selections / expected_count
  - p_value:           exact Poisson-binomial upper-tail probability

Usage:
    uv run python analysis/reselection/post_debut_performance.py
    uv run python analysis/reselection/post_debut_performance.py --include-excerpts
    uv run python analysis/reselection/post_debut_performance.py --alpha 0.01
    uv run python analysis/reselection/post_debut_performance.py --save-csv
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable

import pandas as pd

from afam import DATA_DIR
from afam.cli import add_root_works_flag, add_save_csv_flag
from afam.db import query
from afam.sql import query_path

OUT_WORKS = "post_debut_performance_works.csv"
OUT_AUTHORS = "post_debut_performance_authors.csv"
DEFAULT_ALPHA = 0.05


def _clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _is_root_parent(value: object) -> bool:
    return _clean_id(value) == ""


def _join_unique(values: Iterable[object]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return "; ".join(out)


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.1f}%"


def fmt_float(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    if math.isinf(x):
        return "inf"
    return f"{x:.2f}"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def load_data(only_root_works: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load AFAM anthology selections from PostgreSQL.

    Returns (df_full, df_appearances). The full frame defines the anthology
    opportunity universe; df_appearances applies the optional root-work filter.
    """
    df_full = query(query_path("post-debut-performance"))
    df_full["anthology_publication_year"] = df_full[
        "anthology_publication_year"
    ].astype(int)
    if only_root_works:
        df = filter_root_works(df_full)
    else:
        df = df_full
    return df_full, df


def filter_root_works(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["parent_id"].map(_is_root_parent)].copy()


def build_edition_table(df: pd.DataFrame) -> pd.DataFrame:
    editions = (
        df[["edition_id", "anthology_publication_year"]]
        .dropna(subset=["edition_id"])
        .drop_duplicates("edition_id")
        .sort_values(["anthology_publication_year", "edition_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    editions["edition_ordinal"] = range(len(editions))
    return editions.rename(columns={"anthology_publication_year": "year"})


def build_entity_pairs(
    df: pd.DataFrame, id_col: str, editions: pd.DataFrame
) -> pd.DataFrame:
    pairs = df[[id_col, "edition_id"]].dropna(subset=[id_col, "edition_id"]).copy()
    pairs = pairs[pairs[id_col].map(_clean_id) != ""]
    pairs = pairs.drop_duplicates([id_col, "edition_id"])
    return pairs.merge(editions, on="edition_id", how="left")


def poisson_binomial_upper_tail(k: int, probabilities: Iterable[float]) -> float:
    """Return P(X >= k) for independent Bernoulli trials with unequal p."""
    probs = [min(max(float(p), 0.0), 1.0) for p in probabilities]
    n = len(probs)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0

    dist = [0.0] * (n + 1)
    dist[0] = 1.0
    for i, p in enumerate(probs, start=1):
        for successes in range(i, 0, -1):
            dist[successes] = dist[successes] * (1.0 - p) + dist[successes - 1] * p
        dist[0] *= 1.0 - p
    return float(sum(dist[k:]))


def compute_entity_stats(
    pairs: pd.DataFrame,
    id_col: str,
    editions: pd.DataFrame,
) -> list[dict]:
    """Compute post-debut observed/expected reselection stats for one entity type.

    The null probability for an entity in a later edition is the share of other
    already-debuted entities selected in that edition.
    """
    if pairs.empty:
        return []

    entity_sets: dict[object, set] = (
        pairs.groupby(id_col)["edition_id"].apply(set).to_dict()
    )
    edition_sets: dict[object, set] = (
        pairs.groupby("edition_id")[id_col].apply(set).to_dict()
    )
    debut_ord = pairs.groupby(id_col)["edition_ordinal"].min().to_dict()
    debut_year = (
        pairs.sort_values(["edition_ordinal", "edition_id"])
        .drop_duplicates(id_col)
        .set_index(id_col)["year"]
        .to_dict()
    )
    all_ids = set(entity_sets)

    rows: list[dict] = []
    for entity_id in sorted(all_ids, key=lambda x: str(x)):
        first_ordinal = int(debut_ord[entity_id])
        probabilities: list[float] = []
        selections = 0

        for edition in editions.itertuples(index=False):
            current_ordinal = int(edition.edition_ordinal)
            if current_ordinal <= first_ordinal:
                continue

            eligible = {eid for eid in all_ids if int(debut_ord[eid]) < current_ordinal}
            selected_eligible = edition_sets.get(edition.edition_id, set()) & eligible

            selected = entity_id in selected_eligible
            if selected:
                selections += 1

            denominator = len(eligible) - 1
            if denominator <= 0:
                probabilities.append(0.0)
                continue

            numerator = len(selected_eligible) - (1 if selected else 0)
            probabilities.append(numerator / denominator)

        opportunities = len(probabilities)
        expected_count = float(sum(probabilities))
        selection_rate = selections / opportunities if opportunities else float("nan")
        expected_rate = (
            expected_count / opportunities if opportunities else float("nan")
        )

        if expected_count == 0.0:
            obs_over_expected = float("inf") if selections > 0 else float("nan")
        else:
            obs_over_expected = selections / expected_count

        p_value = (
            poisson_binomial_upper_tail(selections, probabilities)
            if opportunities
            else float("nan")
        )

        rows.append(
            {
                "entity_id": entity_id,
                "first_year": int(debut_year[entity_id]),
                "debut_edition_ordinal": first_ordinal,
                "opportunities": opportunities,
                "selections": selections,
                "selection_rate": selection_rate,
                "expected_count": expected_count,
                "expected_rate": expected_rate,
                "obs_over_expected": obs_over_expected,
                "p_value": p_value,
            }
        )

    return rows


def build_works_df(df: pd.DataFrame, stats_rows: list[dict]) -> pd.DataFrame:
    work_meta = (
        df[
            [
                "work_id",
                "work_title",
                "parent_id",
                "parent_work_title",
            ]
        ]
        .drop_duplicates("work_id")
        .set_index("work_id")
    )
    author_meta = (
        df.dropna(subset=["work_id"])
        .groupby("work_id", sort=False)
        .agg(
            author_ids=("author_id", _join_unique),
            author_names=("author_name", _join_unique),
        )
    )
    stats = (
        pd.DataFrame(stats_rows)
        .rename(columns={"entity_id": "work_id"})
        .set_index("work_id")
    )
    result = work_meta.join(author_meta).join(stats).reset_index()
    return result[
        [
            "work_id",
            "work_title",
            "parent_id",
            "parent_work_title",
            "author_ids",
            "author_names",
            "first_year",
            "opportunities",
            "selections",
            "selection_rate",
            "expected_count",
            "expected_rate",
            "obs_over_expected",
            "p_value",
        ]
    ]


def build_authors_df(df: pd.DataFrame, stats_rows: list[dict]) -> pd.DataFrame:
    author_rows = df[df["author_id"].map(_clean_id) != ""]
    meta = (
        author_rows[["author_id", "author_name", "author_birth_year"]]
        .drop_duplicates("author_id")
        .set_index("author_id")
    )
    stats = (
        pd.DataFrame(stats_rows)
        .rename(columns={"entity_id": "author_id"})
        .set_index("author_id")
    )
    result = meta.join(stats).reset_index()
    return result[
        [
            "author_id",
            "author_name",
            "author_birth_year",
            "first_year",
            "opportunities",
            "selections",
            "selection_rate",
            "expected_count",
            "expected_rate",
            "obs_over_expected",
            "p_value",
        ]
    ]


def aggregate_summary(frame: pd.DataFrame) -> dict[str, float]:
    opportunities = int(frame["opportunities"].sum())
    selections = int(frame["selections"].sum())
    expected = float(frame["expected_count"].sum())
    return {
        "entities": len(frame),
        "opportunities": opportunities,
        "selections": selections,
        "selection_rate": selections / opportunities if opportunities else float("nan"),
        "expected_count": expected,
        "obs_over_expected": selections / expected if expected else float("inf"),
    }


def print_summary(
    works_df: pd.DataFrame, authors_df: pd.DataFrame, alpha: float
) -> None:
    work = aggregate_summary(works_df)
    author = aggregate_summary(authors_df)
    work_sig = int((works_df["p_value"] <= alpha).sum())
    author_sig = int((authors_df["p_value"] <= alpha).sum())

    print("===== POST-DEBUT RESELECTION VS EXPECTED =====")
    print(f"{'':28} {'Works':>12} {'Authors':>12}")
    print(f"{'Entities':28} {work['entities']:>12,} {author['entities']:>12,}")
    print(
        f"{'Later opportunities':28} {work['opportunities']:>12,} {author['opportunities']:>12,}"
    )
    print(
        f"{'Observed reselections':28} {work['selections']:>12,} {author['selections']:>12,}"
    )
    print(
        f"{'Expected reselections':28} {work['expected_count']:>12.1f} {author['expected_count']:>12.1f}"
    )
    print(
        f"{'Observed rate':28} {fmt_pct(work['selection_rate']):>12} {fmt_pct(author['selection_rate']):>12}"
    )
    print(
        f"{'Obs / Expected':28} {fmt_float(work['obs_over_expected']):>12} {fmt_float(author['obs_over_expected']):>12}"
    )
    print(f"{f'Significant p <= {alpha}':28} {work_sig:>12,} {author_sig:>12,}")


def print_works_table(works_df: pd.DataFrame, alpha: float) -> None:
    sig = works_df[works_df["p_value"] <= alpha].sort_values(
        ["obs_over_expected", "selections"],
        ascending=[False, False],
        na_position="last",
    )
    print(f"\n===== TOP WORKS (p <= {alpha}) =====")
    if sig.empty:
        print("  (no significant results)")
        return

    header = (
        f"{'Work Title':40} | {'Author(s)':25} | {'1st Yr':>6} | {'Opps':>4} | "
        f"{'Sel':>3} | {'Exp':>5} | {'Obs/Exp':>7} | {'p-value':>9}"
    )
    print(header)
    print("-" * len(header))
    for _, row in sig.head(25).iterrows():
        print(
            " | ".join(
                [
                    str(row["work_title"])[:40].ljust(40),
                    str(row["author_names"])[:25].ljust(25),
                    str(int(row["first_year"])).rjust(6),
                    str(int(row["opportunities"])).rjust(4),
                    str(int(row["selections"])).rjust(3),
                    f"{row['expected_count']:.1f}".rjust(5),
                    fmt_float(row["obs_over_expected"]).rjust(7),
                    fmt_p(row["p_value"]).rjust(9),
                ]
            )
        )


def print_authors_table(authors_df: pd.DataFrame, alpha: float) -> None:
    sig = authors_df[authors_df["p_value"] <= alpha].sort_values(
        ["obs_over_expected", "selections"],
        ascending=[False, False],
        na_position="last",
    )
    print(f"\n===== TOP AUTHORS (p <= {alpha}) =====")
    if sig.empty:
        print("  (no significant results)")
        return

    header = (
        f"{'Author':35} | {'Birth Yr':>8} | {'1st Yr':>6} | {'Opps':>4} | "
        f"{'Sel':>3} | {'Exp':>5} | {'Obs/Exp':>7} | {'p-value':>9}"
    )
    print(header)
    print("-" * len(header))
    for _, row in sig.head(25).iterrows():
        birth_year = _clean_id(row["author_birth_year"])
        print(
            " | ".join(
                [
                    str(row["author_name"])[:35].ljust(35),
                    birth_year.rjust(8),
                    str(int(row["first_year"])).rjust(6),
                    str(int(row["opportunities"])).rjust(4),
                    str(int(row["selections"])).rjust(3),
                    f"{row['expected_count']:.1f}".rjust(5),
                    fmt_float(row["obs_over_expected"]).rjust(7),
                    fmt_p(row["p_value"]).rjust(9),
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_works_flag(parser)
    add_save_csv_flag(parser)
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        metavar="FLOAT",
        help=f"Significance threshold for stdout tables (default: {DEFAULT_ALPHA}).",
    )
    args = parser.parse_args()

    df_full, df = load_data(args.only_root_works)
    editions = build_edition_table(df_full)

    work_pairs = build_entity_pairs(df, "work_id", editions)
    author_pairs = build_entity_pairs(df, "author_id", editions)

    works_df = build_works_df(df, compute_entity_stats(work_pairs, "work_id", editions))
    authors_df = build_authors_df(
        df, compute_entity_stats(author_pairs, "author_id", editions)
    )

    if args.save_csv:
        works_path = DATA_DIR / OUT_WORKS
        authors_path = DATA_DIR / OUT_AUTHORS
        works_df.to_csv(works_path, index=False)
        authors_df.to_csv(authors_path, index=False)
        print(f"Saved -> {works_path}  ({len(works_df):,} works)")
        print(f"Saved -> {authors_path}  ({len(authors_df):,} authors)")
        print()

    print_summary(works_df, authors_df, args.alpha)
    print_works_table(works_df, args.alpha)
    print_authors_table(authors_df, args.alpha)


if __name__ == "__main__":
    main()
