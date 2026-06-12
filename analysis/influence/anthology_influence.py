"""
anthology_influence.py
----------------------
Models the influence of each African American literary anthology edition on
subsequent editions: how much more often do an edition's selections (authors
and works, separately) appear in later editions than we would expect from the
corpus as a whole?

For each source edition A and each subsequent edition B (strictly later
publication year — same-year editions were prepared concurrently and cannot
have influenced each other), every item attributed to A is a trial and its
appearance in B is a success. The chance baseline for B is the rate at which
B picks up *any* already-anthologized item: |B ∩ Pool(B)| / |Pool(B)|, where
Pool(B) is the union of all selections from editions published strictly
before B. Because year(A) < year(B), all of A's items are automatically in
Pool(B), so no per-pair eligibility filtering is needed.

Two attribution variants are reported side by side:
  - all:    every item in A ("taste confirmation")
  - debut:  only items whose first corpus appearance (in (year, edition_id)
            order, the repo's debut convention) was in A ("agenda setting")
For the chronologically first edition the two variants coincide by
construction.

Pooling across B's follows reselection_vs_chance.py: successes and trials are
summed, the expected rate is the trial-weighted mean of per-B chance rates,
and lift = observed rate / expected rate. p-values come from a two-sided
scipy.stats.binomtest of pooled successes against the weighted expected rate.

The uniform-pool baseline overstates influence for editions whose slates sit
in the high-popularity tail (item pickup propensity is fat-tailed: the median
corpus item appears in one edition). --stratify-prior-count replaces it with
indirect standardization: for each target B, pool items are stratified by how
many qualifying editions selected them strictly before B, and A's expected
successes are the sum over its items of their stratum's pickup rate by B.
Lift is then a standardized selection ratio (observed / expected successes,
SMR-style). Prior counts use only pre-B information, so there is no outcome
circularity; but because influence that operates *through* intermediate
reselections is absorbed into the strata, the stratified lift is a
conservative lower bound, while the uniform-pool lift is an upper bound.

Caveats: trials are not independent (the same item recurs across B's, and
items within an edition compete for slots), so p-values are approximate.
With 26 editions x 2 entities x 2 variants there are ~104 tests; the stdout
footer prints the Bonferroni-adjusted significance threshold.

Usage:
    uv run python analysis/influence/anthology_influence.py
    uv run python analysis/influence/anthology_influence.py --include-excerpts
    uv run python analysis/influence/anthology_influence.py --exclude-within-series
    uv run python analysis/influence/anthology_influence.py --stratify-prior-count
    uv run python analysis/influence/anthology_influence.py --save-csv
"""

from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import binomtest

from afam import DATA_DIR
from afam.cli import add_root_works_flag, add_save_csv_flag
from afam.db import query
from afam.editions import EDITION_LABELS
from afam.sql import query_path

N_TESTS_ALPHA = 0.05


def load_data(only_root_works: bool = True) -> pd.DataFrame:
    df = query(query_path("works-per-afam-edition"))
    if only_root_works:
        df = df[df["parent_id"].isna()]
    return df


# ── Edition decomposition ─────────────────────────────────────────────────────


def sorted_edition_ids(df: pd.DataFrame) -> list:
    ed_year = df.groupby("edition_id")["anthology_publication_year"].min()
    return [eid for eid, _ in sorted(ed_year.items(), key=lambda x: (x[1], x[0]))]


def edition_years(df: pd.DataFrame) -> dict[object, int]:
    return (
        df.groupby("edition_id")["anthology_publication_year"]
        .min()
        .astype(int)
        .to_dict()
    )


def edition_work_sets(df: pd.DataFrame) -> dict[object, set]:
    return {eid: set(g["work_id"]) for eid, g in df.groupby("edition_id")}


def edition_author_sets(df: pd.DataFrame) -> dict[object, set]:
    return {eid: set(g["author_id"].dropna()) for eid, g in df.groupby("edition_id")}


def edition_entry_groups(df: pd.DataFrame) -> dict[object, str]:
    """Map edition_id → series group key.

    Editions in a series share ``series:<id>``; standalone anthologies get
    ``edition:<id>`` so each is its own group.
    """
    firsts = df.drop_duplicates("edition_id").set_index("edition_id")["series_id"]
    return {
        eid: f"series:{int(sid)}" if pd.notna(sid) else f"edition:{eid}"
        for eid, sid in firsts.items()
    }


# ── Attribution and baseline ──────────────────────────────────────────────────


def compute_debut_sets(
    ed_items: dict[object, set],
    sorted_editions: list,
) -> dict[object, set]:
    """Items whose first appearance (in (year, edition_id) order) was in each edition."""
    debuts: dict[object, set] = {}
    seen: set = set()
    for eid in sorted_editions:
        items = ed_items.get(eid, set())
        debuts[eid] = items - seen
        seen |= items
    return debuts


def compute_expected_rates(
    ed_items: dict[object, set],
    ed_year: dict[object, int],
    sorted_editions: list,
    entry_group: dict[object, str] | None = None,
    exclude_within_series: bool = False,
) -> dict[object, tuple[int, int]]:
    """Return {edition_id: (|B ∩ Pool(B)|, |Pool(B)|)}.

    Pool(B) is the union of selections from editions published strictly
    before B. With exclude_within_series, editions in B's own series are
    dropped from the pool so a series reprinting itself does not inflate the
    chance rate every cross-series source is judged against.
    """
    expected: dict[object, tuple[int, int]] = {}
    for b in sorted_editions:
        pool: set = set()
        for c in sorted_editions:
            if ed_year[c] >= ed_year[b]:
                continue
            if (
                exclude_within_series
                and entry_group is not None
                and entry_group[c] == entry_group[b]
            ):
                continue
            pool |= ed_items.get(c, set())
        expected[b] = (len(ed_items.get(b, set()) & pool), len(pool))
    return expected


def compute_prior_counts(
    ed_items: dict[object, set],
    ed_year: dict[object, int],
    sorted_editions: list,
    entry_group: dict[object, str] | None = None,
    exclude_within_series: bool = False,
) -> dict[object, dict]:
    """Return {edition_id: {item: n qualifying editions selecting it strictly before}}.

    The qualifying-edition rule matches compute_expected_rates, so the keys of
    each inner dict are exactly Pool(B).
    """
    counts: dict[object, dict] = {}
    for b in sorted_editions:
        c_b: dict = {}
        for c in sorted_editions:
            if ed_year[c] >= ed_year[b]:
                continue
            if (
                exclude_within_series
                and entry_group is not None
                and entry_group[c] == entry_group[b]
            ):
                continue
            for item in ed_items.get(c, set()):
                c_b[item] = c_b.get(item, 0) + 1
        counts[b] = c_b
    return counts


def compute_stratum_rates(
    prior_counts_b: dict,
    items_b: set,
) -> dict[int, tuple[int, int]]:
    """Return {prior count: (n pool items picked by B, n pool items)}."""
    strata: dict[int, list[int]] = {}
    for item, count in prior_counts_b.items():
        tally = strata.setdefault(count, [0, 0])
        tally[1] += 1
        if item in items_b:
            tally[0] += 1
    return {count: (picked, total) for count, (picked, total) in strata.items()}


# ── Influence table ───────────────────────────────────────────────────────────


def compute_influence_table(
    ed_items: dict[object, set],
    attributed: dict[object, set],
    ed_year: dict[object, int],
    sorted_editions: list,
    expected: dict[object, tuple[int, int]],
    entry_group: dict[object, str] | None = None,
    exclude_within_series: bool = False,
) -> pd.DataFrame:
    """One row per source edition A with pooled forward-pickup statistics."""
    rows: list[dict] = []
    for a in sorted_editions:
        s_a = attributed.get(a, set())
        n_subsequent = 0
        trials = 0
        successes = 0
        weighted_p = 0.0
        for b in sorted_editions:
            if ed_year[b] <= ed_year[a]:
                continue
            if (
                exclude_within_series
                and entry_group is not None
                and entry_group[b] == entry_group[a]
            ):
                continue
            n_subsequent += 1
            n_ab = len(s_a)
            inter, pool_size = expected[b]
            trials += n_ab
            successes += len(s_a & ed_items.get(b, set()))
            if pool_size:
                weighted_p += n_ab * (inter / pool_size)

        if trials:
            obs_rate = successes / trials
            exp_rate = weighted_p / trials
            lift = obs_rate / exp_rate if exp_rate else float("nan")
            p_value = (
                binomtest(successes, trials, p=exp_rate, alternative="two-sided").pvalue
                if 0.0 <= exp_rate <= 1.0 and exp_rate > 0.0
                else float("nan")
            )
        else:
            obs_rate = exp_rate = lift = p_value = float("nan")

        rows.append(
            {
                "edition_id": a,
                "year": ed_year[a],
                "n_items": len(s_a),
                "n_subsequent": n_subsequent,
                "trials": trials,
                "successes": successes,
                "obs_rate": obs_rate,
                "exp_rate": exp_rate,
                "lift": lift,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def compute_stratified_influence_table(
    ed_items: dict[object, set],
    attributed: dict[object, set],
    ed_year: dict[object, int],
    sorted_editions: list,
    prior_counts: dict[object, dict],
    entry_group: dict[object, str] | None = None,
    exclude_within_series: bool = False,
) -> pd.DataFrame:
    """Influence table with expectations standardized by prior selection count.

    For each target B, A's expected successes are the sum over A's items of
    the pickup rate B shows for pool items with the same prior count
    (indirect standardization); lift = observed / expected successes.
    """
    stratum_rates = {
        b: compute_stratum_rates(prior_counts[b], ed_items.get(b, set()))
        for b in sorted_editions
    }

    rows: list[dict] = []
    for a in sorted_editions:
        s_a = attributed.get(a, set())
        n_subsequent = 0
        trials = 0
        successes = 0
        expected_successes = 0.0
        for b in sorted_editions:
            if ed_year[b] <= ed_year[a]:
                continue
            if (
                exclude_within_series
                and entry_group is not None
                and entry_group[b] == entry_group[a]
            ):
                continue
            n_subsequent += 1
            trials += len(s_a)
            successes += len(s_a & ed_items.get(b, set()))
            counts_b = prior_counts[b]
            rates_b = stratum_rates[b]
            for item in s_a:
                picked, total = rates_b[counts_b[item]]
                expected_successes += picked / total

        if trials:
            obs_rate = successes / trials
            exp_rate = expected_successes / trials
            lift = obs_rate / exp_rate if exp_rate else float("nan")
            p_value = (
                binomtest(successes, trials, p=exp_rate, alternative="two-sided").pvalue
                if 0.0 < exp_rate <= 1.0
                else float("nan")
            )
        else:
            obs_rate = exp_rate = lift = p_value = float("nan")

        rows.append(
            {
                "edition_id": a,
                "year": ed_year[a],
                "n_items": len(s_a),
                "n_subsequent": n_subsequent,
                "trials": trials,
                "successes": successes,
                "obs_rate": obs_rate,
                "exp_rate": exp_rate,
                "lift": lift,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def compute_all_variants(
    df: pd.DataFrame,
    exclude_within_series: bool = False,
    stratify_prior_count: bool = False,
) -> pd.DataFrame:
    """Long-format influence table: one row per (edition, entity, variant)."""
    sorted_editions = sorted_edition_ids(df)
    ed_year = edition_years(df)
    entry_group = edition_entry_groups(df)

    blocks: list[pd.DataFrame] = []
    for entity, ed_items in [
        ("works", edition_work_sets(df)),
        ("authors", edition_author_sets(df)),
    ]:
        if stratify_prior_count:
            prior_counts = compute_prior_counts(
                ed_items,
                ed_year,
                sorted_editions,
                entry_group=entry_group,
                exclude_within_series=exclude_within_series,
            )
        else:
            expected = compute_expected_rates(
                ed_items,
                ed_year,
                sorted_editions,
                entry_group=entry_group,
                exclude_within_series=exclude_within_series,
            )
        debut_sets = compute_debut_sets(ed_items, sorted_editions)
        for variant, attributed in [("all", ed_items), ("debut", debut_sets)]:
            if stratify_prior_count:
                table = compute_stratified_influence_table(
                    ed_items,
                    attributed,
                    ed_year,
                    sorted_editions,
                    prior_counts,
                    entry_group=entry_group,
                    exclude_within_series=exclude_within_series,
                )
            else:
                table = compute_influence_table(
                    ed_items,
                    attributed,
                    ed_year,
                    sorted_editions,
                    expected,
                    entry_group=entry_group,
                    exclude_within_series=exclude_within_series,
                )
            table.insert(0, "entity", entity)
            table.insert(1, "variant", variant)
            blocks.append(table)

    out = pd.concat(blocks, ignore_index=True)
    out.insert(2, "label", out["edition_id"].map(EDITION_LABELS))
    return out


# ── Output ────────────────────────────────────────────────────────────────────


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:.1f}%"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fmt_lift(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.2f}"


def print_tables(results: pd.DataFrame) -> None:
    for entity in ["works", "authors"]:
        block = results[results["entity"] == entity]
        all_t = block[block["variant"] == "all"].set_index("edition_id")
        deb_t = block[block["variant"] == "debut"].set_index("edition_id")

        print(f"===== {entity.upper()}: FORWARD PICKUP vs CORPUS BASELINE =====")
        header = (
            f"{'Edition':28} {'Year':>4} "
            f"{'N':>5} {'Obs':>7} {'Exp':>7} {'Lift':>6} {'p':>9}  "
            f"{'N.deb':>5} {'Obs':>7} {'Exp':>7} {'Lift':>6} {'p':>9}"
        )
        sub = f"{'':28} {'':>4} {'──── all selections ────':>37}  {'────── debuts only ──────':>37}"
        print(sub)
        print(header)
        for eid in all_t.index:
            a = all_t.loc[eid]
            d = deb_t.loc[eid]
            label = (a["label"] if pd.notna(a["label"]) else str(eid))[:28]
            print(
                f"{label:28} {int(a['year']):>4} "
                f"{int(a['n_items']):>5} {fmt_pct(a['obs_rate']):>7} {fmt_pct(a['exp_rate']):>7}"
                f" {fmt_lift(a['lift']):>6} {fmt_p(a['p_value']):>9}  "
                f"{int(d['n_items']):>5} {fmt_pct(d['obs_rate']):>7} {fmt_pct(d['exp_rate']):>7}"
                f" {fmt_lift(d['lift']):>6} {fmt_p(d['p_value']):>9}"
            )
        print()

    n_tests = int(results["p_value"].notna().sum())
    if n_tests:
        print(
            f"Note: {n_tests} binomial tests reported; "
            f"Bonferroni-adjusted threshold for α={N_TESTS_ALPHA}: "
            f"{N_TESTS_ALPHA / n_tests:.2e}"
        )
    print(
        "Trials are not independent (items recur across editions), "
        "so p-values are approximate."
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_works_flag(parser)
    add_save_csv_flag(parser)
    parser.add_argument(
        "--exclude-within-series",
        action="store_true",
        help=(
            "Skip source/target pairs within the same anthology series and "
            "drop a series' own prior selections from its baseline pool."
        ),
    )
    parser.add_argument(
        "--stratify-prior-count",
        action="store_true",
        help=(
            "Standardize expectations by per-item prior selection count "
            "(popularity-adjusted baseline; conservative lower bound on influence)."
        ),
    )
    args = parser.parse_args()

    df = load_data(only_root_works=args.only_root_works)
    results = compute_all_variants(
        df,
        exclude_within_series=args.exclude_within_series,
        stratify_prior_count=args.stratify_prior_count,
    )

    if args.save_csv:
        suffix = ("_cross_series" if args.exclude_within_series else "") + (
            "_stratified" if args.stratify_prior_count else ""
        )
        out_path = DATA_DIR / f"anthology_influence{suffix}.csv"
        results.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")
    else:
        if args.stratify_prior_count:
            print(
                "Baseline: stratified by per-item prior selection count "
                "(indirect standardization)\n"
            )
        print_tables(results)


if __name__ == "__main__":
    main()
