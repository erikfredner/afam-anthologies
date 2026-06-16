#!/usr/bin/env python3
"""
work_network.py
--------------------

Build a bipartite work ⇄ anthology-edition network, export a work–work
co-occurrence table, and draw the graph.

Based on viz/anthology_network.py, but nodes represent works instead of authors.
"""

from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from adjustText import adjust_text
from matplotlib import colormaps, colors


# ---------------------------------------------------------------------
# 1. Helper: collapsed edition ID
# ---------------------------------------------------------------------
def make_group_id(series: str, edition) -> str:
    return f"{series}||{edition}"


# ---------------------------------------------------------------------
# 2. Build work–work co-occurrence rows
# ---------------------------------------------------------------------
def build_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame of pairs of works that co-occur in the same anthology edition.
    Columns: group_id, work1_id, work2_id
    """
    rows: list[dict[str, object]] = []
    for gid, sub in df.groupby("group_id"):
        # unique works in this group
        work_ids = sub["work_id"].drop_duplicates()
        for w1, w2 in itertools.combinations(work_ids, 2):
            rows.append({"group_id": gid, "work1_id": w1, "work2_id": w2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. Build bipartite graph
# ---------------------------------------------------------------------
def build_graph(df: pd.DataFrame, cooc: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    # anthology-edition nodes
    for row in (
        df[["group_id", "series_name", "anthology_edition"]]
        .drop_duplicates(subset=["group_id"])
        .itertuples(index=False)
    ):
        G.add_node(
            f"a{row.group_id}",
            type="anthology",
            series=row.series_name,
            label=str(row.anthology_edition),
        )

    # work nodes
    for r in (
        df[["work_id", "work_title", "author_name"]]
        .drop_duplicates(subset=["work_id"])
        .itertuples(index=False)
    ):
        G.add_node(
            f"w{r.work_id}",
            type="work",
            name=r.work_title,
            author=r.author_name,
        )

    # membership edges (work → anthology edition)
    for r in df.itertuples(index=False):
        G.add_edge(f"w{r.work_id}", f"a{r.group_id}", edge_type="membership")

    # work–work co-occurrence edges
    weights: defaultdict[tuple[str, str], int] = defaultdict(int)
    for r in cooc.itertuples(index=False):
        u, v = sorted((f"w{r.work1_id}", f"w{r.work2_id}"))
        weights[(u, v)] += 1
    for (u, v), w in weights.items():
        G.add_edge(u, v, edge_type="cooccurrence", weight=w)

    return G


# ---------------------------------------------------------------------
# 4. Draw graph
# ---------------------------------------------------------------------
def draw_graph(G: nx.Graph, outfile: pathlib.Path, seed: int = 42) -> None:
    """
    Draw the bipartite work–anthology network and co-occurrence subgraph.
    """
    random.seed(seed)

    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.45, seed=seed)

    # dynamic series palette
    series_set = {
        d["series"] for _, d in G.nodes(data=True) if d["type"] == "anthology"
    }
    cmap_tab = colormaps.get_cmap("tab10")
    series_palette = {
        s: colors.to_hex(cmap_tab(i % cmap_tab.N))
        for i, s in enumerate(sorted(series_set))
    }
    viridis = colormaps.get_cmap("viridis")

    anth_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "anthology"]
    work_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "work"]

    # work colour by total co-occurrence strength
    work_strength = {
        n: sum(
            G.edges[n, nbr]["weight"]
            for nbr in G.neighbors(n)
            if G.nodes[nbr]["type"] == "work"
        )
        for n in work_nodes
    }
    vmin, vmax = (
        min(work_strength.values(), default=0),
        max(work_strength.values(), default=0),
    )
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    work_colours = [colors.to_hex(viridis(norm(work_strength[n]))) for n in work_nodes]

    # anthology nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=anth_nodes,
        node_color=[series_palette[G.nodes[n]["series"]] for n in anth_nodes],
        node_shape="s",
        node_size=550,
        edgecolors="black",
        linewidths=0.8,
        ax=ax,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: G.nodes[n]["label"] for n in anth_nodes},
        font_size=8,
        font_weight="bold",
        ax=ax,
    )

    # work nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=work_nodes,
        node_color=work_colours,
        node_size=150,
        edgecolors="black",
        linewidths=0.3,
        alpha=0.65,
        ax=ax,
    )

    # edges: membership
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[e for e, d in G.edges.items() if d.get("edge_type") == "membership"],
        width=0.2,
        alpha=0.15,
        edge_color="lightgrey",
        ax=ax,
    )
    # edges: co-occurrence
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[
            e for e, d in G.edges.items() if d.get("edge_type") == "cooccurrence"
        ],
        width=[
            0.4 * G.edges[e]["weight"]
            for e in G.edges
            if G.edges[e].get("edge_type") == "cooccurrence"
        ],
        alpha=0.35,
        edge_color="grey",
        ax=ax,
    )

    # annotate a random sample of works (up to 10)
    texts: list = []
    sample_size = min(10, len(work_nodes))
    if sample_size > 0:
        sampled = random.sample(work_nodes, sample_size)
        for n in sampled:
            x, y = pos[n]
            text = ax.text(
                x,
                y,
                f"{G.nodes[n]['name']}",
                fontsize=7,
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
            )
            texts.append(text)
        # repel labels & draw white leader lines
        adjust_text(
            texts,
            [pos[n][0] for n in sampled],
            [pos[n][1] for n in sampled],
            ax=ax,
            expand_points=(1.2, 1.2),
            expand_text=(1.2, 1.4),
            arrowprops=dict(
                arrowstyle="-",
                lw=0.5,
                color="white",
                shrinkA=5,
                shrinkB=5,
            ),
        )

    # colour-bar for co-occurrence strength
    sm = plt.cm.ScalarMappable(norm=norm, cmap=viridis)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label("Total work–work co-occurrences", fontsize=9)

    # legend for series
    handles = [
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            color=series_palette[s],
            markersize=8,
            label=s,
        )
        for s in series_palette
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    print(f"[INFO] Network plot saved → {outfile}")


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input works CSV file")
    parser.add_argument("--out", default="work_network.png", help="Output PNG file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for layout")
    parser.add_argument(
        "--top-percent",
        type=float,
        default=0.01,
        help="Fraction of works to keep by top co-occurrence strength (e.g. 0.01 for top 1%%)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # create group id for anthology-edition
    df["group_id"] = df.apply(
        lambda r: make_group_id(r["series_name"], r["anthology_edition"]), axis=1
    )

    # build full co-occurrence to compute work strengths
    cooc_full = build_cooccurrence(df)
    # count co-occurrences per work (including zeros)
    # start with zero counts for all works
    all_work_ids = pd.Series(df["work_id"].unique())
    counts = pd.Series(0, index=all_work_ids.values).astype(int)
    # tally from cooc_full
    m1 = cooc_full[["work1_id"]].rename(columns={"work1_id": "work_id"})
    m2 = cooc_full[["work2_id"]].rename(columns={"work2_id": "work_id"})
    all_ids = pd.concat([m1, m2], ignore_index=True)
    vc = all_ids["work_id"].value_counts()
    counts.loc[vc.index] = vc
    # determine threshold for top works
    pct = args.top_percent
    if not (0 < pct < 1):
        raise ValueError("--top-percent must be between 0 and 1")
    threshold = counts.quantile(1 - pct)
    top_ids = counts[counts >= threshold].index
    kept = len(top_ids)
    total = len(counts)
    print(
        f"[INFO] Keeping top {kept}/{total} works (>= {threshold:.0f} co-occurrences, top {pct * 100:.1f}%)"
    )
    # filter df to only top works
    df = df[df["work_id"].isin(top_ids)]
    # rebuild co-occurrence on filtered data
    cooc = build_cooccurrence(df)
    # save filtered co-occurrence
    os.makedirs("data", exist_ok=True)
    cooc_path = pathlib.Path("data") / "work_cooccurrence_top1.csv"
    cooc.to_csv(cooc_path, index=False)
    print(f"[INFO] Top co-occurrence table saved → {cooc_path}")

    G = build_graph(df, cooc)
    draw_graph(G, pathlib.Path(args.out), seed=args.seed)


if __name__ == "__main__":  # noqa: C901
    main()
