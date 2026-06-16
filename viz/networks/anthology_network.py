#!/usr/bin/env python3
"""
anthology_network.py
--------------------

Build a bipartite author ⇄ anthology‑edition network, export an author‑author
co‑occurrence table, and draw the graph.

April 21 2025 — revision 4
• Volumes collapsed → one node per anthology *series + edition*.
• Author nodes coloured (viridis) by total co‑occurrences.
• Annotate 5 random authors at the global max and 5 at the global min.
  ‑ Labels now show **author name only** (count removed).
  ‑ Leader lines drawn in **white**, distinct from grey network edges.
  ‑ `shrinkA`/`shrinkB` increased to silence FancyArrowPatch warning.
• Thinner, fainter edges.

Install
-------
pip install pandas networkx matplotlib adjustText
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
# 2. Build author–author co‑occurrence rows
# ---------------------------------------------------------------------
def build_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gid, sub in df.groupby("group_id"):
        authors = sub[["author_id", "author_name"]].drop_duplicates()
        for (a1_id, a1_name), (a2_id, a2_name) in itertools.combinations(
            authors.itertuples(index=False), 2
        ):
            rows.append(
                {
                    "group_id": gid,
                    "author1_id": a1_id,
                    "author1_name": a1_name,
                    "author2_id": a2_id,
                    "author2_name": a2_name,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. Build bipartite graph
# ---------------------------------------------------------------------
def build_graph(df: pd.DataFrame, cooc: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    # anthology‑edition nodes
    for row in (
        df[["group_id", "anthology_series", "anthology_edition"]]
        .drop_duplicates("group_id")
        .itertuples(index=False)
    ):
        G.add_node(
            f"a{row.group_id}",
            type="anthology",
            series=row.anthology_series,
            label=str(row.anthology_edition),
        )

    # author nodes
    for r in df[["author_id", "author_name"]].drop_duplicates().itertuples(index=False):
        G.add_node(f"u{r.author_id}", type="author", name=r.author_name)

    # membership edges
    for r in df.itertuples(index=False):
        G.add_edge(f"u{r.author_id}", f"a{r.group_id}", edge_type="membership")

    # author‑author co‑occurrence edges
    weights: defaultdict[tuple[str, str], int] = defaultdict(int)
    for r in cooc.itertuples(index=False):
        u, v = sorted((f"u{r.author1_id}", f"u{r.author2_id}"))
        weights[(u, v)] += 1
    for (u, v), w in weights.items():
        G.add_edge(u, v, edge_type="cooccurrence", weight=w)

    return G


# ---------------------------------------------------------------------
# 4. Draw graph
# ---------------------------------------------------------------------
def draw_graph(G: nx.Graph, outfile: pathlib.Path, seed: int = 42) -> None:
    random.seed(seed)

    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.45, seed=seed)

    # palettes
    series_palette = {
        "The Norton Anthology of American Literature": "#1f77b4",
        "The Norton Anthology of African American Literature": "#ff7f0e",
    }
    viridis = colormaps.get_cmap("viridis")

    anth_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "anthology"]
    author_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "author"]

    # author colour by total co‑occurrence strength
    author_strength = {
        n: sum(
            G.edges[n, nbr]["weight"]
            for nbr in G.neighbors(n)
            if G.nodes[nbr]["type"] == "author"
        )
        for n in author_nodes
    }
    vmin, vmax = min(author_strength.values()), max(author_strength.values())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    author_colours = [
        colors.to_hex(viridis(norm(author_strength[n]))) for n in author_nodes
    ]

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

    # author nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=author_nodes,
        node_color=author_colours,
        node_size=150,
        edgecolors="black",
        linewidths=0.3,
        alpha=0.65,
        ax=ax,
    )

    # edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[e for e, d in G.edges.items() if d["edge_type"] == "membership"],
        width=0.2,
        alpha=0.15,
        edge_color="lightgrey",
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[e for e, d in G.edges.items() if d["edge_type"] == "cooccurrence"],
        width=[
            0.4 * G.edges[e]["weight"]
            for e in G.edges
            if G.edges[e].get("edge_type") == "cooccurrence"
        ],
        alpha=0.35,
        edge_color="grey",
        ax=ax,
    )

    # annotate extremes
    texts = []
    max_val, min_val = vmax, vmin
    max_auths = [n for n, v in author_strength.items() if v == max_val]
    min_auths = [n for n, v in author_strength.items() if v == min_val]

    for group in [
        random.sample(max_auths, min(5, len(max_auths))),
        random.sample(min_auths, min(5, len(min_auths))),
    ]:
        for n in group:
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
        [pos[n][0] for n in G.nodes],
        [pos[n][1] for n in G.nodes],
        ax=ax,
        expand_points=(1.2, 1.2),
        expand_text=(1.2, 1.4),
        arrowprops=dict(
            arrowstyle="-",
            lw=0.5,
            color="white",  # distinct from grey edges
            shrinkA=5,  # increased to avoid overlap (silences warning)
            shrinkB=5,
        ),
    )

    # colour‑bar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=viridis)
    fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02).set_label(
        "Total author–author co‑occurrences", fontsize=9
    )

    # legend
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
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--out", default="network.png", help="Output PNG file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["group_id"] = df.apply(
        lambda r: make_group_id(r["anthology_series"], r["anthology_edition"]), axis=1
    )

    cooc = build_cooccurrence(df)

    os.makedirs("data", exist_ok=True)
    cooc_path = pathlib.Path("data") / "author_cooccurrence.csv"
    cooc.to_csv(cooc_path, index=False)
    print(f"[INFO] Co‑occurrence table saved → {cooc_path}")

    G = build_graph(df, cooc)
    draw_graph(G, pathlib.Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()
