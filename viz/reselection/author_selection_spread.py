import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator

from afam.db import query as query_db
from afam.sql import query_path
from afam.viz_style import OUTPUT_DIR

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


def load_data() -> pd.DataFrame:
    """Load author × work × form × edition rows from the live DB, shaped to the
    columns the stats builder consumes (work_author, work_id, edition_key,
    work_form, parent_work_id). edition_key = edition_id."""
    df = query_db(query_path("author-page-share-reselection"))
    df = df.dropna(subset=["author_id"]).copy()
    df["work_author"] = df["author_name"]
    df["edition_key"] = df["edition_id"]
    df["work_form"] = df["form_name"].fillna("Unknown")
    df["parent_work_id"] = df["parent_id"].apply(
        lambda v: "" if pd.isna(v) else str(int(v))
    )
    return df


def compute_author_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute X, Y, number of works, and modal form per author."""
    records = []
    for author_id, group in df.groupby("author_id"):
        author_name = group["work_author"].iloc[0]
        # Unique anthologies selecting this author
        edition_keys = group["edition_key"].unique()
        X = len(edition_keys)
        # Count unique anthologies per work
        work_counts = group.groupby("work_id")["edition_key"].nunique()
        modal_work_count = work_counts.max() if not work_counts.empty else 0
        # Y = share of selections that repeat same work
        Y = modal_work_count / X if X > 0 else 0.0
        # Distinct works by author
        num_works = group["work_id"].nunique()
        # Modal work form
        form_counts = group["work_form"].value_counts()
        modal_form = form_counts.idxmax() if not form_counts.empty else ""
        records.append(
            {
                "author_id": author_id,
                "author": author_name,
                "X": X,
                "Y": Y,
                "num_works": num_works,
                "modal_form": modal_form,
            }
        )
    return pd.DataFrame.from_records(records)


def make_plot(
    df_stats: pd.DataFrame, period: str, root_flag: str, n_thresh: int, viz_dir: Path
) -> None:
    """Generate and save scatter plot for a given threshold and filter."""
    x = df_stats["X"].to_numpy()
    y = df_stats["Y"].to_numpy() * 100
    sizes = df_stats["num_works"].to_numpy() * 30
    forms = sorted(df_stats["modal_form"].unique())
    cmap = plt.get_cmap("Dark2")
    palette_colors = getattr(
        cmap, "colors", [cmap(i / (len(forms) - 1)) for i in range(len(forms))]
    )
    form_palette = dict(zip(forms, palette_colors[: len(forms)]))
    # Create figure with specified size and resolution
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    # -------- Scatter ------------
    ax.scatter(
        x,
        y,
        s=sizes,
        c=df_stats["modal_form"].map(form_palette),
        alpha=0.6,
        edgecolors="black",
        linewidths=0.4,
    )
    # Integer x-axis only
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Y-axis fixed and reference line
    ax.set_ylim(0, 110)
    ax.axhline(100, color="grey", alpha=0.3, lw=1, zorder=0)
    # Labels and title
    ax.set_xlabel(
        "Number of African American literature anthologies selecting an author"
    )
    ax.set_ylabel("Percentage of selections repeating the same work")
    ax.set_title(
        "Author selection across African American literature anthologies"
        "\n(Root works only; n ≥ 13)"
    )
    # Legend for forms
    for form, color in form_palette.items():
        ax.scatter([], [], c=[color], label=form, s=50, edgecolors="black")
    ax.legend(
        title="Most common form\n(size ∝ total works selected)",
        frameon=False,
        loc="best",
    )
    # -------- Labels -------------
    # -------- Labels -------------
    texts = [
        ax.text(xi, yi + 1, author, fontsize=6, color=form_palette[form])
        for xi, yi, author, form in zip(
            x, y, df_stats["author"], df_stats["modal_form"]
        )
    ]
    # prevent overlaps (adjustText required)
    if adjust_text:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", lw=0.3, color="grey"))
    # Save figure with layout and resolution
    out_path = viz_dir / f"author_spread_{period}_root{root_flag}_n{n_thresh}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    df = load_data()
    df_sub = df[df["parent_work_id"] == ""]
    stats = compute_author_stats(df_sub)
    stats_n = stats[stats["X"] >= 13]
    make_plot(stats_n, "all_time", "root_only", 13, OUTPUT_DIR)
    print("Created figure: viz/author_spread_all_time_rootroot_only_n13.png")


if __name__ == "__main__":
    main()
