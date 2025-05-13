import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score
from matplotlib.ticker import MaxNLocator
try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


def load_data() -> pd.DataFrame:
    """Load author-work-form data and prepare edition keys."""
    csv_path = Path(__file__).parent / 'data' / '202505131423 author work form.csv'
    df = pd.read_csv(csv_path, dtype=str, na_filter=False)
    # filter out authorless rows
    df = df[df["work_author"].str.strip() != ""]
    # Convert year to integer
    df['anthology_year'] = df['anthology_year'].astype(int)
    # Ensure parent_work_id exists
    if 'parent_work_id' not in df.columns:
        df['parent_work_id'] = ''
    # Build unique anthology edition key
    # Use series_id + edition when series_id is present, else anthology_id
    if 'series_id' not in df.columns:
        df['series_id'] = ''
    if 'anthology_edition' not in df.columns:
        df['anthology_edition'] = ''
    if 'anthology_id' not in df.columns:
        df['anthology_id'] = ''
    df['edition_key'] = df.apply(
        lambda row: f"{row['series_id']}_{row['anthology_edition']}"
        if row['series_id'] else row['anthology_id'],
        axis=1,
    )
    return df


def compute_author_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute X, Y, number of works, and modal form per author."""
    records = []
    for author_id, group in df.groupby('author_id'):
        author_name = group['work_author'].iloc[0]
        # Unique anthologies selecting this author
        edition_keys = group['edition_key'].unique()
        X = len(edition_keys)
        # Count unique anthologies per work
        work_counts = group.groupby('work_id')['edition_key'].nunique()
        modal_work_count = work_counts.max() if not work_counts.empty else 0
        # Y = share of selections that repeat same work
        Y = modal_work_count / X if X > 0 else 0.0
        # Distinct works by author
        num_works = group['work_id'].nunique()
        # Modal work form
        form_counts = group['work_form'].value_counts()
        modal_form = form_counts.idxmax() if not form_counts.empty else ''
        records.append({
            'author_id': author_id,
            'author': author_name,
            'X': X,
            'Y': Y,
            'num_works': num_works,
            'modal_form': modal_form,
        })
    return pd.DataFrame.from_records(records)


def make_plot(df_stats: pd.DataFrame, period: str, root_flag: str,
              n_thresh: int, viz_dir: Path) -> None:
    """Generate and save scatter plot for a given threshold and filter."""
    # Jitter X to reduce overplotting
    x = df_stats['X'].to_numpy()
    jitter_x = x + np.random.uniform(-0.3, 0.3, size=len(df_stats))
    # Y in percent
    y = df_stats['Y'].to_numpy() * 100
    # Point sizes
    sizes = df_stats['num_works'].to_numpy() * 30
    # Color by modal_form using Dark2 palette
    forms = sorted(df_stats['modal_form'].unique())
    # replace the existing form_palette line
    form_palette = dict(
        zip(forms, sns.color_palette("Dark2", n_colors=len(forms)))
    )
    # Create figure with specified size and resolution
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    # -------- Scatter ------------
    points = ax.scatter(
        jitter_x, y,
        s=sizes,
        c=df_stats["modal_form"].map(form_palette),
        alpha=0.6,
        edgecolors="black",
        linewidths=0.4
    )
    # Integer x-axis only
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Y-axis fixed and reference line
    ax.set_ylim(0, 110)
    ax.axhline(100, color="grey", alpha=0.3, lw=1, zorder=0)
    # Labels and title
    ax.set_xlabel("# anthologies selecting author (jittered)")
    ax.set_ylabel("% selections repeating the same work")
    ax.set_title(f"{period} | root={root_flag} | n≥{n_thresh}")
    # Legend for forms
    for form, color in form_palette.items():
        ax.scatter([], [], c=[color], label=form, s=50, edgecolors="black")
    ax.legend(title="Most-common form", frameon=False, loc="best")
    # -------- Labels -------------
    # -------- Labels -------------
    texts = [
        ax.text(
            jx, yv + 1,
            author,
            fontsize=6,
            color=form_palette[form]
        )
        for jx, yv, author, form in zip(
            jitter_x, y, df_stats['author'], df_stats['modal_form']
        )
    ]
    # prevent overlaps (adjustText required)
    if adjust_text:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', lw=0.3, color='grey'))
    # Save figure with layout and resolution
    out_path = viz_dir / f"author_spread_{period}_root{root_flag}_n{n_thresh}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    df = load_data()
    # Define periods and filters
    periods = {
        'all_time': df,
        'pre_1996': df[df['anthology_year'] < 1996],
    }
    root_filters = {
        'all_works': False,
        'root_only': True,
    }
    thresholds = [10, 11, 12, 13, 14, 15, 16, 17]
    # Ensure output directory
    viz_dir = Path(__file__).parent / 'viz'
    viz_dir.mkdir(exist_ok=True)
    count = 0
    # Loop over combinations
    for period, df_period in periods.items():
        for flag_name, root_only in root_filters.items():
            if root_only:
                df_sub = df_period[df_period['parent_work_id'] == '']
            else:
                df_sub = df_period
            stats = compute_author_stats(df_sub)
            for n in thresholds:
                stats_n = stats[stats['X'] >= n]
                if stats_n.empty:
                    continue
                make_plot(stats_n, period, flag_name, n, viz_dir)
                count += 1
    total = len(periods) * len(root_filters) * len(thresholds)
    print(f"Created {count} figures in viz/ (periods={len(periods)} × roots={len(root_filters)} × thresholds={len(thresholds)}).")


if __name__ == '__main__':
    main()