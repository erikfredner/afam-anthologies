import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

def bucket(c: int) -> str:
    """Assign a prior appearance count to a frequency bucket."""
    if c == 0:
        return "0"
    elif 1 <= c <= 5:
        return str(c)
    else:
        return "6+"

def main() -> None:
    """Compute inclusion recall by prior appearance frequency."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute bucketed recall for works and authors for NAAAL1996 inclusion."
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save output tables to CSV in data/ directory instead of printing."
    )
    parser.add_argument(
        "--only-root-works",
        action="store_true",
        help="Limit work-level analysis to works without parent works (parent_work_title empty)."
    )
    args = parser.parse_args()
    # Load data
    csv_path = Path(__file__).parent.parent / "data" / "202505121539 authors works.csv"
    df = pd.read_csv(csv_path, dtype=str, na_filter=False)
    df["anthology_year"] = df["anthology_year"].astype(int)
    # Determine which works to include in work-level analysis
    if args.only_root_works:
        root_work_ids = set(df.loc[df["parent_work_title"] == "", "work_id"].unique())
    else:
        root_work_ids = set(df["work_id"].unique())

    # Identify NAAAL 1996 works
    mask_naaal1996 = (
        (df["anthology_series"] == "The Norton Anthology of African American Literature")
        & (df["anthology_edition"] == "1")
        & (df["anthology_year"] == 1996)
    )
    works_naaal1996 = set(df.loc[mask_naaal1996, "work_id"])
    if not works_naaal1996:
        raise ValueError("No NAAAL 1996 rows found.")

    # Count pre-1996 appearances
    pre1996 = df[df["anthology_year"] < 1996]
    freq: Dict[str, int] = pre1996.groupby("work_id").size().to_dict()

    # Prepare per-work summaries (optionally only root works)
    all_work_ids = sorted(root_work_ids)
    work_df = pd.DataFrame({"work_id": all_work_ids})
    work_df["prior_count"] = work_df["work_id"].map(lambda w: freq.get(w, 0)).astype(int)
    work_df["in_naaal1996"] = work_df["work_id"].isin(works_naaal1996).astype(int)

    # Compute cumulative recall by threshold: n or more prior appearances for works
    thresholds = [1, 2, 3, 4, 5, 6]
    result_rows = []
    for t in thresholds:
        masked = work_df[work_df["prior_count"] >= t]
        total = masked["work_id"].nunique()
        in_naaal = masked["in_naaal1996"].sum()
        pct = 100 * in_naaal / total if total > 0 else 0.0
        result_rows.append({
            "bucket": f"{t}+",
            "works_with_prior": total,
            "works_in_naaal": in_naaal,
            "recall_pct": pct,
        })
    result = pd.DataFrame(result_rows)

    # Compute author-level bucketed recall
    # Identify authors in NAAAL 1996
    authors_naaal1996 = set(df.loc[mask_naaal1996, "work_author"])
    # Count pre-1996 appearances per author
    freq_auth: Dict[str, int] = pre1996.groupby("work_author").size().to_dict()
    # Prepare per-author summaries
    all_authors = df["work_author"].unique()
    author_df = pd.DataFrame({"author": all_authors})
    author_df["prior_count"] = author_df["author"].map(lambda a: freq_auth.get(a, 0)).astype(int)
    author_df["in_naaal1996"] = author_df["author"].isin(authors_naaal1996).astype(int)

    # Compute cumulative recall by threshold: n or more prior appearances for authors
    author_result_rows = []
    for t in thresholds:
        masked = author_df[author_df["prior_count"] >= t]
        total = masked["author"].nunique()
        in_naaal = masked["in_naaal1996"].sum()
        pct = 100 * in_naaal / total if total > 0 else 0.0
        author_result_rows.append({
            "bucket": f"{t}+",
            "authors_with_prior": total,
            "authors_in_naaal": in_naaal,
            "recall_pct": pct,
        })
    author_result = pd.DataFrame(author_result_rows)

    # Handle CSV saving mode
    if args.save_csv:
        data_dir = Path(__file__).parent.parent / "data"
        works_csv = data_dir / "freq_bucket_predictability_works.csv"
        authors_csv = data_dir / "freq_bucket_predictability_authors.csv"
        result.to_csv(works_csv, index=False)
        author_result.to_csv(authors_csv, index=False)
        print(f"Saved works table to {works_csv}")
        print(f"Saved authors table to {authors_csv}")
        return
    # Otherwise, print works and authors tables

    # Print table
    cols = ["Bucket", "Works w/ prior count", "Works in NAAAL 1996", "Recall (%)"]
    col_widths = [len(c) for c in cols]
    header = (
        f"{cols[0].ljust(col_widths[0])} | "
        f"{cols[1].ljust(col_widths[1])} | "
        f"{cols[2].ljust(col_widths[2])} | "
        f"{cols[3].ljust(col_widths[3])}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for _, row in result.iterrows():
        b = row["bucket"]
        wpri = row["works_with_prior"]
        wnaa = row["works_in_naaal"]
        rec = row["recall_pct"]
        rec_str = f"{rec:.1f}" if wpri > 0 else f"{rec:.1f} (n=0)"
        print(
            f"{b.ljust(col_widths[0])} | "
            f"{str(wpri).rjust(col_widths[1])} | "
            f"{str(wnaa).rjust(col_widths[2])} | "
            f"{rec_str.rjust(col_widths[3])}"
        )

    # Summary bullets comparing highest and lowest non-zero buckets
    recall_dict = result.set_index("bucket")["recall_pct"].to_dict()
    rec_high = recall_dict.get("6+", 0.0)
    rec_low = recall_dict.get("1+", 0.0)
    print()
    print(f"* Works with 6+ earlier shout-outs had a **{rec_high:.1f}%** chance of appearing in NAAAL 1996.")
    print(f"* By contrast, works with 1+ earlier shout-outs had only **{rec_low:.1f}%** representation.")
    # Print author table
    print()
    auth_cols = ["Bucket", "Authors w/ prior count", "Authors in NAAAL 1996", "Recall (%)"]
    auth_col_widths = [len(c) for c in auth_cols]
    auth_header = (
        f"{auth_cols[0].ljust(auth_col_widths[0])} | "
        f"{auth_cols[1].ljust(auth_col_widths[1])} | "
        f"{auth_cols[2].ljust(auth_col_widths[2])} | "
        f"{auth_cols[3].ljust(auth_col_widths[3])}"
    )
    print(auth_header)
    print("-" * len(auth_header))
    for _, row in author_result.iterrows():
        b = row["bucket"]
        apri = row["authors_with_prior"]
        anaa = row["authors_in_naaal"]
        rec = row["recall_pct"]
        rec_str = f"{rec:.1f}" if apri > 0 else f"{rec:.1f} (n=0)"
        print(
            f"{b.ljust(auth_col_widths[0])} | "
            f"{str(apri).rjust(auth_col_widths[1])} | "
            f"{str(anaa).rjust(auth_col_widths[2])} | "
            f"{rec_str.rjust(auth_col_widths[3])}"
        )
    # Summary bullets for authors
    auth_rec_dict = author_result.set_index("bucket")["recall_pct"].to_dict()
    auth_rec_high = auth_rec_dict.get("6+", 0.0)
    auth_rec_low = auth_rec_dict.get("1+", 0.0)
    print()
    print(f"* Authors with 6+ earlier shout-outs had a **{auth_rec_high:.1f}%** chance of appearing in NAAAL 1996.")
    print(f"* By contrast, authors with 1+ earlier shout-outs had only **{auth_rec_low:.1f}%** representation.")

if __name__ == "__main__":
    main()