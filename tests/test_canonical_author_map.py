from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "viz" / "misc"))

from canonical_author_map import (  # noqa: E402
    assign_canonical_work,
    build_selection_frame,
    build_label_text,
    choose_exemplars,
    compute_author_metrics,
    make_plot,
)


def make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "work_id": "",
        "work_title": "",
        "parent_work_id": "",
        "parent_work_title": "",
        "author_id": "",
        "author_name": "",
        "edition_key": "",
    }
    df = pd.DataFrame(rows)
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    return df.astype(str)


def test_build_selection_frame_preserves_multiple_authors_per_work():
    """The DB source is one row per (work, author); co-authored works arrive
    as multiple rows rather than a delimited field, and build_selection_frame
    must keep all of them rather than collapsing to one author."""
    df = make_df(
        [
            {
                "work_id": "w1",
                "work_title": "Joint Text",
                "author_id": "1",
                "author_name": "Author One",
                "edition_key": "a1",
            },
            {
                "work_id": "w1",
                "work_title": "Joint Text",
                "author_id": "2",
                "author_name": "Author Two",
                "edition_key": "a1",
            },
        ]
    )

    result = build_selection_frame(df)

    assert len(result) == 2
    assert set(result["author_name"]) == {"Author One", "Author Two"}


def test_build_selection_frame_deduplicates_identical_author_edition_work():
    df = make_df(
        [
            {
                "work_id": "w1",
                "work_title": "Text",
                "author_id": "1",
                "author_name": "Author One",
                "edition_key": "8|1",
            },
            {
                "work_id": "w1",
                "work_title": "Text",
                "author_id": "1",
                "author_name": "Author One",
                "edition_key": "8|1",
            },
        ]
    )

    result = build_selection_frame(df)

    assert len(result) == 1


def test_assign_canonical_work_rolls_child_to_top_level_parent():
    df = make_df(
        [
            {
                "work_id": "root",
                "work_title": "Long Book",
                "author_id": "1",
                "author_name": "Author One",
            },
            {
                "work_id": "chapter",
                "work_title": "Chapter I",
                "parent_work_id": "root",
                "parent_work_title": "Long Book",
                "author_id": "1",
                "author_name": "Author One",
            },
            {
                "work_id": "excerpt",
                "work_title": "Excerpt from Chapter I",
                "parent_work_id": "chapter",
                "parent_work_title": "Chapter I",
                "author_id": "1",
                "author_name": "Author One",
            },
        ]
    )

    result = assign_canonical_work(df)

    assert list(result["canonical_work_id"]) == ["root", "root", "root"]
    assert list(result["canonical_work_title"]) == [
        "Long Book",
        "Long Book",
        "Long Book",
    ]


def test_compute_author_metrics_entropy_and_signature_share():
    selection_df = pd.DataFrame(
        [
            {
                "author_id": "1",
                "author_name": "Author One",
                "edition_key": "ed1",
                "work_id": "w1",
                "work_title": "Work A",
            },
            {
                "author_id": "1",
                "author_name": "Author One",
                "edition_key": "ed2",
                "work_id": "w1",
                "work_title": "Work A",
            },
            {
                "author_id": "1",
                "author_name": "Author One",
                "edition_key": "ed3",
                "work_id": "w2",
                "work_title": "Work B",
            },
        ]
    )

    metrics = compute_author_metrics(selection_df)
    row = metrics.iloc[0]

    expected_entropy = -(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3)
    assert row["anthology_appearances"] == 3
    assert row["total_work_selections"] == 3
    assert row["distinct_works_selected"] == 2
    assert math.isclose(row["work_selection_entropy"], expected_entropy)
    assert math.isclose(row["signature_work_share"], 2 / 3)
    assert row["signature_work_id"] == "w1"
    assert row["signature_work_title"] == "Work A"


def test_parent_work_rule_aggregates_excerpt_variants():
    df = make_df(
        [
            {
                "work_id": "w1",
                "work_title": "Chapter I",
                "parent_work_id": "p1",
                "parent_work_title": "Long Book",
                "author_id": "1",
                "author_name": "Author One",
                "edition_key": "a1",
            },
            {
                "work_id": "w2",
                "work_title": "Chapter II",
                "parent_work_id": "p1",
                "parent_work_title": "Long Book",
                "author_id": "1",
                "author_name": "Author One",
                "edition_key": "a2",
            },
        ]
    )

    metrics = compute_author_metrics(build_selection_frame(df))
    row = metrics.iloc[0]

    assert row["distinct_works_selected"] == 1
    assert math.isclose(row["signature_work_share"], 1.0)
    assert row["signature_work_title"] == "Long Book"


def test_choose_exemplars_selects_requested_authors_in_fixed_order():
    metrics_df = pd.DataFrame(
        [
            {
                "author_id": "1",
                "author_name": "Langston Hughes",
                "anthology_appearances": 20,
                "signature_work_share": 0.20,
                "signature_work_title": "Mother to Son",
            },
            {
                "author_id": "2",
                "author_name": "Alain Locke",
                "anthology_appearances": 18,
                "signature_work_share": 0.80,
                "signature_work_title": "The New Negro <essay>",
                "total_work_selections": 25,
            },
            {
                "author_id": "22",
                "author_name": "Alain Locke",
                "anthology_appearances": 1,
                "signature_work_share": 1.00,
                "signature_work_title": "The New Negro <edited collection>",
                "total_work_selections": 1,
            },
            {
                "author_id": "3",
                "author_name": "Olaudah Equiano",
                "anthology_appearances": 22,
                "signature_work_share": 0.30,
                "signature_work_title": "The Interesting Narrative",
            },
            {
                "author_id": "33",
                "author_name": "Phillis Wheatley",
                "anthology_appearances": 24,
                "signature_work_share": 0.52,
                "signature_work_title": "Poems on Various Subjects, Religious and Moral",
            },
            {
                "author_id": "4",
                "author_name": "Gwendolyn Brooks",
                "anthology_appearances": 16,
                "signature_work_share": 0.08,
                "signature_work_title": "We Real Cool",
            },
        ]
    )

    result = choose_exemplars(metrics_df)

    assert list(result["author_name"]) == [
        "Olaudah Equiano",
        "Alain Locke",
        "Langston Hughes",
        "Phillis Wheatley",
    ]
    assert list(result["signature_work_title"]) == [
        "The Interesting Narrative",
        "The New Negro <essay>",
        "Mother to Son",
        "Poems on Various Subjects, Religious and Moral",
    ]


def test_build_label_text_includes_signature_work_title():
    row = pd.Series(
        {
            "author_name": "Gwendolyn Brooks",
            "signature_work_title": "We Real Cool",
        }
    )

    assert build_label_text(row) == 'Gwendolyn Brooks, "We Real Cool"'


def test_build_label_text_uses_custom_short_titles_for_target_authors():
    row = pd.Series(
        {
            "author_name": "Alain Locke",
            "signature_work_title": "The New Negro <essay>",
        }
    )

    assert build_label_text(row) == 'Locke, "The New Negro"'
    wheatley_row = pd.Series(
        {
            "author_name": "Phillis Wheatley",
            "signature_work_title": "Poems on Various Subjects, Religious and Moral",
        }
    )
    assert (
        build_label_text(wheatley_row)
        == r"Wheatley, $\it{Poems\ on\ Various\ Subjects}$"
    )


def test_smoke_writes_metrics_csv_and_png(tmp_path: Path):
    df = make_df(
        [
            {
                "work_id": "w1",
                "work_title": "Work A",
                "author_id": "1",
                "author_name": "Author One",
                "edition_key": "a1",
            },
            {
                "work_id": "w2",
                "work_title": "Work B",
                "author_id": "1",
                "author_name": "Author One",
                "edition_key": "a2",
            },
            {
                "work_id": "w3",
                "work_title": "Work C",
                "author_id": "2",
                "author_name": "Author Two",
                "edition_key": "a3",
            },
        ]
    )

    selection_df = build_selection_frame(df)
    metrics_df = compute_author_metrics(selection_df)
    exemplars_df = choose_exemplars(metrics_df)

    out_csv = tmp_path / "canonical_author_metrics.csv"
    out_png = tmp_path / "canonical_author_map.png"
    metrics_df.to_csv(out_csv, index=False)
    make_plot(metrics_df, exemplars_df, out_png)

    assert out_csv.exists()
    assert out_png.exists()
