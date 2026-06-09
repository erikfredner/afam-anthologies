from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "reselection"))

from authors_without_frequent_works import classify_authors  # noqa: E402


def test_classify_authors_uses_same_threshold_for_authors_and_works():
    rows = [
        {
            "author_id": 1,
            "author_name": "Author With Frequent Work",
            "author_edition_count": 13,
            "work_id": 10,
            "work_title": "Frequent Work",
            "best_work_edition_count": 13,
        },
        {
            "author_id": 2,
            "author_name": "Author Without Frequent Work",
            "author_edition_count": 13,
            "work_id": 20,
            "work_title": "Infrequent Work",
            "best_work_edition_count": 12,
        },
        {
            "author_id": 3,
            "author_name": "Infrequent Author",
            "author_edition_count": 12,
            "work_id": 30,
            "work_title": "Frequent Work",
            "best_work_edition_count": 13,
        },
    ]

    explained, unexplained = classify_authors(pd.DataFrame(rows), threshold=13)

    assert explained["author_name"].tolist() == ["Author With Frequent Work"]
    assert unexplained["author_name"].tolist() == ["Author Without Frequent Work"]
