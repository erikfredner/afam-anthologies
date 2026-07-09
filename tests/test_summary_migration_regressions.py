from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "analysis" / "summaries"))
sys.path.insert(0, str(ROOT / "viz" / "reselection"))

import edition_stats  # noqa: E402
import naal_exclusive_works  # noqa: E402
import series_pair_reselection  # noqa: E402


def test_naal_exclusive_works_preserves_all_authors(monkeypatch, capsys):
    rows = []
    for edition_id in (1, 2):
        for author in ('Andy Razaf', 'Thomas "Fats" Waller'):
            rows.append(
                {
                    "series_id": 3,
                    "edition_id": edition_id,
                    "work_id": 6365,
                    "work_title": "Honeysuckle Rose",
                    "author_name": author,
                }
            )

    monkeypatch.setattr(naal_exclusive_works, "query_db", lambda _: pd.DataFrame(rows))
    monkeypatch.setattr(sys, "argv", ["naal_exclusive_works.py"])

    naal_exclusive_works.main()

    output = capsys.readouterr().out
    assert 'Andy Razaf: "Honeysuckle Rose"' in output
    assert 'Thomas "Fats" Waller: "Honeysuckle Rose"' in output


def test_series_pair_summary_sorts_edition_numbers_numerically(monkeypatch, tmp_path):
    summaries = {
        "Full": {
            edition: {"authors": 1, "reselected": 0, "reselected_pct": 0.0}
            for edition in ("1", "10", "2")
        },
        "Shorter": {},
    }
    monkeypatch.setattr(
        series_pair_reselection,
        "query_db",
        lambda _: pd.DataFrame({"series_id": [1]}),
    )
    monkeypatch.setattr(
        series_pair_reselection, "classify_editions", lambda _: pd.DataFrame()
    )
    monkeypatch.setattr(
        series_pair_reselection,
        "compute_reselection_by_role",
        lambda _s1, _meta, role: summaries[role],
    )
    out_csv = tmp_path / "summary.csv"
    monkeypatch.setattr(series_pair_reselection, "OUT_CSV", out_csv)
    monkeypatch.setattr(sys, "argv", ["series_pair_reselection.py"])

    series_pair_reselection.main()

    assert pd.read_csv(out_csv)["Edition"].tolist() == [1, 2, 10]


def test_edition_stats_creates_custom_output_directory(monkeypatch, tmp_path):
    stats = pd.DataFrame([{"Edition": 1, "Total": 10}])
    monkeypatch.setattr(edition_stats, "query_db", lambda _: pd.DataFrame())
    monkeypatch.setattr(edition_stats, "compute_stats", lambda _: stats)
    out_csv = tmp_path / "nested" / "output" / "stats.csv"
    monkeypatch.setattr(sys, "argv", ["edition_stats.py", "--out", str(out_csv)])

    edition_stats.main()

    assert out_csv.is_file()
    pd.testing.assert_frame_equal(pd.read_csv(out_csv), stats)
