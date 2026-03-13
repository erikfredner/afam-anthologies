from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

from logistic_predictability_naaal2025_works import (  # noqa: E402
    DATA_FILE,
    TARGET_KEY,
    assign_edition_key,
    build_work_frame,
    empirical_rates,
    find_target_key,
    fit_logit_safe,
    load_data,
    plot_probability_curve,
    predicted_probabilities,
)


def make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "anthology_id": "",
        "anthology_title": "",
        "series": "",
        "edition_number": "",
        "volume": "",
        "publication_year": "",
        "work_id": "",
    }
    df = pd.DataFrame(rows)
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    return df.astype(str)


def test_assign_edition_key_collapses_untitled_multivolume_target():
    df = make_df(
        [
            {"anthology_id": "138", "edition_number": "4", "volume": "1", "publication_year": "2025"},
            {"anthology_id": "139", "edition_number": "4", "volume": "2", "publication_year": "2025"},
        ]
    )

    result = assign_edition_key(df)

    assert result["edition_key"].nunique() == 1
    assert set(result["edition_key"]) == {TARGET_KEY}


def test_assign_edition_key_keeps_unrelated_untitled_editions_distinct():
    df = make_df(
        [
            {"anthology_id": "35", "edition_number": "1", "publication_year": "1996"},
            {"anthology_id": "36", "edition_number": "2", "publication_year": "2004"},
        ]
    )

    result = assign_edition_key(df)

    assert list(result["edition_key"]) == ["untitled|1996|1", "untitled|2004|2"]


def test_build_work_frame_counts_collapsed_editions_once():
    df = make_df(
        [
            {"anthology_id": "52", "edition_number": "1", "volume": "1", "publication_year": "1972", "work_id": "w1"},
            {"anthology_id": "53", "edition_number": "1", "volume": "2", "publication_year": "1972", "work_id": "w1"},
            {"anthology_id": "45", "edition_number": "2", "publication_year": "1985", "work_id": "w1"},
            {"anthology_id": "138", "edition_number": "4", "volume": "1", "publication_year": "2025", "work_id": "w1"},
            {"anthology_id": "139", "edition_number": "4", "volume": "2", "publication_year": "2025", "work_id": "w1"},
        ]
    )

    frame = build_work_frame(assign_edition_key(df), target_key=TARGET_KEY, min_prior=1)

    assert len(frame) == 1
    assert frame.iloc[0]["prior_count"] == 2
    assert frame.iloc[0]["in_target"] == 1


def test_build_work_frame_excludes_zero_prior_target_only_works():
    df = make_df(
        [
            {"anthology_id": "45", "edition_number": "2", "publication_year": "1985", "work_id": "w1"},
            {"anthology_id": "138", "edition_number": "4", "volume": "1", "publication_year": "2025", "work_id": "w1"},
            {"anthology_id": "139", "edition_number": "4", "volume": "2", "publication_year": "2025", "work_id": "w2"},
        ]
    )

    frame = build_work_frame(assign_edition_key(df), target_key=TARGET_KEY, min_prior=1)

    assert set(frame["work_id"]) == {"w1"}
    assert frame.iloc[0]["prior_count"] == 1


def test_fit_logit_safe_recovers_positive_slope():
    rng = np.random.default_rng(0)
    x_arr = rng.integers(1, 10, size=500)
    log_odds = -3 + 0.9 * x_arr
    p = 1 / (1 + np.exp(-log_odds))
    y = pd.Series(rng.binomial(1, p).astype(float))
    x = pd.Series(x_arr.astype(float))

    result = fit_logit_safe(y, x)

    assert result.params["prior_count"] > 0


def test_plot_probability_curve_smoke(tmp_path: Path):
    works = pd.DataFrame(
        {
            "work_id": [f"w{i}" for i in range(1, 11)],
            "prior_count": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "in_target": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        }
    )
    result = fit_logit_safe(works["in_target"], works["prior_count"])
    out_path = tmp_path / "curve.png"

    plot_probability_curve(works, result, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


@pytest.mark.skipif(not DATA_FILE.exists(), reason="gitignored real dataset is unavailable")
def test_real_data_positive_slope_and_monotonic_curve():
    df = load_data(DATA_FILE)
    target_key = find_target_key(df)
    frame = build_work_frame(df, target_key=target_key, min_prior=1)

    result = fit_logit_safe(frame["in_target"], frame["prior_count"])
    curve_df = predicted_probabilities(result, int(frame["prior_count"].max()), min_prior_count=0)

    assert target_key == TARGET_KEY
    assert result.params["prior_count"] > 0
    assert np.all(np.diff(curve_df["probability"]) >= -1e-12)
    assert empirical_rates(frame)["probability"].iloc[0] < empirical_rates(frame)["probability"].iloc[-1]
