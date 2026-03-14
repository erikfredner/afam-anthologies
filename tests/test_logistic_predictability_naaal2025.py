from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

from logistic_predictability_naaal2025 import (  # noqa: E402
    DATA_FILE,
    TARGET_KEY,
    _clean_str,
    _strip_volume,
    assign_edition_key,
    build_author_frame,
    empirical_rates,
    empirical_rates_cumulative,
    find_target_key,
    fit_logit_safe,
    load_data,
    plot_probability_curve,
    predicted_probabilities,
)


# ---------------------------------------------------------------------------
# _strip_volume
# ---------------------------------------------------------------------------

def test_strip_volume_removes_vol_suffix():
    assert _strip_volume("Norton Anthology, Vol. 2") == "Norton Anthology"


def test_strip_volume_removes_vol_without_dot():
    assert _strip_volume("Black Fire, Vol 1") == "Black Fire"


def test_strip_volume_leaves_no_volume_title_unchanged():
    assert _strip_volume("Call and Response") == "Call and Response"


# ---------------------------------------------------------------------------
# _clean_str
# ---------------------------------------------------------------------------

def test_clean_str_converts_nan_to_empty():
    assert _clean_str(float("nan")) == ""


def test_clean_str_strips_surrounding_whitespace():
    assert _clean_str("  hello  ") == "hello"


def test_clean_str_converts_non_string_to_str():
    assert _clean_str(42) == "42"


# ---------------------------------------------------------------------------
# make_df helper
# ---------------------------------------------------------------------------

def make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "anthology_id": "",
        "anthology_title": "",
        "series": "",
        "edition_number": "",
        "volume": "",
        "publication_year": "",
        "author_id": "",
    }
    df = pd.DataFrame(rows)
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    return df.astype(str)


def test_assign_edition_key_titled_with_volume_uses_stripped_title_and_edition():
    df = make_df(
        [
            {"anthology_id": "10", "anthology_title": "Black Fire, Vol. 1", "edition_number": "1", "volume": "1"},
            {"anthology_id": "11", "anthology_title": "Black Fire, Vol. 2", "edition_number": "1", "volume": "2"},
        ]
    )
    result = assign_edition_key(df)
    assert result["edition_key"].nunique() == 1
    assert set(result["edition_key"]) == {"Black Fire|1"}


def test_assign_edition_key_falls_back_to_anthology_id():
    # No title, no year, no edition → can't build a meaningful key → use anthology_id
    df = make_df([{"anthology_id": "99"}])
    result = assign_edition_key(df)
    assert result.iloc[0]["edition_key"] == "99"


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


def test_find_target_key_raises_when_no_match():
    df = make_df([{"anthology_id": "1", "edition_number": "1", "publication_year": "1996"}])
    df = assign_edition_key(df)
    with pytest.raises(ValueError, match="Expected exactly one"):
        find_target_key(df)


def test_find_target_key_raises_when_multiple_keys():
    # Two different anthology_ids that both resolve to different keys but match year/edition
    df = make_df(
        [
            {"anthology_id": "A", "edition_number": "4", "publication_year": "2025"},
            {"anthology_id": "B", "edition_number": "4", "publication_year": "2025"},
        ]
    )
    df = assign_edition_key(df)
    # Manually give them different edition_keys to simulate ambiguity
    df.loc[df["anthology_id"] == "A", "edition_key"] = "key_A"
    df.loc[df["anthology_id"] == "B", "edition_key"] = "key_B"
    with pytest.raises(ValueError, match="Expected exactly one"):
        find_target_key(df)


def test_build_author_frame_counts_collapsed_editions_once():
    df = make_df(
        [
            {"anthology_id": "52", "edition_number": "1", "volume": "1", "publication_year": "1972", "author_id": "a1"},
            {"anthology_id": "53", "edition_number": "1", "volume": "2", "publication_year": "1972", "author_id": "a1"},
            {"anthology_id": "45", "edition_number": "2", "publication_year": "1985", "author_id": "a1"},
            {"anthology_id": "138", "edition_number": "4", "volume": "1", "publication_year": "2025", "author_id": "a1"},
            {"anthology_id": "139", "edition_number": "4", "volume": "2", "publication_year": "2025", "author_id": "a1"},
        ]
    )

    frame = build_author_frame(assign_edition_key(df), target_key=TARGET_KEY, min_prior=1)

    assert len(frame) == 1
    assert frame.iloc[0]["prior_count"] == 2
    assert frame.iloc[0]["in_target"] == 1


def test_build_author_frame_excludes_zero_prior_target_only_authors():
    df = make_df(
        [
            {"anthology_id": "45", "edition_number": "2", "publication_year": "1985", "author_id": "a1"},
            {"anthology_id": "138", "edition_number": "4", "volume": "1", "publication_year": "2025", "author_id": "a1"},
            {"anthology_id": "139", "edition_number": "4", "volume": "2", "publication_year": "2025", "author_id": "a2"},
        ]
    )

    frame = build_author_frame(assign_edition_key(df), target_key=TARGET_KEY, min_prior=1)

    assert set(frame["author_id"]) == {"a1"}
    assert frame.iloc[0]["prior_count"] == 1


def test_fit_logit_safe_recovers_positive_slope():
    rng = np.random.default_rng(0)
    x_arr = rng.integers(1, 10, size=500)
    log_odds = -3 + 0.7 * x_arr
    p = 1 / (1 + np.exp(-log_odds))
    y = pd.Series(rng.binomial(1, p).astype(float))
    x = pd.Series(x_arr.astype(float))

    result = fit_logit_safe(y, x)

    assert result.params["prior_count"] > 0


def test_empirical_rates_returns_one_row_per_bucket():
    frame = pd.DataFrame(
        {"prior_count": [1, 1, 2, 2, 3], "in_target": [0, 1, 0, 1, 1]}
    )
    rates = empirical_rates(frame)
    assert list(rates["prior_count"]) == [1, 2, 3]
    assert rates.loc[rates["prior_count"] == 1, "n"].iloc[0] == 2
    assert rates.loc[rates["prior_count"] == 3, "probability"].iloc[0] == pytest.approx(1.0)


def test_empirical_rates_cumulative_n_is_monotone_decreasing():
    frame = pd.DataFrame(
        {"prior_count": [1, 1, 2, 3, 3], "in_target": [0, 1, 1, 1, 1]}
    )
    rates = empirical_rates_cumulative(frame)
    assert list(rates["n"]) == sorted(rates["n"], reverse=True)


def test_empirical_rates_cumulative_first_row_covers_all():
    frame = pd.DataFrame(
        {"prior_count": [1, 2, 3], "in_target": [0, 1, 1]}
    )
    rates = empirical_rates_cumulative(frame)
    assert rates.iloc[0]["n"] == len(frame)
    assert rates.iloc[0]["probability"] == pytest.approx(2 / 3)


def test_empirical_rates_cumulative_last_row_covers_highest_only():
    frame = pd.DataFrame(
        {"prior_count": [1, 2, 3], "in_target": [0, 0, 1]}
    )
    rates = empirical_rates_cumulative(frame)
    assert rates.iloc[-1]["prior_count"] == 3
    assert rates.iloc[-1]["n"] == 1
    assert rates.iloc[-1]["probability"] == pytest.approx(1.0)


def test_predicted_probabilities_correct_length_and_range():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.integers(1, 6, 200).astype(float))
    y = pd.Series(rng.binomial(1, 0.5, 200).astype(float))
    result = fit_logit_safe(y, x)
    curve = predicted_probabilities(result, max_prior_count=5, min_prior_count=1)
    assert len(curve) == 5
    assert curve["prior_count"].tolist() == list(range(1, 6))
    assert curve["probability"].between(0, 1).all()


def test_predicted_probabilities_monotone_when_slope_positive():
    rng = np.random.default_rng(2)
    x_arr = rng.integers(1, 10, 500)
    p = 1 / (1 + np.exp(-(-4 + 1.2 * x_arr)))
    y = pd.Series(rng.binomial(1, p).astype(float))
    x = pd.Series(x_arr.astype(float))
    result = fit_logit_safe(y, x)
    curve = predicted_probabilities(result, max_prior_count=9)
    assert np.all(np.diff(curve["probability"]) >= -1e-12)


def test_plot_probability_curve_smoke(tmp_path: Path):
    authors = pd.DataFrame(
        {
            "author_id": [f"a{i}" for i in range(1, 11)],
            "prior_count": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "in_target": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        }
    )
    result = fit_logit_safe(authors["in_target"], authors["prior_count"])
    out_path = tmp_path / "curve.png"

    plot_probability_curve(authors, result, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_probability_curve_cumulative_smoke(tmp_path: Path):
    authors = pd.DataFrame(
        {
            "author_id": [f"a{i}" for i in range(1, 11)],
            "prior_count": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "in_target": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        }
    )
    result = fit_logit_safe(authors["in_target"], authors["prior_count"])
    out_path = tmp_path / "curve_cumulative.png"

    plot_probability_curve(authors, result, out_path, cumulative=True)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


@pytest.mark.skipif(not DATA_FILE.exists(), reason="gitignored real dataset is unavailable")
def test_real_data_positive_slope_and_monotonic_curve():
    df = load_data(DATA_FILE)
    target_key = find_target_key(df)
    frame = build_author_frame(df, target_key=target_key, min_prior=1)

    result = fit_logit_safe(frame["in_target"], frame["prior_count"])
    curve_df = predicted_probabilities(result, int(frame["prior_count"].max()), min_prior_count=0)

    assert target_key == TARGET_KEY
    assert result.params["prior_count"] > 0
    assert np.all(np.diff(curve_df["probability"]) >= -1e-12)
    assert empirical_rates(frame)["probability"].iloc[0] < empirical_rates(frame)["probability"].iloc[-1]
