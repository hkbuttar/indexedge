import numpy as np
import pandas as pd
import pytest

from smartbeta.multi_factor import (
    composite_score,
    forward_returns_panel,
    information_coefficient_series,
    tilt_weights,
    trailing_ic_weights,
)


def test_forward_returns_panel_matches_hand_computation():
    dates = pd.date_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=dates)
    fwd = forward_returns_panel(prices, horizon=2)
    assert fwd["A"].iloc[0] == pytest.approx(12.0 / 10.0 - 1)
    assert pd.isna(fwd["A"].iloc[-1])


def test_ic_series_detects_perfect_predictive_signal():
    dates = pd.date_range("2024-01-01", periods=30)
    symbols = [f"S{i}" for i in range(20)]
    rng = np.random.default_rng(0)
    factor = pd.DataFrame(rng.normal(0, 1, (30, 20)), index=dates, columns=symbols)
    # forward return exactly monotonic in the factor score each day -> IC should be ~1
    forward_returns = factor.rank(axis=1)

    ic = information_coefficient_series(factor, forward_returns, dates)
    assert (ic > 0.99).all()


def test_trailing_ic_weights_clips_negative_ic_to_zero():
    dates = pd.date_range("2024-01-01", periods=50)
    symbols = [f"S{i}" for i in range(20)]
    rng = np.random.default_rng(1)
    good_factor = pd.DataFrame(rng.normal(0, 1, (50, 20)), index=dates, columns=symbols)
    forward_returns = good_factor.rank(axis=1)  # perfectly predicted by good_factor
    bad_factor = -good_factor  # perfectly anti-predictive -> negative IC

    weights = trailing_ic_weights(
        {"good": good_factor, "bad": bad_factor}, forward_returns, dates, refit_date=dates[40], lookback_days=30
    )
    assert weights["bad"] == 0.0
    assert weights["good"] == pytest.approx(1.0)


def test_trailing_ic_weights_falls_back_to_equal_when_all_nonpositive():
    dates = pd.date_range("2024-01-01", periods=50)
    symbols = [f"S{i}" for i in range(20)]
    rng = np.random.default_rng(2)
    factor = pd.DataFrame(rng.normal(0, 1, (50, 20)), index=dates, columns=symbols)
    forward_returns = (-factor).rank(axis=1)  # perfectly anti-correlated with the factor
    factor_b = factor * 2  # same rank ordering as factor -> also perfectly anti-correlated

    weights = trailing_ic_weights({"a": factor, "b": factor_b}, forward_returns, dates, refit_date=dates[40], lookback_days=30)
    assert weights == {"a": 0.5, "b": 0.5}


def test_composite_score_renormalizes_over_available_factors():
    date = pd.Timestamp("2024-01-05")
    factor_a = pd.DataFrame({"X": [1.0], "Y": [2.0]}, index=[date])
    factor_b = pd.DataFrame({"X": [np.nan], "Y": [4.0]}, index=[date])  # missing for X

    composite = composite_score({"a": factor_a, "b": factor_b}, {"a": 0.5, "b": 0.5}, date)
    assert composite["X"] == pytest.approx(1.0)  # only factor a available -> its value alone
    assert composite["Y"] == pytest.approx(3.0)  # 0.5*2 + 0.5*4


def test_tilt_weights_scales_cap_by_composite_and_sums_to_one():
    members = {"A", "B", "C"}
    market_cap_row = pd.Series({"A": 100.0, "B": 100.0, "C": 100.0})
    composite_row = pd.Series({"A": 1.0, "B": 0.0, "C": -1.0})

    weights = tilt_weights(members, composite_row, market_cap_row, tilt_strength=1.0)
    assert weights.sum() == pytest.approx(1.0)
    assert weights["A"] > weights["B"] > weights["C"]


def test_tilt_weights_excludes_names_missing_cap_or_score():
    members = {"A", "NOCAP", "NOSCORE"}
    market_cap_row = pd.Series({"A": 100.0, "NOSCORE": 100.0})
    composite_row = pd.Series({"A": 1.0, "NOCAP": 1.0})
    weights = tilt_weights(members, composite_row, market_cap_row)
    assert set(weights.index) == {"A"}
