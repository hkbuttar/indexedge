import numpy as np
import pandas as pd
import pytest

from liquidity.impact import avg_daily_dollar_volume, estimate_transaction_cost


def test_cost_matches_hand_computed_sqrt_law():
    # cost_fraction = Y * vol * sqrt(participation_rate); Y=1.0 default
    trade_value = 100_000.0
    vol = 0.30
    dollar_vol = 10_000_000.0
    result = estimate_transaction_cost(trade_value, vol, dollar_vol)
    expected_participation = 100_000 / 10_000_000
    expected_cost_fraction = 1.0 * vol * np.sqrt(expected_participation)
    assert result.participation_rate == pytest.approx(expected_participation)
    assert result.cost_fraction == pytest.approx(expected_cost_fraction)
    assert result.dollar_cost == pytest.approx(trade_value * expected_cost_fraction)


def test_cost_scales_with_sqrt_of_trade_size():
    base = estimate_transaction_cost(10_000, 0.25, 5_000_000)
    quadrupled = estimate_transaction_cost(40_000, 0.25, 5_000_000)
    # 4x trade size -> 2x cost_fraction (sqrt law)
    assert quadrupled.cost_fraction == pytest.approx(base.cost_fraction * 2, rel=1e-9)


def test_returns_none_for_zero_or_missing_liquidity_data():
    assert estimate_transaction_cost(1000, 0.2, 0) is None
    assert estimate_transaction_cost(1000, float("nan"), 1_000_000) is None
    assert estimate_transaction_cost(1000, 0.2, None) is None


def test_avg_daily_dollar_volume_uses_trailing_window():
    dates = pd.date_range("2024-01-01", periods=100)
    prices = pd.DataFrame({"A": 10.0}, index=dates)
    volumes = pd.DataFrame({"A": 1000.0}, index=dates)
    volumes.iloc[:50] = 999999.0  # far outside the trailing window, must not affect result
    result = avg_daily_dollar_volume(prices, volumes, window=10)
    assert result["A"] == pytest.approx(10.0 * 1000.0)
