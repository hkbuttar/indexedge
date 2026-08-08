import pandas as pd
import pytest

from costs.transaction_costs import apply_costs_to_returns, compute_rebalance_costs, cost_adjusted_returns


def _liquidity_data():
    daily_vol = pd.Series({"A": 0.2, "B": 0.2})
    dollar_vol = pd.Series({"A": 5_000_000.0, "B": 5_000_000.0})
    return daily_vol, dollar_vol


def test_compute_rebalance_costs_prices_first_rebalance_as_establishment():
    weights_by_date = {
        pd.Timestamp("2024-01-01"): pd.Series({"A": 1.0}),
        pd.Timestamp("2024-04-01"): pd.Series({"A": 1.0}),  # unchanged -> zero incremental cost
    }
    daily_vol, dollar_vol = _liquidity_data()
    costs = compute_rebalance_costs(weights_by_date, aum=1_000_000, daily_vol_by_symbol=daily_vol, dollar_volume_by_symbol=dollar_vol)
    assert costs[pd.Timestamp("2024-01-01")] > 0
    assert costs[pd.Timestamp("2024-04-01")] == pytest.approx(0.0)


def test_apply_costs_hits_first_trading_day_on_or_after_rebalance():
    dates = pd.date_range("2024-01-02", periods=5)  # rebalance date itself (Jan 1) is not a trading day
    returns = pd.Series(0.01, index=dates)
    cost_by_date = pd.Series({pd.Timestamp("2024-01-01"): 0.05})

    adjusted = apply_costs_to_returns(returns, cost_by_date)
    assert adjusted.iloc[0] == pytest.approx(0.01 - 0.05)
    assert (adjusted.iloc[1:] == 0.01).all()


def test_apply_costs_skips_rebalance_after_series_end():
    dates = pd.date_range("2024-01-01", periods=3)
    returns = pd.Series(0.01, index=dates)
    cost_by_date = pd.Series({pd.Timestamp("2025-01-01"): 0.05})  # far past the series
    adjusted = apply_costs_to_returns(returns, cost_by_date)
    pd.testing.assert_series_equal(adjusted, returns)


def test_cost_adjusted_returns_reduces_total_return_versus_gross():
    dates = pd.date_range("2024-01-01", periods=200)
    returns = pd.Series(0.001, index=dates)
    weights_by_date = {
        dates[0]: pd.Series({"A": 0.5, "B": 0.5}),
        dates[100]: pd.Series({"A": 0.9, "B": 0.1}),  # a real rebalance trade mid-series
    }
    daily_vol, dollar_vol = _liquidity_data()

    adjusted, cost_by_date = cost_adjusted_returns(returns, weights_by_date, aum=50_000_000, daily_vol_by_symbol=daily_vol, dollar_volume_by_symbol=dollar_vol)
    assert (1 + adjusted).prod() < (1 + returns).prod()
    assert len(cost_by_date) == 2
    assert cost_by_date.iloc[1] > 0  # the mid-series rebalance actually traded
