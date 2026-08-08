import numpy as np
import pandas as pd
import pytest

from liquidity.capacity import estimate_portfolio_trade_cost


def test_establishment_cost_treats_prev_weights_as_zero():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    daily_vol = pd.Series({"A": 0.2, "B": 0.3})
    dollar_vol = pd.Series({"A": 5_000_000.0, "B": 5_000_000.0})

    result = estimate_portfolio_trade_cost(weights, None, aum=1_000_000, daily_vol_by_symbol=daily_vol, dollar_volume_by_symbol=dollar_vol)
    assert set(result.per_symbol.keys()) == {"A", "B"}
    assert result.total_dollar_cost > 0
    assert result.total_cost_fraction == pytest.approx(result.total_dollar_cost / 1_000_000)


def test_rebalance_only_trades_the_delta():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    prev_weights = pd.Series({"A": 0.5, "B": 0.5})  # no change -> zero cost
    daily_vol = pd.Series({"A": 0.2, "B": 0.2})
    dollar_vol = pd.Series({"A": 5_000_000.0, "B": 5_000_000.0})

    result = estimate_portfolio_trade_cost(weights, prev_weights, aum=1_000_000, daily_vol_by_symbol=daily_vol, dollar_volume_by_symbol=dollar_vol)
    assert result.total_dollar_cost == pytest.approx(0.0)
    assert result.per_symbol == {}


def test_cost_fraction_scales_as_sqrt_of_aum():
    """The core Step 7 finding, verified directly: total cost as a fraction
    of AUM grows with sqrt(AUM), since dollar cost per trade scales with
    sqrt(trade size) and trade size scales linearly with AUM."""
    weights = pd.Series({"A": 0.5, "B": 0.5})
    daily_vol = pd.Series({"A": 0.25, "B": 0.25})
    dollar_vol = pd.Series({"A": 20_000_000.0, "B": 20_000_000.0})

    small = estimate_portfolio_trade_cost(weights, None, aum=10_000_000, daily_vol_by_symbol=daily_vol, dollar_volume_by_symbol=dollar_vol)
    large = estimate_portfolio_trade_cost(weights, None, aum=1_000_000_000, daily_vol_by_symbol=daily_vol, dollar_volume_by_symbol=dollar_vol)
    # AUM up 100x -> cost fraction up sqrt(100) = 10x
    assert large.total_cost_fraction == pytest.approx(small.total_cost_fraction * 10, rel=1e-6)


def test_missing_liquidity_data_excluded_and_noted():
    weights = pd.Series({"A": 0.5, "NODATA": 0.5})
    daily_vol = pd.Series({"A": 0.2})
    dollar_vol = pd.Series({"A": 5_000_000.0})

    result = estimate_portfolio_trade_cost(weights, None, aum=1_000_000, daily_vol_by_symbol=daily_vol, dollar_volume_by_symbol=dollar_vol)
    assert "NODATA" not in result.per_symbol
    assert any("NODATA" in note for note in result.notes)
