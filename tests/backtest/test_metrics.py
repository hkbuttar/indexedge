import numpy as np
import pandas as pd
import pytest

from backtest.metrics import cagr, max_drawdown, sharpe_ratio, sortino_ratio, win_rate


def test_cagr_matches_hand_calc_for_doubling_over_one_year():
    # 252 points = exactly 1 year in TRADING_DAYS_PER_YEAR terms; start=1.0, end=2.0 -> 100% CAGR
    equity = pd.Series([1.0] * 251 + [2.0])
    assert cagr(equity) == pytest.approx(1.0)


def test_cagr_nan_for_nonpositive_start():
    assert np.isnan(cagr(pd.Series([0.0, 1.0])))
    assert np.isnan(cagr(pd.Series([1.0])))


def test_sharpe_ratio_zero_vol_is_nan():
    returns = pd.Series([0.001] * 50)
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_ratio_matches_hand_calc():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
    expected = returns.mean() / returns.std() * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sortino_only_penalizes_downside():
    all_up = pd.Series([0.01, 0.02, 0.015, 0.03])
    assert np.isnan(sortino_ratio(all_up))  # no downside observations -> undefined


def test_max_drawdown_matches_hand_calc():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])
    # peak 120, trough 90 -> drawdown = 1 - 90/120 = 0.25
    assert max_drawdown(equity) == pytest.approx(0.25)


def test_win_rate_matches_hand_calc():
    returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.0])
    assert win_rate(returns) == pytest.approx(2 / 5)
