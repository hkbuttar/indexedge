import numpy as np
import pandas as pd

from risk.kill_switch import (
    KillSwitch,
    check_relative_drawdown_limit,
    check_tracking_error_limit,
    relative_value_series,
)


def test_tracking_error_limit_not_breached_when_identical_to_benchmark():
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 100), index=dates)
    result = check_tracking_error_limit(returns, returns, limit=0.05)
    assert not result.breached


def test_tracking_error_limit_breached_for_wildly_different_series():
    dates = pd.date_range("2024-01-01", periods=100)
    rng = np.random.default_rng(0)
    portfolio = pd.Series(rng.normal(0.0005, 0.03, 100), index=dates)
    benchmark = pd.Series(rng.normal(0.0005, 0.001, 100), index=dates)
    result = check_tracking_error_limit(portfolio, benchmark, limit=0.05)
    assert result.breached


def test_relative_value_series_flat_when_portfolio_equals_benchmark():
    dates = pd.date_range("2024-01-01", periods=20)
    returns = pd.Series(0.001, index=dates)
    rv = relative_value_series(returns, returns)
    assert np.allclose(rv.to_numpy(), 1.0)


def test_relative_drawdown_breached_when_portfolio_persistently_underperforms():
    dates = pd.date_range("2024-01-01", periods=60)
    benchmark = pd.Series(0.001, index=dates)
    portfolio = pd.Series(0.001, index=dates)
    portfolio.iloc[30:] = -0.01  # sharp, sustained underperformance after day 30
    result = check_relative_drawdown_limit(portfolio, benchmark, limit=0.10)
    assert result.breached


def test_relative_drawdown_not_breached_when_portfolio_tracks_closely():
    dates = pd.date_range("2024-01-01", periods=60)
    benchmark = pd.Series(0.001, index=dates)
    portfolio = pd.Series(0.001, index=dates)
    result = check_relative_drawdown_limit(portfolio, benchmark, limit=0.10)
    assert not result.breached


def test_relative_drawdown_detects_a_prior_breach_after_recovery():
    dates = pd.date_range("2024-01-01", periods=4)
    benchmark = pd.Series(0.0, index=dates)
    portfolio = pd.Series([-0.12, 0.14, 0.0, 0.0], index=dates)
    result = check_relative_drawdown_limit(portfolio, benchmark, limit=0.10)
    assert result.breached
    assert "max relative drawdown" in result.detail


def test_relative_drawdown_counts_loss_on_first_observation_from_initial_value():
    dates = pd.date_range("2024-01-01", periods=2)
    benchmark = pd.Series(0.0, index=dates)
    portfolio = pd.Series([-0.11, 0.0], index=dates)
    assert check_relative_drawdown_limit(portfolio, benchmark, limit=0.10).breached


def test_kill_switch_is_sticky_and_requires_manual_reset():
    switch = KillSwitch()
    from risk.kill_switch import LimitCheckResult

    switch.check([LimitCheckResult("tracking_error", breached=True, detail="x")])
    assert switch.triggered
    assert switch.trigger_reasons == ["tracking_error"]

    # a subsequent all-clear check must NOT un-trip it
    switch.check([LimitCheckResult("tracking_error", breached=False, detail="y")])
    assert switch.triggered

    switch.reset()
    assert not switch.triggered
    assert switch.trigger_reasons == []


def test_kill_switch_records_each_distinct_reason_once():
    from risk.kill_switch import LimitCheckResult

    switch = KillSwitch()
    switch.check([LimitCheckResult("tracking_error", breached=True, detail="x")])
    switch.check([LimitCheckResult("tracking_error", breached=True, detail="x again")])
    switch.check([LimitCheckResult("relative_drawdown", breached=True, detail="z")])
    assert switch.trigger_reasons == ["tracking_error", "relative_drawdown"]
