"""Converts a strategy's per-rebalance weight history into real,
cost-adjusted daily returns, by applying `liquidity.capacity`'s square-root-
law impact model to every actual rebalancing trade across the
whole backtest, not just a single snapshot rebalance.

The one-time cost incurred at a rebalance is subtracted from the *first*
trading day's return at or after that rebalance date -- modeling the cost
as hitting the portfolio instantaneously at the moment of rebalancing,
which is the standard convention (and the same one
`liquidity/run_capacity_analysis.py`'s single-snapshot
analysis implicitly used). This is a disclosed simplification: real
execution happens gradually over the trading day(s) around a rebalance, not
in one instant, but spreading it out would need intraday data this project
doesn't have.
"""

from __future__ import annotations

import pandas as pd

from liquidity.capacity import estimate_portfolio_trade_cost


def compute_rebalance_costs(
    weights_by_date: dict[pd.Timestamp, pd.Series],
    aum: float,
    daily_vol_by_symbol: pd.Series,
    dollar_volume_by_symbol: pd.Series,
) -> pd.Series:
    """Real per-rebalance cost fraction (of AUM) for every consecutive pair
    of weight vectors in `weights_by_date`, first rebalance priced as
    establishment (from zero)."""
    dates = sorted(weights_by_date.keys())
    cost_by_date = {}
    prev_weights = pd.Series(dtype=float)
    for t in dates:
        weights = weights_by_date[t]
        cost = estimate_portfolio_trade_cost(weights, prev_weights, aum, daily_vol_by_symbol, dollar_volume_by_symbol)
        cost_by_date[t] = cost.total_cost_fraction
        prev_weights = weights
    return pd.Series(cost_by_date)


def apply_costs_to_returns(returns: pd.Series, cost_by_date: pd.Series) -> pd.Series:
    """Subtracts each rebalance's cost fraction from the first trading day
    at or after that rebalance date in `returns`. Rebalance dates with no
    matching trading day on/after them in `returns`' index (e.g. a
    rebalance after the return series' last observed day) are skipped."""
    adjusted = returns.copy()
    for rebalance_date, cost_fraction in cost_by_date.items():
        pos = adjusted.index.searchsorted(rebalance_date)
        if pos < len(adjusted.index):
            hit_date = adjusted.index[pos]
            adjusted.loc[hit_date] = adjusted.loc[hit_date] - cost_fraction
    return adjusted


def cost_adjusted_returns(
    returns: pd.Series,
    weights_by_date: dict[pd.Timestamp, pd.Series],
    aum: float,
    daily_vol_by_symbol: pd.Series,
    dollar_volume_by_symbol: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Returns (cost_adjusted_returns, per_rebalance_cost_fraction) -- the
    latter kept alongside so callers/tests can inspect the actual per-period
    cost drag, not just its aggregate effect on the return series."""
    cost_by_date = compute_rebalance_costs(weights_by_date, aum, daily_vol_by_symbol, dollar_volume_by_symbol)
    return apply_costs_to_returns(returns, cost_by_date), cost_by_date
