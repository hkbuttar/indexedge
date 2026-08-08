"""Walk-forward comparison of the three Step 3 sampling methods
(stratified, cvxpy optimization, LASSO) against Step 2's full-replication
portfolio, producing the tracking-error-vs-name-count curve.

Walk-forward, not in-sample: at each historical rebalance date, every
method is fit (or, for stratified, structurally built) using only a
trailing lookback window of returns *ending at* that date, then its
resulting weights are held through the next quarter and tracking error is
computed on THAT forward, out-of-sample period. Reporting in-sample fit
quality as if it were achievable tracking error would be a fabricated
result -- the whole reason for the trailing/forward split.

The benchmark being tracked is Step 2's own full-replication portfolio
value series, not the raw S&P 500 index directly. This is a deliberate
layering, not a simplification: it isolates "how much tracking error does
sampling N<full names cost" from "how much tracking error does this
project's full replication already carry against the real index due to
data-coverage/free-float limitations" (documented in
`full_replication.py`). Step 10 recombines both layers for the honest
end-to-end comparison against the real index.

A candidate is only eligible for the optimization and LASSO methods at a
given date if it has a *complete* trailing lookback window of return data
(no gaps from a recent IPO or a data hole) -- both methods need a clean
numeric return matrix, and silently zero-filling gaps would fabricate
history that doesn't exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from replication.candidate_selection import top_n_by_market_cap
from replication.lasso_sampling import fit_for_target_count
from replication.optimized_sampling import solve_min_tracking_error_weights
from replication.stratified import stratified_sample_weights
from risk.tracking_error import annualized_tracking_error

DEFAULT_LOOKBACK_DAYS = 252


def simulate_holding_period(prices: pd.DataFrame, weights: pd.Series, period_start: pd.Timestamp, period_end: pd.Timestamp) -> pd.Series:
    price_at_start = prices.loc[period_start, weights.index]
    shares = weights / price_at_start
    period_dates = prices.index[(prices.index >= period_start) & (prices.index <= period_end)]
    period_prices = prices.loc[period_dates, weights.index].ffill()
    values = period_prices.mul(shares, axis=1).sum(axis=1)
    return values / values.iloc[0]


def _forward_tracking_error(prices, weights, period_start, period_end, benchmark_returns) -> float:
    if weights.empty:
        return float("nan")
    value = simulate_holding_period(prices, weights, period_start, period_end)
    returns = value.pct_change().dropna()
    return annualized_tracking_error(returns, benchmark_returns)


def evaluate_sampling_methods(
    prices: pd.DataFrame,
    market_caps: pd.DataFrame,
    membership_history: pd.DataFrame,
    benchmark_value: pd.Series,
    sector_by_symbol: dict[str, str],
    target_counts: list[int],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    rebalance_dates = sorted(membership_history["rebalance_date"].unique())
    all_returns = prices.pct_change()
    benchmark_returns_full = benchmark_value.pct_change().dropna()

    records = []
    for i, t in enumerate(rebalance_dates[:-1]):
        pos = all_returns.index.searchsorted(t)
        if pos < lookback_days:
            continue
        trailing_returns = all_returns.iloc[pos - lookback_days: pos]
        trailing_bench = benchmark_returns_full.reindex(trailing_returns.index)
        if trailing_bench.isna().mean() > 0.2:
            continue

        period_start = prices.index[pos]
        period_end = rebalance_dates[i + 1]
        forward_bench = benchmark_returns_full[(benchmark_returns_full.index > period_start) & (benchmark_returns_full.index <= period_end)]
        if len(forward_bench) < 5:
            continue

        members = set(membership_history.loc[membership_history["rebalance_date"] == t, "symbol"])
        cap_row = market_caps.loc[market_caps.index[market_caps.index.searchsorted(t)]]
        complete_history = [s for s in members if s in trailing_returns.columns and trailing_returns[s].notna().all()]

        for n in target_counts:
            sector_counts = {sector_by_symbol[s] for s in members if s in sector_by_symbol}
            buckets_per_sector = max(1, round(n / max(len(sector_counts), 1)))
            strat_weights = stratified_sample_weights(members, cap_row, sector_by_symbol, buckets_per_sector)

            candidates = [s for s in top_n_by_market_cap(members, cap_row, n) if s in complete_history]
            opt_weights = solve_min_tracking_error_weights(trailing_returns[candidates], trailing_bench) if candidates else pd.Series(dtype=float)

            lasso_weights = fit_for_target_count(trailing_returns[complete_history], trailing_bench, n) if complete_history else pd.Series(dtype=float)

            for method, weights in [("stratified", strat_weights), ("optimization", opt_weights), ("lasso", lasso_weights)]:
                te = _forward_tracking_error(prices, weights, period_start, period_end, forward_bench)
                records.append({
                    "rebalance_date": t, "method": method, "target_n": n,
                    "actual_n": len(weights), "tracking_error": te,
                })

    return pd.DataFrame(records)


def summarize_curve(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.dropna(subset=["tracking_error"])
        .groupby(["method", "target_n"])
        .agg(mean_actual_n=("actual_n", "mean"), mean_tracking_error=("tracking_error", "mean"),
             median_tracking_error=("tracking_error", "median"), n_periods=("tracking_error", "count"))
        .reset_index()
    )
