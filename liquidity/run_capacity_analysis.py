"""Runs the capacity/impact analysis against real cached data: for
every sampling method and smart-beta variant, computes the
real rebalancing cost (one actual turnover event, from one real quarterly
rebalance to the next) at three disclosed hypothetical AUM levels, and
reports how the annualized-return ranking shifts once that cost is priced
in.

Quarterly rebalancing cost is annualized as `4 * per-rebalance cost
fraction` (four rebalances/year) -- a disclosed simplifying assumption that
this one observed turnover event is representative of the strategy's
typical quarterly turnover, not a claim that turnover is literally constant
across all periods.

Usage: `python -m liquidity.run_capacity_analysis`
"""

from __future__ import annotations

import pandas as pd

from liquidity.capacity import estimate_portfolio_trade_cost
from liquidity.impact import avg_daily_dollar_volume
from regime.volatility_tercile import rolling_realized_vol
from replication.candidate_selection import top_n_by_market_cap
from replication.lasso_sampling import fit_for_target_count
from replication.optimized_sampling import solve_min_tracking_error_weights
from smartbeta.equal_weight import equal_weights
from smartbeta.min_vol import solve_min_variance_weights
from smartbeta.multi_factor import composite_score, tilt_weights, trailing_ic_weights
from smartbeta.quality import quality_weights
from smartbeta.run_smartbeta_comparison import build_backtest_inputs

AUM_LEVELS = [10_000_000, 100_000_000, 1_000_000_000]
SAMPLING_N = 60
MIN_VOL_LOOKBACK = 252


def main(start: str, end: str) -> pd.DataFrame:
    prices, market_caps, membership, benchmark_value, quality_scores, factor_scores, fwd_returns = build_backtest_inputs(start, end)
    all_returns = prices.pct_change()
    volumes = pd.DataFrame({s: pd.read_parquet(f"data/cache/prices/{s}.parquet")["volume"] for s in prices.columns}).reindex(prices.index)
    dollar_volume = avg_daily_dollar_volume(prices, volumes)

    daily_vol = {}
    for col in prices.columns:
        v = rolling_realized_vol(prices[col])
        daily_vol[col] = v.iloc[-1] if v.notna().any() else float("nan")
    daily_vol = pd.Series(daily_vol)

    dates = sorted(membership["rebalance_date"].unique())
    t_prev, t_curr = dates[-3], dates[-2]

    def _members_and_context(t):
        pos = prices.index.searchsorted(t)
        period_start = prices.index[pos]
        members = set(membership.loc[membership["rebalance_date"] == t, "symbol"])
        cap_row = market_caps.loc[market_caps.index[market_caps.index.searchsorted(t)]]
        price_row = prices.loc[period_start]
        return period_start, members, cap_row, price_row

    def _smartbeta_weights(t):
        period_start, members, cap_row, price_row = _members_and_context(t)
        pos = prices.index.searchsorted(t)
        weights = {"equal_weight": equal_weights(members, price_row), "quality": quality_weights(members, quality_scores)}

        if pos >= MIN_VOL_LOOKBACK:
            trailing = all_returns.iloc[pos - MIN_VOL_LOOKBACK: pos]
            candidates = [s for s in members if s in trailing.columns and trailing[s].notna().all()]
            weights["min_vol"] = solve_min_variance_weights(trailing[candidates], max_weight=0.05) if candidates else pd.Series(dtype=float)

        ic_weights = trailing_ic_weights(factor_scores, fwd_returns, prices.index, t)
        composite = composite_score(factor_scores, ic_weights, period_start)
        weights["multi_factor"] = tilt_weights(members, composite, cap_row)
        return weights

    def _sampling_weights(t):
        _, members, cap_row, _ = _members_and_context(t)
        pos = all_returns.index.searchsorted(t)
        if pos < MIN_VOL_LOOKBACK:
            return {}
        trailing = all_returns.iloc[pos - MIN_VOL_LOOKBACK: pos]
        complete = [s for s in members if s in trailing.columns and trailing[s].notna().all()]
        # stratified skipped here (needs a sector map); optimization + lasso only

        candidates = top_n_by_market_cap(members, cap_row, SAMPLING_N)
        candidates = [c for c in candidates if c in complete]
        opt = solve_min_tracking_error_weights(trailing[candidates], benchmark_value.pct_change().reindex(trailing.index).dropna()) if candidates else pd.Series(dtype=float)
        lasso = fit_for_target_count(trailing[complete], benchmark_value.pct_change().reindex(trailing.index).dropna(), SAMPLING_N) if complete else pd.Series(dtype=float)
        return {"sampling_optimization_n60": opt, "sampling_lasso_n60": lasso}

    prev_weights = {**_smartbeta_weights(t_prev), **_sampling_weights(t_prev)}
    curr_weights = {**_smartbeta_weights(t_curr), **_sampling_weights(t_curr)}

    records = []
    for name, weights in curr_weights.items():
        if weights.empty:
            continue
        prev = prev_weights.get(name, pd.Series(dtype=float))
        turnover = float((weights.reindex(weights.index.union(prev.index)).fillna(0)
                           - prev.reindex(weights.index.union(prev.index)).fillna(0)).abs().sum())
        row = {"strategy": name, "n_holdings": int((weights > 1e-6).sum()), "turnover": turnover}
        for aum in AUM_LEVELS:
            cost = estimate_portfolio_trade_cost(weights, prev, aum, daily_vol, dollar_volume)
            row[f"rebalance_cost_frac_aum_{aum:,}"] = cost.total_cost_fraction
            row[f"annualized_cost_drag_aum_{aum:,}"] = cost.total_cost_fraction * 4
        records.append(row)

    results = pd.DataFrame(records).set_index("strategy")
    print(f"Rebalance from {t_prev.date()} to {t_curr.date()}:")
    print(results.to_string())

    # Gross annualized returns from smartbeta/run_smartbeta_comparison.py,
    # reused here rather than re-simulated, to show the net-of-cost ranking directly.
    gross_returns = {"equal_weight": 0.147149, "min_vol": 0.103465, "quality": 0.190321, "multi_factor": 0.175754}
    print("\nGross vs cost-adjusted annualized return (smart-beta variants only):")
    for aum in AUM_LEVELS:
        col = f"annualized_cost_drag_aum_{aum:,}"
        print(f"\n  AUM = ${aum:,}")
        ranked = sorted(gross_returns, key=lambda n: gross_returns[n] - results.loc[n, col] if n in results.index else -999, reverse=True)
        for name in ranked:
            if name not in results.index:
                continue
            drag = results.loc[name, col]
            print(f"    {name:14s} gross={gross_returns[name]:+.4f}  drag={drag:.4f}  net={gross_returns[name] - drag:+.4f}")

    return results


if __name__ == "__main__":
    main("2016-01-01", "2026-08-07")
