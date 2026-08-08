"""Runs the regime-conditional smart-beta comparison against real
cached data: classifies calm/normal/volatile regimes on ^GSPC (real S&P 500
index level), then reports each smart-beta variant's return/vol/Sharpe
*separately per regime* -- the direct test of whether min-vol's real-world
performance backs up its own textbook justification (outperformance
specifically when markets are stressed), rather than a pooled number that
could hide the effect either way.

Usage: `python -m regime.run_regime_conditional`
"""

from __future__ import annotations

import pandas as pd

from data.prices import fetch_index_level
from regime.conditional import summarize_regime_conditional_performance
from regime.volatility_tercile import classify_regimes, rolling_realized_vol
from smartbeta.backtest import simulate_all_variants
from smartbeta.run_smartbeta_comparison import build_backtest_inputs


def main(start: str, end: str) -> pd.DataFrame:
    prices, market_caps, membership, benchmark_value, quality_scores, factor_scores, fwd_returns = build_backtest_inputs(start, end)
    returns_by_strategy = simulate_all_variants(prices, market_caps, membership, quality_scores, factor_scores, fwd_returns)
    returns_by_strategy["full_replication"] = benchmark_value.pct_change().dropna()

    index_level = fetch_index_level("^GSPC")["close"]
    vol = rolling_realized_vol(index_level)
    regimes = classify_regimes(vol)
    print("Regime day counts (full ^GSPC history):")
    print(regimes.value_counts())

    summary = summarize_regime_conditional_performance(returns_by_strategy, regimes.labels)
    pivot_return = summary.pivot(index="strategy", columns="regime", values="annualized_return")
    pivot_vol = summary.pivot(index="strategy", columns="regime", values="annualized_vol")
    pivot_sharpe = summary.pivot(index="strategy", columns="regime", values="sharpe_zero_rate")

    regime_order = [r for r in ["calm", "normal", "volatile"] if r in pivot_return.columns]
    print("\nAnnualized return by regime:")
    print(pivot_return[regime_order].to_string())
    print("\nAnnualized vol by regime:")
    print(pivot_vol[regime_order].to_string())
    print("\nSharpe (zero risk-free rate) by regime:")
    print(pivot_sharpe[regime_order].to_string())

    if "min_vol" in pivot_return.index and "full_replication" in pivot_return.index:
        excess = pivot_return.loc["min_vol"] - pivot_return.loc["full_replication"]
        print("\nmin_vol excess annualized return over full_replication, by regime:")
        print(excess[regime_order].to_string())

    return summary


if __name__ == "__main__":
    main("2016-01-01", "2026-08-07")
