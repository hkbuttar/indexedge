"""Runs the Step 8 cost-adjusted, block-bootstrap backtest against real
cached data: for each smart-beta variant, applies real rebalancing costs
(Step 7's impact model, via `costs/transaction_costs.py`) at a disclosed
AUM, then block-bootstraps CAGR, Sharpe, Sortino, max drawdown, win rate,
and tracking error (vs Step 2's full replication) with 95% confidence
intervals -- the statistical-rigor standard reused from BookMaker,
ExecEdge, and PairTrade Lab (`backtest/bootstrap.py`).

Usage: `python -m backtest.run_bootstrap_backtest`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.bootstrap import bootstrap_backtest_metrics
from costs.transaction_costs import cost_adjusted_returns
from liquidity.impact import avg_daily_dollar_volume
from regime.volatility_tercile import rolling_realized_vol
from smartbeta.backtest import simulate_all_variants_with_weights
from smartbeta.run_smartbeta_comparison import build_backtest_inputs

AUM = 100_000_000  # disclosed representative institutional size for this step's headline table
BLOCK_LENGTH = 20
N_RESAMPLES = 2000


def main(start: str, end: str) -> pd.DataFrame:
    prices, market_caps, membership, benchmark_value, quality_scores, factor_scores, fwd_returns = build_backtest_inputs(start, end)
    returns_by_strategy, weights_by_date_by_strategy = simulate_all_variants_with_weights(
        prices, market_caps, membership, quality_scores, factor_scores, fwd_returns
    )
    benchmark_returns = benchmark_value.pct_change().dropna()

    volumes = pd.DataFrame({s: pd.read_parquet(f"data/cache/prices/{s}.parquet")["volume"] for s in prices.columns}).reindex(prices.index)
    dollar_volume = avg_daily_dollar_volume(prices, volumes)
    daily_vol = pd.Series({col: (lambda v: v.iloc[-1] if v.notna().any() else float("nan"))(rolling_realized_vol(prices[col])) for col in prices.columns})

    print(f"Cost-adjusted block-bootstrap backtest, AUM=${AUM:,}, block_length={BLOCK_LENGTH}, n_resamples={N_RESAMPLES}\n")

    records = []
    for name, returns in returns_by_strategy.items():
        weights_by_date = weights_by_date_by_strategy[name]
        adjusted_returns, cost_by_date = cost_adjusted_returns(returns, weights_by_date, AUM, daily_vol, dollar_volume)
        mean_rebalance_cost = float(cost_by_date.mean())

        results = bootstrap_backtest_metrics(
            adjusted_returns, benchmark_returns=benchmark_returns,
            block_length=BLOCK_LENGTH, n_resamples=N_RESAMPLES, seed=42,
        )
        row = {"strategy": name, "mean_rebalance_cost_frac": mean_rebalance_cost}
        for metric, result in results.items():
            row[f"{metric}"] = result.point_estimate
            row[f"{metric}_ci_low"] = result.ci_low
            row[f"{metric}_ci_high"] = result.ci_high
        records.append(row)

    results_df = pd.DataFrame(records).set_index("strategy")
    for metric in ["cagr", "sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate", "tracking_error"]:
        cols = ["mean_rebalance_cost_frac", metric, f"{metric}_ci_low", f"{metric}_ci_high"] if metric == "cagr" else [metric, f"{metric}_ci_low", f"{metric}_ci_high"]
        print(f"\n{metric} (95% CI):")
        print(results_df[cols].to_string())

    return results_df


if __name__ == "__main__":
    main("2016-01-01", "2026-08-07")
