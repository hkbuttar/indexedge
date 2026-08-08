"""Runs the Step 9 risk layer against real cached data:

1. Brinson-Fachler active-risk decomposition (sector allocation vs.
   security selection vs. interaction) for the multi-factor tilt strategy
   against Step 2's full replication, over one real recent rebalance
   period, plus its factor-exposure differential.
2. Kill-switch checks (tracking error limit, relative-drawdown limit) run
   over each smart-beta variant's FULL cost-adjusted backtest history
   (Step 8), showing which variants would actually have tripped it on real
   data and which stayed within the disclosed limits.
3. A pointer to Step 6's regime-conditional breakdown as the fourth lens
   the plan asks for (calm/normal/volatile performance), already built.

Usage: `python -m risk.run_risk_layer`
"""

from __future__ import annotations

import pandas as pd

from costs.transaction_costs import cost_adjusted_returns
from data.wikipedia_constituents import fetch_constituents_and_changes
from liquidity.impact import avg_daily_dollar_volume
from regime.volatility_tercile import rolling_realized_vol
from replication.full_replication import rebalance_weights
from risk.attribution import brinson_fachler_attribution, factor_exposure_differential
from risk.kill_switch import KillSwitch, check_relative_drawdown_limit, check_tracking_error_limit, relative_value_series
from risk.tracking_error import summarize_tracking
from smartbeta.backtest import simulate_all_variants_with_weights
from smartbeta.multi_factor import composite_score, trailing_ic_weights
from smartbeta.run_smartbeta_comparison import build_backtest_inputs

AUM = 100_000_000


def main(start: str, end: str) -> None:
    prices, market_caps, membership, benchmark_value, quality_scores, factor_scores, fwd_returns = build_backtest_inputs(start, end)
    returns_by_strategy, weights_by_date_by_strategy = simulate_all_variants_with_weights(
        prices, market_caps, membership, quality_scores, factor_scores, fwd_returns
    )
    benchmark_returns = benchmark_value.pct_change().dropna()
    current, _ = fetch_constituents_and_changes()
    sector_by_symbol = dict(zip(current["yfinance_symbol"], current["gics_sector"]))

    # --- 1. Brinson-Fachler attribution, multi_factor vs full replication, most recent real period ---
    dates = sorted(weights_by_date_by_strategy["multi_factor"].keys())
    t_prev, t_curr = dates[-2], dates[-1]
    portfolio_weights = weights_by_date_by_strategy["multi_factor"][t_prev]

    pos_prev, pos_curr = prices.index.searchsorted(t_prev), prices.index.searchsorted(t_curr)
    period_start, period_end = prices.index[pos_prev], prices.index[min(pos_curr, len(prices.index) - 1)]
    period_returns = prices.loc[period_end] / prices.loc[period_start] - 1

    members_prev = set(membership.loc[membership["rebalance_date"] == t_prev, "symbol"])
    cap_row = market_caps.loc[market_caps.index[market_caps.index.searchsorted(t_prev)]]
    benchmark_weights = rebalance_weights(members_prev, cap_row)

    attribution = brinson_fachler_attribution(portfolio_weights, benchmark_weights, period_returns, sector_by_symbol)
    print(f"Brinson-Fachler attribution, multi_factor vs full replication, {t_prev.date()} -> {t_curr.date()}:")
    print(f"  portfolio_return={attribution.portfolio_return:+.4f}  benchmark_return={attribution.benchmark_return:+.4f}  "
          f"active={attribution.total_active_return:+.4f}")
    print(f"  allocation={attribution.allocation:+.4f}  selection={attribution.selection:+.4f}  "
          f"interaction={attribution.interaction:+.4f}  reconciles={attribution.reconciles()}")
    print(f"  excluded {len(attribution.excluded_symbols)} symbols, {attribution.excluded_weight:.4f} of portfolio weight")

    ic_weights = trailing_ic_weights(factor_scores, fwd_returns, prices.index, t_prev)
    composite = composite_score(factor_scores, ic_weights, period_start)
    factor_diff = factor_exposure_differential(portfolio_weights, benchmark_weights, composite)
    print(f"  factor exposure differential (portfolio - benchmark) = {factor_diff:+.4f}")

    # --- 2. Kill-switch checks over each variant's full cost-adjusted history ---
    volumes = pd.DataFrame({s: pd.read_parquet(f"data/cache/prices/{s}.parquet")["volume"] for s in prices.columns}).reindex(prices.index)
    dollar_volume = avg_daily_dollar_volume(prices, volumes)
    daily_vol = pd.Series({col: (lambda v: v.iloc[-1] if v.notna().any() else float("nan"))(rolling_realized_vol(prices[col])) for col in prices.columns})

    print(f"\nKill-switch checks (TE limit=5%, relative drawdown limit=10%), AUM=${AUM:,}:")
    for name, returns in returns_by_strategy.items():
        weights_by_date = weights_by_date_by_strategy[name]
        adjusted_returns, _ = cost_adjusted_returns(returns, weights_by_date, AUM, daily_vol, dollar_volume)

        te_check = check_tracking_error_limit(adjusted_returns, benchmark_returns)
        dd_check = check_relative_drawdown_limit(adjusted_returns, benchmark_returns)
        relative_value = relative_value_series(adjusted_returns, benchmark_returns)
        from backtest.metrics import running_drawdown
        max_relative_drawdown = (
            float((1 - relative_value / relative_value.cummax().clip(lower=1.0)).max())
            if not relative_value.empty
            else float("nan")
        )

        switch = KillSwitch()
        switch.check([te_check, dd_check])
        print(f"  {name:14s} {te_check.detail:45s} | {dd_check.detail:50s} | "
              f"max_relative_drawdown_ever={max_relative_drawdown:.4f} | triggered={switch.triggered} {switch.trigger_reasons}")

    print("\nSee regime/run_regime_conditional.py for the fourth lens (calm/normal/volatile performance breakdown, Step 6).")


if __name__ == "__main__":
    main("2016-01-01", "2026-08-07")
