"""Runs the smart-beta comparison end-to-end against real cached
data and reports each variant's annualized return, volatility, and tracking
error against full replication over the whole backtest window.
Simulation itself lives in `backtest.simulate_all_variants`, shared with
`regime/`'s per-regime performance breakdown.

Min-vol and multi-factor's momentum/low-vol components are fit walk-forward
(trailing lookback only, as of each rebalance date -- no look-ahead).
Quality and multi-factor's value/quality components use current-snapshot
fundamentals at every historical date, per the disclosed limitation in
`data/fundamentals.py` and `smartbeta/quality.py`.

Usage: `python -m smartbeta.run_smartbeta_comparison`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.fundamentals import fetch_universe_fundamentals
from data.prices import to_wide_panel
from data.shares_outstanding import fetch_universe_shares_anchors, fetch_universe_shares_outstanding
from data.wikipedia_constituents import fetch_constituents_and_changes
from replication.full_replication import simulate_cap_weighted_replication
from replication.market_cap import build_market_cap_panel
from replication.point_in_time import build_membership_history, quarterly_rebalance_dates
from risk.tracking_error import summarize_tracking
from smartbeta.backtest import simulate_all_variants
from smartbeta.factors import low_vol_panel, momentum_panel, value_panel
from smartbeta.multi_factor import forward_returns_panel
from smartbeta.quality import compute_quality_scores

PRICE_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "prices"


def _load_cached_prices() -> pd.DataFrame:
    histories = {}
    for path in PRICE_CACHE_DIR.glob("*.parquet"):
        df = pd.read_parquet(path)
        if not df.empty:
            histories[path.stem] = df
    return to_wide_panel(histories, field="close")


def build_backtest_inputs(start: str, end: str):
    """Everything needed to run `simulate_all_variants`, gathered once so
    both this script and `regime/run_regime_conditional.py` build it
    identically instead of duplicating the wiring."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    prices = _load_cached_prices()

    current, changes = fetch_constituents_and_changes()
    rebalance_dates = quarterly_rebalance_dates(start_ts, end_ts)
    membership = build_membership_history(rebalance_dates, current, changes)

    symbols = sorted(prices.columns)
    shares_map, _ = fetch_universe_shares_outstanding(symbols)
    anchors = fetch_universe_shares_anchors(symbols)
    market_caps = build_market_cap_panel(prices, shares_map, class_specific_anchors=anchors)
    benchmark_value, _, _ = simulate_cap_weighted_replication(prices, market_caps, membership)

    fundamentals, _ = fetch_universe_fundamentals(sorted(current["yfinance_symbol"].unique()))
    quality_scores = compute_quality_scores(fundamentals)
    factor_scores = {
        "momentum": momentum_panel(prices), "low_vol": low_vol_panel(prices),
        "value": value_panel(prices, fundamentals["trailingEps"]), "quality": quality_scores,
    }
    fwd_returns = forward_returns_panel(prices)

    return prices, market_caps, membership, benchmark_value, quality_scores, factor_scores, fwd_returns


def main(start: str, end: str) -> pd.DataFrame:
    prices, market_caps, membership, benchmark_value, quality_scores, factor_scores, fwd_returns = build_backtest_inputs(start, end)
    returns_by_strategy = simulate_all_variants(prices, market_caps, membership, quality_scores, factor_scores, fwd_returns)
    benchmark_returns = benchmark_value.pct_change().dropna()

    records = []
    for name, returns in returns_by_strategy.items():
        summary = summarize_tracking(returns, benchmark_returns)
        annualized_return = float((1 + returns.mean()) ** 252 - 1)
        annualized_vol = float(returns.std(ddof=1) * np.sqrt(252))
        records.append({
            "strategy": name, "annualized_return": annualized_return, "annualized_vol": annualized_vol,
            "tracking_error_vs_full_replication": summary.tracking_error_annualized,
            "correlation": summary.correlation, "n_days": len(returns),
        })

    results = pd.DataFrame(records)
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main("2016-01-01", "2026-08-07")
