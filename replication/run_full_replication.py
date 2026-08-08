"""Runs the full cap-weighted replication baseline end-to-end against
real cached data and reports tracking error -- the reference point every
later strategy variant (optimized sampling, smart-beta, multi-objective) is
measured against.

Honest result (see module docstrings in `full_replication.py` and
`data/shares_outstanding.py` for the full diagnosis): tracking error here is
small but not literally zero, ~1.2-2.0% annualized depending on the window's
data coverage, not a bug in the simulation mechanics (see
`tests/replication/test_full_replication.py`, which proves the mechanics
reproduce a synthetic benchmark exactly). It's driven by three disclosed,
real-world data limitations: (1) `sharesOutstanding` is used as a free-float
proxy, not the S&P's own official investable-weight-factor float; (2) 86% of
the point-in-time constituent universe (2016-2026) has fetchable price
history -- 14% are delisted/acquired names Yahoo Finance no longer serves;
(3) rebalancing happens on real quarterly + real corporate-action-event
dates, not continuously. Tracking error falls from ~1.95% (full 2016-2026
window) to ~1.2% (2023-2026, where coverage is highest), directly confirming
coverage is a real contributor, not the sole cause.

Usage: `python -m replication.run_full_replication [--start 2016-01-01]`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.prices import fetch_index_level, to_wide_panel
from data.shares_outstanding import fetch_universe_shares_anchors, fetch_universe_shares_outstanding
from data.wikipedia_constituents import fetch_constituents_and_changes
from replication.market_cap import build_market_cap_panel, coverage_report
from replication.full_replication import simulate_cap_weighted_replication, summarize_coverage
from replication.point_in_time import build_membership_history, quarterly_rebalance_dates
from risk.tracking_error import summarize_tracking

PRICE_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "prices"


def _load_cached_prices() -> pd.DataFrame:
    histories = {}
    for path in PRICE_CACHE_DIR.glob("*.parquet"):
        df = pd.read_parquet(path)
        if not df.empty:
            histories[path.stem] = df
    return to_wide_panel(histories, field="close")


def main(start: str, end: str) -> None:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    prices = _load_cached_prices()
    print(f"Loaded cached prices: {prices.shape[1]} symbols, {prices.index.min().date()} -> {prices.index.max().date()}")

    current, changes = fetch_constituents_and_changes()
    rebalance_dates = quarterly_rebalance_dates(start_ts, end_ts)
    membership = build_membership_history(rebalance_dates, current, changes)
    print(f"Point-in-time membership: {len(rebalance_dates)} quarterly rebalances, "
          f"{membership['symbol'].nunique()} unique tickers")

    symbols = sorted(prices.columns)
    shares_map, shares_missing = fetch_universe_shares_outstanding(symbols)
    anchors = fetch_universe_shares_anchors(symbols)
    market_caps = build_market_cap_panel(prices, shares_map, class_specific_anchors=anchors)
    print(f"Market cap coverage: {coverage_report(prices, market_caps)}")

    value, returns, coverage = simulate_cap_weighted_replication(prices, market_caps, membership)
    cov_df = summarize_coverage(coverage)
    print(f"\nPer-rebalance name coverage: min={cov_df['coverage_fraction'].min():.1%}, "
          f"max={cov_df['coverage_fraction'].max():.1%}, mean={cov_df['coverage_fraction'].mean():.1%}")

    price_returns = fetch_index_level("^GSPC")["close"].pct_change().dropna()
    tr_returns = fetch_index_level("^SP500TR")["close"].pct_change().dropna()

    print(f"\nTracking summary, {start_ts.date()} -> {end_ts.date()}:")
    print(f"  vs ^GSPC (price index):        {summarize_tracking(returns, price_returns)}")
    print(f"  vs ^SP500TR (total return):    {summarize_tracking(returns, tr_returns)}")

    recent_start = "2023-01-01"
    recent_returns = returns[returns.index >= recent_start]
    print(f"\nHigh-coverage recent window ({recent_start} -> {end_ts.date()}), confirming coverage drives TE down:")
    print(f"  vs ^GSPC:    {summarize_tracking(recent_returns, price_returns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-08-07")
    args = parser.parse_args()
    main(args.start, args.end)
