"""Runs the Step 3 walk-forward comparison of stratified, cvxpy-optimized,
and LASSO sampling against Step 2's full replication, and prints the
tracking-error-vs-name-count curve for all three methods.

Target-count grid is capped at 200: LASSO's achievable name count is
mathematically bounded by the trailing lookback window's sample size
(~252 observations) -- see `lasso_sampling.py`'s docstring -- so a shared
grid above that would let LASSO's curve silently plateau while the other
two methods kept improving, which would misrepresent the comparison at the
top end rather than being a fair three-way read at every grid point.

Usage: `python -m replication.run_sampling_comparison`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.prices import to_wide_panel
from data.shares_outstanding import fetch_universe_shares_anchors, fetch_universe_shares_outstanding
from data.wikipedia_constituents import fetch_constituents_and_changes
from replication.full_replication import simulate_cap_weighted_replication
from replication.market_cap import build_market_cap_panel
from replication.point_in_time import build_membership_history, quarterly_rebalance_dates
from replication.sampling_evaluation import evaluate_sampling_methods, summarize_curve

PRICE_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "prices"
TARGET_COUNTS = [20, 40, 60, 100, 150, 200]


def _load_cached_prices() -> pd.DataFrame:
    histories = {}
    for path in PRICE_CACHE_DIR.glob("*.parquet"):
        df = pd.read_parquet(path)
        if not df.empty:
            histories[path.stem] = df
    return to_wide_panel(histories, field="close")


def main(start: str, end: str) -> pd.DataFrame:
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
    sector_by_symbol = dict(zip(current["yfinance_symbol"], current["gics_sector"]))

    print(f"Evaluating {len(TARGET_COUNTS)} name-count targets across "
          f"{len(rebalance_dates)} rebalance dates (walk-forward, trailing-fit -> forward-test)...")
    results = evaluate_sampling_methods(
        prices, market_caps, membership, benchmark_value, sector_by_symbol, TARGET_COUNTS
    )
    curve = summarize_curve(results)
    print(f"\n{len(results['rebalance_date'].unique())} rebalance dates had sufficient trailing history "
          f"({results['rebalance_date'].nunique()} used out of {len(rebalance_dates)} total)")
    print("\nTracking-error-vs-name-count curve (mean annualized TE vs full replication, out-of-sample):")
    print(curve.to_string(index=False))
    return curve


if __name__ == "__main__":
    main("2016-01-01", "2026-08-07")
