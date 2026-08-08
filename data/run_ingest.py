"""Single entry point for this project's real data pull: point-in-time
constituent reconstruction, index level (both ^GSPC and ^SP500TR --
`replication/full_replication.py` and `backend/main.py`'s
`/api/replication/full` compare against both), constituent price history,
historical shares outstanding + dual-class anchors (needed by
`replication/market_cap.py` for point-in-time market caps -- see
`data/shares_outstanding.py`'s docstring), and current-snapshot
fundamentals. Re-running is cheap -- every fetch below is cached
per-symbol on disk (`data/cache/`, gitignored) and only hits the network
for what's missing.

Deliberately fetches everything `backend/main.py`'s `_STATE` will need at
startup, not just prices/fundamentals: this is also what
`render.yaml`'s buildCommand runs, specifically so the deployed backend's
one-time startup computation (see that module's docstring) doesn't
silently spend several extra minutes live-fetching shares-outstanding
history and dual-class anchors on its first real request instead.

Usage: `python -m data.run_ingest [--start 2016-01-01] [--end today]`
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from data.fundamentals import fetch_universe_fundamentals
from data.prices import fetch_constituent_history, fetch_index_level
from data.shares_outstanding import fetch_universe_shares_anchors, fetch_universe_shares_outstanding
from data.wikipedia_constituents import fetch_constituents_and_changes
from replication.point_in_time import build_membership_history, coverage_notes, quarterly_rebalance_dates


def main(start: str, end: str) -> None:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    print("Fetching Wikipedia constituent list + changes history...")
    current, changes = fetch_constituents_and_changes()
    print(f"  {len(current)} current constituents, {len(changes)} logged changes "
          f"({changes['effective_date'].min().date()} -> {changes['effective_date'].max().date()})")
    for note in coverage_notes(current, changes):
        print(f"  NOTE: {note}")

    print(f"\nReconstructing point-in-time membership, {start_ts.date()} -> {end_ts.date()}...")
    dates = quarterly_rebalance_dates(start_ts, end_ts)
    history = build_membership_history(dates, current, changes)
    print(f"  {len(dates)} quarterly rebalance dates, {history['symbol'].nunique()} unique tickers ever in the index")

    symbol_map = dict(zip(current["symbol"], current["yfinance_symbol"]))
    yf_symbols = sorted({symbol_map.get(s, s.replace(".", "-")) for s in history["symbol"].unique()})

    print(f"\nFetching index level (^GSPC, ^SP500TR)...")
    fetch_index_level("^GSPC")
    fetch_index_level("^SP500TR")

    print(f"\nFetching price history for {len(yf_symbols)} symbols...")
    price_histories, price_missing = fetch_constituent_history(yf_symbols)
    coverage = len(price_histories) / len(yf_symbols)
    print(f"  {len(price_histories)}/{len(yf_symbols)} symbols fetched ({coverage:.1%} coverage)")
    print(f"  {len(price_missing)} unreachable (delisted/acquired names yfinance has no history for): {price_missing[:15]}{'...' if len(price_missing) > 15 else ''}")

    priced_symbols = sorted(price_histories.keys())
    print(f"\nFetching historical shares outstanding for {len(priced_symbols)} priced symbols...")
    shares_map, shares_missing = fetch_universe_shares_outstanding(priced_symbols)
    print(f"  {len(shares_map)}/{len(priced_symbols)} fetched, {len(shares_missing)} missing")

    print(f"\nFetching dual-class share anchors for {len(priced_symbols)} priced symbols...")
    anchors = fetch_universe_shares_anchors(priced_symbols)
    print(f"  {len(anchors)}/{len(priced_symbols)} fetched")

    print(f"\nFetching current-snapshot fundamentals for {len(current)} current constituents...")
    fundamentals, fundamentals_missing = fetch_universe_fundamentals(sorted(current["yfinance_symbol"].unique()))
    print(f"  {len(fundamentals)}/{len(current)} fetched, {len(fundamentals_missing)} missing: {fundamentals_missing}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=str(date.today()))
    args = parser.parse_args()
    main(args.start, args.end)
