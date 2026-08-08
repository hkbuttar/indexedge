"""Builds a point-in-time market-cap panel from real price and shares-
outstanding data: market_cap(t) = price(t) * shares_outstanding(t), with
`shares_outstanding` forward-filled from Yahoo's irregular filing-date
series (`data.shares_outstanding`) onto the daily price calendar.

Forward-fill is the disclosed approximation here: a share count is treated
as constant between two filed values, which is correct in expectation but
can lag same-day corporate actions (buybacks, offerings) by however long
the filing lag was. No backfill -- dates before a symbol's first filed
share count are left NaN rather than guessed.

`class_specific_anchors`, if passed, rescales each symbol's whole historical
share series by (anchor / most-recent raw value) before multiplying by
price. This corrects a real bug in the upstream `get_shares_full` data for
dual-class tickers (GOOGL/GOOG, FOX/FOXA, NWS/NWSA), which otherwise return
the company-wide combined share count identically for each class and
roughly double-count that company's weight -- see
`data.shares_outstanding`'s module docstring for how this was diagnosed.
"""

from __future__ import annotations

import pandas as pd


def build_market_cap_panel(
    prices: pd.DataFrame,
    shares_by_symbol: dict[str, pd.Series],
    class_specific_anchors: dict[str, float] | None = None,
) -> pd.DataFrame:
    class_specific_anchors = class_specific_anchors or {}
    caps = {}
    for symbol in prices.columns:
        shares = shares_by_symbol.get(symbol)
        if shares is None or shares.empty:
            continue
        combined_index = prices.index.union(shares.index)
        aligned_shares = shares.reindex(combined_index).sort_index().ffill().reindex(prices.index)

        anchor = class_specific_anchors.get(symbol)
        if anchor and shares.iloc[-1] > 0:
            aligned_shares = aligned_shares * (anchor / shares.iloc[-1])

        caps[symbol] = prices[symbol] * aligned_shares
    return pd.DataFrame(caps).sort_index(axis=1)


def coverage_report(prices: pd.DataFrame, market_caps: pd.DataFrame) -> dict:
    priced_symbols = set(prices.columns)
    capped_symbols = set(market_caps.columns)
    missing = sorted(priced_symbols - capped_symbols)
    return {
        "priced_symbols": len(priced_symbols),
        "market_cap_symbols": len(capped_symbols),
        "missing_shares_data": missing,
        "coverage_fraction": len(capped_symbols) / len(priced_symbols) if priced_symbols else float("nan"),
    }
