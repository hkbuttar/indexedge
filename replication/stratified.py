"""Stratified sampling: bucket point-in-time constituents by GICS sector x
market-cap tercile-within-sector, pick the largest-cap name in each bucket
as its representative, and weight that representative by the *bucket's*
total market cap (not just its own) -- so the representative stands in for
the whole bucket's cap exposure. This is a structural method: it uses no
historical return data at all, unlike the other two methods in this
comparison.

Sector labels are a disclosed snapshot limitation, same as
`data/fundamentals.py`: this project only has *current* GICS sector
assignments (from the Wikipedia current-constituent table), not each
company's sector as of a historical rebalance date. A name that was
reclassified between sectors during the backtest window is bucketed by its
present-day sector at every historical date, not its historical one.
Delisted names with no current sector label are excluded from stratified
candidacy entirely (they simply can't form or join a bucket) -- a narrower
version of the same gap.
"""

from __future__ import annotations

import pandas as pd


def build_buckets(
    members: set[str], market_cap_row: pd.Series, sector_by_symbol: dict[str, str], buckets_per_sector: int
) -> dict[tuple[str, int], list[str]]:
    available = [
        s for s in members
        if s in market_cap_row.index and pd.notna(market_cap_row[s]) and market_cap_row[s] > 0
        and s in sector_by_symbol
    ]
    by_sector: dict[str, list[str]] = {}
    for symbol in available:
        by_sector.setdefault(sector_by_symbol[symbol], []).append(symbol)

    buckets: dict[tuple[str, int], list[str]] = {}
    for sector, symbols in by_sector.items():
        ranked = market_cap_row[symbols].sort_values(ascending=False)
        n_buckets = min(buckets_per_sector, len(ranked))
        # split the sector's cap-ranked names into n_buckets contiguous size tiers
        tiers = pd.qcut(range(len(ranked)), n_buckets, labels=False, duplicates="drop") if len(ranked) > 1 else [0]
        for symbol, tier in zip(ranked.index, tiers):
            buckets.setdefault((sector, int(tier)), []).append(symbol)
    return buckets


def stratified_sample_weights(
    members: set[str], market_cap_row: pd.Series, sector_by_symbol: dict[str, str], buckets_per_sector: int
) -> pd.Series:
    buckets = build_buckets(members, market_cap_row, sector_by_symbol, buckets_per_sector)
    weights = {}
    for _, symbols in buckets.items():
        representative = market_cap_row[symbols].idxmax()
        bucket_cap = market_cap_row[symbols].sum()
        weights[representative] = weights.get(representative, 0.0) + bucket_cap
    total = sum(weights.values())
    return pd.Series(weights).sort_values(ascending=False) / total if total > 0 else pd.Series(dtype=float)
