"""Real fundamentals via yfinance's `Ticker.info`, for the quality smart-beta
factor in `smartbeta/quality.py`. This is current-snapshot data
only -- yfinance does not expose point-in-time historical fundamentals, so
every symbol's fundamentals here reflect whatever yfinance currently
reports, not the fundamentals as of a specific historical rebalance date.

This is a real, disclosed limitation carried into every downstream use: a
quality-tilted portfolio backtested over, say, 2016-2026 using these
fundamentals is implicitly using look-ahead information (today's ROE/margin
profile) at every historical rebalance, not what was actually knowable at
the time. `smartbeta/quality.py` and the README state this plainly rather
than presenting the backtest as point-in-time-correct on the fundamentals
axis the way `replication.point_in_time` is for index membership.

One `.info` call per symbol (yfinance has no real batch fundamentals
endpoint); cached per-symbol to disk since this is the slow, rate-limit-
sensitive part of the pipeline and the underlying data changes infrequently
relative to how often it'll be re-read here.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache" / "fundamentals"

# Fields used by smartbeta/quality.py's quality score, plus sector/market cap
# for context and stratified sampling (replication/stratified.py).
FIELDS = [
    "sector", "industry", "marketCap",
    "returnOnEquity", "debtToEquity", "profitMargins",
    "earningsGrowth", "earningsQuarterlyGrowth", "trailingEps",
    "freeCashflow", "totalDebt", "totalCash",
]


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('/', '_')}.parquet"


def fetch_symbol_fundamentals(symbol: str, force_refresh: bool = False) -> dict | None:
    path = _cache_path(symbol)
    if not force_refresh and path.exists():
        cached = pd.read_parquet(path)
        return None if cached.empty else cached.iloc[0].to_dict()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        info = {}

    row = {field: info.get(field) for field in FIELDS}
    has_data = any(v is not None for v in row.values())
    pd.DataFrame([row] if has_data else []).to_parquet(path)
    return row if has_data else None


def fetch_universe_fundamentals(
    symbols: list[str], force_refresh: bool = False, pause_seconds: float = 0.0
) -> tuple[pd.DataFrame, list[str]]:
    """Sequential fetch (no batch endpoint exists) with per-symbol caching,
    so a re-run after a partial failure only re-fetches what's missing.
    `pause_seconds` throttles between *uncached* network calls only, to stay
    polite to yfinance's backend on a large universe."""
    rows, missing = {}, []
    for symbol in sorted(set(symbols)):
        was_cached = _cache_path(symbol).exists() and not force_refresh
        result = fetch_symbol_fundamentals(symbol, force_refresh=force_refresh)
        if result is None:
            missing.append(symbol)
        else:
            rows[symbol] = result
        if not was_cached and pause_seconds:
            time.sleep(pause_seconds)

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "symbol"
    return df, missing
