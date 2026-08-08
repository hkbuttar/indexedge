"""Real historical shares-outstanding series via yfinance's
`Ticker.get_shares_full()`, which returns Yahoo's own irregular-interval
share-count history (updated whenever a company files a change, not daily).

This exists so `replication.market_cap` can compute genuine point-in-time
market cap (price(t) * shares_outstanding(t)) instead of the much weaker
approximation of multiplying historical price by *today's* share count --
which would silently misweight any company that did a buyback, secondary
offering, or split-adjustment-independent share issuance during the
backtest window. Forward-filling this irregular series onto the daily price
index (done in `market_cap.py`, not here) is itself a disclosed
approximation: the true share count on a given day is whatever was last
filed at or before that day, which is exactly what forward-fill gives you,
but it can lag a same-day change by however long the filing lag was.

## A real bug this module works around: dual-class tickers

`get_shares_full()` was found (by comparing the initial full-replication
tracking error against a sanity check on portfolio weights -- GOOGL+GOOG
alone came out to ~12% combined, roughly 2.6x Alphabet's real ~4.5% S&P
weight) to return the company-WIDE combined share count identically for
*each* listed class of a multi-class company, not the class's own share
count. GOOGL and GOOG (Alphabet), FOX/FOXA (Fox Corp), and NWS/NWSA (News
Corp) are the S&P 500 names this affects; using the raw series for both
tickers of a pair double-counts that company's true market cap.

`fetch_symbol_shares_anchor()` below pulls the class-specific count from
`Ticker.info['sharesOutstanding']` (confirmed distinct per class, e.g.
GOOGL 5.87B vs GOOG 5.53B, vs. `get_shares_full`'s identical 12.23B for
both) as a single current-day anchor, and `market_cap.py` rescales the
whole historical `get_shares_full` series by (anchor / most-recent raw
value) before computing market cap. This assumes the class mix has been
roughly stable historically -- a disclosed approximation, not a claim of
exact historical class-level share counts, but a large improvement over the
2.6x error it replaces. Applied uniformly to every symbol (not just the
known dual-class names) since it's a no-op wherever `info.sharesOutstanding`
already equals the single-class total.

Same cache-per-symbol pattern as `prices.py`/`fundamentals.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from data._retry import retry_with_backoff

CACHE_DIR = Path(__file__).parent / "cache" / "shares_outstanding"
ANCHOR_CACHE_DIR = Path(__file__).parent / "cache" / "shares_anchor"


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('/', '_')}.parquet"


def fetch_symbol_shares_outstanding(symbol: str, force_refresh: bool = False) -> pd.Series | None:
    path = _cache_path(symbol)
    if not force_refresh and path.exists():
        cached = pd.read_parquet(path)
        return None if cached.empty else cached["shares"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = retry_with_backoff(lambda: yf.Ticker(symbol).get_shares_full(start="1990-01-01"))

    if raw is None or len(raw) == 0:
        pd.DataFrame().to_parquet(path)
        return None

    series = raw.copy()
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "shares"

    series.to_frame().to_parquet(path)
    return series


def fetch_universe_shares_outstanding(
    symbols: list[str], force_refresh: bool = False
) -> tuple[dict[str, pd.Series], list[str]]:
    series_by_symbol, missing = {}, []
    for symbol in sorted(set(symbols)):
        result = fetch_symbol_shares_outstanding(symbol, force_refresh=force_refresh)
        if result is None:
            missing.append(symbol)
        else:
            series_by_symbol[symbol] = result
    return series_by_symbol, missing


def _anchor_cache_path(symbol: str) -> Path:
    return ANCHOR_CACHE_DIR / f"{symbol.replace('/', '_')}.parquet"


def fetch_symbol_shares_anchor(symbol: str, force_refresh: bool = False) -> float | None:
    """Single current-day class-specific share count from `Ticker.info`,
    used only to rescale `get_shares_full`'s combined-class series -- see
    module docstring."""
    path = _anchor_cache_path(symbol)
    if not force_refresh and path.exists():
        cached = pd.read_parquet(path)
        return None if cached.empty else float(cached["shares"].iloc[0])

    ANCHOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    value = retry_with_backoff(lambda: yf.Ticker(symbol).info.get("sharesOutstanding"))

    pd.DataFrame({"shares": [value]} if value else []).to_parquet(path)
    return float(value) if value else None


def fetch_universe_shares_anchors(symbols: list[str], force_refresh: bool = False) -> dict[str, float]:
    anchors = {}
    for symbol in sorted(set(symbols)):
        value = fetch_symbol_shares_anchor(symbol, force_refresh=force_refresh)
        if value is not None:
            anchors[symbol] = value
    return anchors
