"""Real daily price data via yfinance: the S&P 500 index level itself
(`^GSPC`) and constituent adjusted-close/volume history for whatever ticker
universe the caller passes in (typically the point-in-time union from
`replication.point_in_time.build_membership_history`).

Caching is per-symbol, full-history, on disk (`data/cache/prices/`,
gitignored): each symbol is fetched once for its entire available history
and re-sliced in memory for whatever date window a given call needs, rather
than re-fetching per (symbol, window) pair. This matters here specifically
because later steps (replication, smart-beta, bootstrap backtesting) all
re-slice the same universe over many different rebalance windows.

Not every historical constituent is reachable through yfinance -- tickers
that were acquired/delisted long enough ago sometimes have no data under
their old symbol. This is a real, disclosed gap (surfaced via the
`missing` list every fetch function returns), not silently dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache" / "prices"
INDEX_CACHE_DIR = Path(__file__).parent / "cache" / "index"

DEFAULT_INDEX_TICKER = "^GSPC"


def _cache_path(base_dir: Path, symbol: str) -> Path:
    return base_dir / f"{symbol.replace('/', '_')}.parquet"


def fetch_index_level(ticker: str = DEFAULT_INDEX_TICKER, force_refresh: bool = False) -> pd.DataFrame:
    """Full available daily history for the index level itself, columns
    open/high/low/close/volume, auto-adjusted (irrelevant for an index level
    with no dividends/splits, but kept consistent with constituent fetches)."""
    path = _cache_path(INDEX_CACHE_DIR, ticker)
    if not force_refresh and path.exists():
        return pd.read_parquet(path)

    df = yf.download(ticker, period="max", auto_adjust=True, progress=False, group_by="ticker")
    if df.empty:
        raise ValueError(f"yfinance returned no data for index ticker {ticker!r}")
    df = df[ticker].rename(columns=str.lower)
    df.index.name = "date"

    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def fetch_symbol_history(symbol: str, force_refresh: bool = False) -> pd.DataFrame | None:
    """Full available daily history for one constituent symbol, or None if
    yfinance has no data for it (delisted/unreachable)."""
    path = _cache_path(CACHE_DIR, symbol)
    if not force_refresh and path.exists():
        cached = pd.read_parquet(path)
        return None if cached.empty else cached

    df = yf.download(symbol, period="max", auto_adjust=True, progress=False, group_by="ticker")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        pd.DataFrame().to_parquet(path)  # cache the miss so re-runs don't re-hit the network
        return None

    df = df[symbol].rename(columns=str.lower)
    df.index.name = "date"
    df.to_parquet(path)
    return df


def fetch_constituent_history(
    symbols: list[str], force_refresh: bool = False, batch_size: int = 100
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Fetches full history per symbol, batched via yfinance's own
    multi-ticker download (much faster than one `Ticker.history()` call per
    symbol) for whatever symbols aren't already cached, falling back to a
    single-symbol retry for any symbol a batch response is missing so one
    bad ticker in a batch doesn't silently blank out its batch-mates.

    Returns (histories keyed by symbol, list of symbols with no usable data).
    """
    symbols = sorted(set(symbols))
    histories: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    to_fetch = []
    for symbol in symbols:
        path = _cache_path(CACHE_DIR, symbol)
        if not force_refresh and path.exists():
            cached = pd.read_parquet(path)
            if cached.empty:
                missing.append(symbol)
            else:
                histories[symbol] = cached
        else:
            to_fetch.append(symbol)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i + batch_size]
        raw = yf.download(batch, period="max", auto_adjust=True, progress=False,
                           group_by="ticker", threads=True)
        top_level = set(raw.columns.get_level_values(0)) if not raw.empty else set()

        for symbol in batch:
            if symbol not in top_level:
                result = fetch_symbol_history(symbol, force_refresh=force_refresh)
                if result is not None:
                    histories[symbol] = result
                else:
                    missing.append(symbol)
                continue
            df = raw[symbol].dropna(how="all")
            if df.empty:
                pd.DataFrame().to_parquet(_cache_path(CACHE_DIR, symbol))
                missing.append(symbol)
                continue
            df = df.rename(columns=str.lower)
            df.index.name = "date"
            df.to_parquet(_cache_path(CACHE_DIR, symbol))
            histories[symbol] = df

    return histories, missing


def to_wide_panel(histories: dict[str, pd.DataFrame], field: str = "close") -> pd.DataFrame:
    """Long dict-of-DataFrames -> wide date x symbol panel for one field
    (default adjusted close). Missing (symbol, date) cells are NaN, not
    forward-filled -- callers decide how to handle gaps (e.g. a symbol not
    yet public, or a genuine trading halt)."""
    series = {symbol: df[field] for symbol, df in histories.items() if field in df.columns}
    return pd.DataFrame(series).sort_index()
