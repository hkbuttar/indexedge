"""Fetches the two source tables this project's point-in-time constituent
reconstruction is built from: Wikipedia's "List of S&P 500 companies" page
carries (1) the current constituent list, each row dated with the company's
actual index-entry date, and (2) a "Selected changes to the components"
table logging every add/remove event with an effective date, back to 1976.

This is a real, dated, cross-checkable source (each row cites a press
release or S&P announcement), not a survivorship-biased snapshot -- which is
exactly why `replication.point_in_time` rolls the current list backward
through the changes table instead of just using today's constituents for
all historical dates. See that module's docstring for the reconstruction
logic and its own disclosed gaps (pre-1976 coverage, undetected pure ticker
renames).

Results are cached to `data/cache/` (gitignored) since Wikipedia's own
etiquette asks bots not to hammer the same page repeatedly, and this data
changes at most a few times a month.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "indexedge-research-project (point-in-time S&P 500 constituent reconstruction)"

CACHE_DIR = Path(__file__).parent / "cache"
CURRENT_CACHE_PATH = CACHE_DIR / "current_constituents.csv"
CHANGES_CACHE_PATH = CACHE_DIR / "sp500_changes.csv"


def _to_yfinance_symbol(symbol: str) -> str:
    """yfinance/most US equity data feeds use a dash for the class suffix
    (BRK-B, BF-B); Wikipedia's table uses the dot form S&P itself uses
    (BRK.B, BF.B). Both are kept: `symbol` is the canonical/citable one,
    `yfinance_symbol` is the lookup key for price/fundamentals fetches."""
    return symbol.replace(".", "-") if isinstance(symbol, str) else symbol


def fetch_raw_tables(url: str = WIKI_URL, timeout: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    if len(tables) < 2:
        raise ValueError(f"expected >=2 tables on the Wikipedia page, got {len(tables)} -- page layout may have changed")
    return tables[0], tables[1]


def clean_current(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={
        "Symbol": "symbol", "Security": "security", "GICS Sector": "gics_sector",
        "GICS Sub-Industry": "gics_sub_industry", "Headquarters Location": "headquarters",
        "Date added": "date_added", "CIK": "cik", "Founded": "founded",
    })
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["yfinance_symbol"] = df["symbol"].map(_to_yfinance_symbol)
    return df[["symbol", "yfinance_symbol", "security", "gics_sector", "gics_sub_industry",
               "headquarters", "date_added", "cik", "founded"]]


def clean_changes(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    # Real production failure, not hypothetical: Wikipedia's changes table
    # fetched with 7 columns instead of the usual 6. Traced to one specific
    # historical row (a 2023 JEF/SPB spinoff) whose reason text was split
    # across two <td> cells by malformed markup on the page itself -- not a
    # genuine schema change, and the extra column was ~100% empty except for
    # that one row's overflow text. A hardcoded 6-name assignment crashed on
    # this (`ValueError: Length mismatch`) rather than either dropping real
    # content or failing gracefully. Fix: every column from `core_columns`
    # onward is treated as "reason" and concatenated -- a no-op on the
    # normal 6-column fetch, and content-preserving (not data-dropping) on
    # a fetch like this one.
    core_columns = ["effective_date", "added_ticker", "added_security", "removed_ticker", "removed_security"]
    if df.shape[1] <= len(core_columns):
        raise ValueError(
            f"expected at least {len(core_columns) + 1} columns in Wikipedia's changes "
            f"table (5 core fields + reason), got {df.shape[1]}: {df.columns.tolist()} -- "
            "the page's real structure may have changed."
        )
    reason = df.iloc[:, len(core_columns):].apply(
        lambda row: " ".join(str(v).strip() for v in row if pd.notna(v)), axis=1
    )
    df = df.iloc[:, :len(core_columns)].copy()
    df["reason"] = reason
    df.columns = ["effective_date", "added_ticker", "added_security",
                  "removed_ticker", "removed_security", "reason"]
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df = df.dropna(subset=["effective_date"])
    df["added_yfinance"] = df["added_ticker"].map(_to_yfinance_symbol)
    df["removed_yfinance"] = df["removed_ticker"].map(_to_yfinance_symbol)
    return df.sort_values("effective_date").reset_index(drop=True)


def fetch_constituents_and_changes(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached fetch. Returns (current_constituents, changes_history), both
    already cleaned via `clean_current`/`clean_changes`."""
    if not force_refresh and CURRENT_CACHE_PATH.exists() and CHANGES_CACHE_PATH.exists():
        current = pd.read_csv(CURRENT_CACHE_PATH, parse_dates=["date_added"])
        changes = pd.read_csv(CHANGES_CACHE_PATH, parse_dates=["effective_date"])
        return current, changes

    raw_current, raw_changes = fetch_raw_tables()
    current, changes = clean_current(raw_current), clean_changes(raw_changes)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    current.to_csv(CURRENT_CACHE_PATH, index=False)
    changes.to_csv(CHANGES_CACHE_PATH, index=False)
    return current, changes
