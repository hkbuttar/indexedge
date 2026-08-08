import pandas as pd
import pytest

from data.wikipedia_constituents import (
    _to_yfinance_symbol,
    clean_changes,
    clean_current,
    fetch_constituents_and_changes,
)


def test_to_yfinance_symbol_replaces_dot_with_dash():
    assert _to_yfinance_symbol("BRK.B") == "BRK-B"
    assert _to_yfinance_symbol("AAPL") == "AAPL"


def test_clean_current_renames_and_parses_dates():
    raw = pd.DataFrame({
        "Symbol": ["AAPL", "BRK.B"],
        "Security": ["Apple", "Berkshire"],
        "GICS Sector": ["Technology", "Financials"],
        "GICS Sub-Industry": ["Hardware", "Insurance"],
        "Headquarters Location": ["Cupertino", "Omaha"],
        "Date added": ["1982-11-30", "2010-02-01"],
        "CIK": [320193, 1067983],
        "Founded": [1976, 1839],
    })
    cleaned = clean_current(raw)
    assert list(cleaned.columns) == [
        "symbol", "yfinance_symbol", "security", "gics_sector", "gics_sub_industry",
        "headquarters", "date_added", "cik", "founded",
    ]
    assert cleaned.loc[1, "yfinance_symbol"] == "BRK-B"
    assert cleaned["date_added"].dtype.kind == "M"  # datetime


def test_clean_changes_parses_dates_and_normalizes_tickers():
    raw = pd.DataFrame({
        "Effective Date": ["2020-12-21", "not a date"],
        "col_added_ticker": ["TSLA", "BRK.B"],
        "col_added_security": ["Tesla", "Berkshire"],
        "col_removed_ticker": ["AIV", None],
        "col_removed_security": ["Apt Inv Mgmt", None],
        "Reason": ["Market cap change.", "Whatever"],
    })
    cleaned = clean_changes(raw)
    # the invalid date row must be dropped, not silently coerced to NaT and kept
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["added_ticker"] == "TSLA"
    assert cleaned.iloc[0]["added_yfinance"] == "TSLA"


def test_clean_changes_yfinance_normalization_for_dotted_tickers():
    raw = pd.DataFrame({
        "Effective Date": ["2020-01-01"],
        "a": ["BF.B"], "b": ["Brown-Forman"], "c": [None], "d": [None], "e": ["reason"],
    })
    cleaned = clean_changes(raw)
    assert cleaned.iloc[0]["added_yfinance"] == "BF-B"


def test_clean_changes_coalesces_extra_trailing_columns_into_reason():
    # Real production case: Wikipedia's live page returned a 7th column for
    # exactly one historical row, whose reason text got split across two
    # <td> cells by malformed markup on the page itself -- not a genuine
    # schema change. A hardcoded 6-column assignment crashed on this
    # (ValueError: Length mismatch); the fix coalesces any columns beyond
    # the 5 core fields into `reason` instead of dropping data or crashing.
    raw = pd.DataFrame({
        "Effective Date": ["2020-01-01", "2020-02-01"],
        "a": ["TSLA", None], "b": ["Tesla", None], "c": [None, "AIV"], "d": [None, "Apt Inv Mgmt"],
        "Reason": ["Market cap change", "Removed."],
        "Overflow": [None, "extra split text"],
    })
    cleaned = clean_changes(raw)
    assert len(cleaned) == 2
    assert cleaned.iloc[0]["reason"] == "Market cap change"
    assert cleaned.iloc[1]["reason"] == "Removed. extra split text"


def test_clean_changes_raises_on_too_few_columns():
    raw = pd.DataFrame({
        "Effective Date": ["2020-01-01"],
        "a": ["TSLA"], "b": ["Tesla"], "c": [None], "d": [None],
        # no reason column at all -- only 5 total columns
    })
    with pytest.raises(ValueError):
        clean_changes(raw)


def test_fetch_constituents_and_changes_uses_cache_when_present(tmp_path, monkeypatch):
    import data.wikipedia_constituents as wc

    monkeypatch.setattr(wc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(wc, "CURRENT_CACHE_PATH", tmp_path / "current.csv")
    monkeypatch.setattr(wc, "CHANGES_CACHE_PATH", tmp_path / "changes.csv")

    current = pd.DataFrame({"symbol": ["AAPL"], "date_added": pd.to_datetime(["1982-11-30"])})
    changes = pd.DataFrame({"effective_date": pd.to_datetime(["2020-01-01"]), "added_ticker": ["TSLA"]})
    current.to_csv(wc.CURRENT_CACHE_PATH, index=False)
    changes.to_csv(wc.CHANGES_CACHE_PATH, index=False)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("fetch_raw_tables should not be called when cache exists")

    monkeypatch.setattr(wc, "fetch_raw_tables", _should_not_be_called)

    loaded_current, loaded_changes = fetch_constituents_and_changes()
    assert loaded_current["symbol"].tolist() == ["AAPL"]
    assert loaded_changes["added_ticker"].tolist() == ["TSLA"]


def test_fetch_constituents_and_changes_fetches_and_caches_when_absent(tmp_path, monkeypatch):
    import data.wikipedia_constituents as wc

    monkeypatch.setattr(wc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(wc, "CURRENT_CACHE_PATH", tmp_path / "current.csv")
    monkeypatch.setattr(wc, "CHANGES_CACHE_PATH", tmp_path / "changes.csv")

    raw_current = pd.DataFrame({
        "Symbol": ["AAPL"], "Security": ["Apple"], "GICS Sector": ["Tech"], "GICS Sub-Industry": ["HW"],
        "Headquarters Location": ["CA"], "Date added": ["1982-11-30"], "CIK": [1], "Founded": [1976],
    })
    raw_changes = pd.DataFrame({
        "Effective Date": ["2020-01-01"], "a": ["TSLA"], "b": ["Tesla"], "c": [None], "d": [None], "e": ["x"],
    })
    monkeypatch.setattr(wc, "fetch_raw_tables", lambda: (raw_current, raw_changes))

    current, changes = fetch_constituents_and_changes()
    assert wc.CURRENT_CACHE_PATH.exists()
    assert wc.CHANGES_CACHE_PATH.exists()
    assert current["symbol"].tolist() == ["AAPL"]
