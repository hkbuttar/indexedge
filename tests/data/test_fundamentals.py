import pandas as pd
import pytest

from data.fundamentals import fetch_symbol_fundamentals, fetch_universe_fundamentals


class _FakeTicker:
    def __init__(self, symbol, info=None, raise_on_info=False):
        self._info = info or {}
        self._raise_on_info = raise_on_info

    @property
    def info(self):
        if self._raise_on_info:
            raise ValueError("simulated yfinance failure")
        return self._info


def test_fetch_symbol_fundamentals_extracts_only_known_fields(tmp_path, monkeypatch):
    import data.fundamentals as fnd

    monkeypatch.setattr(fnd, "CACHE_DIR", tmp_path)
    info = {"sector": "Technology", "returnOnEquity": 0.3, "irrelevantField": "ignored"}
    monkeypatch.setattr(fnd.yf, "Ticker", lambda s: _FakeTicker(s, info=info))

    result = fetch_symbol_fundamentals("AAPL")
    assert result["sector"] == "Technology"
    assert result["returnOnEquity"] == 0.3
    assert "irrelevantField" not in result
    assert result["debtToEquity"] is None  # present in FIELDS but absent from info -> None, not dropped


def test_fetch_symbol_fundamentals_caches_miss_when_all_fields_none(tmp_path, monkeypatch):
    import data.fundamentals as fnd

    monkeypatch.setattr(fnd, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fnd.yf, "Ticker", lambda s: _FakeTicker(s, info={}))

    result = fetch_symbol_fundamentals("NODATA")
    assert result is None
    assert (tmp_path / "NODATA.parquet").exists()


def test_fetch_symbol_fundamentals_treats_exception_as_no_data(tmp_path, monkeypatch):
    import data.fundamentals as fnd

    monkeypatch.setattr(fnd, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fnd.yf, "Ticker", lambda s: _FakeTicker(s, raise_on_info=True))

    assert fetch_symbol_fundamentals("BADSYM") is None


def test_fetch_symbol_fundamentals_uses_cache_when_present(tmp_path, monkeypatch):
    import data.fundamentals as fnd

    monkeypatch.setattr(fnd, "CACHE_DIR", tmp_path)
    pd.DataFrame([{"sector": "Health", "returnOnEquity": 0.1}]).to_parquet(tmp_path / "AAPL.parquet")

    def _should_not_be_called(s):
        raise AssertionError("should not construct a Ticker when cache exists")
    monkeypatch.setattr(fnd.yf, "Ticker", _should_not_be_called)

    result = fetch_symbol_fundamentals("AAPL")
    assert result["sector"] == "Health"


def test_fetch_universe_fundamentals_builds_indexed_dataframe(tmp_path, monkeypatch):
    import data.fundamentals as fnd

    monkeypatch.setattr(fnd, "CACHE_DIR", tmp_path)

    def _fake_ticker(symbol):
        info = {"sector": "Tech"} if symbol == "AAPL" else {}
        return _FakeTicker(symbol, info=info)
    monkeypatch.setattr(fnd.yf, "Ticker", _fake_ticker)

    df, missing = fetch_universe_fundamentals(["AAPL", "NODATA"])
    assert df.index.name == "symbol"
    assert df.loc["AAPL", "sector"] == "Tech"
    assert missing == ["NODATA"]
