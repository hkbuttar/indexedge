import pandas as pd
import pytest

from data.shares_outstanding import (
    fetch_symbol_shares_anchor,
    fetch_symbol_shares_outstanding,
    fetch_universe_shares_anchors,
    fetch_universe_shares_outstanding,
)


class _FakeTicker:
    def __init__(self, symbol, shares_series=None, info=None, raise_on_shares=False):
        self.symbol = symbol
        self._shares_series = shares_series
        self.info = info or {}
        self._raise_on_shares = raise_on_shares

    def get_shares_full(self, start=None):
        if self._raise_on_shares:
            raise ValueError("simulated yfinance failure")
        return self._shares_series


def test_fetch_symbol_shares_outstanding_caches_and_normalizes_index(tmp_path, monkeypatch):
    import data.shares_outstanding as so

    monkeypatch.setattr(so, "CACHE_DIR", tmp_path)
    raw = pd.Series(
        [100.0, 200.0],
        index=pd.DatetimeIndex([
            pd.Timestamp("2020-01-01", tz="US/Eastern"),
            pd.Timestamp("2020-06-01", tz="US/Eastern"),
        ]),
    )
    monkeypatch.setattr(so.yf, "Ticker", lambda s: _FakeTicker(s, shares_series=raw))

    result = fetch_symbol_shares_outstanding("AAPL")
    assert result is not None
    assert result.index.tz is None  # tz-localized input normalized to naive
    assert (tmp_path / "AAPL.parquet").exists()


def test_fetch_symbol_shares_outstanding_caches_miss_on_empty_or_exception(tmp_path, monkeypatch):
    import data.shares_outstanding as so

    monkeypatch.setattr(so, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(so.yf, "Ticker", lambda s: _FakeTicker(s, raise_on_shares=True))

    result = fetch_symbol_shares_outstanding("BADSYM")
    assert result is None
    assert (tmp_path / "BADSYM.parquet").exists()


def test_fetch_symbol_shares_outstanding_uses_cache_when_present(tmp_path, monkeypatch):
    import data.shares_outstanding as so

    monkeypatch.setattr(so, "CACHE_DIR", tmp_path)
    dates = pd.to_datetime(["2020-01-01"])
    pd.DataFrame({"shares": [500.0]}, index=dates).to_parquet(tmp_path / "AAPL.parquet")

    def _should_not_be_called(s):
        raise AssertionError("should not construct a Ticker when cache exists")
    monkeypatch.setattr(so.yf, "Ticker", _should_not_be_called)

    result = fetch_symbol_shares_outstanding("AAPL")
    assert result.iloc[0] == 500.0


def test_fetch_universe_shares_outstanding_separates_hits_and_misses(tmp_path, monkeypatch):
    import data.shares_outstanding as so

    monkeypatch.setattr(so, "CACHE_DIR", tmp_path)
    raw = pd.Series([100.0], index=pd.to_datetime(["2020-01-01"]))

    def _fake_ticker(symbol):
        return _FakeTicker(symbol, shares_series=raw if symbol == "AAPL" else None)

    monkeypatch.setattr(so.yf, "Ticker", _fake_ticker)

    series_by_symbol, missing = fetch_universe_shares_outstanding(["AAPL", "NODATA"])
    assert "AAPL" in series_by_symbol
    assert missing == ["NODATA"]


def test_fetch_symbol_shares_anchor_reads_info_field(tmp_path, monkeypatch):
    import data.shares_outstanding as so

    monkeypatch.setattr(so, "ANCHOR_CACHE_DIR", tmp_path)
    monkeypatch.setattr(so.yf, "Ticker", lambda s: _FakeTicker(s, info={"sharesOutstanding": 5_867_155_790}))

    anchor = fetch_symbol_shares_anchor("GOOGL")
    assert anchor == pytest.approx(5_867_155_790)
    assert (tmp_path / "GOOGL.parquet").exists()


def test_fetch_symbol_shares_anchor_none_when_field_missing(tmp_path, monkeypatch):
    import data.shares_outstanding as so

    monkeypatch.setattr(so, "ANCHOR_CACHE_DIR", tmp_path)
    monkeypatch.setattr(so.yf, "Ticker", lambda s: _FakeTicker(s, info={}))

    assert fetch_symbol_shares_anchor("NOFIELD") is None


def test_fetch_universe_shares_anchors_only_includes_found_values(tmp_path, monkeypatch):
    import data.shares_outstanding as so

    monkeypatch.setattr(so, "ANCHOR_CACHE_DIR", tmp_path)

    def _fake_ticker(symbol):
        info = {"sharesOutstanding": 1000.0} if symbol == "A" else {}
        return _FakeTicker(symbol, info=info)
    monkeypatch.setattr(so.yf, "Ticker", _fake_ticker)

    anchors = fetch_universe_shares_anchors(["A", "B"])
    assert anchors == {"A": 1000.0}
