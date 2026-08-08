import pandas as pd
import pytest

from data.prices import (
    fetch_constituent_history,
    fetch_index_level,
    fetch_symbol_history,
    to_wide_panel,
)


def _fake_download_multi(tickers, **kwargs):
    if isinstance(tickers, str):
        tickers = [tickers]
    dates = pd.bdate_range("2022-01-03", "2022-01-07")
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
    data = {(t, field): (100.0 if field != "Volume" else 1000) for t in tickers for field in ["Open", "High", "Low", "Close", "Volume"]}
    return pd.DataFrame({k: [v] * len(dates) for k, v in data.items()}, index=dates, columns=cols)


def _fake_download_empty(*args, **kwargs):
    return pd.DataFrame()


# --- to_wide_panel (pure function, no mocking needed) ---

def test_to_wide_panel_extracts_one_field_across_symbols():
    dates = pd.date_range("2024-01-01", periods=3)
    histories = {
        "A": pd.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [10, 20, 30]}, index=dates),
        "B": pd.DataFrame({"close": [4.0, 5.0, 6.0], "volume": [40, 50, 60]}, index=dates),
    }
    panel = to_wide_panel(histories, field="close")
    assert list(panel.columns) == ["A", "B"]
    assert panel.loc[dates[0], "A"] == 1.0


def test_to_wide_panel_skips_symbols_missing_the_field():
    dates = pd.date_range("2024-01-01", periods=2)
    histories = {"A": pd.DataFrame({"close": [1.0, 2.0]}, index=dates), "B": pd.DataFrame({"open": [1.0, 2.0]}, index=dates)}
    panel = to_wide_panel(histories, field="close")
    assert list(panel.columns) == ["A"]


# --- fetch_index_level ---

def test_fetch_index_level_uses_cache_when_present(tmp_path, monkeypatch):
    import data.prices as prices_module

    monkeypatch.setattr(prices_module, "INDEX_CACHE_DIR", tmp_path)
    cache_path = tmp_path / "^GSPC.parquet"
    dates = pd.date_range("2024-01-01", periods=2)
    pd.DataFrame({"close": [1.0, 2.0]}, index=dates).to_parquet(cache_path)

    def _should_not_be_called(*a, **k):
        raise AssertionError("yf.download should not be called when cache exists")
    monkeypatch.setattr(prices_module.yf, "download", _should_not_be_called)

    result = fetch_index_level("^GSPC")
    assert result["close"].tolist() == [1.0, 2.0]


def test_fetch_index_level_fetches_and_caches_on_miss(tmp_path, monkeypatch):
    import data.prices as prices_module

    monkeypatch.setattr(prices_module, "INDEX_CACHE_DIR", tmp_path)
    monkeypatch.setattr(prices_module.yf, "download", _fake_download_multi)

    result = fetch_index_level("^GSPC")
    assert "close" in result.columns
    assert (tmp_path / "^GSPC.parquet").exists()


def test_fetch_index_level_raises_on_empty_response(monkeypatch):
    import data.prices as prices_module
    monkeypatch.setattr(prices_module.yf, "download", _fake_download_empty)
    with pytest.raises(ValueError):
        fetch_index_level("^NOPE", force_refresh=True)


# --- fetch_symbol_history ---

def test_fetch_symbol_history_caches_a_miss_and_returns_none(tmp_path, monkeypatch):
    import data.prices as prices_module

    monkeypatch.setattr(prices_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(prices_module.yf, "download", _fake_download_empty)

    result = fetch_symbol_history("ZZZINVALID")
    assert result is None
    assert (tmp_path / "ZZZINVALID.parquet").exists()

    # re-running should hit the cached miss, not call yf.download again
    def _should_not_be_called(*a, **k):
        raise AssertionError("should use cached miss, not re-fetch")
    monkeypatch.setattr(prices_module.yf, "download", _should_not_be_called)
    assert fetch_symbol_history("ZZZINVALID") is None


def test_fetch_symbol_history_caches_and_returns_data(tmp_path, monkeypatch):
    import data.prices as prices_module

    monkeypatch.setattr(prices_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(prices_module.yf, "download", _fake_download_multi)

    result = fetch_symbol_history("AAPL")
    assert result is not None
    assert "close" in result.columns
    assert (tmp_path / "AAPL.parquet").exists()


# --- fetch_constituent_history ---

def test_fetch_constituent_history_batches_and_reports_missing(tmp_path, monkeypatch):
    import data.prices as prices_module

    monkeypatch.setattr(prices_module, "CACHE_DIR", tmp_path)

    def _partial_download(tickers, **kwargs):
        # simulate AAPL present, MISSING absent from the batch (and single-symbol retry) response entirely
        tickers = [tickers] if isinstance(tickers, str) else tickers
        present = [t for t in tickers if t != "MISSING"]
        if not present:
            return pd.DataFrame()
        return _fake_download_multi(present)

    monkeypatch.setattr(prices_module.yf, "download", _partial_download)

    histories, missing = fetch_constituent_history(["AAPL", "MISSING"])
    assert "AAPL" in histories
    assert "MISSING" in missing


def test_fetch_constituent_history_skips_already_cached_symbols(tmp_path, monkeypatch):
    import data.prices as prices_module

    monkeypatch.setattr(prices_module, "CACHE_DIR", tmp_path)
    dates = pd.date_range("2024-01-01", periods=2)
    pd.DataFrame({"close": [1.0, 2.0], "volume": [1, 2]}, index=dates).to_parquet(tmp_path / "AAPL.parquet")

    def _should_not_be_called(*a, **k):
        raise AssertionError("cached symbol should not trigger a fetch")
    monkeypatch.setattr(prices_module.yf, "download", _should_not_be_called)

    histories, missing = fetch_constituent_history(["AAPL"])
    assert histories["AAPL"]["close"].tolist() == [1.0, 2.0]
    assert missing == []
