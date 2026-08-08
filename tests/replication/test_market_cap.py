import pandas as pd

from replication.market_cap import build_market_cap_panel, coverage_report


def test_dual_class_anchor_rescales_combined_share_series():
    # GOOGL and GOOG both report Alphabet's combined ~12.2B share count from
    # get_shares_full (the real, diagnosed bug) -- without the anchor
    # correction this double-counts Alphabet's true market cap.
    dates = pd.date_range("2026-01-01", periods=3)
    prices = pd.DataFrame({"GOOGL": [100.0, 101.0, 102.0], "GOOG": [99.0, 100.0, 101.0]}, index=dates)
    combined_shares = pd.Series([12_229_934_831] * 3, index=dates)
    shares_by_symbol = {"GOOGL": combined_shares, "GOOG": combined_shares}

    uncorrected = build_market_cap_panel(prices, shares_by_symbol)
    assert uncorrected["GOOGL"].iloc[-1] == prices["GOOGL"].iloc[-1] * combined_shares.iloc[-1]

    anchors = {"GOOGL": 5_867_155_790, "GOOG": 5_527_000_000}
    corrected = build_market_cap_panel(prices, shares_by_symbol, class_specific_anchors=anchors)

    assert corrected["GOOGL"].iloc[-1] == prices["GOOGL"].iloc[-1] * anchors["GOOGL"]
    assert corrected["GOOG"].iloc[-1] == prices["GOOG"].iloc[-1] * anchors["GOOG"]
    # combined weight roughly halves once each class uses its own count
    combined_uncorrected = uncorrected["GOOGL"].iloc[-1] + uncorrected["GOOG"].iloc[-1]
    combined_corrected = corrected["GOOGL"].iloc[-1] + corrected["GOOG"].iloc[-1]
    assert combined_corrected < combined_uncorrected * 0.6


def test_anchor_rescale_is_noop_when_anchor_matches_latest_raw_value():
    dates = pd.date_range("2026-01-01", periods=3)
    prices = pd.DataFrame({"AAPL": [10.0, 11.0, 12.0]}, index=dates)
    shares = pd.Series([100.0, 100.0, 100.0], index=dates)

    no_anchor = build_market_cap_panel(prices, {"AAPL": shares})
    with_matching_anchor = build_market_cap_panel(prices, {"AAPL": shares}, class_specific_anchors={"AAPL": 100.0})

    pd.testing.assert_series_equal(no_anchor["AAPL"], with_matching_anchor["AAPL"])


def test_symbol_with_no_shares_data_excluded_and_reported():
    dates = pd.date_range("2026-01-01", periods=2)
    prices = pd.DataFrame({"AAPL": [10.0, 11.0], "NODATA": [5.0, 5.5]}, index=dates)
    caps = build_market_cap_panel(prices, {"AAPL": pd.Series([100.0, 100.0], index=dates)})

    assert "NODATA" not in caps.columns
    report = coverage_report(prices, caps)
    assert report["missing_shares_data"] == ["NODATA"]
    assert report["coverage_fraction"] == 0.5
