import pandas as pd
import pytest

from replication.point_in_time import (
    build_membership_history,
    coverage_notes,
    quarterly_rebalance_dates,
    reconstruct_membership,
)


@pytest.fixture
def current():
    return pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "TSLA"],
        "date_added": pd.to_datetime(["1982-11-30", "1994-06-01", "2020-12-21"]),
    })


@pytest.fixture
def changes():
    # TSLA joined 2020-12-21, replacing AAPL... (synthetic; real event added TSLA
    # and removed Apartment Investment & Management, kept minimal here on purpose)
    return pd.DataFrame({
        "effective_date": pd.to_datetime(["2020-12-21", "2010-01-01"]),
        "added_ticker": ["TSLA", "MSFT"],
        "removed_ticker": ["AIV", pd.NA],
    })


def test_current_date_returns_full_current_set(current, changes):
    membership = reconstruct_membership(pd.Timestamp("2026-01-01"), current, changes)
    assert membership == {"AAPL", "MSFT", "TSLA"}


def test_rolling_back_past_addition_removes_ticker(current, changes):
    membership = reconstruct_membership(pd.Timestamp("2020-01-01"), current, changes)
    assert "TSLA" not in membership
    assert "AIV" in membership
    assert membership == {"AAPL", "MSFT", "AIV"}


def test_rolling_back_before_changes_history_start(current, changes):
    membership = reconstruct_membership(pd.Timestamp("1976-01-01"), current, changes)
    assert "TSLA" not in membership
    assert "AIV" in membership


def test_same_ticker_added_then_removed_before_asof_nets_absent():
    # MRNA added 2005-01-01, then removed 2015-01-01 (e.g. delisting).
    # as_of predates the addition entirely -> must NOT be present, regardless
    # of processing order, which is exactly what descending-date order guarantees.
    current = pd.DataFrame({"symbol": ["AAPL"], "date_added": pd.to_datetime(["1982-11-30"])})
    changes = pd.DataFrame({
        "effective_date": pd.to_datetime(["2015-01-01", "2005-01-01"]),
        "added_ticker": [pd.NA, "MRNA"],
        "removed_ticker": ["MRNA", pd.NA],
    })
    membership = reconstruct_membership(pd.Timestamp("2000-01-01"), current, changes)
    assert "MRNA" not in membership


def test_build_membership_history_long_format(current, changes):
    dates = [pd.Timestamp("2019-01-01"), pd.Timestamp("2021-01-01")]
    history = build_membership_history(dates, current, changes)
    assert set(history.columns) == {"rebalance_date", "symbol"}
    at_2019 = set(history[history["rebalance_date"] == dates[0]]["symbol"])
    at_2021 = set(history[history["rebalance_date"] == dates[1]]["symbol"])
    assert "TSLA" not in at_2019
    assert "TSLA" in at_2021


def test_quarterly_rebalance_dates_are_third_fridays():
    dates = quarterly_rebalance_dates(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
    assert len(dates) == 4
    for d in dates:
        assert d.weekday() == 4  # Friday
        assert d.month in (3, 6, 9, 12)
        assert 15 <= d.day <= 21  # third Friday always falls in this range


def test_coverage_notes_flags_pre_1976_additions(current, changes):
    notes = coverage_notes(current, changes)
    assert any("1976" in note for note in notes)
    assert len(notes) >= 2
