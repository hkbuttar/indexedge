import numpy as np
import pandas as pd
import pytest

from replication.full_replication import compute_weights_by_date, rebalance_weights, simulate_cap_weighted_replication
from risk.tracking_error import annualized_tracking_error


def test_rebalance_weights_normalizes_over_available_members_only():
    members = {"A", "B", "C", "MISSING"}
    row = pd.Series({"A": 100.0, "B": 200.0, "C": 300.0, "OTHER": 999.0})
    weights = rebalance_weights(members, row)
    assert set(weights.index) == {"A", "B", "C"}
    assert weights["C"] == pytest.approx(0.5)
    assert weights.sum() == pytest.approx(1.0)


def test_a_nan_priced_symbol_does_not_poison_every_date():
    # Real bug class, caught on real data (see
    # tests/replication/test_sampling_evaluation.py's sibling test for the
    # case that actually failed in production): a plain `period_prices @
    # shares` matrix multiply, unlike pandas' `.mul(...).sum(axis=1)`
    # (skipna=True by default), lets ONE symbol's NaN price poison every
    # date's portfolio value, not just exclude that symbol's contribution.
    #
    # In `simulate_cap_weighted_replication` specifically, `rebalance_weights`
    # already filters candidates to a valid (non-NaN) market cap at the
    # rebalance date, and prices are globally ffilled before the loop, so
    # this exact poisoning can't arise from realistic (mutually consistent)
    # inputs -- this test deliberately passes an inconsistent `market_caps`
    # (claims B is valid) against `prices` (B is NaN throughout) specifically
    # to exercise the downstream sum step in isolation and lock in the fix
    # as defensive correctness, not to model a real pipeline state.
    dates = pd.bdate_range("2022-01-03", periods=10)
    prices = pd.DataFrame({
        "A": np.linspace(100, 110, 10), "B": [np.nan] * 10, "C": np.linspace(50, 55, 10),
    }, index=dates)
    market_caps = pd.DataFrame({"A": 1000.0, "B": 1000.0, "C": 1000.0}, index=dates)  # B claimed valid despite NaN price
    membership = pd.DataFrame({"rebalance_date": [dates[0]] * 3, "symbol": ["A", "B", "C"]})

    value, returns, coverage = simulate_cap_weighted_replication(prices, market_caps, membership)
    assert value.notna().all()
    assert len(returns) > 0
    assert coverage[0].weighted_members == 3  # all three pass the (inconsistent) market-cap filter


def test_full_replication_reproduces_benchmark_exactly_with_full_coverage():
    """The actual code-correctness check: build a synthetic 4-stock universe
    with 100% coverage and no data gaps, define the "index" as exactly the
    cap-weighted combination of those same 4 stocks, and confirm the
    simulator's tracking error against it is ~0 (floating point only).

    This is deliberately synthetic rather than real data: real full
    replication (see replication/run_full_replication.py) still shows ~1-2%
    tracking error against the real S&P 500 even after fixing the dual-class
    share-count bug, driven by disclosed real-world data gaps (imperfect
    free-float proxy, <100% historical price coverage for delisted names) --
    not a bug in the simulation mechanics themselves. This test isolates the
    mechanics from that real-world data noise.
    """
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=500)
    symbols = ["A", "B", "C", "D"]

    prices = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    prices.iloc[0] = [100.0, 50.0, 200.0, 10.0]
    for i in range(1, len(dates)):
        daily_returns = rng.normal(0.0003, 0.01, size=4)
        prices.iloc[i] = prices.iloc[i - 1].to_numpy() * (1 + daily_returns)

    shares = pd.DataFrame(1.0, index=dates, columns=symbols)  # constant share counts, no corporate actions
    market_caps = prices * shares

    quarterly = pd.date_range("2020-01-01", "2021-12-31", freq="QS")
    membership = pd.DataFrame([
        (date, symbol) for date in quarterly for symbol in symbols
    ], columns=["rebalance_date", "symbol"])

    value, returns, coverage = simulate_cap_weighted_replication(prices, market_caps, membership)
    assert all(c.coverage_fraction == 1.0 for c in coverage)

    # true cap-weighted benchmark: literally the same weighting scheme,
    # recomputed independently (not reusing simulate_cap_weighted_replication)
    total_cap = market_caps.sum(axis=1)
    weights = market_caps.div(total_cap, axis=0)
    benchmark_value = (weights.shift(1) * prices.pct_change()).sum(axis=1).add(1).cumprod()
    benchmark_returns = benchmark_value.pct_change().dropna()

    te = annualized_tracking_error(returns, benchmark_returns)
    assert te < 0.01  # not exactly 0: benchmark rebalances daily, simulator only quarterly


def test_full_replication_matches_static_cap_weighted_portfolio_when_no_rebalancing_needed():
    """Simpler exact-equality check: with only ONE rebalance date (buy and
    hold the whole period, no interim reweighting), the simulator's value
    path must exactly equal a hand-computed static buy-and-hold portfolio."""
    dates = pd.bdate_range("2022-01-01", periods=50)
    symbols = ["X", "Y"]
    prices = pd.DataFrame({
        "X": np.linspace(100, 120, 50),
        "Y": np.linspace(50, 45, 50),
    }, index=dates)
    shares = pd.DataFrame({"X": 10.0, "Y": 40.0}, index=dates)
    market_caps = prices * shares

    membership = pd.DataFrame({"rebalance_date": [dates[0]] * 2, "symbol": symbols})
    value, returns, coverage = simulate_cap_weighted_replication(prices, market_caps, membership)

    start_cap = market_caps.iloc[0]
    weights = start_cap / start_cap.sum()
    hand_shares = weights / prices.iloc[0]
    expected_value = prices.mul(hand_shares, axis=1).sum(axis=1)
    expected_value = expected_value / expected_value.iloc[0]

    pd.testing.assert_series_equal(value, expected_value, check_names=False)


def test_compute_weights_by_date_matches_rebalance_weights_per_date():
    dates = pd.bdate_range("2022-01-01", periods=10)
    prices = pd.DataFrame({"X": 100.0, "Y": 50.0}, index=dates)
    shares = pd.DataFrame({"X": 10.0, "Y": 40.0}, index=dates)
    market_caps = prices * shares

    rebalance_dates = [dates[0], dates[5]]
    membership = pd.DataFrame({
        "rebalance_date": [rebalance_dates[0], rebalance_dates[0], rebalance_dates[1]],
        "symbol": ["X", "Y", "X"],
    })

    weights_by_date = compute_weights_by_date(membership, market_caps)
    assert set(weights_by_date.keys()) == set(rebalance_dates)

    expected_first = rebalance_weights({"X", "Y"}, market_caps.loc[rebalance_dates[0]])
    pd.testing.assert_series_equal(weights_by_date[rebalance_dates[0]].sort_index(), expected_first.sort_index())

    # second rebalance only has X in membership -> weight should be 100% X
    assert weights_by_date[rebalance_dates[1]].to_dict() == {"X": 1.0}
