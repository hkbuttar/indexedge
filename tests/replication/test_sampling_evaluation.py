import numpy as np
import pandas as pd

from replication.sampling_evaluation import evaluate_sampling_methods, simulate_holding_period


def _synthetic_universe(n_symbols=8, n_days=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    symbols = [f"S{i}" for i in range(n_symbols)]
    prices = pd.DataFrame(100.0, index=dates, columns=symbols)
    for i in range(1, n_days):
        prices.iloc[i] = prices.iloc[i - 1] * (1 + rng.normal(0.0002, 0.01, n_symbols))
    shares = pd.DataFrame(1.0, index=dates, columns=symbols)
    return prices, prices * shares


def test_walk_forward_has_no_lookahead():
    """Perturbing prices strictly AFTER a rebalance date must not change the
    weights fitted or reported for that rebalance date -- if it did, the
    trailing-fit window would be leaking future data. This is the actual
    correctness property the walk-forward split exists to guarantee."""
    prices, market_caps = _synthetic_universe()
    dates = sorted(prices.index)
    rebalance_dates = [dates[300], dates[350]]
    membership = pd.DataFrame([
        (d, s) for d in rebalance_dates for s in prices.columns
    ], columns=["rebalance_date", "symbol"])
    sector_by_symbol = {s: "Sector" for s in prices.columns}
    benchmark_value = market_caps.sum(axis=1)
    benchmark_value = benchmark_value / benchmark_value.iloc[0]

    results_a = evaluate_sampling_methods(
        prices, market_caps, membership, benchmark_value, sector_by_symbol, target_counts=[4]
    )

    perturbed_prices = prices.copy()
    cutoff = rebalance_dates[0]
    future_mask = perturbed_prices.index > cutoff
    perturbed_prices.loc[future_mask] *= 5.0  # wildly change everything after the first rebalance
    perturbed_caps = perturbed_prices * (market_caps / prices)
    perturbed_benchmark = perturbed_caps.sum(axis=1)
    perturbed_benchmark = perturbed_benchmark / perturbed_benchmark.iloc[0]

    results_b = evaluate_sampling_methods(
        perturbed_prices, perturbed_caps, membership, perturbed_benchmark, sector_by_symbol, target_counts=[4]
    )

    first_a = results_a[results_a.rebalance_date == cutoff].set_index("method")["actual_n"]
    first_b = results_b[results_b.rebalance_date == cutoff].set_index("method")["actual_n"]
    # weight *selection* (name count chosen) at the first rebalance must be identical:
    # it can only have been influenced by data up to and including that date.
    pd.testing.assert_series_equal(first_a.sort_index(), first_b.sort_index())


def test_simulate_holding_period_normalizes_to_one_at_start():
    dates = pd.date_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"A": np.linspace(10, 20, 10), "B": np.linspace(5, 4, 10)}, index=dates)
    weights = pd.Series({"A": 0.5, "B": 0.5})
    value = simulate_holding_period(prices, weights, dates[0], dates[-1])
    assert value.iloc[0] == 1.0
    assert len(value) == 10


def test_simulate_holding_period_forward_fills_gaps_like_pandas_ffill():
    # Numpy-integer-indexed implementation (see its own docstring for why:
    # a real production memory-fragmentation fix), verified here against
    # hand-computed pandas .loc[...].ffill() on data with actual NaN gaps in
    # both columns, since the other tests in this file have no gaps to
    # exercise this path at all.
    dates = pd.date_range("2024-01-01", periods=10)
    prices = pd.DataFrame({
        "A": [10.0, 11.0, np.nan, np.nan, 13.0, 14.0, 15.0, np.nan, 17.0, 18.0],
        "B": [5.0, np.nan, 5.5, 6.0, np.nan, np.nan, 6.5, 7.0, 7.5, 8.0],
    }, index=dates)
    weights = pd.Series({"A": 0.5, "B": 0.5})

    result = simulate_holding_period(prices, weights, dates[0], dates[-1])

    price_at_start = prices.loc[dates[0], weights.index]
    shares = weights / price_at_start
    period_prices = prices.loc[dates[0]:dates[-1], weights.index].ffill()
    values = period_prices.mul(shares, axis=1).sum(axis=1)
    expected = values / values.iloc[0]

    pd.testing.assert_series_equal(result, expected, check_names=False, check_freq=False)


def test_simulate_holding_period_symbol_missing_at_period_start_does_not_poison_others():
    # The actual production failure this reproduces: quality-weighted's real
    # 345-symbol weight set included at least one symbol with no valid price
    # AT the rebalance date itself (price_at_start -> NaN -> shares -> NaN
    # for that symbol only). A plain `period_prices @ shares` dot product
    # let that single NaN propagate through every date's sum, turning
    # quality's ENTIRE return series into 0 real observations after
    # .dropna() -- not caught by the other tests here since none of them
    # have a NaN exactly at period_start. B is NaN throughout below,
    # including at dates[0]; A alone must still produce a real value series.
    dates = pd.date_range("2024-01-01", periods=10)
    prices = pd.DataFrame({
        "A": np.linspace(10, 20, 10),
        "B": [np.nan] * 10,
    }, index=dates)
    weights = pd.Series({"A": 0.6, "B": 0.4})

    result = simulate_holding_period(prices, weights, dates[0], dates[-1])

    assert result.notna().all()
    assert len(result) == 10
    # matches a hand-computed A-only portfolio (B's NaN contribution excluded,
    # same as pandas' skipna=True .sum(axis=1) would give)
    a_only_shares = weights["A"] / prices["A"].iloc[0]
    expected_values = prices["A"].to_numpy() * a_only_shares
    expected = expected_values / expected_values[0]
    assert np.allclose(result.to_numpy(), expected)
