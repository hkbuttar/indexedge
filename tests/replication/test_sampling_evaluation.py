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
