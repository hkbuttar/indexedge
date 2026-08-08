import numpy as np
import pandas as pd
import pytest

from smartbeta.backtest import simulate_all_variants, simulate_all_variants_with_weights


def _synthetic_inputs(n_symbols=8, n_days=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    symbols = [f"S{i}" for i in range(n_symbols)]

    prices = pd.DataFrame(100.0, index=dates, columns=symbols)
    for i in range(1, n_days):
        prices.iloc[i] = prices.iloc[i - 1] * (1 + rng.normal(0.0003, 0.01, n_symbols))
    market_caps = prices * 1_000_000  # constant share counts

    rebalance_dates = [dates[0], dates[150], dates[300]]
    membership = pd.DataFrame([(d, s) for d in rebalance_dates for s in symbols], columns=["rebalance_date", "symbol"])

    quality_scores = pd.Series(rng.normal(0, 1, n_symbols), index=symbols)
    momentum_panel = pd.DataFrame(rng.normal(0, 1, (n_days, n_symbols)), index=dates, columns=symbols)
    factor_scores = {"momentum": momentum_panel, "quality": quality_scores}
    fwd_returns = prices.shift(-21) / prices - 1

    return prices, market_caps, membership, quality_scores, factor_scores, fwd_returns


def test_simulate_all_variants_returns_nonempty_series_for_each_variant():
    prices, market_caps, membership, quality_scores, factor_scores, fwd_returns = _synthetic_inputs()
    # short lookback so min_vol actually gets a chance to fire within this synthetic window
    returns_by_strategy = simulate_all_variants(
        prices, market_caps, membership, quality_scores, factor_scores, fwd_returns,
        min_vol_lookback=100, min_vol_max_weight=0.25,  # small synthetic universe -> 5% default cap is infeasible
    )
    assert set(returns_by_strategy.keys()) >= {"equal_weight", "quality", "multi_factor"}
    for name, returns in returns_by_strategy.items():
        assert len(returns) > 0, name
        assert returns.notna().all()


def test_min_vol_absent_before_lookback_window_satisfied():
    prices, market_caps, membership, quality_scores, factor_scores, fwd_returns = _synthetic_inputs()
    # lookback longer than the whole series -> min_vol should never fire
    returns_by_strategy = simulate_all_variants(
        prices, market_caps, membership, quality_scores, factor_scores, fwd_returns, min_vol_lookback=10_000
    )
    assert "min_vol" not in returns_by_strategy


def test_simulate_all_variants_with_weights_matches_returns_from_plain_variant():
    prices, market_caps, membership, quality_scores, factor_scores, fwd_returns = _synthetic_inputs()
    returns_plain = simulate_all_variants(
        prices, market_caps, membership, quality_scores, factor_scores, fwd_returns, min_vol_lookback=100, min_vol_max_weight=0.25
    )
    returns_with_weights, weights_by_date = simulate_all_variants_with_weights(
        prices, market_caps, membership, quality_scores, factor_scores, fwd_returns, min_vol_lookback=100, min_vol_max_weight=0.25
    )

    for name in returns_plain:
        pd.testing.assert_series_equal(returns_plain[name], returns_with_weights[name])

    assert set(weights_by_date.keys()) == {"equal_weight", "min_vol", "quality", "multi_factor"}
    for name, by_date in weights_by_date.items():
        for t, weights in by_date.items():
            assert weights.sum() == pytest.approx(1.0), f"{name} at {t}"


def test_weights_by_date_keyed_by_actual_rebalance_dates_used():
    prices, market_caps, membership, quality_scores, factor_scores, fwd_returns = _synthetic_inputs()
    _, weights_by_date = simulate_all_variants_with_weights(
        prices, market_caps, membership, quality_scores, factor_scores, fwd_returns, min_vol_lookback=100, min_vol_max_weight=0.25
    )
    rebalance_dates = sorted(membership["rebalance_date"].unique())
    # equal_weight has no data requirement -> should hold at every rebalance except the last (no forward period)
    assert set(weights_by_date["equal_weight"].keys()) == set(rebalance_dates[:-1])
