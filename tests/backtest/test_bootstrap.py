import numpy as np
import pandas as pd
import pytest

from backtest.bootstrap import (
    BootstrapResult,
    block_bootstrap_resample,
    block_bootstrap_resample_paired,
    bootstrap_backtest_metrics,
)


def _ar1_series(seed: int, n: int = 2000, phi: float = 0.7) -> np.ndarray:
    """Synthetic series with known lag-1 autocorrelation ~phi -- the
    standard way to test a block bootstrap actually preserves
    autocorrelation structure rather than resampling i.i.d. Ported from
    pairtrade-lab-1's own bootstrap test, same construction."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = phi * values[t - 1] + rng.normal(0, 1)
    return values


def _lag1_autocorr(values: np.ndarray) -> float:
    return float(pd.Series(values).autocorr(lag=1))


def test_block_bootstrap_preserves_known_autocorrelation():
    series = _ar1_series(seed=0, phi=0.7)
    true_autocorr = _lag1_autocorr(series)

    resampled = block_bootstrap_resample(series, block_length=20, n_resamples=200, seed=1)
    block_autocorrs = [_lag1_autocorr(row) for row in resampled]

    assert np.mean(block_autocorrs) == pytest.approx(true_autocorr, abs=0.15)


def test_block_bootstrap_preserves_autocorrelation_better_than_iid():
    series = _ar1_series(seed=2, phi=0.7)
    true_autocorr = _lag1_autocorr(series)

    block_resampled = block_bootstrap_resample(series, block_length=20, n_resamples=200, seed=3)
    block_autocorrs = [_lag1_autocorr(row) for row in block_resampled]

    iid_rng = np.random.default_rng(4)
    n = len(series)
    iid_autocorrs = [_lag1_autocorr(series[iid_rng.integers(0, n, size=n)]) for _ in range(200)]

    block_error = abs(np.mean(block_autocorrs) - true_autocorr)
    iid_error = abs(np.mean(iid_autocorrs) - true_autocorr)
    assert block_error < iid_error
    assert np.mean(iid_autocorrs) == pytest.approx(0.0, abs=0.1)


def test_block_bootstrap_resample_shape():
    series = np.arange(100, dtype=float)
    resampled = block_bootstrap_resample(series, block_length=10, n_resamples=50, seed=0)
    assert resampled.shape == (50, 100)


def test_paired_resample_uses_identical_indices_across_series():
    a = np.arange(100, dtype=float)
    b = np.arange(100, dtype=float) * 10  # b[i] = 10 * a[i] always, for any index

    resampled = block_bootstrap_resample_paired({"a": a, "b": b}, block_length=10, n_resamples=20, seed=0)
    # if the same block indices were used for both series, this relationship must hold everywhere
    assert np.allclose(resampled["b"], resampled["a"] * 10)


def test_paired_resample_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        block_bootstrap_resample_paired({"a": np.zeros(10), "b": np.zeros(20)}, block_length=5, n_resamples=10)


def test_bootstrap_backtest_metrics_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 300))
    results = bootstrap_backtest_metrics(returns, block_length=20, n_resamples=200, seed=1)

    for name, result in results.items():
        assert isinstance(result, BootstrapResult)
        if np.isnan(result.point_estimate):
            continue
        # CI should bracket the point estimate for a reasonably well-behaved metric/series
        assert result.ci_low <= result.point_estimate <= result.ci_high or result.ci_width < 1e-6


def test_bootstrap_backtest_metrics_includes_tracking_error_with_benchmark():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=300)
    returns = pd.Series(rng.normal(0.0005, 0.01, 300), index=dates)
    benchmark = pd.Series(rng.normal(0.0004, 0.009, 300), index=dates)

    results = bootstrap_backtest_metrics(returns, benchmark_returns=benchmark, block_length=20, n_resamples=100, seed=2)
    assert "tracking_error" in results
    assert results["tracking_error"].point_estimate >= 0
    assert results["tracking_error"].ci_low >= 0  # tracking error is non-negative by construction


def test_bootstrap_backtest_metrics_raises_below_block_length():
    returns = pd.Series([0.01] * 5)
    with pytest.raises(ValueError):
        bootstrap_backtest_metrics(returns, block_length=20)
