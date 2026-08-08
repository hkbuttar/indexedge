import numpy as np
import pandas as pd

from regime.volatility_tercile import classify_regimes, rolling_realized_vol


def test_rolling_realized_vol_is_nan_during_warmup():
    dates = pd.date_range("2024-01-01", periods=30)
    close = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 30)), index=dates)
    vol = rolling_realized_vol(close, window=21)
    assert vol.iloc[:21].isna().all()
    assert vol.iloc[21:].notna().all()


def test_rolling_realized_vol_scales_with_return_std():
    dates = pd.date_range("2024-01-01", periods=100)
    rng = np.random.default_rng(1)
    calm_returns = rng.normal(0, 0.005, 100)
    wild_returns = rng.normal(0, 0.03, 100)
    calm_close = pd.Series(100 * np.cumprod(1 + calm_returns), index=dates)
    wild_close = pd.Series(100 * np.cumprod(1 + wild_returns), index=dates)

    calm_vol = rolling_realized_vol(calm_close, window=21).dropna()
    wild_vol = rolling_realized_vol(wild_close, window=21).dropna()
    assert wild_vol.mean() > calm_vol.mean()


def test_classify_regimes_labels_by_tercile():
    # construct a vol series with exactly known terciles: 30 values, 10 in each third
    values = list(range(1, 11)) + list(range(11, 21)) + list(range(21, 31))
    vol = pd.Series(values, index=pd.date_range("2024-01-01", periods=30))

    result = classify_regimes(vol)
    counts = result.value_counts()
    assert counts["calm"] == 10
    assert counts["volatile"] == 10
    assert counts["normal"] == 10
    assert result.labels.iloc[0] == "calm"
    assert result.labels.iloc[-1] == "volatile"


def test_classify_regimes_preserves_nan_from_warmup():
    vol = pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0])
    result = classify_regimes(vol)
    assert pd.isna(result.labels.iloc[0])
    assert pd.isna(result.labels.iloc[1])


def test_current_returns_last_valid_label():
    vol = pd.Series([1.0, 2.0, 3.0, np.nan])
    result = classify_regimes(vol)
    assert result.current() in ("calm", "normal", "volatile")
