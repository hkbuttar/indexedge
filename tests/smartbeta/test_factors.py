import numpy as np
import pandas as pd
import pytest

from smartbeta.factors import low_vol_panel, momentum_panel, value_panel


def test_momentum_panel_ranks_the_faster_grower_higher():
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    fast = 100 * (1.001 ** np.arange(300))
    slow = 100 * (1.0001 ** np.arange(300))
    prices = pd.DataFrame({"FAST": fast, "SLOW": slow}, index=dates)

    panel = momentum_panel(prices)
    last_valid = panel.dropna().iloc[-1]
    assert last_valid["FAST"] > last_valid["SLOW"]
    # cross-sectional z-score of a 2-column row: symmetric around 0
    assert last_valid.sum() == pytest.approx(0.0, abs=1e-8)


def test_low_vol_panel_ranks_the_calmer_asset_higher():
    dates = pd.date_range("2020-01-01", periods=150, freq="B")
    rng = np.random.default_rng(0)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.002, 150))
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.03, 150))
    prices = pd.DataFrame({"CALM": calm, "WILD": wild}, index=dates)

    panel = low_vol_panel(prices, window=60)
    last_valid = panel.dropna().iloc[-1]
    assert last_valid["CALM"] > last_valid["WILD"]


def test_value_panel_ranks_cheaper_earnings_yield_higher():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"CHEAP": [10.0] * 10, "EXPENSIVE": [100.0] * 10}, index=dates)
    eps = pd.Series({"CHEAP": 1.0, "EXPENSIVE": 1.0})  # same EPS, very different price -> different yield

    panel = value_panel(prices, eps)
    assert (panel["CHEAP"] > panel["EXPENSIVE"]).all()
