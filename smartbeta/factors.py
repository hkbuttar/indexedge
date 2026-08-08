"""Three of the four factors combined in `multi_factor.py`'s tilt, computed
as full date x symbol panels directly from price history so they're
genuinely point-in-time (no look-ahead: the value on date t only uses
prices up to and including t). The fourth factor, quality, is
`smartbeta.quality.compute_quality_scores` -- a single current-day snapshot,
not a panel, because fundamentals data has no historical time series (see
that module's docstring).

- Momentum: classic 12-1 month momentum (t-252 to t-21 trading days),
  skipping the most recent month to avoid short-term reversal contaminating
  the signal -- the standard Jegadeesh-Titman definition, not an invented
  variant.
- Low-volatility: negative trailing realized volatility (`window` days,
  annualized), so a higher score means lower risk -- sign-flipped here
  (before cross-sectional z-scoring) so every factor in the composite
  points the same direction ("higher score = more desirable").
- Value: earnings yield, trailing EPS / price. EPS is the same disclosed
  current-snapshot limitation as quality, but price varies daily, so this
  factor is *partially* point-in-time -- more so than quality, which has no
  price-varying component at all.

All three are returned already cross-sectionally z-scored per date (mean 0,
std 1 across symbols on each date), ready to feed into `multi_factor.py`'s
IC-weighted combination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_LOOKBACK_DAYS = 252
MOMENTUM_SKIP_DAYS = 21
LOW_VOL_WINDOW_DAYS = 90
PERIODS_PER_YEAR = 252


def _cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    mean = panel.mean(axis=1)
    std = panel.std(axis=1)
    return panel.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


def momentum_panel(prices: pd.DataFrame) -> pd.DataFrame:
    raw = prices.shift(MOMENTUM_SKIP_DAYS) / prices.shift(MOMENTUM_LOOKBACK_DAYS) - 1
    return _cross_sectional_zscore(raw)


def low_vol_panel(prices: pd.DataFrame, window: int = LOW_VOL_WINDOW_DAYS) -> pd.DataFrame:
    returns = prices.pct_change()
    realized_vol = returns.rolling(window, min_periods=window).std() * np.sqrt(PERIODS_PER_YEAR)
    return _cross_sectional_zscore(-realized_vol)


def value_panel(prices: pd.DataFrame, trailing_eps: pd.Series) -> pd.DataFrame:
    eps = trailing_eps.reindex(prices.columns)
    earnings_yield = prices.rtruediv(eps, axis=1)
    return _cross_sectional_zscore(earnings_yield)
