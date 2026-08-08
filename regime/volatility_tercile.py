"""Calm/normal/volatile regime classification via rolling realized
volatility terciles.

Reused directly, unchanged, from riskdesk's `regime/volatility_tercile.py`
(which itself adapted execedge's original crypto-hourly methodology to
daily equity/crypto bars: `window=21` trading days ~= one month,
`periods_per_year=252`) -- direct methodological continuity across the
portfolio, per this project's plan. Same rolling-window log-return
volatility estimator, same static (computed-once, not re-fit per bar)
tercile split via `Series.quantile(1/3)` / `Series.quantile(2/3)` over the
whole available history, same label rule.

One choice made here rather than inherited: the reference series is
`^GSPC`, the real S&P 500 index level itself (already cached by
`data/prices.py`), not SPY (riskdesk's proxy ETF) or this project's own
strategy returns. Using the actual index this whole project replicates and
tilts away from is more direct than a proxy fund for an index-replication
project specifically, and keeps with riskdesk's own stated principle: a
regime label should describe the market condition a strategy is exposed to,
not be circularly defined by that strategy's own returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIME_LABELS = ("calm", "normal", "volatile")

DEFAULT_WINDOW_DAYS = 21
DEFAULT_PERIODS_PER_YEAR = 252


def rolling_realized_vol(
    close: pd.Series, window: int = DEFAULT_WINDOW_DAYS, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    rolling_std = log_returns.rolling(window=window, min_periods=window).std()
    return rolling_std * np.sqrt(periods_per_year)


@dataclass
class TercileResult:
    labels: pd.Series
    thresholds: dict[str, float]
    vol: pd.Series

    def value_counts(self) -> pd.Series:
        return self.labels.value_counts()

    def current(self) -> str | None:
        valid = self.labels.dropna()
        return valid.iloc[-1] if len(valid) else None


def classify_regimes(vol: pd.Series, low_q: float = 1 / 3, high_q: float = 2 / 3) -> TercileResult:
    valid = vol.dropna()
    low_thresh = float(valid.quantile(low_q))
    high_thresh = float(valid.quantile(high_q))

    def label(v: float) -> str | None:
        if pd.isna(v):
            return None
        if v <= low_thresh:
            return "calm"
        if v >= high_thresh:
            return "volatile"
        return "normal"

    labels = vol.apply(label)
    thresholds = {
        "low_quantile": low_q, "high_quantile": high_q,
        "low_threshold": low_thresh, "high_threshold": high_thresh,
    }
    return TercileResult(labels=labels, thresholds=thresholds, vol=vol)
