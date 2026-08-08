"""Multi-factor tilt: cap-weighted holdings tilted by an IC-weighted
composite of four factor z-scores (momentum, low-vol, value from
`factors.py`; quality from `quality.py`). This is a tilt, not a standalone
weighting scheme like `quality.py`'s quality-weighted variant -- weight is
market cap times a positive function of the composite score, so a name
never gets excluded purely for a mediocre score the way the other three
smart-beta variants can end up doing.

Combination methodology is adapted directly from alpha-signal-lab's
`factors/composite.py`: each factor's combination weight is its trailing
mean Information Coefficient (Spearman rank correlation between the
factor's cross-sectional score and forward returns) over a lookback window,
clipped at zero (a factor with no observed positive predictive power in
that window gets no active weight, not negative weight), refit
periodically and held constant between refits -- walk-forward, so the
weight used at any date was derived entirely from data available before
that date. Per-symbol, only factors with a non-NaN score that day are
blended, weights renormalized over the available subset (mirrors quality's
missing-fundamentals name having its weight absorbed by the other
factors rather than the composite going to zero for that name). If every
factor has non-positive trailing IC, weights fall back to equal weighting.

One adaptation from alpha-signal-lab's setup, disclosed: quality's score
doesn't vary by date (fundamentals are a snapshot -- see `quality.py`), so
it's passed as a plain Series rather than a date x symbol panel. The IC
loop below handles both uniformly: a constant score is legitimately
testable for trailing correlation against varying forward returns, it's
just answering "would today's quality snapshot, held constant, have been
predictive over this trailing window" -- consistent with, not a new
instance of, the already-disclosed fundamentals look-ahead limitation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CROSS_SECTION = 10
IC_HORIZON_DAYS = 21
IC_LOOKBACK_DAYS = 126
DEFAULT_TILT_STRENGTH = 0.5
ZSCORE_CLIP = 3.0


def forward_returns_panel(prices: pd.DataFrame, horizon: int = IC_HORIZON_DAYS) -> pd.DataFrame:
    return prices.shift(-horizon) / prices - 1


def _row_at(factor: pd.DataFrame | pd.Series, date: pd.Timestamp) -> pd.Series:
    return factor.loc[date] if isinstance(factor, pd.DataFrame) else factor


def information_coefficient_series(
    factor: pd.DataFrame | pd.Series, forward_returns: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.Series:
    ics = {}
    for date in dates:
        if date not in forward_returns.index:
            continue
        score = _row_at(factor, date)
        fwd = forward_returns.loc[date]
        valid = score.notna() & fwd.notna()
        if valid.sum() >= MIN_CROSS_SECTION:
            ics[date] = score[valid].astype(float).corr(fwd[valid].astype(float), method="spearman")
    return pd.Series(ics, dtype=float)


def trailing_ic_weights(
    factor_scores: dict[str, pd.DataFrame | pd.Series],
    forward_returns: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    refit_date: pd.Timestamp,
    lookback_days: int = IC_LOOKBACK_DAYS,
) -> dict[str, float]:
    pos = trading_dates.searchsorted(refit_date)
    window_dates = trading_dates[max(0, pos - lookback_days): pos]

    ics = {}
    for name, factor in factor_scores.items():
        ic_series = information_coefficient_series(factor, forward_returns, window_dates)
        ics[name] = max(0.0, float(ic_series.mean())) if len(ic_series) else 0.0

    total = sum(ics.values())
    if total <= 0:
        n = len(factor_scores)
        return dict.fromkeys(factor_scores, 1.0 / n)
    return {name: ic / total for name, ic in ics.items()}


def composite_score(factor_scores: dict[str, pd.DataFrame | pd.Series], weights: dict[str, float], date: pd.Timestamp) -> pd.Series:
    rows = {name: _row_at(factor, date) for name, factor in factor_scores.items()}
    df = pd.DataFrame(rows)
    available_weight = df.notna().mul(pd.Series(weights), axis=1)
    weight_sum = available_weight.sum(axis=1)
    normalized = available_weight.div(weight_sum.replace(0, np.nan), axis=0)
    return (df.fillna(0) * normalized).sum(axis=1, min_count=1)


def tilt_weights(
    members: set[str], composite_row: pd.Series, market_cap_row: pd.Series, tilt_strength: float = DEFAULT_TILT_STRENGTH
) -> pd.Series:
    available = [
        s for s in members
        if s in market_cap_row.index and pd.notna(market_cap_row[s]) and market_cap_row[s] > 0
        and s in composite_row.index and pd.notna(composite_row[s])
    ]
    if not available:
        return pd.Series(dtype=float)

    caps = market_cap_row[available].astype(float)
    scores = composite_row[available].clip(-ZSCORE_CLIP, ZSCORE_CLIP)
    tilted = caps * np.exp(tilt_strength * scores)
    total = tilted.sum()
    return tilted / total if total > 0 else pd.Series(dtype=float)
