"""Shared simulation loop for all four smart-beta variants: at each real
quarterly rebalance date, build equal-weight, min-vol, quality-weighted,
and multi-factor-tilt weights and hold to the next rebalance (buy-and-hold,
same mechanic as `replication.full_replication` and
`replication.sampling_evaluation`). Extracted out of
`run_smartbeta_comparison.py` so `regime/`'s per-regime performance
evaluation (Step 6) can reuse the exact same daily return series each
variant's own summary stats are computed from, rather than re-deriving a
second, potentially-inconsistent simulation.
"""

from __future__ import annotations

import pandas as pd

from replication.sampling_evaluation import simulate_holding_period
from smartbeta.equal_weight import equal_weights
from smartbeta.min_vol import solve_min_variance_weights
from smartbeta.multi_factor import composite_score, tilt_weights, trailing_ic_weights
from smartbeta.quality import quality_weights

MIN_VOL_LOOKBACK_DAYS = 252
MIN_VOL_MAX_WEIGHT = 0.05


def simulate_all_variants(
    prices: pd.DataFrame,
    market_caps: pd.DataFrame,
    membership: pd.DataFrame,
    quality_scores: pd.Series,
    factor_scores: dict[str, pd.DataFrame | pd.Series],
    fwd_returns: pd.DataFrame,
    min_vol_lookback: int = MIN_VOL_LOOKBACK_DAYS,
    min_vol_max_weight: float = MIN_VOL_MAX_WEIGHT,
) -> dict[str, pd.Series]:
    """Returns {strategy_name: daily_return_series} for all four variants,
    each series spanning whatever portion of the backtest window that
    variant had enough data to be constructed over (min-vol needs a full
    trailing lookback window before its first holding period)."""
    all_returns = prices.pct_change()
    dates = sorted(membership["rebalance_date"].unique())
    value_segments: dict[str, list[pd.Series]] = {name: [] for name in ["equal_weight", "min_vol", "quality", "multi_factor"]}

    for i, t in enumerate(dates[:-1]):
        pos = prices.index.searchsorted(t)
        period_start = prices.index[pos]
        period_end = dates[i + 1]
        members = set(membership.loc[membership["rebalance_date"] == t, "symbol"])
        price_row = prices.loc[period_start]
        cap_row = market_caps.loc[market_caps.index[market_caps.index.searchsorted(t)]]

        ew = equal_weights(members, price_row)

        if pos >= min_vol_lookback:
            trailing = all_returns.iloc[pos - min_vol_lookback: pos]
            candidates = [s for s in members if s in trailing.columns and trailing[s].notna().all()]
            mv = solve_min_variance_weights(trailing[candidates], max_weight=min_vol_max_weight) if candidates else pd.Series(dtype=float)
        else:
            mv = pd.Series(dtype=float)

        qw = quality_weights(members, quality_scores)

        ic_weights = trailing_ic_weights(factor_scores, fwd_returns, prices.index, t)
        composite = composite_score(factor_scores, ic_weights, period_start)
        mf = tilt_weights(members, composite, cap_row)

        for name, weights in [("equal_weight", ew), ("min_vol", mv), ("quality", qw), ("multi_factor", mf)]:
            if weights.empty:
                continue
            value = simulate_holding_period(prices, weights, period_start, period_end)
            if value_segments[name]:
                value = value * value_segments[name][-1].iloc[-1]
            value_segments[name].append(value)

    returns_by_strategy = {}
    for name, segments in value_segments.items():
        if not segments:
            continue
        full_value = pd.concat(segments).sort_index()
        full_value = full_value[~full_value.index.duplicated(keep="first")]
        returns_by_strategy[name] = full_value.pct_change().dropna()

    return returns_by_strategy
