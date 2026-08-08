"""Runs the Step 4 smart-beta comparison end-to-end against real cached
data: at each real quarterly rebalance date, build all four variants
(equal-weight, min-vol, quality-weighted, multi-factor tilt), hold to the
next rebalance (same buy-and-hold mechanic as `replication.full_replication`
and `replication.sampling_evaluation`), and report each variant's
annualized return, volatility, and tracking error against Step 2's full
replication over the whole backtest window.

Min-vol and multi-factor's momentum/low-vol components are fit walk-forward
(trailing lookback only, as of each rebalance date -- no look-ahead).
Quality and multi-factor's value/quality components use current-snapshot
fundamentals at every historical date, per the disclosed limitation in
`data/fundamentals.py` and `smartbeta/quality.py`.

Usage: `python -m smartbeta.run_smartbeta_comparison`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.fundamentals import fetch_universe_fundamentals
from data.prices import to_wide_panel
from data.shares_outstanding import fetch_universe_shares_anchors, fetch_universe_shares_outstanding
from data.wikipedia_constituents import fetch_constituents_and_changes
from replication.full_replication import simulate_cap_weighted_replication
from replication.market_cap import build_market_cap_panel
from replication.point_in_time import build_membership_history, quarterly_rebalance_dates
from replication.sampling_evaluation import simulate_holding_period
from risk.tracking_error import summarize_tracking
from smartbeta.equal_weight import equal_weights
from smartbeta.factors import low_vol_panel, momentum_panel, value_panel
from smartbeta.min_vol import solve_min_variance_weights
from smartbeta.multi_factor import composite_score, forward_returns_panel, tilt_weights, trailing_ic_weights
from smartbeta.quality import compute_quality_scores, quality_weights

PRICE_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "prices"
MIN_VOL_LOOKBACK_DAYS = 252
MIN_VOL_MAX_WEIGHT = 0.05


def _load_cached_prices() -> pd.DataFrame:
    histories = {}
    for path in PRICE_CACHE_DIR.glob("*.parquet"):
        df = pd.read_parquet(path)
        if not df.empty:
            histories[path.stem] = df
    return to_wide_panel(histories, field="close")


def main(start: str, end: str) -> pd.DataFrame:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    prices = _load_cached_prices()
    all_returns = prices.pct_change()

    current, changes = fetch_constituents_and_changes()
    rebalance_dates = quarterly_rebalance_dates(start_ts, end_ts)
    membership = build_membership_history(rebalance_dates, current, changes)

    symbols = sorted(prices.columns)
    shares_map, _ = fetch_universe_shares_outstanding(symbols)
    anchors = fetch_universe_shares_anchors(symbols)
    market_caps = build_market_cap_panel(prices, shares_map, class_specific_anchors=anchors)
    benchmark_value, _, _ = simulate_cap_weighted_replication(prices, market_caps, membership)

    fundamentals, _ = fetch_universe_fundamentals(sorted(current["yfinance_symbol"].unique()))
    quality_scores = compute_quality_scores(fundamentals)
    trailing_eps = fundamentals["trailingEps"]

    mom_panel = momentum_panel(prices)
    vol_panel = low_vol_panel(prices)
    val_panel = value_panel(prices, trailing_eps)
    fwd_returns = forward_returns_panel(prices)
    factor_scores = {"momentum": mom_panel, "low_vol": vol_panel, "value": val_panel, "quality": quality_scores}

    dates = sorted(membership["rebalance_date"].unique())
    value_segments = {name: [] for name in ["equal_weight", "min_vol", "quality", "multi_factor"]}

    for i, t in enumerate(dates[:-1]):
        pos = prices.index.searchsorted(t)
        period_start = prices.index[pos]
        period_end = dates[i + 1]
        members = set(membership.loc[membership["rebalance_date"] == t, "symbol"])
        price_row = prices.loc[period_start]
        cap_row = market_caps.loc[market_caps.index[market_caps.index.searchsorted(t)]]

        ew = equal_weights(members, price_row)

        if pos >= MIN_VOL_LOOKBACK_DAYS:
            trailing = all_returns.iloc[pos - MIN_VOL_LOOKBACK_DAYS: pos]
            candidates = [s for s in members if s in trailing.columns and trailing[s].notna().all()]
            mv = solve_min_variance_weights(trailing[candidates], max_weight=MIN_VOL_MAX_WEIGHT) if candidates else pd.Series(dtype=float)
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

    benchmark_returns = benchmark_value.pct_change().dropna()
    records = []
    for name, segments in value_segments.items():
        if not segments:
            continue
        full_value = pd.concat(segments).sort_index()
        full_value = full_value[~full_value.index.duplicated(keep="first")]
        returns = full_value.pct_change().dropna()
        summary = summarize_tracking(returns, benchmark_returns)
        annualized_return = float((1 + returns.mean()) ** 252 - 1)
        annualized_vol = float(returns.std(ddof=1) * np.sqrt(252))
        records.append({
            "strategy": name, "annualized_return": annualized_return, "annualized_vol": annualized_vol,
            "tracking_error_vs_full_replication": summary.tracking_error_annualized,
            "correlation": summary.correlation, "n_days": len(returns),
        })

    results = pd.DataFrame(records)
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main("2016-01-01", "2026-08-07")
