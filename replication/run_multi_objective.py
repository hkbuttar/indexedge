"""Runs the Step 5 multi-objective portfolio construction against real
cached data: at the most recent real quarterly rebalance date, trace the
tracking-error-vs-turnover Pareto frontier at a few fixed factor-exposure
targets, starting from the prior rebalance's realized multi-factor tilt
weights (so turnover is measured against an actual prior holding).

Usage: `python -m replication.run_multi_objective`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.fundamentals import fetch_universe_fundamentals
from data.prices import to_wide_panel
from data.shares_outstanding import fetch_universe_shares_anchors, fetch_universe_shares_outstanding
from data.wikipedia_constituents import fetch_constituents_and_changes
from replication.market_cap import build_market_cap_panel
from replication.multi_objective import trace_pareto_frontier
from replication.point_in_time import build_membership_history, quarterly_rebalance_dates
from smartbeta.factors import low_vol_panel, momentum_panel, value_panel
from smartbeta.multi_factor import composite_score, forward_returns_panel, tilt_weights, trailing_ic_weights
from smartbeta.quality import compute_quality_scores

PRICE_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "prices"
LOOKBACK_DAYS = 252
TURNOVER_BUDGETS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


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

    fundamentals, _ = fetch_universe_fundamentals(sorted(current["yfinance_symbol"].unique()))
    quality_scores = compute_quality_scores(fundamentals)
    factor_scores = {
        "momentum": momentum_panel(prices), "low_vol": low_vol_panel(prices),
        "value": value_panel(prices, fundamentals["trailingEps"]), "quality": quality_scores,
    }
    fwd_returns = forward_returns_panel(prices)

    dates = sorted(membership["rebalance_date"].unique())
    t_prev, t_curr = dates[-3], dates[-2]  # leave one date free as a forward period for context

    def _tilt_weights_at(t):
        pos = prices.index.searchsorted(t)
        period_start = prices.index[pos]
        members = set(membership.loc[membership["rebalance_date"] == t, "symbol"])
        cap_row = market_caps.loc[market_caps.index[market_caps.index.searchsorted(t)]]
        ic_weights = trailing_ic_weights(factor_scores, fwd_returns, prices.index, t)
        composite = composite_score(factor_scores, ic_weights, period_start)
        return tilt_weights(members, composite, cap_row), composite, period_start

    prev_weights, _, _ = _tilt_weights_at(t_prev)
    curr_weights, composite_curr, period_start = _tilt_weights_at(t_curr)

    pos = all_returns.index.searchsorted(t_curr)
    trailing_returns = all_returns.iloc[pos - LOOKBACK_DAYS: pos]
    members = set(membership.loc[membership["rebalance_date"] == t_curr, "symbol"])
    candidates = [s for s in members if s in trailing_returns.columns and trailing_returns[s].notna().all()
                  and s in composite_curr.index and pd.notna(composite_curr[s])]

    # use the full-replication cap-weighted return as the trailing benchmark proxy for this snapshot
    cap_row = market_caps.loc[market_caps.index[market_caps.index.searchsorted(t_curr)]]
    bench_candidates = [s for s in members if s in cap_row.index and pd.notna(cap_row[s]) and cap_row[s] > 0]
    bench_weights = cap_row[bench_candidates] / cap_row[bench_candidates].sum()
    benchmark_returns = trailing_returns[bench_candidates].mul(bench_weights, axis=1).sum(axis=1)

    exposure_median = float(composite_curr[candidates].median())
    exposure_high = float(composite_curr[candidates].quantile(0.75))
    factor_targets = [-10.0, exposure_median, exposure_high]  # -10 ~= unconstrained (always feasible)

    print(f"Tracing Pareto frontier as of {t_curr.date()}, {len(candidates)} candidates, "
          f"prior holding = {int((prev_weights > 1e-6).sum())} names")
    frontier = trace_pareto_frontier(
        trailing_returns[candidates], benchmark_returns, prev_weights, composite_curr[candidates],
        factor_targets=factor_targets, turnover_budgets=TURNOVER_BUDGETS,
    )
    print(frontier.to_string(index=False))
    return frontier


if __name__ == "__main__":
    main("2016-01-01", "2026-08-07")
