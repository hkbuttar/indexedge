"""Full cap-weighted replication: at each real quarterly rebalance date, buy
every point-in-time constituent in proportion to its market cap; hold share
counts fixed (buy-and-hold, weights drifting with price) until the next
rebalance. This is the project's correctness baseline -- with the full,
correctly point-in-time constituent set and no sampling, tracking error
against the real index should be near zero. Everything in later steps
(optimized sampling, smart-beta, multi-objective) is judged by how much
tracking error, turnover, or cost it trades away relative to this baseline.

Two disclosed approximations, both inherited from upstream modules rather
than new here: (1) prices are forward-filled before simulating, so a name
that stops trading mid-period (acquired, delisted) is held at its last
known price until the next rebalance rather than cashed out at a real deal
price -- `replication.market_cap`'s and `data.prices`' docstrings cover the
data-availability side of this. (2) a constituent with no usable market cap
at a rebalance date (missing shares-outstanding data) is excluded from that
rebalance's weights, which are renormalized over whatever's left --
`coverage_notes` reports how many names and what fraction of intended
membership this affected at every rebalance, so a low-coverage period is
visible rather than silently absorbed into the weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RebalanceCoverage:
    rebalance_date: pd.Timestamp
    intended_members: int
    weighted_members: int

    @property
    def coverage_fraction(self) -> float:
        return self.weighted_members / self.intended_members if self.intended_members else float("nan")


def rebalance_weights(members: set[str], market_cap_row: pd.Series) -> pd.Series:
    available = [s for s in members if s in market_cap_row.index and pd.notna(market_cap_row[s]) and market_cap_row[s] > 0]
    caps = market_cap_row[available].astype(float)
    if caps.sum() <= 0:
        return pd.Series(dtype=float)
    return caps / caps.sum()


def simulate_cap_weighted_replication(
    prices: pd.DataFrame, market_caps: pd.DataFrame, membership_history: pd.DataFrame
) -> tuple[pd.Series, pd.Series, list[RebalanceCoverage]]:
    """Returns (portfolio_value indexed by trading date, normalized to 1.0 at
    the first rebalance date; daily returns derived from it; per-rebalance
    coverage diagnostics).

    Indexes into a single numpy array by integer position rather than
    repeated pandas `.loc[period_dates, weights.index]` label-based fancy
    indexing per rebalance period -- a real fix, not a style preference:
    profiling a production out-of-memory failure traced ~700MB of peak RSS
    growth to exactly this loop, one rebalance period at a time, even though
    `gc.collect()` confirmed the per-iteration objects *were* being freed at
    the Python level. Pandas' label-based `.loc` with a list of names
    rebuilds a new Index/hash table every call; that churn of many
    differently-shaped temporary allocations fragments the memory allocator
    (freed memory kept mapped for reuse rather than returned to the OS),
    which shows up as growing RSS despite no real leak. Plain integer
    indexing into one already-allocated numpy array avoids that churn.
    """
    prices = prices.ffill()
    rebalance_dates = sorted(membership_history["rebalance_date"].unique())
    if not rebalance_dates:
        raise ValueError("membership_history has no rebalance dates")

    trading_dates = prices.index
    price_values = prices.to_numpy()
    column_position = {symbol: i for i, symbol in enumerate(prices.columns)}

    value_segments = []
    coverage: list[RebalanceCoverage] = []
    portfolio_value = 1.0

    for i, rebal_date in enumerate(rebalance_dates):
        pos = trading_dates.searchsorted(rebal_date)
        if pos >= len(trading_dates):
            break
        period_start = trading_dates[pos]
        period_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else trading_dates[-1]

        members = set(membership_history.loc[membership_history["rebalance_date"] == rebal_date, "symbol"])
        cap_row = market_caps.loc[period_start] if period_start in market_caps.index else market_caps.reindex([period_start]).iloc[0]
        weights = rebalance_weights(members, cap_row)
        coverage.append(RebalanceCoverage(rebal_date, len(members), len(weights)))
        if weights.empty:
            continue

        col_idx = np.fromiter((column_position[s] for s in weights.index), dtype=np.intp, count=len(weights.index))
        end_pos = trading_dates.searchsorted(period_end, side="right")
        row_idx = slice(pos, end_pos)

        price_at_start = price_values[pos, col_idx]
        shares = (portfolio_value * weights.to_numpy()) / price_at_start

        period_prices = price_values[row_idx][:, col_idx]
        # np.nansum, not `@` (matrix multiply) -- see
        # `replication.sampling_evaluation.simulate_holding_period`'s
        # docstring for the real bug this avoids: pandas' original
        # `.mul(shares, axis=1).sum(axis=1)` skips NaN per element
        # (skipna=True default), so one symbol with a NaN price contributes
        # zero for that date rather than poisoning every date's sum via a
        # plain dot product. `rebalance_weights` already filters to symbols
        # with a valid market cap at the rebalance date, so this is a
        # defensive-correctness fix here rather than one caught failing on
        # real data the way `simulate_holding_period`'s was -- relying on
        # today's candidate-filtering happening to avoid the gap is fragile.
        period_values_arr = np.nansum(period_prices * shares, axis=1)
        period_values = pd.Series(period_values_arr, index=trading_dates[row_idx])
        value_segments.append(period_values)
        portfolio_value = float(period_values_arr[-1])

    value_series = pd.concat(value_segments).sort_index()
    value_series = value_series[~value_series.index.duplicated(keep="first")]
    returns = value_series.pct_change().dropna()
    return value_series, returns, coverage


def compute_weights_by_date(membership: pd.DataFrame, market_caps: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    """Full replication's own weight vector at every rebalance date --
    needed wherever full replication is treated as a strategy in its own
    right (e.g. `costs/transaction_costs.py` cost-adjustment),
    not just as the benchmark other strategies are measured against."""
    weights_by_date = {}
    for t in sorted(membership["rebalance_date"].unique()):
        members = set(membership.loc[membership["rebalance_date"] == t, "symbol"])
        cap_row = market_caps.loc[market_caps.index[market_caps.index.searchsorted(t)]]
        weights_by_date[t] = rebalance_weights(members, cap_row)
    return weights_by_date


def summarize_coverage(coverage: list[RebalanceCoverage]) -> pd.DataFrame:
    return pd.DataFrame([
        {"rebalance_date": c.rebalance_date, "intended_members": c.intended_members,
         "weighted_members": c.weighted_members, "coverage_fraction": c.coverage_fraction}
        for c in coverage
    ])
