"""Portfolio-level capacity/impact cost: applies `impact.py`'s per-name
square-root-law cost to every trade a rebalance requires (from `prev_weights`
to `weights`, sized in real dollars at a disclosed hypothetical AUM), and
aggregates to a total cost as a fraction of AUM.

`prev_weights=None` (or all-zero) prices the cost of *establishing* a
portfolio from cash; a real `prev_weights` prices the cost of one
*rebalancing* trade -- the same distinction the plan's Step 7 draws
("establishing and rebalancing"). Because cost_fraction scales with
sqrt(participation_rate) and participation_rate scales linearly with AUM,
total dollar cost scales roughly as AUM^1.5, not AUM^1 -- cost as a
*fraction* of the portfolio still grows (as sqrt(AUM)) even though it's a
sublinear relationship, which is the concrete "a strategy that looks best
on paper can lose its edge at realistic size" mechanism the plan points at.

Names with no usable liquidity data (missing dollar-volume or realized-vol
estimate) are excluded from the cost total and reported separately via
`notes`, not silently zeroed -- consistent with every other disclosed-gap
pattern in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from liquidity.impact import TransactionCostEstimate, estimate_transaction_cost


@dataclass
class PortfolioCostEstimate:
    total_cost_fraction: float  # total dollar cost / AUM
    total_dollar_cost: float
    aum: float
    per_symbol: dict[str, TransactionCostEstimate] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def estimate_portfolio_trade_cost(
    weights: pd.Series,
    prev_weights: pd.Series | None,
    aum: float,
    daily_vol_by_symbol: pd.Series,
    dollar_volume_by_symbol: pd.Series,
) -> PortfolioCostEstimate:
    prev_weights = prev_weights if prev_weights is not None else pd.Series(dtype=float)
    all_symbols = sorted(set(weights.index) | set(prev_weights.index))

    per_symbol: dict[str, TransactionCostEstimate] = {}
    notes: list[str] = []
    total_dollar_cost = 0.0

    for symbol in all_symbols:
        trade_weight = weights.get(symbol, 0.0) - prev_weights.get(symbol, 0.0)
        if abs(trade_weight) < 1e-12:
            continue
        trade_value = trade_weight * aum

        vol = daily_vol_by_symbol.get(symbol)
        dollar_vol = dollar_volume_by_symbol.get(symbol)
        cost = estimate_transaction_cost(trade_value, vol, dollar_vol) if vol is not None and dollar_vol is not None else None

        if cost is None:
            notes.append(f"{symbol}: no usable liquidity data -- excluded from cost total.")
            continue

        per_symbol[symbol] = cost
        total_dollar_cost += cost.dollar_cost

    return PortfolioCostEstimate(
        total_cost_fraction=total_dollar_cost / aum if aum > 0 else float("nan"),
        total_dollar_cost=total_dollar_cost, aum=aum, per_symbol=per_symbol, notes=notes,
    )
