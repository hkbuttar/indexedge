"""Real dollar-cost estimate of trading a position, reusing execedge's
square-root-law market-impact model directly rather than inventing a fresh
cost formula -- specifically riskdesk's own already-adapted version
(`liquidity/impact.py` in that sibling repo), which itself reuses
execedge's `algos/impact_calibration.py`:

    cost_fraction = Y * daily_realized_vol * sqrt(participation_rate)

`Y=1.0` is kept as execedge's own disclosed "textbook order-of-magnitude"
convention, not a value fitted here: neither the Almgren & Chriss (2000)
nor the Almgren-Thum-Hauptmann-Li (2005) papers' exact fitted linear-impact
coefficients were extractable in execedge's environment (no PDF-to-text
tooling, two attempts failed -- see that module's docstring), so it falls
back to the square-root law's functional form with Y as an order-1 constant
(independent studies converge on roughly 0.5-1.5, not a single precisely-
sourced number). Reusing the same disclosed gap here rather than presenting
a more confident-looking number this project has no better basis for.

Participation rate is in DOLLAR terms, riskdesk's own convention:
`|trade_dollar_value| / average_daily_dollar_volume`, uniform regardless of
share price -- not share/contract counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SQRT_LAW_Y = 1.0


@dataclass
class TransactionCostEstimate:
    cost_fraction: float  # Y * daily_realized_vol * sqrt(participation_rate)
    dollar_cost: float
    participation_rate: float
    avg_daily_dollar_volume: float


def avg_daily_dollar_volume(prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 63) -> pd.Series:
    """Trailing-window average daily dollar volume per symbol (close *
    volume, ~3 trading months by default -- execedge/riskdesk's own default
    lookback for this kind of liquidity estimate)."""
    dollar_volume = prices.mul(volumes)
    return dollar_volume.tail(window).mean()


def estimate_transaction_cost(
    trade_dollar_value: float, daily_realized_vol: float, avg_daily_dollar_vol: float, y: float = SQRT_LAW_Y
) -> TransactionCostEstimate | None:
    if avg_daily_dollar_vol is None or avg_daily_dollar_vol <= 0 or pd.isna(daily_realized_vol):
        return None

    participation_rate = abs(trade_dollar_value) / avg_daily_dollar_vol
    cost_fraction = y * daily_realized_vol * np.sqrt(participation_rate)
    return TransactionCostEstimate(
        cost_fraction=cost_fraction,
        dollar_cost=abs(trade_dollar_value) * cost_fraction,
        participation_rate=participation_rate,
        avg_daily_dollar_volume=avg_daily_dollar_vol,
    )
