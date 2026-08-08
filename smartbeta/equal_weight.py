"""Equal-weight smart-beta: 1/N across point-in-time constituents with a
tradeable price at the rebalance date. No return or fundamentals data used
-- this is the "dumbest" baseline every other smart-beta variant should beat
to justify its own added complexity, deliberately included to test whether
any smart-beta variant beats simpler alternatives once real costs are
included; equal-weight is the simplest possible one.
"""

from __future__ import annotations

import pandas as pd


def equal_weights(members: set[str], price_row: pd.Series) -> pd.Series:
    tradeable = sorted(s for s in members if s in price_row.index and pd.notna(price_row[s]) and price_row[s] > 0)
    if not tradeable:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(tradeable), index=tradeable)
