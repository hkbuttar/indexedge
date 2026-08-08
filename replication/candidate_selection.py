"""Shared candidate-preselection helper used by both the optimization-based
and stratified sampling methods. LASSO (the third
method) deliberately does NOT use this -- its whole point is that sparsity
emerges endogenously from L1 regularization over the *full* membership, not
from a market-cap preselection step, which is the real methodological
difference being tested (see `lasso_sampling.py`).
"""

from __future__ import annotations

import pandas as pd


def top_n_by_market_cap(members: set[str], market_cap_row: pd.Series, n: int) -> list[str]:
    available = [s for s in members if s in market_cap_row.index and pd.notna(market_cap_row[s]) and market_cap_row[s] > 0]
    ranked = market_cap_row[available].sort_values(ascending=False)
    return list(ranked.index[:n])
