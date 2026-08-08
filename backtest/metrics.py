"""Backtest performance metrics: CAGR, Sharpe, Sortino, drawdown, win rate.
Ported directly from pairtrade-lab-1's `backtest/metrics.py` (same formulas,
same 252-trading-day annualization, same zero-vol epsilon guard) for
methodological continuity with the statistical-rigor standard the plan asks
this step to match. Tracking error is intentionally NOT duplicated here --
`risk/tracking_error.py` already owns that definition project-wide, and
`backtest/bootstrap.py` calls it directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
ZERO_VOL_EPSILON = 1e-9


def running_drawdown(equity_curve: pd.Series) -> pd.Series:
    peak = equity_curve.cummax()
    return 1 - equity_curve / peak


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = len(equity) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns.dropna() - risk_free / TRADING_DAYS_PER_YEAR
    if excess.empty or excess.std() < ZERO_VOL_EPSILON:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns.dropna() - risk_free / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    if downside.empty or downside.std() < ZERO_VOL_EPSILON:
        return float("nan")
    return float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float(running_drawdown(equity).max())


def win_rate(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float((clean > 0).mean())


def compute_metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change()
    return {
        "cagr": cagr(equity), "sharpe_ratio": sharpe_ratio(returns), "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity), "win_rate": win_rate(returns),
    }
