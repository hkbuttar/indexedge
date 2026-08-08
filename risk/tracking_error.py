"""Tracking error and related active-return statistics, shared by every
strategy variant built in this project (full replication, optimized
sampling, smart-beta, multi-objective, regime-conditional, bootstrap). One
definition used everywhere so numbers across steps stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_PERIODS_PER_YEAR = 252


def active_returns(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    """Aligns on the intersection of both indices (inner join) before
    differencing -- silently comparing on a mismatched calendar would
    understate or fabricate tracking error rather than error out."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner")
    aligned.columns = ["portfolio", "benchmark"]
    return aligned["portfolio"] - aligned["benchmark"]


def annualized_tracking_error(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> float:
    diffs = active_returns(portfolio_returns, benchmark_returns)
    if len(diffs) < 2:
        return float("nan")
    return float(diffs.std(ddof=1) * np.sqrt(periods_per_year))


@dataclass
class TrackingSummary:
    tracking_error_annualized: float
    mean_active_return_annualized: float
    correlation: float
    n_periods: int
    cumulative_active_return: float


def summarize_tracking(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> TrackingSummary:
    diffs = active_returns(portfolio_returns, benchmark_returns)
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner")
    aligned.columns = ["portfolio", "benchmark"]

    cumulative_portfolio = (1 + aligned["portfolio"]).prod() - 1
    cumulative_benchmark = (1 + aligned["benchmark"]).prod() - 1

    return TrackingSummary(
        tracking_error_annualized=annualized_tracking_error(portfolio_returns, benchmark_returns, periods_per_year),
        mean_active_return_annualized=float(diffs.mean() * periods_per_year),
        correlation=float(aligned["portfolio"].corr(aligned["benchmark"])),
        n_periods=len(diffs),
        cumulative_active_return=float(cumulative_portfolio - cumulative_benchmark),
    )
