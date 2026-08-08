"""Regime-conditional smart-beta performance: re-estimate each strategy's
return/vol/Sharpe separately per volatility regime, rather than pooling
every historical day into one number regardless of market condition --
`align_regime_labels`/`split_by_regime` reused directly from riskdesk's
`regime/conditional.py` (plain reindex, no forward-fill: a day inside the
21-day vol warm-up window stays unlabeled rather than silently inheriting a
neighboring day's regime).

This is the module that answers the real question: does
min-vol *specifically* outperform in the "volatile" regime -- the textbook
theoretical justification for the low-volatility anomaly -- or does that
effect wash out once returns are split by regime instead of pooled across
the whole backtest. `summarize_regime_conditional_performance` reports every
strategy in every regime side by side specifically so that comparison is
visible directly, not argued for in prose.

Sharpe here is annualized mean return / annualized vol with no risk-free
rate subtracted (this project has no cached risk-free series) -- a
disclosed zero-rate approximation, not a claim of true excess-return Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_DAYS_FOR_CONDITIONAL_STATS = 30
PERIODS_PER_YEAR = 252


def align_regime_labels(returns_index: pd.DatetimeIndex, regime_labels: pd.Series) -> pd.Series:
    return regime_labels.reindex(returns_index)


def split_by_regime(returns: pd.Series, aligned_labels: pd.Series) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for regime in aligned_labels.dropna().unique():
        mask = aligned_labels == regime
        out[regime] = returns[mask]
    return out


def regime_conditional_stats(returns: pd.Series, aligned_labels: pd.Series) -> tuple[dict[str, dict], list[str]]:
    notes: list[str] = []
    results: dict[str, dict] = {}
    for regime, series in split_by_regime(returns, aligned_labels).items():
        if len(series) < MIN_DAYS_FOR_CONDITIONAL_STATS:
            notes.append(f"{regime}: only {len(series)} days (< {MIN_DAYS_FOR_CONDITIONAL_STATS}) -- skipped, too few observations.")
            continue
        ann_return = float((1 + series.mean()) ** PERIODS_PER_YEAR - 1)
        ann_vol = float(series.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
        results[regime] = {
            "annualized_return": ann_return, "annualized_vol": ann_vol,
            "sharpe_zero_rate": ann_return / ann_vol if ann_vol > 0 else float("nan"),
            "n_days": len(series),
        }
    return results, notes


def summarize_regime_conditional_performance(
    returns_by_strategy: dict[str, pd.Series], regime_labels: pd.Series
) -> pd.DataFrame:
    records = []
    for strategy, returns in returns_by_strategy.items():
        aligned = align_regime_labels(returns.index, regime_labels)
        stats, _ = regime_conditional_stats(returns, aligned)
        for regime, values in stats.items():
            records.append({"strategy": strategy, "regime": regime, **values})
    return pd.DataFrame(records)
