"""Quality-weighted smart-beta: a standalone weighting scheme (like
equal-weight, unlike multi-factor tilt) where weight is proportional to a
fundamentals-based quality score, not market cap at all.

Quality score = the coverage-aware mean of four cross-sectionally z-scored
fields from `data/fundamentals.py`: profitability (`returnOnEquity`,
`profitMargins`), earnings growth (`earningsGrowth`), and low leverage
(negated `debtToEquity`) -- the standard academic "quality" factor
ingredients (profitability, low leverage, earnings stability/growth), not
an arbitrary field selection. "Coverage-aware": a name missing one field
(e.g. no reported `debtToEquity`) is scored on the z-scored mean of
whichever fields it *does* have, rather than being dropped or scored 0 --
same averaging philosophy as alpha-signal-lab's composite factor blending.

Weight transform: `exp(z-scored quality score)`, clipped to +/-3 std before
exponentiating (bounding how much a single outlier name can dominate),
renormalized to sum to 1. This is a deliberate, disclosed choice among
several valid ways to turn a signed z-score into a positive weight -- it
keeps every name's weight strictly positive (no name is fully excluded for
being merely below-average quality, only down-weighted) while still being
meaningfully more sensitive to quality *magnitude* than a pure rank-based
weighting would be.

Inherits `data/fundamentals.py`'s disclosed limitation: fundamentals are a
current-day snapshot, not point-in-time historical values, so a historical
backtest of this strategy implicitly uses look-ahead fundamentals data at
every past rebalance date. That caveat is not repeated at every call site
below -- see that module's docstring for the full statement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY_FIELDS_POSITIVE = ("returnOnEquity", "profitMargins", "earningsGrowth")
QUALITY_FIELDS_NEGATIVE = ("debtToEquity",)  # lower is better -> sign-flipped before scoring
ZSCORE_CLIP = 3.0


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not std or np.isnan(std):
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def compute_quality_scores(fundamentals: pd.DataFrame) -> pd.Series:
    zscored = {}
    for field in QUALITY_FIELDS_POSITIVE:
        if field in fundamentals.columns:
            zscored[field] = _cross_sectional_zscore(fundamentals[field].astype(float))
    for field in QUALITY_FIELDS_NEGATIVE:
        if field in fundamentals.columns:
            zscored[f"neg_{field}"] = _cross_sectional_zscore(-fundamentals[field].astype(float))

    if not zscored:
        return pd.Series(dtype=float, index=fundamentals.index)
    return pd.DataFrame(zscored).mean(axis=1, skipna=True)


def quality_weights(members: set[str], quality_scores: pd.Series) -> pd.Series:
    available = quality_scores.reindex(sorted(members)).dropna()
    if available.empty:
        return pd.Series(dtype=float)

    transformed = np.exp(available.clip(-ZSCORE_CLIP, ZSCORE_CLIP))
    total = transformed.sum()
    return transformed / total if total > 0 else pd.Series(dtype=float)
