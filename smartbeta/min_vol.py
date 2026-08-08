"""Minimum-volatility smart-beta: long-only, fully-invested portfolio
minimizing variance under a Ledoit-Wolf shrunk covariance estimate,
reusing riskdesk's own `correlation/shrinkage.py` methodology directly
(`sklearn.covariance.LedoitWolf`, same default shrinkage target) rather than
the raw sample covariance -- with N candidate names often approaching or
exceeding the trailing observation count, the sample covariance is exactly
the ill-conditioned regime Ledoit-Wolf shrinkage exists to fix (see that
module's docstring for why: inflated largest eigenvalues, deflated
smallest, which here would otherwise push the optimizer toward
spuriously-low-vol combinations that are really just estimation noise).

    minimize_w   w' Sigma_shrunk w
    subject to   sum(w) == 1,  0 <= w <= max_weight

`max_weight` (default 5%) is a disclosed, real practical constraint, not an
artifact: unconstrained minimum-variance optimization tends to concentrate
heavily in whatever handful of names look least volatile in-sample, which
is exactly the degenerate outcome real min-vol indices (e.g. MSCI's
Minimum Volatility methodology) cap constituent weight specifically to
avoid.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

DEFAULT_MAX_WEIGHT = 0.05


def shrunk_covariance(trailing_returns: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf().fit(trailing_returns.to_numpy())
    return pd.DataFrame(lw.covariance_, index=trailing_returns.columns, columns=trailing_returns.columns)


def solve_min_variance_weights(
    trailing_returns: pd.DataFrame, max_weight: float = DEFAULT_MAX_WEIGHT
) -> pd.Series:
    clean = trailing_returns.dropna(axis=1, how="any")
    n = clean.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    if n * max_weight < 1.0:
        raise ValueError(f"{n} candidates x max_weight={max_weight} cannot sum to 1 -- raise max_weight or widen the candidate set")

    cov = shrunk_covariance(clean).to_numpy()
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if w.value is None:
        return pd.Series(dtype=float)

    weights = pd.Series(np.clip(w.value, 0, None), index=clean.columns)
    total = weights.sum()
    return weights / total if total > 0 else pd.Series(dtype=float)
