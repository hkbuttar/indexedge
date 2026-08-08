"""Optimization-based sampling: preselect a fixed candidate universe of N
names by market cap (`candidate_selection.top_n_by_market_cap`), then solve
the convex QP

    minimize_w   || R @ w - b ||_2^2
    subject to   sum(w) == 1,  w >= 0

over trailing daily returns R (T x N, candidates) and benchmark daily
returns b (T,) -- i.e. minimize realized historical tracking-error variance
over the lookback window, long-only, fully invested. Name count (N) is
fixed by the preselection step, not jointly optimized with the weights:
true cardinality-constrained ("pick the best N *and* their weights")
tracking-error minimization is a mixed-integer QP, NP-hard at this
universe size. Fixing the candidate set by market cap first and solving an
exact convex QP for weights within it is the standard simplification real
index-sampling desks use, not a shortcut invented for this project -- and
it's exactly what makes this method's tracking-error-vs-name-count curve
comparable to LASSO's, which selects its own N differently (see
`lasso_sampling.py`).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


def solve_min_tracking_error_weights(candidate_returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    aligned = candidate_returns.join(benchmark_returns.rename("__benchmark__"), how="inner").dropna()
    if len(aligned) < 2 or candidate_returns.shape[1] == 0:
        return pd.Series(dtype=float)

    R = aligned[candidate_returns.columns].to_numpy()
    b = aligned["__benchmark__"].to_numpy()
    n = R.shape[1]

    w = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(R @ w - b))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if w.value is None:
        return pd.Series(dtype=float)

    weights = pd.Series(np.clip(w.value, 0, None), index=candidate_returns.columns)
    total = weights.sum()
    return weights / total if total > 0 else pd.Series(dtype=float)
