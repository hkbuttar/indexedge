"""ML-based sampling: LASSO-regularized regression of the benchmark's
trailing daily returns on ALL point-in-time constituents' trailing daily
returns (no market-cap preselection -- the whole point of this method is
that sparsity emerges endogenously from L1 shrinkage, not from filtering
candidates down first the way `optimized_sampling.py` and `stratified.py`
both do). `positive=True` keeps coefficients long-only, matching real index
construction; there's no native sum-to-1 constraint in LASSO, so weights are
renormalized to sum to 1 after fitting -- a standard post-hoc step in the
"index tracking via LASSO" literature (e.g. Fastrich et al.), not part of
LASSO's own optimization objective.

`sklearn.linear_model.lasso_path` computes the full regularization path in
one call, which is what lets `fit_for_target_count` pick whichever alpha on
that path gives (approximately) the requested name count, rather than
manually bisecting on alpha.

## A real ceiling on achievable name count, not a tunable bug

The number of nonzero coefficients anywhere on a LASSO path is bounded by
the number of observations used to fit it (a known property of the
coordinate-descent solution path for underdetermined regressions): with a
252-trading-day trailing lookback, `target_count` above roughly 200-250
simply isn't reachable no matter how low `eps` (the path's minimum-alpha
fraction) is pushed -- confirmed empirically here (eps=1e-5 reached only
~246 of 498 available candidates at one test date, barely more than
eps=1e-4's ~246, i.e. already at the ceiling). `sampling_evaluation.py`
deliberately caps its shared target-count grid well under this ceiling so
the three methods' tracking-error-vs-name-count curves stay comparable on
the same axes, rather than letting LASSO's line silently plateau.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path


def fit_for_target_count(
    candidate_returns: pd.DataFrame, benchmark_returns: pd.Series, target_count: int, n_alphas: int = 150, eps: float = 1e-4
) -> pd.Series:
    aligned = candidate_returns.join(benchmark_returns.rename("__benchmark__"), how="inner").dropna()
    if len(aligned) < 2 or candidate_returns.shape[1] == 0:
        return pd.Series(dtype=float)

    X = aligned[candidate_returns.columns].to_numpy()
    y = aligned["__benchmark__"].to_numpy()

    _, coefs, _ = lasso_path(X, y, positive=True, alphas=n_alphas, eps=eps)
    nnz_counts = (coefs != 0).sum(axis=0)
    idx = int(np.argmin(np.abs(nnz_counts - target_count)))
    coef = coefs[:, idx]

    weights = pd.Series(coef, index=candidate_returns.columns)
    weights = weights[weights > 0]
    total = weights.sum()
    return weights / total if total > 0 else pd.Series(dtype=float)
