"""Multi-objective portfolio construction: jointly balances tracking error,
turnover, and factor exposure in a single convex problem, rather than
picking a name-count or a factor tilt strength first and only checking
turnover/tracking error afterward the way Steps 3-4 do.

## Why epsilon-constraint, not weighted-sum scalarization

The plan calls for either "weighted-sum or Pareto-frontier approach."
Weighted-sum scalarization (minimize `(1-gamma)*TE + gamma*turnover`) was
tried first and rejected: tracking-error-variance terms here are ~1e-4 to
1e-6 in magnitude (squared daily return differences) while L1 turnover
terms are O(1) (weight changes sum to at most 2), so a gamma sweep would be
dominated entirely by turnover's scale unless both terms were first
normalized by some reference value -- an extra, somewhat arbitrary,
disclosed choice. Epsilon-constraint sidesteps this: fix a turnover BUDGET
(directly interpretable: "how much trading am I willing to do"), minimize
tracking error subject to staying under it, and sweep the budget to trace
the exact frontier. This is also the more directly actionable form of the
question a real portfolio manager asks ("what's the best tracking error I
can get for X% turnover"), and for a convex problem like this one it traces
the same frontier weighted-sum would, without the scaling problem.

    minimize_w   || R @ w - b ||_2^2
    subject to   sum(w) == 1
                 0 <= w <= max_weight
                 f' @ w >= factor_target      (portfolio factor exposure floor)
                 || w - w_prev ||_1 <= turnover_budget

`f` is the multi-factor composite score from `smartbeta/multi_factor.py`
(same IC-weighted composite, reused rather than a new factor definition
invented for this step) and `w_prev` is the portfolio's weights entering
this rebalance -- turnover is measured against an actual prior holding, not
against zero.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


def solve_for_turnover_budget(
    candidate_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    prev_weights: pd.Series,
    factor_exposure: pd.Series,
    factor_target: float,
    turnover_budget: float,
    max_weight: float = 1.0,
) -> pd.Series:
    candidates = [c for c in candidate_returns.columns if c in factor_exposure.index and pd.notna(factor_exposure[c])]
    if not candidates:
        return pd.Series(dtype=float)

    aligned = candidate_returns[candidates].join(benchmark_returns.rename("__benchmark__"), how="inner").dropna()
    if len(aligned) < 2:
        return pd.Series(dtype=float)

    R = aligned[candidates].to_numpy()
    b = aligned["__benchmark__"].to_numpy()
    f = factor_exposure[candidates].to_numpy()
    w_prev = prev_weights.reindex(candidates).fillna(0.0).to_numpy()
    n = len(candidates)

    if n * max_weight < 1.0:
        return pd.Series(dtype=float)

    w = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(R @ w - b))
    constraints = [
        cp.sum(w) == 1, w >= 0, w <= max_weight,
        f @ w >= factor_target,
        cp.norm1(w - w_prev) <= turnover_budget,
    ]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return pd.Series(dtype=float)

    if w.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        return pd.Series(dtype=float)

    weights = pd.Series(np.clip(w.value, 0, None), index=candidates)
    total = weights.sum()
    return weights / total if total > 0 else pd.Series(dtype=float)


def realized_tracking_error(weights: pd.Series, candidate_returns: pd.DataFrame, benchmark_returns: pd.Series) -> float:
    aligned = candidate_returns[weights.index].join(benchmark_returns.rename("__benchmark__"), how="inner").dropna()
    active = aligned[weights.index].to_numpy() @ weights.to_numpy() - aligned["__benchmark__"].to_numpy()
    return float(np.std(active, ddof=1) * np.sqrt(252))


def trace_pareto_frontier(
    candidate_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    prev_weights: pd.Series,
    factor_exposure: pd.Series,
    factor_targets: list[float],
    turnover_budgets: list[float],
    max_weight: float = 1.0,
) -> pd.DataFrame:
    records = []
    for target in factor_targets:
        for budget in turnover_budgets:
            weights = solve_for_turnover_budget(
                candidate_returns, benchmark_returns, prev_weights, factor_exposure,
                target, budget, max_weight,
            )
            if weights.empty:
                continue
            realized_turnover = float((weights.reindex(weights.index.union(prev_weights.index)).fillna(0)
                                        - prev_weights.reindex(weights.index.union(prev_weights.index)).fillna(0)).abs().sum())
            realized_exposure = float((weights * factor_exposure.reindex(weights.index)).sum())
            records.append({
                "factor_target": target, "turnover_budget": budget,
                "realized_turnover": realized_turnover, "realized_factor_exposure": realized_exposure,
                "tracking_error": realized_tracking_error(weights, candidate_returns, benchmark_returns),
                "n_holdings": int((weights > 1e-6).sum()),
            })
    return pd.DataFrame(records)
