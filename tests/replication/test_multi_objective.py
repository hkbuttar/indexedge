import numpy as np
import pandas as pd

from replication.multi_objective import (
    realized_tracking_error,
    solve_for_turnover_budget,
    trace_pareto_frontier,
)


def _synthetic_setup(n_assets=15, n_days=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days)
    symbols = [f"S{i}" for i in range(n_assets)]
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (n_days, n_assets)), columns=symbols, index=dates)
    benchmark = returns.mean(axis=1) + rng.normal(0, 0.0005, n_days)
    prev_weights = pd.Series(1.0 / n_assets, index=symbols)
    factor_exposure = pd.Series(rng.normal(0, 1, n_assets), index=symbols)
    return returns, benchmark, prev_weights, factor_exposure


def test_relaxing_turnover_budget_never_increases_tracking_error():
    """The core correctness property the plan calls for: relaxing a
    constraint (larger turnover budget = strictly larger feasible region,
    since the old feasible set is a subset of the new one) can only weakly
    improve -- never worsen -- the achieved tracking error. This must hold
    by construction for any correct convex epsilon-constraint solve."""
    returns, benchmark, prev_weights, factor_exposure = _synthetic_setup()
    budgets = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 2.0]

    tes = []
    for budget in budgets:
        weights = solve_for_turnover_budget(returns, benchmark, prev_weights, factor_exposure, factor_target=-10, turnover_budget=budget)
        assert not weights.empty
        tes.append(realized_tracking_error(weights, returns, benchmark))

    # allow tiny numerical solver noise, but must be non-increasing overall
    for earlier, later in zip(tes, tes[1:]):
        assert later <= earlier + 1e-6


def test_factor_exposure_constraint_is_respected():
    returns, benchmark, prev_weights, factor_exposure = _synthetic_setup()
    target = float(factor_exposure.median())
    weights = solve_for_turnover_budget(returns, benchmark, prev_weights, factor_exposure, factor_target=target, turnover_budget=2.0)
    assert not weights.empty
    achieved = float((weights * factor_exposure.reindex(weights.index)).sum())
    assert achieved >= target - 1e-6


def test_tight_turnover_budget_keeps_weights_close_to_previous():
    returns, benchmark, prev_weights, factor_exposure = _synthetic_setup()
    weights = solve_for_turnover_budget(returns, benchmark, prev_weights, factor_exposure, factor_target=-10, turnover_budget=0.02)
    assert not weights.empty
    turnover = (weights.reindex(prev_weights.index).fillna(0) - prev_weights).abs().sum()
    assert turnover <= 0.02 + 1e-6


def test_infeasible_factor_target_returns_empty():
    returns, benchmark, prev_weights, factor_exposure = _synthetic_setup()
    impossible_target = float(factor_exposure.max()) + 100  # no portfolio can achieve this
    weights = solve_for_turnover_budget(returns, benchmark, prev_weights, factor_exposure, factor_target=impossible_target, turnover_budget=2.0)
    assert weights.empty


def test_trace_pareto_frontier_produces_a_row_per_feasible_combination():
    returns, benchmark, prev_weights, factor_exposure = _synthetic_setup()
    frontier = trace_pareto_frontier(
        returns, benchmark, prev_weights, factor_exposure,
        factor_targets=[-10, 0.0], turnover_budgets=[0.1, 0.5, 1.5],
    )
    assert set(frontier["factor_target"].unique()) <= {-10, 0.0}
    assert (frontier["realized_turnover"] <= frontier["turnover_budget"] + 1e-6).all()
    # within each factor target, tracking error must be non-increasing as the turnover budget relaxes
    for _, group in frontier.groupby("factor_target"):
        tes = group.sort_values("turnover_budget")["tracking_error"].tolist()
        for earlier, later in zip(tes, tes[1:]):
            assert later <= earlier + 1e-6
