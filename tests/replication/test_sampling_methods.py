import numpy as np
import pandas as pd
import pytest

from replication.candidate_selection import top_n_by_market_cap
from replication.lasso_sampling import fit_for_target_count
from replication.optimized_sampling import solve_min_tracking_error_weights
from replication.stratified import stratified_sample_weights


def test_top_n_by_market_cap_ranks_correctly():
    row = pd.Series({"A": 300.0, "B": 100.0, "C": 200.0, "D": 50.0})
    assert top_n_by_market_cap({"A", "B", "C", "D"}, row, 2) == ["A", "C"]


def test_top_n_by_market_cap_excludes_missing_and_nonpositive():
    row = pd.Series({"A": 100.0, "B": -5.0, "C": np.nan})
    assert top_n_by_market_cap({"A", "B", "C", "MISSING"}, row, 5) == ["A"]


def test_stratified_weights_sum_to_one_and_pick_largest_per_bucket():
    members = {"A", "B", "C", "D"}
    row = pd.Series({"A": 100.0, "B": 90.0, "C": 10.0, "D": 5.0})
    sectors = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Health"}
    weights = stratified_sample_weights(members, row, sectors, buckets_per_sector=1)
    # one bucket per sector -> each sector's largest-cap name represents the whole sector
    assert set(weights.index) == {"A", "C"}
    assert weights["A"] == pytest.approx(190 / 205)
    assert weights.sum() == pytest.approx(1.0)


def test_stratified_excludes_symbols_with_no_sector():
    members = {"A", "NOSECTOR"}
    row = pd.Series({"A": 100.0, "NOSECTOR": 100.0})
    weights = stratified_sample_weights(members, row, {"A": "Tech"}, buckets_per_sector=1)
    assert set(weights.index) == {"A"}


def test_optimized_sampling_recovers_exact_replica_when_benchmark_is_one_asset():
    dates = pd.date_range("2024-01-01", periods=60)
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.01, 60)
    candidate_returns = pd.DataFrame({"A": a, "B": rng.normal(0, 0.01, 60)}, index=dates)
    benchmark = pd.Series(a, index=dates)  # benchmark == candidate A exactly

    weights = solve_min_tracking_error_weights(candidate_returns, benchmark)
    assert weights["A"] == pytest.approx(1.0, abs=1e-3)
    assert weights.get("B", 0.0) == pytest.approx(0.0, abs=1e-3)


def test_optimized_sampling_weights_are_long_only_and_sum_to_one():
    dates = pd.date_range("2024-01-01", periods=100)
    rng = np.random.default_rng(2)
    candidate_returns = pd.DataFrame(rng.normal(0, 0.01, (100, 5)), columns=list("ABCDE"), index=dates)
    benchmark = pd.Series(candidate_returns.mean(axis=1) + rng.normal(0, 0.001, 100), index=dates)

    weights = solve_min_tracking_error_weights(candidate_returns, benchmark)
    assert (weights >= -1e-9).all()
    assert weights.sum() == pytest.approx(1.0)


def test_lasso_sparsity_increases_as_target_count_decreases():
    dates = pd.date_range("2024-01-01", periods=200)
    rng = np.random.default_rng(3)
    candidate_returns = pd.DataFrame(rng.normal(0, 0.01, (200, 20)), columns=[f"S{i}" for i in range(20)], index=dates)
    benchmark = pd.Series(candidate_returns.iloc[:, :3].mean(axis=1), index=dates)

    sparse = fit_for_target_count(candidate_returns, benchmark, target_count=3)
    dense = fit_for_target_count(candidate_returns, benchmark, target_count=15)
    assert len(sparse) <= len(dense)
    assert sparse.sum() == pytest.approx(1.0)


def test_lasso_recovers_true_drivers_with_low_noise():
    dates = pd.date_range("2024-01-01", periods=300)
    rng = np.random.default_rng(4)
    candidate_returns = pd.DataFrame(rng.normal(0, 0.01, (300, 10)), columns=[f"S{i}" for i in range(10)], index=dates)
    true_weights = pd.Series({"S0": 0.6, "S1": 0.4})
    benchmark = candidate_returns[["S0", "S1"]].dot(true_weights) + rng.normal(0, 0.0001, 300)

    weights = fit_for_target_count(candidate_returns, benchmark, target_count=2)
    assert set(weights.index) <= {"S0", "S1", "S2", "S3"}  # allow minor path granularity, not arbitrary names
    assert weights.sum() == pytest.approx(1.0)
