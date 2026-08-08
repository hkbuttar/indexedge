import numpy as np
import pandas as pd
import pytest

from results.comparison import average_portfolio_metadata, build_comparison_table, derive_honest_findings


def test_average_portfolio_metadata_uses_one_way_turnover():
    weights = {
        pd.Timestamp("2024-01-01"): pd.Series({"A": 0.5, "B": 0.5}),
        pd.Timestamp("2024-04-01"): pd.Series({"A": 0.25, "C": 0.75}),
    }
    names, turnover = average_portfolio_metadata(weights)
    assert names == 2
    assert turnover == pytest.approx(0.75)


def test_comparison_table_has_overall_and_regime_rows_with_intervals():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=120)
    benchmark = pd.Series(rng.normal(0.0003, 0.008, len(dates)), index=dates)
    strategy = benchmark + pd.Series(rng.normal(0.0001, 0.002, len(dates)), index=dates)
    labels = pd.Series(np.repeat(["calm", "normal", "volatile"], 40), index=dates)
    table = build_comparison_table(
        {"full_replication": benchmark, "quality": strategy},
        {10_000_000: {"full_replication": benchmark, "quality": strategy}},
        benchmark,
        labels,
        {},
        block_length=10,
        n_resamples=30,
        seed=1,
    )
    assert set(table["regime"]) == {"all", "calm", "normal", "volatile"}
    assert table["cost_adjusted_return_ci_low"].notna().all()
    benchmark_rows = table[table["strategy"] == "full_replication"]
    assert np.allclose(benchmark_rows["tracking_error"], 0.0)


def test_findings_are_derived_from_values_and_compare_lasso_to_optimization():
    table = pd.DataFrame(
        [
            {"strategy": "full_replication", "strategy_family": "benchmark", "aum": 10.0,
             "regime": "all", "cost_adjusted_return": 0.08},
            {"strategy": "quality", "strategy_family": "smart_beta", "aum": 10.0,
             "regime": "all", "cost_adjusted_return": 0.10},
        ]
    )
    sampling = pd.DataFrame(
        [
            {"method": "lasso", "target_n": 50, "mean_tracking_error": 0.03},
            {"method": "optimization", "target_n": 50, "mean_tracking_error": 0.02},
        ]
    )
    findings = derive_honest_findings(table, sampling)
    assert any("quality beat" in finding for finding in findings)
    assert any("LASSO minus optimization" in finding for finding in findings)
