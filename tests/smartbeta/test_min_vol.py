import numpy as np
import pandas as pd
import pytest

from smartbeta.min_vol import solve_min_variance_weights


def test_min_variance_favors_the_lower_vol_asset():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=200)
    low_vol = rng.normal(0.0002, 0.005, 200)
    high_vol = rng.normal(0.0002, 0.03, 200)
    returns = pd.DataFrame({"LOW": low_vol, "HIGH": high_vol, "MID": rng.normal(0.0002, 0.015, 200)}, index=dates)

    weights = solve_min_variance_weights(returns, max_weight=1.0)
    assert weights["LOW"] > weights["HIGH"]
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= -1e-9).all()


def test_max_weight_cap_is_respected():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=300)
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, (300, 10)), columns=[f"S{i}" for i in range(10)], index=dates)
    weights = solve_min_variance_weights(returns, max_weight=0.15)
    assert (weights <= 0.15 + 1e-6).all()
    assert weights.sum() == pytest.approx(1.0)


def test_raises_when_max_weight_infeasible_for_candidate_count():
    dates = pd.date_range("2024-01-01", periods=50)
    returns = pd.DataFrame(np.random.randn(50, 5) * 0.01, columns=list("ABCDE"), index=dates)
    with pytest.raises(ValueError):
        solve_min_variance_weights(returns, max_weight=0.1)  # 5 * 0.1 = 0.5 < 1.0


def test_drops_candidates_with_any_missing_returns():
    dates = pd.date_range("2024-01-01", periods=50)
    returns = pd.DataFrame(np.random.randn(50, 3) * 0.01, columns=["A", "B", "C"], index=dates)
    returns.loc[dates[5], "C"] = np.nan
    weights = solve_min_variance_weights(returns, max_weight=1.0)
    assert "C" not in weights.index
